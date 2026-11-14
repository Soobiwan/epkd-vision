"""Decisive check of the training update on torch_tpu, per compile scope (none/model/fwdbwd/step).
Per step, with plain SGD (momentum 0, weight decay 0, so the expected update is exactly -lr * grad):
  grad_rel = ||g_dev - g_ref|| / ||g_ref||  where g_ref = CPU fp32 reference loop at the SAME weights (copied from the device)
  upd_rel  = ||(p_after - p_before) + lr * g_dev|| / ||lr * g_dev||   (0 = the optimizer applied the update; 1 = lost)
  run_rel  = BN running-stat change vs the reference loop's change from the same starting stats
(Multi-step trajectories are not compared: tiny ghost-BN groups make them chaotic.)  Also native BN running stats.
  python3 bench/tpu_update_probe.py --device tpu
"""
from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import epkd_colab  # noqa: F401,E402
from epkd_colab.ghost_bn import GhostHolder, convert_to_ghost_bn  # noqa: E402
from epkd_colab.step import StepFn  # noqa: E402
from model.python.resnet_imagenet import ResNet18, ResNet34  # noqa: E402
from tests.test_step_equivalence import build_step_batch  # noqa: E402
from tgsc_kd.trainer import KDLoss  # noqa: E402
from tgsc_kd.utils import forward_logits  # noqa: E402


def norm(ts):
    return float(torch.sqrt(sum((t.double() ** 2).sum() for t in ts)))


def ref_grads_and_running(model, teacher, sb, mean, std, kd, lam, G, S):
    """Reference loop (sequential micro-batches, loss*n/512, anchors as their own BN batch): grads + new running stats."""
    model.train()
    model.zero_grad(set_to_none=True)
    for g in range(G):
        n, a = sb.counts_v[g], sb.counts_a[g]
        if n == 0:
            continue
        x = ((sb.xv[g * S: g * S + n].float() / 255.0) - mean) / std
        y = sb.yv[g * S: g * S + n]
        with torch.no_grad():
            t = forward_logits(teacher, x)
        loss = kd(forward_logits(model, x), t, y, weights=torch.ones(n))["loss"]
        if sb.anchor is not None:
            xa = ((sb.xa[g * S: g * S + a].float() / 255.0) - mean) / std
            ya = sb.ya[g * S: g * S + a]
            with torch.no_grad():
                ta = forward_logits(teacher, xa)
            loss = loss + lam * kd(forward_logits(model, xa), ta, ya, weights=None)["loss"]
        (loss * n / (G * S)).backward()
    grads = [p.grad.detach().clone() for p in model.parameters()]
    running = [b.detach().clone() for nme, b in model.named_buffers() if "running" in nme]
    return grads, running


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="tpu")
    ap.add_argument("--scopes", default="none,model,fwdbwd,step")
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--crop", type=int, default=64)
    ap.add_argument("--anchor", type=int, default=96)
    ap.add_argument("--dtypes", default="fp32,bf16", help="fp32 = autocast off; bf16 = torch.autocast(device, bf16) as in training")
    a = ap.parse_args()
    if a.device == "tpu":
        import torch_tpu  # noqa: F401
    dev = torch.device(a.device)
    torch.set_float32_matmul_precision("highest")
    torch.manual_seed(0)

    bn = nn.BatchNorm2d(8).train()
    z = torch.randn(16, 8, 6, 6)
    for name, compiled in (("eager", False), ("compiled", True)):
        m = copy.deepcopy(bn).to(dev)
        f = torch.compile(m, dynamic=False) if compiled else m
        r = copy.deepcopy(bn)
        r(z)
        f(z.to(dev))
        rel = norm([m.running_mean.cpu() - r.running_mean, m.running_var.cpu() - r.running_var]) / \
            norm([r.running_mean - bn.running_mean, r.running_var - bn.running_var])
        print(f"BN native {name}: running-stat rel err {rel:.2e} (1 = not updated)", flush=True)

    G, S, n_cls, lr, lam = 4, 8, 10, 0.01, 1.0
    counts_v = [S] * G
    counts_v[-1] = S - 3
    counts_a = [max(1, S - 1 - (g % 3)) for g in range(G)]
    teacher = ResNet34(num_classes=n_cls).eval()
    student0 = ResNet18(num_classes=n_cls).train()
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    kd = KDLoss(1.0, 0.5, 0.5)
    batches = [build_step_batch(G, S, a.crop, a.anchor, counts_v, counts_a, torch.Generator().manual_seed(1 + i), n_cls, lam)
               for i in range(a.steps)]
    for scope, dt in [(sc, d) for d in a.dtypes.split(",") for sc in a.scopes.split(",")]:
        try:
            holder = GhostHolder()
            mine = copy.deepcopy(student0).to(dev)
            convert_to_ghost_bn(mine, holder)
            opt = torch.optim.SGD(mine.parameters(), lr=lr, momentum=0.0, weight_decay=0.0)
            fn = StepFn(student=mine, teacher=copy.deepcopy(teacher).to(dev), holder=holder, optimizer=opt, device=dev,
                        mean=mean.view(-1), std=std.view(-1), temperature=1.0, ce_weight=0.5, kd_weight=0.5, full_anchor=lam,
                        groups=G, slot=S, dtype=torch.bfloat16 if dt == "bf16" else torch.float32, scaler=None,
                        channels_last=False, compile_scope=scope, amp=dt == "bf16")
            rows = []
            for i, sb in enumerate(batches):
                ref = copy.deepcopy(student0)
                with torch.no_grad():
                    for pr, pm in zip(ref.parameters(), mine.parameters()):
                        pr.copy_(pm.detach().cpu())
                    for br, bm in zip(ref.buffers(), mine.buffers()):
                        br.copy_(bm.detach().cpu())
                run0 = [b.detach().clone() for nme, b in ref.named_buffers() if "running" in nme]
                g_ref, run_ref = ref_grads_and_running(ref, teacher, sb, mean, std, kd, lam, G, S)
                p_before = [p.detach().cpu().clone() for p in mine.parameters()]
                fn.run(sb)
                p_after = [p.detach().cpu() for p in mine.parameters()]
                g_dev = [(p.grad.detach().cpu() if p.grad is not None else torch.zeros_like(q))
                         for p, q in zip(mine.parameters(), p_after)]
                run_dev = [b.detach().cpu() for nme, b in mine.named_buffers() if "running" in nme]
                grad_rel = norm([x - y for x, y in zip(g_dev, g_ref)]) / norm(g_ref)
                step_vec = [x - y for x, y in zip(p_after, p_before)]
                upd_rel = norm([d + lr * g for d, g in zip(step_vec, g_dev)]) / max(norm([lr * g for g in g_dev]), 1e-30)
                moved_rel = norm(step_vec) / (lr * norm(g_ref))
                run_rel = norm([x - y for x, y in zip(run_dev, run_ref)]) / norm([x - y for x, y in zip(run_ref, run0)])
                rows.append((grad_rel, upd_rel, moved_rel, run_rel))
                print(f"  scope={scope} {dt} step {i}: grad_rel {grad_rel:.2e} upd_rel {upd_rel:.2e} moved/expected {moved_rel:.3f} "
                      f"running_rel {run_rel:.2e} grads_present {sum(p.grad is not None for p in mine.parameters())}", flush=True)
            tol = 2e-2 if dt == "fp32" else 1.5e-1  # bf16 autocast vs the fp32 reference
            ok = all(r[0] < tol and r[1] < 2e-2 and abs(r[2] - 1) < max(5e-2, tol) and r[3] < max(5e-2, tol) for r in rows)
            lost = all(r[2] < 0.05 for r in rows)
            print(f"StepFn scope={scope} {dt}: {'OK' if ok else ('UPDATES_LOST' if lost else 'WRONG')}", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"StepFn scope={scope} {dt}: ERROR {e!r}"[:400], flush=True)
    print("UPDATE_PROBE_DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
