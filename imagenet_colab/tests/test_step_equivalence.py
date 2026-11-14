"""One optimisation step of StepFn (batched, ghost BN, padded/masked anchors) == the reference per-micro-batch
loop (DDPKDTrainer.train body: KDLoss per micro-batch, loss*n/512, anchors as their own BN batch, SGD) on identical
crops and weights.  CPU fp32 (also runnable on cuda/tpu: --device).  From imagenet_colab: python tests/test_step_equivalence.py"""
from __future__ import annotations

import argparse
import copy
import sys
import types
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import epkd_colab  # noqa: F401,E402
from epkd_colab.ghost_bn import GhostHolder, convert_to_ghost_bn, interleaved_ema_weights  # noqa: E402
from epkd_colab.loader import StepBatch  # noqa: E402
from epkd_colab.step import StepFn  # noqa: E402
from model.python.resnet_imagenet import ResNet18, ResNet34  # noqa: E402
from tgsc_kd.trainer import KDLoss  # noqa: E402
from tgsc_kd.utils import forward_logits  # noqa: E402


def smooth_u8(n, size, gen):
    """Smooth random images (bilinearly upsampled low-res noise): pixel noise makes an untrained ResNet ill-conditioned
    in fp32 (fp32-vs-fp64 gradient differences of several %), which would hide real differences."""
    import torch.nn.functional as F

    lo = torch.randint(0, 256, (n, 3, 6, 6), generator=gen).float()
    return F.interpolate(lo, size=(size, size), mode="bilinear", align_corners=False).round().clamp(0, 255).to(torch.uint8)


def build_step_batch(G, S, crop, anchor, counts_v, counts_a, gen, n_classes, lam):
    sb = StepBatch(G, S, crop, anchor, pin=False)
    step_rows = G * S
    for g in range(G):
        n, a = counts_v[g], counts_a[g]
        sl = slice(g * S, g * S + n)
        sb.xv[sl] = smooth_u8(n, crop, gen)
        sb.yv[sl] = torch.randint(0, n_classes, (n,), generator=gen)
        sb.wv[sl] = 1.0 / step_rows
        sb.mv[sl] = 1.0
        sb.kdw[sl] = 1.0
        if anchor:
            sa = slice(g * S, g * S + a)
            sb.xa[sa] = smooth_u8(a, anchor, gen)
            sb.ya[sa] = torch.randint(0, n_classes, (a,), generator=gen)
            sb.wa[sa] = n / (step_rows * a) if a else 0.0
            sb.ma[sa] = 1.0
    nv = torch.tensor([float(c) for c in counts_v])
    sb.n_v.copy_(nv)
    sb.inv_v.copy_(torch.where(nv > 0, 1 / nv.clamp_min(1), torch.zeros_like(nv)))
    sb.kdw.mul_(sb.mv)
    if anchor:
        na = torch.tensor([float(c) for c in counts_a])
        sb.n_a.copy_(na)
        sb.inv_a.copy_(torch.where(na > 0, 1 / na.clamp_min(1), torch.zeros_like(na)))
        w_v, k_v, w_a, k_a = interleaved_ema_weights(counts_v, counts_a, 0.1)
        sb.ema_a.copy_(torch.tensor(w_a))
        sb.keep_a.fill_(k_a)
    else:
        from epkd_colab.ghost_bn import single_ema_weights

        w_v, k_v = single_ema_weights(counts_v, 0.1)
    sb.ema_v.copy_(torch.tensor(w_v))
    sb.keep_v.fill_(k_v)
    sb.counts_v, sb.counts_a, sb.rows = list(counts_v), list(counts_a), int(sum(counts_v))
    return sb


def reference_step(student, teacher, sb, mean, std, kd, lr, lam, G, S):
    """The reference loop (world 1): sequential micro-batches, loss*n/512, fresh SGD step."""
    opt = torch.optim.SGD(student.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    opt.zero_grad(set_to_none=True)
    step_rows = G * S
    tot = 0.0
    for g in range(G):
        n, a = sb.counts_v[g], sb.counts_a[g]
        if n == 0:
            continue
        x = ((sb.xv[g * S: g * S + n].float() / 255.0) - mean) / std
        y = sb.yv[g * S: g * S + n]
        with torch.no_grad():
            t = forward_logits(teacher, x)
        s = forward_logits(student, x)
        out = kd(s, t, y, weights=torch.ones(n))
        loss = out["loss"]
        if sb.anchor is not None:
            xa = ((sb.xa[g * S: g * S + a].float() / 255.0) - mean) / std
            ya = sb.ya[g * S: g * S + a]
            with torch.no_grad():
                ta = forward_logits(teacher, xa)
            sa = forward_logits(student, xa)
            loss = loss + lam * kd(sa, ta, ya, weights=None)["loss"]
        (loss * n / step_rows).backward()
        tot += float(out["loss"]) * n
    opt.step()
    return tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dtype", default="fp32")
    ap.add_argument("--compile-scope", default="none")
    ap.add_argument("--crop", type=int, default=64)
    ap.add_argument("--anchor", type=int, default=96)
    ap.add_argument("--groups", type=int, default=4)
    ap.add_argument("--slot", type=int, default=8)
    ap.add_argument("--tf32", type=int, default=1, help="CUDA: allow TF32 convolutions (PyTorch default)")
    a = ap.parse_args()
    if a.device.startswith("cuda"):
        torch.backends.cudnn.allow_tf32 = bool(a.tf32)
        torch.backends.cuda.matmul.allow_tf32 = bool(a.tf32)
    if a.device == "tpu":
        import torch_tpu  # noqa: F401
    dev = torch.device(a.device)
    torch.manual_seed(0)
    G, S, n_cls, lam, lr = a.groups, a.slot, 50, 1.0, 0.05
    counts_v = [S] * G
    counts_v[-1] = max(1, S - 3)  # partial last micro-batch
    counts_a = [max(1, S - 1 - (g % 3)) for g in range(G)]
    gen = torch.Generator().manual_seed(1)
    teacher = ResNet34(num_classes=n_cls).eval()
    student = ResNet18(num_classes=n_cls).train()
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    kd = KDLoss(1.0, 0.5, 0.5)
    sb = build_step_batch(G, S, a.crop, a.anchor if a.anchor > 0 else None, counts_v, counts_a, gen, n_cls, lam)
    # ---- reference (CPU fp32)
    ref = copy.deepcopy(student)
    ref_loss = reference_step(ref, teacher, sb, mean, std, kd, lr, lam, G, S)
    ref64 = copy.deepcopy(student).double()  # fp32 noise floor: the same reference loop in float64
    reference_step(ref64, copy.deepcopy(teacher).double(), sb, mean.double(), std.double(), kd, lr, lam, G, S)
    # ---- StepFn on the requested device
    holder = GhostHolder()
    mine = copy.deepcopy(student).to(dev)
    convert_to_ghost_bn(mine, holder)
    t_dev = copy.deepcopy(teacher).to(dev)
    opt = torch.optim.SGD(mine.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    dtype = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[a.dtype]
    fn = StepFn(student=mine, teacher=t_dev, holder=holder, optimizer=opt, device=dev, mean=mean.view(-1), std=std.view(-1), temperature=1.0,
                ce_weight=0.5, kd_weight=0.5, full_anchor=lam, groups=G, slot=S, dtype=dtype, scaler=None, channels_last=False,
                compile_scope=a.compile_scope, amp=(a.dtype != "fp32"))
    loss = fn.run(sb)
    float(loss)
    st = fn.pop_stats()
    # ---- compare parameters (after the step), running stats, logged loss sum
    worst_p = worst_b = floor = 0.0
    p0 = dict(student.named_parameters())
    p64 = dict(ref64.named_parameters())
    for (n1, p1), (n2, p2) in zip(ref.named_parameters(), mine.named_parameters()):
        upd = (p1 - p0[n1]).abs().max().item() + 1e-12
        d = (p1 - p2.detach().cpu()).abs().max().item()
        worst_p = max(worst_p, d / upd)
        floor = max(floor, (p1 - p64[n1].float()).abs().max().item() / upd)
    for (n1, b1), (n2, b2) in zip(ref.named_buffers(), mine.named_buffers()):
        if b1.dtype.is_floating_point:
            worst_b = max(worst_b, ((b1 - b2.detach().cpu()).abs() / (b1.abs() + 1e-3)).max().item())
    print(f"device={a.device} dtype={a.dtype}: max |param diff| / |update| = {worst_p:.3e} (fp32-vs-fp64 reference floor {floor:.3e}); "
          f"running-stat max rel diff = {worst_b:.3e}; loss sum ref {ref_loss:.5f} mine {st[0]:.5f}")
    ok = worst_p < max(1e-3, 10 * floor) if a.dtype == "fp32" else worst_p < 0.2
    ok = ok and worst_b < (1e-4 if a.dtype == "fp32" else 5e-2)
    print("STEP EQUIVALENCE", "OK" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
