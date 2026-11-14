"""Where does the R50->MNv1 EP-KD step spend its time on the TPU, and do equivalent rewrites help? (11-13)
Synthetic batch shaped like E1 (views --crop px, anchors 224 px, 16x32 rows), the real StepFn, device otherwise idle.
  parts  : teacher(views), teacher(anchors), student fwd+bwd (views), student fwd+bwd (anchors), full step
  variant: depthwise 3x3 as shifted multiply-adds (epkd_colab/dwconv.py; fp32 / bf16 accumulation), channels_last
  check  : rewritten student == original (logits + weight grads, fp32, same weights, same input) -> max rel diff
    python3 bench/e_breakdown.py [--crop 112] [--steps 20] [--variants base,dw32,dw16,cl]
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import torch
import torch_tpu  # noqa: F401

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import epkd_colab  # noqa: F401,E402
from epkd_colab.dwconv import replace_depthwise  # noqa: E402
import epkd_colab.ghost_bn as gbn  # noqa: E402
from epkd_colab.ghost_bn import GhostHolder, convert_to_ghost_bn  # noqa: E402
from epkd_colab.step import StepFn  # noqa: E402
from tgsc_kd.modeling import build_model  # noqa: E402
from tgsc_kd.utils import forward_logits  # noqa: E402
from tests.test_step_equivalence import build_step_batch  # noqa: E402

dev = torch.device("tpu")
G, S = 16, 32


def sync(t: torch.Tensor) -> None:
    float(t.detach().reshape(-1)[0].cpu())


def timed(call, n: int, warm: int, sync_every: int = 10) -> float:
    """Mean ms per call; syncs every ``sync_every`` calls so queued uploads cannot pile up on the device
    (600 unsynced 96 MB uploads ran the TPU out of memory, 11-14)."""
    r = None
    for _ in range(warm):
        r = call()
    sync(r)
    t0 = time.time()
    for i in range(n):
        r = call()
        if sync_every and (i + 1) % sync_every == 0:
            sync(r)
    sync(r)
    return (time.time() - t0) / n * 1e3


def make_fn(student0, teacher0, variant: str):
    holder = GhostHolder()
    st = copy.deepcopy(student0)
    if variant in ("dw32", "dw16"):
        k = replace_depthwise(st, acc_fp32=(variant == "dw32"))
        print(f"INFO {variant}: replaced {k} depthwise convs", flush=True)
    st = st.to(dev)
    gbn._FAST = variant == "gfast"  # read at trace time by GhostBatchNorm2d.forward
    if variant != "plainbn":  # plainbn = native BatchNorm over all 512 rows: NOT equivalent, diagnostic floor only
        convert_to_ghost_bn(st, holder)
    opt = torch.optim.SGD(st.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-5, foreach=True)
    fn = StepFn(student=st, teacher=copy.deepcopy(teacher0).to(dev).eval(), holder=holder, optimizer=opt, device=dev,
                mean=torch.tensor([0.485, 0.456, 0.406]), std=torch.tensor([0.229, 0.224, 0.225]), temperature=1.0,
                ce_weight=0.5, kd_weight=0.5, full_anchor=1.0, groups=G, slot=S, dtype=torch.bfloat16, scaler=None,
                channels_last=(variant == "cl"), compile_scope="model", amp=True)
    return fn, st


def student_fb(fn: StepFn, d, key: str):
    x = fn._norm(fn._img(d, key))
    fn.holder.spec = fn.spec_v if key == "xv" else fn.spec_a
    with torch.autocast("tpu", dtype=torch.bfloat16):
        s = fn.student(x)
    loss = s.float().square().mean()
    fn.opt.zero_grad(set_to_none=True)
    loss.backward()
    return loss.detach()


def equality(student0) -> None:
    """fp32, eager, same weights and input: original vs shifted depthwise (logits and all weight grads)."""
    torch.manual_seed(1)
    x = torch.randn(8, 3, 112, 112)
    for variant in ("dw32",):
        a = copy.deepcopy(student0).to(dev).train()
        b = copy.deepcopy(student0)
        replace_depthwise(b, acc_fp32=True)
        b = b.to(dev).train()
        outs = []
        for m in (a, b):
            xx = x.to(dev)
            y = forward_logits(m, xx)
            y.float().square().mean().backward()
            outs.append((y.detach().cpu(), [p.grad.detach().cpu() for p in m.parameters()]))
        (ya, ga), (yb, gb) = outs
        rel_y = float((ya - yb).abs().max() / ya.abs().max())
        rel_g = max(float((p - q).abs().max() / (p.abs().max() + 1e-12)) for p, q in zip(ga, gb))
        print(f"RESULT equality {variant}: logits max rel diff {rel_y:.2e}, weight-grad max rel diff {rel_g:.2e}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--crop", type=int, default=112)
    ap.add_argument("--anchor", type=int, default=224)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--variants", default="base,dw32,dw16,cl")
    ap.add_argument("--parts", type=int, default=1)
    ap.add_argument("--burn", type=int, default=0, help="N background processes doing memcpy loops (loader-like CPU + memory load)")
    ap.add_argument("--pin", default="", help="cores for THIS process, e.g. 0-3; burners get the rest")
    a = ap.parse_args()
    import os
    import subprocess
    ncpu = os.cpu_count() or 1
    if a.pin:
        lo, hi = (int(v) for v in a.pin.split("-"))
        mine = set(range(lo, hi + 1))
        os.sched_setaffinity(0, mine)
        rest = sorted(set(range(ncpu)) - mine)
        print(f"INFO pinned to {sorted(mine)}; burners on {len(rest)} cores", flush=True)
    else:
        rest = list(range(ncpu))
    burners = []
    for i in range(a.burn):
        code = ("import os,numpy as np\nos.nice(10)\na=np.ones(64_000_000,np.uint8);b=np.empty_like(a)\n"
                "while True: np.copyto(b,a); b[::4096]+=1")
        p = subprocess.Popen([sys.executable, "-c", code])
        os.sched_setaffinity(p.pid, set(rest))
        burners.append(p)
    if burners:
        print(f"INFO {len(burners)} memcpy burners running (nice 10)", flush=True)
    torch.manual_seed(0)
    student0 = build_model("MNv1", num_classes=1000).train()
    teacher0 = build_model("RN50", num_classes=1000).eval()
    sbs = [build_step_batch(G, S, a.crop, a.anchor, [S] * G, [S] * G, torch.Generator().manual_seed(i), 1000, 1.0) for i in range(2)]
    try:
        if a.parts:
            equality(student0)
    except Exception as e:  # noqa: BLE001
        print(f"RESULT equality ERROR {e!r}"[:400], flush=True)
    for variant in a.variants.split(","):
        try:
            fn, st = make_fn(student0, teacher0, variant)
            d = fn.upload(sbs[0], 0)
            fn._bind_specs(d)
            r = {"step_ms": timed(lambda: fn.compute(d, True), a.steps, 4)}
            if variant in ("base", "gfast"):  # host->TPU upload alone, and overlapped with compute (the training pattern)
                r["upload_ms"] = timed(lambda: fn.upload(sbs[1], 1)["yv"], a.steps, 3)
                state = {"d": d, "slot": 0}

                def overlapped():
                    th, dn = fn.upload_async(sbs[state["slot"]], 1 - state["slot"])
                    out = fn.compute(state["d"], True)
                    th.join()
                    state["d"], state["slot"] = dn, 1 - state["slot"]
                    return out
                r["overlapped_ms"] = timed(overlapped, a.steps, 3)
            if a.parts and variant in ("base", "dw32", "dw16", "gfast", "plainbn"):
                if variant == "base":
                    r["teacher_views_ms"] = timed(lambda: fn._teacher(fn._norm(fn._img(d, "xv"))), a.steps, 3)
                    r["teacher_anchors_ms"] = timed(lambda: fn._teacher(fn._norm(fn._img(d, "xa"))), a.steps, 3)
                r["student_views_fb_ms"] = timed(lambda: student_fb(fn, d, "xv"), a.steps, 3)
                r["student_anchors_fb_ms"] = timed(lambda: student_fb(fn, d, "xa"), a.steps, 3)
            r["rows_per_s_ceiling"] = G * S / r["step_ms"] * 1e3
            print(f"RESULT {variant} " + json.dumps({k: round(v, 1) for k, v in r.items()}), flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"RESULT {variant} ERROR {e!r}"[:400], flush=True)
    for p in burners:
        p.kill()
    print("E_BREAKDOWN_DONE", flush=True)


if __name__ == "__main__":
    main()
