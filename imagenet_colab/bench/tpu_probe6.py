#!/usr/bin/env python3
"""torch_tpu H2D follow-up: (1) pinned-host tensor creation attempts (compiled graphs reading host memory at link
speed?); (2) async batch_transfer_h2d overlapped with the 'model'-scope compiled step (double buffering):
effective ms/step for the 112+224 problem."""
from __future__ import annotations

import sys
import time
import threading
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import epkd_colab  # noqa: F401,E402
import torch_tpu  # noqa: E402
from torch_tpu._internal.batch_transfer import batch_transfer_impl as bt  # noqa: E402
from torch_tpu._internal.compile import compiler as tcomp  # noqa: E402

T0 = time.time()


def log(k, v):
    print(f"[{time.time() - T0:6.0f}s] {k}: {v}", flush=True)


dev = torch.device("tpu")
sync = torch.accelerator.synchronize
x = torch.randint(0, 255, (512, 3, 224, 224), dtype=torch.uint8)

# ---- 1. pinned host attempts
for name, fn in [
    ("torch.empty(device=tpu, pin_memory=True)", lambda: torch.empty(x.shape, dtype=torch.uint8, device=dev, pin_memory=True)),
    ("x.pin_memory()", lambda: x.pin_memory()),
    ("x.pin_memory(device=tpu)", lambda: x.pin_memory(device=dev)),
    ("x.to('tpu') then is_pinned", lambda: x.to(dev)),
]:
    try:
        t = fn()
        log(name, f"ok device={t.device} pinned_host={tcomp.is_pinned_host_tensor(t)} is_pinned={t.is_pinned() if hasattr(t, 'is_pinned') else '?'}")
    except Exception as e:  # noqa: BLE001
        log(name, f"ERR {repr(e)[:160]}")
try:
    ph = tcomp.tpu_torch_compile
    log("tpu_torch_compile attrs", [n for n in dir(ph) if "pin" in n.lower() or "host" in n.lower()][:20])
except Exception as e:  # noqa: BLE001
    log("tpu_torch_compile", repr(e)[:200])

# ---- 2. async transfer + compute overlap
from epkd_colab.ghost_bn import GhostHolder, convert_to_ghost_bn  # noqa: E402
from epkd_colab.step import StepFn  # noqa: E402
from model.python.resnet_imagenet import ResNet18, ResNet34  # noqa: E402
from tests.test_step_equivalence import build_step_batch  # noqa: E402

G, S = 16, 32
mean = torch.tensor([0.485, 0.456, 0.406])
std = torch.tensor([0.229, 0.224, 0.225])
teacher = ResNet34(num_classes=1000).eval().to(dev)
gen = torch.Generator().manual_seed(0)
sb = build_step_batch(G, S, 112, 224, [S] * G, [S - (g % 3) for g in range(G)], gen, 1000, 1.0)
student = ResNet18(num_classes=1000).train()
holder = GhostHolder()
convert_to_ghost_bn(student, holder)
student.to(dev)
opt = torch.optim.SGD(student.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
fn = StepFn(student=student, teacher=teacher, holder=holder, optimizer=opt, device=dev, mean=mean, std=std, temperature=1.0,
            ce_weight=0.5, kd_weight=0.5, full_anchor=1.0, groups=G, slot=S, dtype=torch.bfloat16, scaler=None,
            channels_last=False, compile_scope="model", amp=True)
d0 = fn.upload(sb)
sync()
for _ in range(2):
    fn._bind_specs(d0)
    float(fn._run(d0, True))
log("model-scope warm", "ok")
# two device buffer sets
d1 = {k: torch.zeros_like(v) for k, v in d0.items()}
host = {k: v.contiguous() for k, v in sb.tensors().items()}
big = ["xv", "xa"]


def h2d_sync(d, src=None):
    src = src or host
    for k in big:  # the batched API wants uniform shapes -> one call per tensor
        bt.batch_transfer_h2d_sync([src[k]], [d[k]])
    for k in host:
        if k not in big:
            d[k].copy_(host[k])


pinned = {k: v.pin_memory() for k, v in host.items()}
for name, src in [("pageable", host), ("pinned", pinned)]:
    t = time.time()
    for i in range(5):
        h2d_sync(d1, src)
    sync()
    log(f"h2d batch_transfer_h2d_sync per-tensor from {name} (96MB)", f"{(time.time() - t) / 5 * 1000:.0f} ms")
    t = time.time()
    for i in range(5):
        d1["xa"].copy_(src["xa"])
    sync()
    log(f"h2d copy_ xa 77MB from {name}", f"{(time.time() - t) / 5 * 1000:.0f} ms")


# serial: transfer then compute
t = time.time()
for i in range(10):
    h2d_sync(d0)
    fn._bind_specs(d0)
    l = fn._run(d0, True)
float(l)
sync()
log("serial h2d(batch_transfer_h2d_sync)+compute", f"{(time.time() - t) / 10 * 1000:.0f} ms/step")
t = time.time()
for i in range(10):
    h2d_sync(d0)
sync()
log("h2d alone (xv 19MB + xa 77MB + small)", f"{(time.time() - t) / 10 * 1000:.0f} ms")
# overlapped: async transfer of the other buffer while computing
try:
    t = time.time()
    bufs = [d0, d1]
    futs = [bt.batch_transfer_h2d([pinned[k]], [bufs[1][k]]) for k in big]
    for i in range(10):
        cur, nxt = bufs[i % 2], bufs[(i + 1) % 2]
        for fut in futs:
            if hasattr(fut, "wait"):
                fut.wait()
            elif hasattr(fut, "result"):
                fut.result()
        for k in host:
            if k not in big:
                cur[k].copy_(host[k])
        futs = [bt.batch_transfer_h2d([pinned[k]], [nxt[k]]) for k in big]
        fn._bind_specs(cur)
        l = fn._run(cur, True)
    fut = futs[0]
    float(l)
    sync()
    log("overlapped async batch_transfer_h2d + compute", f"{(time.time() - t) / 10 * 1000:.0f} ms/step (future type {type(fut).__name__}, methods {[m for m in dir(fut) if not m.startswith('_')]})")
except Exception as e:  # noqa: BLE001
    import traceback

    log("overlap ERR", repr(e)[:300] + traceback.format_exc().splitlines()[-1][:200])
# thread-overlapped sync transfer
try:
    t = time.time()
    bufs = [d0, d1]
    th = threading.Thread(target=h2d_sync, args=(bufs[1], pinned))
    th.start()
    for i in range(10):
        cur, nxt = bufs[i % 2], bufs[(i + 1) % 2]
        th.join()
        th = threading.Thread(target=h2d_sync, args=(nxt, pinned))
        th.start()
        fn._bind_specs(cur)
        l = fn._run(cur, True)
    th.join()
    float(l)
    sync()
    log("thread-overlapped sync transfer + compute", f"{(time.time() - t) / 10 * 1000:.0f} ms/step")
except Exception as e:  # noqa: BLE001
    log("thread overlap ERR", repr(e)[:300])
log("total", round(time.time() - T0))
