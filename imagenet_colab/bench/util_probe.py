"""TPU utilisation probe for the EP-KD training step (synthetic batch shaped like F1 @ 140 px + 224 px anchors).
Answers, per configuration, with the device otherwise idle:
  E1 device-only  : step time with the batch already on the TPU (the ceiling if input were free)
  E1f fwd/bwd only: the same without the optimizer step (optimizer share)
  E2 upload-only  : host->TPU time of one step's inputs (i32 fresh-tensor path, as in training)
  E3 overlapped   : upload of step N+1 in a thread while step N computes (the training pattern)
  optimizer variants (foreach=False = what runs now, foreach=True) incl. a one-step equality check
  compile scopes (none / model)
Prints RESULT lines; tpu-info is sampled by the caller.
    EPKD_H2D=i32 python3 bench/util_probe.py [--crop 140] [--steps 30]
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
from epkd_colab.ghost_bn import GhostHolder, convert_to_ghost_bn  # noqa: E402
from epkd_colab.step import StepFn  # noqa: E402
from model.python.resnet_imagenet import ResNet18, ResNet34  # noqa: E402
from tests.test_step_equivalence import build_step_batch  # noqa: E402

dev = torch.device("tpu")
G, S = 16, 32


def sync(t: torch.Tensor) -> None:
    float(t.detach().reshape(-1)[0].cpu())


def make_fn(scope: str, foreach: bool, crop: int, anchor: int, student0, teacher0):
    holder = GhostHolder()
    st = copy.deepcopy(student0).to(dev)
    convert_to_ghost_bn(st, holder)
    opt = torch.optim.SGD(st.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4, foreach=foreach)
    mean = torch.tensor([0.485, 0.456, 0.406])
    std = torch.tensor([0.229, 0.224, 0.225])
    fn = StepFn(student=st, teacher=copy.deepcopy(teacher0).to(dev).eval(), holder=holder, optimizer=opt, device=dev,
                mean=mean, std=std, temperature=1.0, ce_weight=0.5, kd_weight=0.5, full_anchor=1.0, groups=G, slot=S,
                dtype=torch.bfloat16, scaler=None, channels_last=False, compile_scope=scope, amp=True)
    return fn, st


def timed(fn_call, n: int, warm: int, sync_t) -> float:
    for _ in range(warm):
        r = fn_call()
    sync(sync_t(r))
    t0 = time.time()
    for _ in range(n):
        r = fn_call()
    sync(sync_t(r))
    return (time.time() - t0) / n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--crop", type=int, default=140)
    ap.add_argument("--anchor", type=int, default=224)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--configs", default="model:0,model:1,none:0")
    a = ap.parse_args()
    torch.manual_seed(0)
    student0, teacher0 = ResNet18(num_classes=1000).train(), ResNet34(num_classes=1000).eval()
    counts_v, counts_a = [S] * G, [S - 1] * G
    sbs = [build_step_batch(G, S, a.crop, a.anchor, counts_v, counts_a, torch.Generator().manual_seed(i), 1000, 1.0) for i in range(2)]
    mb = sum(t.numel() * t.element_size() for t in sbs[0].tensors().values()) / 1e6
    print(f"INFO step inputs {mb:.1f} MB (crop {a.crop}, anchor {a.anchor}, {G}x{S} rows)", flush=True)
    res = {}
    for cfg in a.configs.split(","):
        scope, fe = cfg.split(":")
        fe = bool(int(fe))
        tag = f"scope={scope} foreach={fe}"
        try:
            fn, st = make_fn(scope, fe, a.crop, a.anchor, student0, teacher0)
            d0 = fn.upload(sbs[0], 0)
            n = a.steps
            # E1 device-only (full step) and E1f (forward/backward without the optimizer)
            t1 = timed(lambda: fn.compute(d0, True), n, 5, lambda r: r)
            t1f = timed(lambda: fn._fwd_bwd(d0, True), n, 3, lambda r: r)
            # E2 upload-only
            t2 = timed(lambda: fn.upload(sbs[1], 1), n, 3, lambda d: d["yv"])
            # E3 overlapped (training pattern: next upload in a thread while this step computes)
            state = {"d": d0, "slot": 0}

            def overlapped():
                th, dn = fn.upload_async(sbs[state["slot"]], 1 - state["slot"])
                r = fn.compute(state["d"], True)
                th.join()
                state["d"], state["slot"] = dn, 1 - state["slot"]
                return r
            t3 = timed(overlapped, n, 3, lambda r: r)
            res[tag] = dict(device_only_ms=t1 * 1e3, fwd_bwd_ms=t1f * 1e3, optimizer_ms=(t1 - t1f) * 1e3, upload_ms=t2 * 1e3,
                            overlapped_ms=t3 * 1e3, serial_sum_ms=(t1 + t2) * 1e3, rows_per_s_ceiling=G * S / t1,
                            rows_per_s_overlapped=G * S / t3)
            print("RESULT " + tag + " " + json.dumps({k: round(v, 1) for k, v in res[tag].items()}), flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"RESULT {tag} ERROR {e!r}"[:400], flush=True)
    # optimizer equality: one identical step with foreach False vs True (same start, same batch)
    try:
        outs = []
        for fe in (False, True):
            fn, st = make_fn("model", fe, a.crop, a.anchor, student0, teacher0)
            sync(fn.compute(fn.upload(sbs[0], 0), True))
            outs.append([p.detach().cpu().float() for p in st.parameters()])
        diff = max(float((x - y).abs().max()) for x, y in zip(*outs))
        ref = max(float(x.abs().max()) for x in outs[0])
        print(f"RESULT optimizer foreach equality: max |diff| {diff:.3e} (param scale {ref:.3e})", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"RESULT optimizer equality ERROR {e!r}"[:300], flush=True)
    print("UTIL_PROBE_DONE", flush=True)


if __name__ == "__main__":
    main()
