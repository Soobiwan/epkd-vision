#!/usr/bin/env python3
"""torch_tpu capability + throughput probe for THIS pipeline (also runs on cuda/cpu with --device).
  1. host->device bandwidth of a 512x3x224x224 uint8 step buffer;
  2. StepFn (ghost BN, padded anchors, compiled whole step) at crop 112/140/168/196 (+224 anchors) and 224:
     ms/step, rows/s, compile time; checks that the weights change and that no NaN appears;
  3. precompute ops: quantile / hooks / backward-to-input on the device, per-shape compile cost of the union map
     (eager and compiled) for a few image shapes.
Usage: python bench/tpu_probe3.py --device tpu [--steps 20] [--crops 112,224] [--precompute 1]
"""
from __future__ import annotations

import argparse
import copy
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import epkd_colab  # noqa: F401,E402
from epkd_colab.ghost_bn import GhostHolder, convert_to_ghost_bn  # noqa: E402
from epkd_colab.step import StepFn  # noqa: E402
from model.python.resnet_imagenet import ResNet18, ResNet34  # noqa: E402
from tests.test_step_equivalence import build_step_batch  # noqa: E402

T0 = time.time()


def log(k, v):
    print(f"[{time.time() - T0:6.0f}s] {k}: {v}", flush=True)


def sync(dev):
    if dev.type == "cuda":
        torch.cuda.synchronize()
    elif dev.type == "tpu":
        torch.accelerator.synchronize()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="tpu")
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--crops", default="112,224")
    ap.add_argument("--dtype", default="bf16")
    ap.add_argument("--compile-scope", default="step")
    ap.add_argument("--precompute", type=int, default=1)
    ap.add_argument("--h2d", type=int, default=1)
    a = ap.parse_args()
    if a.device == "tpu":
        import torch_tpu  # noqa: F401
    dev = torch.device(a.device)
    if dev.type == "cuda":
        torch.backends.cudnn.benchmark = True
    log("device", f"{dev} torch {torch.__version__}")
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[a.dtype]

    # ---------------- 1. H2D
    if a.h2d:
        x = torch.randint(0, 255, (512, 3, 224, 224), dtype=torch.uint8)
        xp = x.pin_memory() if dev.type == "cuda" else x
        d = torch.zeros_like(x, device=dev)
        for i in range(3):
            d.copy_(xp, non_blocking=True)
        sync(dev)
        t = time.time()
        for i in range(10):
            d.copy_(xp, non_blocking=True)
        sync(dev)
        el = (time.time() - t) / 10
        log("h2d 77MB uint8", f"{el * 1000:.1f} ms = {x.numel() / el / 1e9:.1f} GB/s (copy_ into static device buffer)")
        t = time.time()
        for i in range(10):
            d2 = xp.to(dev)
        float(d2.view(-1)[0].float())
        sync(dev)
        log("h2d 77MB .to()", f"{(time.time() - t) / 10 * 1000:.1f} ms")

    # ---------------- 2. training step
    G, S = 16, 32
    mean = torch.tensor([0.485, 0.456, 0.406])
    std = torch.tensor([0.229, 0.224, 0.225])
    teacher = ResNet34(num_classes=1000).eval().to(dev)
    for crop in [int(c) for c in a.crops.split(",")]:
        anchor = 224 if crop < 224 else None
        counts_v = [S] * G
        counts_a = [S - (g % 3) for g in range(G)] if anchor else [0] * G
        gen = torch.Generator().manual_seed(0)
        sb = build_step_batch(G, S, crop, anchor, counts_v, counts_a, gen, 1000, 1.0)
        student = ResNet18(num_classes=1000).train()
        holder = GhostHolder()
        convert_to_ghost_bn(student, holder)
        student.to(dev)
        p_before = [p.detach().cpu().clone() for p in student.parameters()]  # HOST snapshot (device clone may alias/lazy)
        opt = torch.optim.SGD(student.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
        scaler = torch.amp.GradScaler("cuda") if (dev.type == "cuda" and dtype == torch.float16) else None
        fn = StepFn(student=student, teacher=teacher, holder=holder, optimizer=opt, device=dev, mean=mean, std=std, temperature=1.0,
                    ce_weight=0.5, kd_weight=0.5, full_anchor=1.0, groups=G, slot=S, dtype=dtype, scaler=scaler,
                    channels_last=dev.type == "cuda", compile_scope=a.compile_scope, amp=dtype != torch.float32)
        t = time.time()
        try:
            p0 = next(student.parameters())
            ptr0 = p0.untyped_storage().data_ptr()
            for i in range(3):
                loss = fn.run(sb)
                lv = float(loss)
            sync(dev)
            log(f"crop {crop} warm/compile", f"{time.time() - t:.1f}s loss {lv:.4f}")
            t = time.time()
            for i in range(a.steps):
                loss = fn.run(sb)
            lv = float(loss)
            sync(dev)
            el = (time.time() - t) / a.steps
            rows = sum(counts_v)
            sync(dev)
            changed = max((p.detach().cpu() - q).abs().max().item() for p, q in zip(student.parameters(), p_before))
            ptr1 = next(student.parameters()).untyped_storage().data_ptr()
            # decisive check: loss on the SAME batch for a few consecutive steps must decrease if updates are applied
            same = []
            for i in range(5):
                same.append(float(fn.run(sb)))
            sync(dev)
            dec = all(b_ < a_ for a_, b_ in zip(same, same[1:]))
            ok = changed > 0 and dec
            verdict = "UPDATES_OK" if ok else "UPDATES_LOST"
            nan = any(torch.isnan(p).any().item() for p in student.parameters())
            log(f"crop {crop}{'+anchor224' if anchor else ''} G{G}xS{S} {a.dtype} {a.compile_scope}",
                f"{el * 1000:.1f} ms/step  {rows / el:.0f} rows/s  (loss {lv:.4f}, max |dW| {changed:.2e} (host), same-batch losses {[round(v, 4) for v in same]}, param0 storage ptr {ptr0:#x}->{ptr1:#x} {'REPLACED' if ptr0 != ptr1 else 'same'}, {verdict}, nan={nan}, "
                f"stats {[round(v, 2) for v in fn.pop_stats()]})")
        except Exception as e:  # noqa: BLE001
            import traceback

            log(f"crop {crop} ERROR", repr(e)[:400] + " | " + traceback.format_exc().splitlines()[-1])
        del student, opt, fn

    # ---------------- 3. precompute ops
    if a.precompute:
        from epkd_colab.precompute import compute_union_batched, emit_stage_rows_batched
        import types

        from tgsc_kd.config import TGSCConfig
        from tgsc_kd.utils import pick_target_layer

        cfg = TGSCConfig(stride=56, candidate_stride=56, dynamic_stride=True, k_per_img=6, top_k_crops=3, low_k_crops=0,
                         edge_align_windows=True, view_selection="saliency", kd_gate="off", eval_chunk=256)
        pre = types.SimpleNamespace(device=dev, teacher=teacher, target_layer=pick_target_layer(teacher), tgsc_cfg=cfg, edge_align=True,
                                    stage_sizes=(112, 140, 168, 196), _union_signature=lambda: (True, 0.7, 0.3, 1.5, 1.2, 1.0, 99.0))
        for B, (H, W) in [(32, (256, 341)), (32, (256, 341)), (32, (341, 256)), (32, (256, 384)), (8, (256, 341)), (8, (300, 256))]:
            xb = torch.rand(B, 3, H, W)
            yb = torch.randint(0, 1000, (B,))
            cand = {s: [] for s in pre.stage_sizes}
            stage = {s: [] for s in pre.stage_sizes}
            t = time.time()
            try:
                an = compute_union_batched(pre, xb, yb)
                float(an["union"].sum())
                t1 = time.time()
                emit_stage_rows_batched(pre, an, list(range(B)), cand, stage)
                t2 = time.time()
                log(f"precompute B{B} {H}x{W} eager", f"union {t1 - t:.2f}s emit {t2 - t1:.2f}s rows {sum(len(v) for v in stage.values())}")
            except Exception as e:  # noqa: BLE001
                import traceback

                log(f"precompute B{B} {H}x{W} ERROR", repr(e)[:300] + " | " + traceback.format_exc().splitlines()[-1])
    log("total", round(time.time() - T0))


if __name__ == "__main__":
    main()
