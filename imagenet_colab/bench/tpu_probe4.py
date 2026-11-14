#!/usr/bin/env python3
"""torch_tpu follow-up probe: (1) host->device transfer variants (uint8 / int32-packed / bf16 / float32, threads);
(2) does the weight update survive torch.compile?  variants: whole step compiled (probe pattern), fwd+bwd compiled
with eager optimizer, fwd compiled only; each reports max |dW| and ms/step for the real G16xS32 112+224 problem;
(3) same for a plain nn.BatchNorm2d student (is ghost BN the slow part?)."""
from __future__ import annotations

import argparse
import copy
import sys
import threading
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="tpu")
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--part", default="h2d,update")
    a = ap.parse_args()
    if a.device == "tpu":
        import torch_tpu  # noqa: F401
    dev = torch.device(a.device)
    sync = torch.accelerator.synchronize if dev.type != "cpu" else (lambda: None)
    parts = a.part.split(",")

    if "h2d" in parts:
        base = torch.randint(0, 255, (512, 3, 224, 224), dtype=torch.uint8)
        variants = {
            "uint8 (77MB)": base,
            "int32 view (77MB)": base.view(torch.int32),
            "int64 view (77MB)": base.view(torch.int64),
            "bf16 (154MB)": base.to(torch.bfloat16),
            "float32 (308MB)": base.float(),
            "uint8 contiguous clone": base.clone(),
        }
        for name, x in variants.items():
            d = torch.empty_like(x, device=dev)
            d.copy_(x)
            sync()
            t = time.time()
            for _ in range(5):
                d.copy_(x, non_blocking=True)
            sync()
            el = (time.time() - t) / 5
            log(f"h2d copy_ {name}", f"{el * 1000:.0f} ms = {x.numel() * x.element_size() / el / 1e9:.2f} GB/s")
            t = time.time()
            for _ in range(5):
                y = x.to(dev, non_blocking=True)
            sync()
            el = (time.time() - t) / 5
            log(f"h2d .to() {name}", f"{el * 1000:.0f} ms = {x.numel() * x.element_size() / el / 1e9:.2f} GB/s")
        # 4 threads, 4 quarters
        qs = list(base.chunk(4))
        ds = [torch.empty_like(q, device=dev) for q in qs]
        t = time.time()
        ths = [threading.Thread(target=lambda q=q, d=d: d.copy_(q)) for q, d in zip(qs, ds)]
        [th.start() for th in ths]
        [th.join() for th in ths]
        sync()
        log("h2d uint8 4 threads", f"{(time.time() - t) * 1000:.0f} ms")
        # device-side unpack cost check: uint8 -> float on device
        d = base.to(dev)
        sync()
        t = time.time()
        for _ in range(5):
            f = d.float()
        sync()
        log("device uint8->float 77MB", f"{(time.time() - t) / 5 * 1000:.1f} ms")

    if "update" in parts:
        G, S, crop, anchor = 16, 32, 112, 224
        mean = torch.tensor([0.485, 0.456, 0.406])
        std = torch.tensor([0.229, 0.224, 0.225])
        teacher = ResNet34(num_classes=1000).eval().to(dev)
        gen = torch.Generator().manual_seed(0)
        sb = build_step_batch(G, S, crop, anchor, [S] * G, [S - (g % 3) for g in range(G)], gen, 1000, 1.0)
        for variant in ["step", "fwdbwd", "model", "none"]:
            student = ResNet18(num_classes=1000).train()
            holder = GhostHolder()
            convert_to_ghost_bn(student, holder)
            student.to(dev)
            p0 = [p.detach().clone() for p in student.parameters()]
            opt = torch.optim.SGD(student.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
            scope = {"step": "step", "fwdbwd": "fwdbwd", "model": "model", "none": "none"}[variant]
            fn = StepFn(student=student, teacher=teacher, holder=holder, optimizer=opt, device=dev, mean=mean, std=std, temperature=1.0,
                        ce_weight=0.5, kd_weight=0.5, full_anchor=1.0, groups=G, slot=S, dtype=torch.bfloat16, scaler=None,
                        channels_last=False, compile_scope=scope, amp=True)
            try:
                d = fn.upload(sb)
                sync()
                t = time.time()
                for i in range(2):
                    loss = fn._run(d, True)
                    lv = float(loss)
                sync()
                log(f"{variant}: warm", f"{time.time() - t:.1f}s loss {lv:.4f}")
                losses = []
                t = time.time()
                for i in range(a.steps):
                    fn._bind_specs(d)
                    loss = fn._run(d, True)
                    losses.append(float(loss))
                sync()
                el = (time.time() - t) / a.steps
                changed = max((p.detach() - q).abs().max().item() for p, q in zip(student.parameters(), p0))
                log(f"{variant}: compute only (no H2D)", f"{el * 1000:.1f} ms/step {512 / el:.0f} rows/s | max|dW| {changed:.3e} | losses {[round(x, 3) for x in losses[:4]]}..{round(losses[-1], 3)}")
            except Exception as e:  # noqa: BLE001
                import traceback

                log(f"{variant}: ERROR", repr(e)[:300] + " | " + traceback.format_exc().splitlines()[-1][:200])
            del student, opt, fn
        # plain BN student (no ghost) whole-step compiled: is ghost BN the slow part?
        student = ResNet18(num_classes=1000).train().to(dev)
        opt = torch.optim.SGD(student.parameters(), lr=0.1, momentum=0.9)
        from epkd_colab.step import LogitsOnly, kd_terms_per_row

        S_ = LogitsOnly(student)
        T_ = LogitsOnly(teacher)
        xv = torch.randn(512, 3, 112, 112, device=dev)
        xa = torch.randn(512, 3, 224, 224, device=dev)
        y = torch.randint(0, 1000, (512,), device=dev)

        def plain_step():
            with torch.no_grad(), torch.autocast("tpu", dtype=torch.bfloat16):
                tv = T_(xv).float()
                ta = T_(xa).float()
            with torch.autocast("tpu", dtype=torch.bfloat16):
                sv = S_(xv).float()
                sa = S_(xa).float()
            ce, kd = kd_terms_per_row(sv, tv, y, 1.0)
            cea, kda = kd_terms_per_row(sa, ta, y, 1.0)
            loss = (0.5 * ce + 0.5 * kd).mean() + (0.5 * cea + 0.5 * kda).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            return loss.detach()

        cs = torch.compile(plain_step)
        p0 = [p.detach().clone() for p in student.parameters()]
        t = time.time()
        for _ in range(2):
            lv = float(cs())
        log("plainBN whole-step compiled warm", f"{time.time() - t:.1f}s")
        t = time.time()
        for _ in range(a.steps):
            l = cs()
        float(l)
        sync()
        el = (time.time() - t) / a.steps
        changed = max((p.detach() - q).abs().max().item() for p, q in zip(student.parameters(), p0))
        log("plainBN whole-step compiled 512 views112 + 512 anchors224", f"{el * 1000:.1f} ms/step {512 / el:.0f} rows/s | max|dW| {changed:.3e}")
    log("total", round(time.time() - T0))


if __name__ == "__main__":
    main()
