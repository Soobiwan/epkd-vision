"""Why is the in-training upload ~0.3 GB/s when a 59 MB i32 upload benchmarks at 1.26 GB/s?
Times i32 fresh-tensor uploads (the training path) by size, single vs chunked, before and after a compiled
ResNet step has run (HBM in use), and checks the chunked path for correctness.
    python3 bench/h2d_bench2.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import torch
import torch_tpu  # noqa: F401

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
dev = torch.device("tpu")


def up_i32(src: torch.Tensor) -> torch.Tensor:
    return src.view(torch.int32).to(dev).view(torch.uint8)


def up_chunked(src: torch.Tensor, k: int) -> torch.Tensor:
    parts = [up_i32(c) for c in src.chunk(k)]
    return torch.cat(parts)


def t_ms(fn, n=8) -> float:
    fn()
    float(fn()[:4].float().sum().cpu())
    t0 = time.time()
    for _ in range(n):
        r = fn()
    float(r[:4].float().sum().cpu())
    return (time.time() - t0) / n * 1e3


def sweep(tag: str) -> None:
    g = torch.Generator().manual_seed(0)
    for mb in (19, 38, 59, 77, 96):
        n = (mb * 1_000_000 // 64) * 64
        host = torch.randint(0, 256, (n,), dtype=torch.uint8, generator=g)
        single = t_ms(lambda: up_i32(host))
        line = f"RESULT {tag} {mb:3d} MB single {single:7.1f} ms ({n / single / 1e6:5.2f} GB/s)"
        for k in (2, 4, 8):
            c = t_ms(lambda: up_chunked(host, k))
            line += f" | x{k} {c:6.1f} ms ({n / c / 1e6:5.2f} GB/s)"
        print(line, flush=True)
    host = torch.randint(0, 256, (77_070_336,), dtype=torch.uint8, generator=g)
    ok = torch.equal(up_chunked(host, 4).cpu(), host)
    print(f"RESULT {tag} chunked x4 correct {ok}", flush=True)


def main() -> None:
    sweep("fresh")
    # load the device the way training does: compile + run a ResNet fwd/bwd at the anchor size
    from model.python.resnet_imagenet import ResNet34
    m = ResNet34(num_classes=1000).to(dev)
    f = torch.compile(m, dynamic=False)
    x = torch.randn(512, 3, 224, 224, device=dev)
    for _ in range(3):
        out = f(x)
        out.float().sum().backward()
    float(out.float().sum().cpu())
    sweep("after_model")
    print("H2D2_DONE", flush=True)


if __name__ == "__main__":
    main()
