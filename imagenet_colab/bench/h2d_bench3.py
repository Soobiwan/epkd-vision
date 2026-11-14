"""Host source-buffer kinds for host->TPU uploads: plain vs pinned (pin_memory) vs THP-backed (mmap + MADV_HUGEPAGE).
Each kind: 77 MB (anchor-sized) upload via .to() and the i32 path, timed over repeats; correctness by round trip.
    python3 bench/h2d_bench3.py
"""
from __future__ import annotations

import mmap
import time

import numpy as np
import torch
import torch_tpu  # noqa: F401

dev = torch.device("tpu")
N = 77_070_336  # 512 x 3 x 224 x 224


def plain() -> torch.Tensor:
    return torch.empty(N, dtype=torch.uint8)


def pinned() -> torch.Tensor:
    return torch.empty(N, dtype=torch.uint8, pin_memory=True)


_keep = []


def thp() -> torch.Tensor:
    size = (N + (2 << 20) - 1) // (2 << 20) * (2 << 20)
    mm = mmap.mmap(-1, size, flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS)
    mm.madvise(mmap.MADV_HUGEPAGE)
    _keep.append(mm)
    arr = np.frombuffer(mm, dtype=np.uint8, count=N)
    t = torch.from_numpy(arr)
    t.fill_(0)  # fault the pages in (as huge pages where possible)
    return t


def bench(kind: str, make) -> None:
    try:
        src = make()
        src.copy_(torch.randint(0, 256, (N,), dtype=torch.uint8))
        for name, up in (("to", lambda s: s.to(dev)), ("i32", lambda s: s.view(torch.int32).to(dev).view(torch.uint8))):
            up(src)
            ok = torch.equal(up(src).cpu(), src)
            ts = []
            for _ in range(6):
                t0 = time.time()
                d = up(src)
                float(d[:4].float().sum().cpu())
                ts.append((time.time() - t0) * 1e3)
            ts.sort()
            print(f"RESULT {kind:7s} {name:4s} correct {ok} median {ts[3]:6.1f} ms  min {ts[0]:6.1f} ms  ({N / ts[3] / 1e6:4.2f} GB/s median)",
                  flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"RESULT {kind} ERROR {e!r}"[:300], flush=True)


if __name__ == "__main__":
    print("pinned allocator:", torch.empty(1, pin_memory=True).is_pinned() if True else None, flush=True)
    for kind, make in (("plain", plain), ("pinned", pinned), ("thp", thp), ("plain2", plain)):
        bench(kind, make)
    print("H2D3_DONE", flush=True)
