"""Tiny CPU test of the flat-image H2D path in StepFn (crop 8 ~ '112', crop 10 ~ '140', both small): flat host buffer ->
device (CPU) -> reshape == original, and the per-key fallback to copy_ when batch_transfer raises.  python tests/test_flat_h2d.py"""
from __future__ import annotations

import sys
import types
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import epkd_colab  # noqa: F401,E402
from epkd_colab.loader import StepBatch  # noqa: E402
from epkd_colab.step import StepFn  # noqa: E402


def make(crop, bt=None):
    G, S = 2, 2
    sb = StepBatch(G, S, crop, 12, pin=False)
    sb.xv.copy_(torch.randint(0, 256, sb.xv.shape, dtype=torch.uint8))
    sb.xa.copy_(torch.randint(0, 256, sb.xa.shape, dtype=torch.uint8))
    fn = object.__new__(StepFn)  # skip the heavy constructor; only the transfer state is needed
    fn.device, fn.dev, fn._bt, fn._bt_bad, fn.N = torch.device("cpu"), {}, bt, set(), G * S
    return fn, sb


def check(fn, sb):
    d = fn.upload(sb)
    for k in ("xv", "xa"):
        assert d[k].dim() == 1 and d[k].dtype == torch.uint8
        assert torch.equal(fn._img(d, k), getattr(sb, k)), k
    for k, v in sb.tensors().items():
        if k not in ("xv", "xa"):
            assert torch.equal(d[k], v), k


def main():
    for crop in (8, 10):
        check(*make(crop))
    # per-key fallback: transfer raises once per key, then is never called again for it
    calls = []

    def boom(src, dst):
        calls.append(1)
        raise RuntimeError("Source buffer too small for raw copy")

    fn, sb = make(10, types.SimpleNamespace(batch_transfer_h2d_sync=boom))
    # the transfer branch needs >= 1 MiB: use one 1.08 MB image buffer
    sb.xv = torch.randint(0, 256, (1, 3, 600, 600), dtype=torch.uint8)  # 1.08 MB
    fn.N = 1
    for _ in range(3):
        d = fn.upload(sb)
        assert torch.equal(fn._img(d, "xv"), sb.xv)
    assert len(calls) == 1 and fn._bt_bad == {"xv"}, (calls, fn._bt_bad)
    print("test_flat_h2d OK (flat roundtrip crops 8/10; fallback called transfer once, then copy_)")


if __name__ == "__main__":
    main()
