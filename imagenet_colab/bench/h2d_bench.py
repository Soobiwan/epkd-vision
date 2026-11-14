"""Host->TPU upload methods for the training step's uint8 image buffers: speed AND correctness.
Correctness = the device tensor equals the host data at upload time, even after the host buffer is overwritten right
after the call (catches lazy/aliasing transfers), checked by a device->host round trip.
    python3 bench/h2d_bench.py            (prints one RESULT line per method/size, then H2D_BENCH_DONE)
"""
from __future__ import annotations

import time

import torch
import torch_tpu  # noqa: F401

dev = torch.device("tpu")
try:
    from torch_tpu._internal.batch_transfer import batch_transfer_impl as BT
except Exception:  # noqa: BLE001
    BT = None


def sync(t: torch.Tensor) -> None:
    t.view(-1)[:1].cpu()


def m_copy(src, state):
    if "buf" not in state:
        state["buf"] = torch.zeros(src.numel(), dtype=src.dtype, device=dev)
    state["buf"].copy_(src)
    return state["buf"]


def m_to(src, state):
    return src.to(dev)


def m_to_nb(src, state):
    return src.to(dev, non_blocking=True)


def m_i32(src, state):
    return src.view(torch.int32).to(dev).view(torch.uint8)


def m_bt_existing(src, state):
    if "buf" not in state:
        state["buf"] = torch.zeros(src.numel(), dtype=src.dtype, device=dev)
    BT.batch_transfer_h2d_sync([src], [state["buf"]])
    return state["buf"]


def m_bt_fresh(src, state):
    dst = torch.empty(src.numel(), dtype=src.dtype, device=dev)
    BT.batch_transfer_h2d_sync([src], [dst])
    return dst


METHODS = {"copy": m_copy, "to": m_to, "to_nb": m_to_nb, "i32": m_i32}
if BT is not None:
    METHODS.update(bt_existing=m_bt_existing, bt_fresh=m_bt_fresh)


def main() -> None:
    g = torch.Generator().manual_seed(0)
    for crop in (112, 196):
        n = 512 * 3 * crop * crop
        host = torch.randint(0, 256, (n,), dtype=torch.uint8, generator=g)
        for name, fn in METHODS.items():
            state: dict = {}
            try:
                # correctness: upload, then immediately clobber the host buffer, then read the device copy back
                ok = True
                for rep in range(3):
                    src = host.clone()
                    src[: 1 << 16].fill_(rep)  # different content per rep (catches stale buffers)
                    want = src.clone()
                    d = fn(src, state)
                    src.fill_(7)
                    got = d.cpu()
                    ok &= bool(torch.equal(got, want))
                # speed: 10 uploads + device sync
                src = host.clone()
                fn(src, state)
                sync(fn(src, state))
                t0 = time.time()
                for _ in range(10):
                    d = fn(src, state)
                sync(d)
                dt = (time.time() - t0) / 10
                print(f"RESULT crop {crop} {name:12s} correct {ok} {dt * 1e3:7.1f} ms/upload  {n / dt / 1e9:6.2f} GB/s", flush=True)
            except Exception as e:  # noqa: BLE001
                print(f"RESULT crop {crop} {name:12s} ERROR {e!r}"[:300], flush=True)
    print("H2D_BENCH_DONE", flush=True)


if __name__ == "__main__":
    main()
