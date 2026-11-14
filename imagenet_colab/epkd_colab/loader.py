"""Order-faithful multiprocess CPU loader producing ready-to-use uint8 crops per 512-row step (device-agnostic).

Reproduces the reference loader (``DistributedStageDataset`` + ``DataLoader(num_workers=6, batch_size=32,
collate_fn=pad_stage_batch)`` on ONE rank) and the reference micro-batch preparation (prep.py):
  * parquet files -> ``balanced_file_assignment`` (world 1) -> logical worker w gets files[w::6], shuffled with
    ``random.Random(seed + epoch*997 + w)``; samples in row order, rows in stage-index order, 512-row shuffle buffer
    popped with the same RNG -> identical row order to the reference for the same seed/epoch;
  * micro-batches of 32 rows round-robin over the active workers (PyTorch DataLoader order), partial last batch
    per worker, zero padding per micro-batch, replicate pad + jitter + flip + anchors (prep.py).
Each logical worker is a process with its own PIL decode sub-pool (reference ``_image_bytes_to_byte_tensor``, so
pixels are identical to the reference); it writes finished micro-batches into a ring of shared-memory buffers.
The main process gathers 16 micro-batches into one pinned step buffer (views (512,3,s,s), anchors
(512,3,224,224) padded to 32 slots per micro-batch, labels, per-row loss weights, ghost-BN group spec).
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import multiprocessing as mp
import os
import queue
import random
import threading
import time
from typing import Any, Optional, Sequence

import numpy as np
import pyarrow.parquet as pq
import torch

from tgsc_kd.ddp_trainer import balanced_file_assignment
from tgsc_kd.imagenet_parquet import StageIndexArrays, StageScoreCalibration, _image_bytes_to_byte_tensor

from .ghost_bn import interleaved_ema_weights, single_ema_weights
from .prep import make_draws, prepare_microbatch

READ_BATCH = 64  # parquet rows per read/decode batch (the reference reads 16; row order unaffected)


def _decode_np(args: tuple[bytes, int]) -> np.ndarray:
    b, short = args
    return _image_bytes_to_byte_tensor(b, short).numpy()


@dataclass
class StreamSpec:
    paths: list[str]
    offsets: list[int]
    row_counts: list[int]
    stage_index_path: str
    total_rows: int
    shard_indices: list[int]
    rng_seed: int  # random.Random seed (reference buffer RNG)
    np_seed: int  # jitter/flip/anchor draws
    buffer_rows: int
    micro_batch: int
    crop: int
    anchor: Optional[int]
    short: int
    jitter: int
    flip_prob: float
    decode_procs: int
    max_microbatches: Optional[int] = None


def _stream_worker(wid: int, spec: StreamSpec, ring: dict[str, torch.Tensor], out_q: mp.Queue, free_q: mp.Queue) -> None:
    """Process body of one logical worker."""
    torch.set_num_threads(1)
    os.environ["OMP_NUM_THREADS"] = "1"
    parent = os.getppid()
    try:
        stage = StageIndexArrays.from_parquet(spec.stage_index_path, spec.total_rows)
        calib = StageScoreCalibration.from_stage_index(stage)
        rng = random.Random(spec.rng_seed)
        nrng = np.random.default_rng(spec.np_seed)
        pool = mp.get_context("fork").Pool(spec.decode_procs) if spec.decode_procs > 0 else None
        shard_indices = list(spec.shard_indices)
        if len(shard_indices) > 1:
            rng.shuffle(shard_indices)
        images: dict[int, np.ndarray] = {}
        refs: dict[int, int] = {}

        def records():
            for shard_idx in shard_indices:
                shard_start = spec.offsets[shard_idx]
                pf = pq.ParquetFile(spec.paths[shard_idx])
                row_in_shard = 0
                pending: list = []  # (async result, todo, meta) of the batches being decoded, oldest first
                shard_rows = spec.row_counts[shard_idx]  # (max_train_samples may truncate the last used shard)
                for batch in pf.iter_batches(columns=["image.bytes", "label"], batch_size=READ_BATCH):
                    if row_in_shard >= shard_rows:
                        break
                    blobs = batch.column(0).field("bytes").to_pylist()
                    labels = batch.column(1).to_pylist()
                    todo, meta = [], []
                    for b, label in zip(blobs, labels):
                        if row_in_shard >= shard_rows:
                            break
                        sid = shard_start + row_in_shard
                        row_in_shard += 1
                        start, end = stage.rows_for(sid)
                        if start == end or b is None or label is None or int(label) < 0:
                            continue
                        todo.append((b, spec.short))
                        meta.append((sid, int(label), start, end))
                    pending.append((pool.map_async(_decode_np, todo, chunksize=4) if pool else None, todo, meta))
                    # keep DECODE_LOOKAHEAD batches decoding ahead (was 1: a worker stalled whenever it caught up with
                    # its decode pool, and the strict round-robin consumption stalls every step with it); order unchanged
                    while len(pending) > DECODE_LOOKAHEAD:
                        yield from emit(pending.pop(0))
                while pending:
                    yield from emit(pending.pop(0))

        def emit(pending):
            res, todo, meta = pending
            arrs = res.get() if res is not None else [_decode_np(t) for t in todo]
            for arr, (sid, _l, start, end) in zip(arrs, meta):
                images[sid] = arr
                refs[sid] = end - start
            for sid, label, start, end in meta:
                for idx in range(start, end):
                    yield (sid, label, calib.normalize_scalar(float(stage.scores[idx])), int(stage.quality_flags[idx]),
                           (float(stage.x0[idx]), float(stage.y0[idx]), float(stage.x1[idx]), float(stage.y1[idx])))

        def buffered():
            buf: list = []
            for item in records():
                buf.append(item)
                if len(buf) >= spec.buffer_rows:
                    yield buf.pop(rng.randrange(len(buf)))
            while buf:
                yield buf.pop(rng.randrange(len(buf)))

        it = buffered()
        produced = 0
        exhausted = False
        while not exhausted:
            if spec.max_microbatches is not None and produced >= spec.max_microbatches:
                break
            if os.getppid() != parent:  # the trainer died: do not linger
                break
            rows = []
            for _ in range(spec.micro_batch):
                try:
                    rows.append(next(it))
                except StopIteration:
                    exhausted = True
                    break
            if not rows:
                break
            n = len(rows)
            sids = np.array([r[0] for r in rows], dtype=np.int64)
            imgs = []
            for r in rows:
                imgs.append(images[r[0]])
                refs[r[0]] -= 1
                if refs[r[0]] == 0:
                    del images[r[0]], refs[r[0]]
            boxes = np.array([r[4] for r in rows], dtype=np.float32)
            n_anchor = len(np.unique(sids)) if spec.anchor is not None else 0
            draws = make_draws(nrng, n, n_anchor, spec.jitter)
            views, anchors, first = prepare_microbatch(imgs, boxes, sids, crop=spec.crop, jitter=spec.jitter,
                                                       flip_prob=spec.flip_prob, anchor_size=spec.anchor, draws=draws)
            while True:  # wait for a free ring slot, but notice a dead parent
                try:
                    slot = free_q.get(timeout=5.0)
                    break
                except queue.Empty:
                    if os.getppid() != parent:
                        return
            ring["xv"][slot, :n].copy_(torch.from_numpy(views))
            ring["yv"][slot, :n].copy_(torch.tensor([r[1] for r in rows], dtype=torch.long))
            ring["score"][slot, :n].copy_(torch.tensor([r[2] for r in rows], dtype=torch.float32))
            ring["qflag"][slot, :n].copy_(torch.tensor([r[3] for r in rows], dtype=torch.int16))
            ring["sid"][slot, :n].copy_(torch.from_numpy(sids))
            a = 0
            if anchors is not None:
                a = anchors.shape[0]
                ring["xa"][slot, :a].copy_(torch.from_numpy(anchors))
                ring["ya"][slot, :a].copy_(torch.tensor([rows[i][1] for i in first], dtype=torch.long))
            out_q.put((slot, n, a))
            produced += 1
        out_q.put(None)
        if pool is not None:
            pool.terminate()
    except BaseException as e:  # noqa: BLE001
        import traceback

        out_q.put(("error", f"worker {wid}: {e!r}\n{traceback.format_exc()}"))


DECODE_LOOKAHEAD = int(os.environ.get("EPKD_DECODE_LOOKAHEAD", "3"))


def _thp_zeros(shape: tuple, dtype: torch.dtype, keep: list) -> torch.Tensor:
    """Zero host tensor backed by transparent huge pages (anonymous mmap + MADV_HUGEPAGE).  On the Colab TPU VM the
    host->TPU upload of a large buffer through the int32 path runs at ~4.4 GB/s from huge pages but ~0.33 GB/s from
    ordinary 4 KB pages once host memory is fragmented (11-12 bench: 17 ms vs 226 ms for 77 MB).  ``keep`` holds the
    mmap alive as long as the tensor's owner."""
    import mmap

    n = int(np.prod(shape)) * torch.empty((), dtype=dtype).element_size()
    size = max((n + (2 << 20) - 1) // (2 << 20) * (2 << 20), 2 << 20)
    mm = mmap.mmap(-1, size, flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS)
    if hasattr(mm, "madvise") and hasattr(mmap, "MADV_HUGEPAGE"):
        mm.madvise(mmap.MADV_HUGEPAGE)
    keep.append(mm)
    t = torch.from_numpy(np.frombuffer(mm, dtype=np.uint8, count=n)).view(dtype).view(shape)
    t.zero_()  # fault the pages in now
    return t


class StepBatch:
    """One 512-row step on the host (pinned): see keys below."""

    def __init__(self, G: int, slot: int, crop: int, anchor: Optional[int], pin: bool) -> None:
        N = G * slot
        kw = dict(pin_memory=pin)
        self.G, self.slot, self.crop, self.anchor = G, slot, crop, anchor
        # image buffers on huge pages (EPKD_HOST_THP=1, default on TPU hosts: the caller passes pin=False there)
        self._mm: list = []
        thp = os.environ.get("EPKD_HOST_THP", "1") == "1" and not pin
        img = (lambda shp: _thp_zeros(shp, torch.uint8, self._mm)) if thp else (lambda shp: torch.zeros(*shp, dtype=torch.uint8, **kw))
        self.xv = img((N, 3, crop, crop))
        self.yv = torch.zeros(N, dtype=torch.long, **kw)
        self.wv = torch.zeros(N, dtype=torch.float32, **kw)  # loss weight (1/512 for real rows)
        self.mv = torch.zeros(N, dtype=torch.float32, **kw)  # 1 for real rows (BN stats, metrics)
        self.kdw = torch.ones(N, dtype=torch.float32, **kw)  # score-gate KD weight (1 with kd_gate off)
        self.sid = torch.zeros(N, dtype=torch.long)
        if anchor is not None:
            self.xa = img((N, 3, anchor, anchor))
            self.ya = torch.zeros(N, dtype=torch.long, **kw)
            self.wa = torch.zeros(N, dtype=torch.float32, **kw)
            self.ma = torch.zeros(N, dtype=torch.float32, **kw)
        # ghost-BN spec (views / anchors): per-group 1/n, n, ema weight; keep factors
        self.inv_v = torch.zeros(G, **kw)
        self.n_v = torch.zeros(G, **kw)
        self.ema_v = torch.zeros(G, **kw)
        self.keep_v = torch.ones((), **kw)
        self.inv_a = torch.zeros(G, **kw)
        self.n_a = torch.zeros(G, **kw)
        self.ema_a = torch.zeros(G, **kw)
        self.keep_a = torch.ones((), **kw)
        self.counts_v: list[int] = [0] * G
        self.counts_a: list[int] = [0] * G
        self.rows = 0

    def tensors(self) -> dict[str, torch.Tensor]:
        names = ["xv", "yv", "wv", "mv", "kdw", "inv_v", "n_v", "ema_v", "keep_v"]
        if self.anchor is not None:
            names += ["xa", "ya", "wa", "ma", "inv_a", "n_a", "ema_a", "keep_a"]
        return {k: getattr(self, k) for k in names}


class StepLoader:
    """Iterates StepBatch objects for one epoch (reference order).  ``kd_weight_fn(scores, qflags)`` is the
    reference ``_score_to_kd_weights`` (CPU tensors)."""

    def __init__(self, *, paths: Sequence[str], offsets: Sequence[int], row_counts: Sequence[int], total_rows: int,
                 stage_index_path: str, file_rows: Sequence[int], seed: int, epoch: int, crop: int, anchor: Optional[int],
                 short: int, jitter: int, flip_prob: float, kd_weight_fn: Any, num_workers: int = 6, micro_batch: int = 32,
                 step_rows: int = 512, buffer_rows: int = 512, decode_procs_per_worker: int = 6, ring_slots: int = 12,
                 bn_momentum: float = 0.1, pin: bool = True, max_steps: Optional[int] = None, n_step_buffers: int = 3,
                 gather_thread: bool = False) -> None:
        self.G = step_rows // micro_batch
        self.slot = micro_batch
        self.crop, self.anchor = int(crop), anchor
        self.kd_weight_fn = kd_weight_fn
        self.bn_momentum = bn_momentum
        self.num_workers = int(num_workers)
        assignment = balanced_file_assignment(list(file_rows), 1, int(seed), int(epoch))[0]
        self.epoch_rows = int(sum(file_rows[i] for i in assignment))
        ctx = mp.get_context("spawn")
        self.procs, self.out_qs, self.free_qs, self.rings = [], [], [], []
        max_mb = (max_steps * self.G) if max_steps else None
        for wid in range(self.num_workers):
            ring = {
                "xv": torch.zeros(ring_slots, micro_batch, 3, crop, crop, dtype=torch.uint8).share_memory_(),
                "yv": torch.zeros(ring_slots, micro_batch, dtype=torch.long).share_memory_(),
                "score": torch.zeros(ring_slots, micro_batch, dtype=torch.float32).share_memory_(),
                "qflag": torch.zeros(ring_slots, micro_batch, dtype=torch.int16).share_memory_(),
                "sid": torch.zeros(ring_slots, micro_batch, dtype=torch.long).share_memory_(),
            }
            if anchor is not None:
                ring["xa"] = torch.zeros(ring_slots, micro_batch, 3, anchor, anchor, dtype=torch.uint8).share_memory_()
                ring["ya"] = torch.zeros(ring_slots, micro_batch, dtype=torch.long).share_memory_()
            out_q, free_q = ctx.Queue(), ctx.Queue()
            for s in range(ring_slots):
                free_q.put(s)
            spec = StreamSpec(
                paths=[str(p) for p in paths], offsets=list(offsets), row_counts=list(row_counts), stage_index_path=str(stage_index_path),
                total_rows=int(total_rows), shard_indices=assignment[wid::self.num_workers],
                rng_seed=int(seed) + int(epoch) * 997 + wid, np_seed=int(seed) * 7919 + int(epoch) * 104729 + wid,
                buffer_rows=int(buffer_rows), micro_batch=int(micro_batch), crop=int(crop), anchor=anchor, short=int(short),
                jitter=int(jitter), flip_prob=float(flip_prob), decode_procs=int(decode_procs_per_worker), max_microbatches=max_mb,
            )
            p = ctx.Process(target=_stream_worker, args=(wid, spec, ring, out_q, free_q), daemon=False, name=f"stream-{wid}")
            p.start()
            self.procs.append(p)
            self.out_qs.append(out_q)
            self.free_qs.append(free_q)
            self.rings.append(ring)
        self.step_bufs = [StepBatch(self.G, self.slot, crop, anchor, pin) for _ in range(n_step_buffers)]
        self._buf_i = 0
        self.steps = 0
        self.t_gather = 0.0
        self.t_wait = 0.0
        # gather_thread: a background thread assembles step buffers ahead of the trainer.  A buffer is refilled only after
        # the trainer hands it back with release() (i.e. after its host->device upload finished), never by rotation.
        self.gather_thread = bool(gather_thread)
        self._free: Optional[queue.Queue] = None

    def _next_microbatch(self, active: list[int], pos: int) -> Optional[tuple[int, int, int, int]]:
        """Returns (worker, slot, n, a) or None when all workers are exhausted; updates ``active`` in place."""
        while active:
            pos %= len(active)
            w = active[pos]
            t0 = time.time()
            item = self.out_qs[w].get()
            self.t_wait += time.time() - t0
            if item is None:
                active.pop(pos)
                continue
            if isinstance(item, tuple) and item and item[0] == "error":
                raise RuntimeError(item[1])
            slot, n, a = item
            self._pos = pos + 1
            return w, slot, n, a
        return None

    def _rotating_buf(self) -> "StepBatch":
        sb = self.step_bufs[self._buf_i % len(self.step_bufs)]
        self._buf_i += 1
        return sb

    def release(self, sb: "StepBatch") -> None:
        """The trainer is done reading ``sb`` (its upload finished): it may be refilled.  No-op without gather_thread."""
        if self._free is not None:
            self._free.put(sb)

    def __iter__(self):
        if not self.gather_thread:
            yield from self._gather_iter(self._rotating_buf)
            return
        self._free = queue.Queue()
        for b in self.step_bufs:
            self._free.put(b)
        ready: queue.Queue = queue.Queue()

        def producer() -> None:
            try:
                for sb in self._gather_iter(self._free.get):
                    ready.put(sb)
                ready.put(None)
            except BaseException as e:  # noqa: BLE001
                ready.put(e)

        threading.Thread(target=producer, daemon=True, name="step-gather").start()
        while True:
            item = ready.get()
            if item is None:
                return
            if isinstance(item, BaseException):
                raise item
            yield item

    def _gather_iter(self, next_buf):
        active = list(range(self.num_workers))
        self._pos = 0
        done = False
        while not done:
            sb = next_buf()
            t0 = time.time()
            counts_v, counts_a = [0] * self.G, [0] * self.G
            g = 0
            while g < self.G:
                nxt = self._next_microbatch(active, self._pos)
                if nxt is None:
                    done = True
                    break
                w, slot, n, a = nxt
                ring = self.rings[w]
                sl = slice(g * self.slot, g * self.slot + n)
                sb.xv[sl].copy_(ring["xv"][slot, :n])
                sb.yv[sl].copy_(ring["yv"][slot, :n])
                sb.sid[sl].copy_(ring["sid"][slot, :n])
                sb.kdw[sl].copy_(self.kd_weight_fn(ring["score"][slot, :n].clone(), ring["qflag"][slot, :n].clone()))
                counts_v[g] = n
                if self.anchor is not None:
                    sa = slice(g * self.slot, g * self.slot + a)
                    sb.xa[sa].copy_(ring["xa"][slot, :a])
                    sb.ya[sa].copy_(ring["ya"][slot, :a])
                    counts_a[g] = a
                self.free_qs[w].put(slot)
                g += 1
            if g == 0:
                break
            self._finish(sb, counts_v, counts_a)
            self.t_gather += time.time() - t0
            self.steps += 1
            yield sb

    def _finish(self, sb: StepBatch, counts_v: list[int], counts_a: list[int]) -> None:
        G, S = self.G, self.slot
        step_rows = G * S
        mv = torch.zeros(G, S)
        wv = torch.zeros(G, S)
        for g, n in enumerate(counts_v):
            mv[g, :n] = 1.0
            wv[g, :n] = 1.0 / step_rows
        sb.mv.copy_(mv.view(-1))
        sb.wv.copy_(wv.view(-1))
        sb.kdw.mul_(sb.mv)  # pad rows: 0
        n_v = torch.tensor([float(c) for c in counts_v])
        sb.n_v.copy_(n_v)
        sb.inv_v.copy_(torch.where(n_v > 0, 1.0 / n_v.clamp_min(1.0), torch.zeros_like(n_v)))
        if self.anchor is not None:
            ma = torch.zeros(G, S)
            wa = torch.zeros(G, S)
            for g, (n, a) in enumerate(zip(counts_v, counts_a)):
                if a > 0:
                    ma[g, :a] = 1.0
                    wa[g, :a] = float(n) / (step_rows * a)
            sb.ma.copy_(ma.view(-1))
            sb.wa.copy_(wa.view(-1))
            # zero the pad anchor slots (they still go through the network; stats and loss are masked)
            for g, a in enumerate(counts_a):
                if a < S:
                    sb.xa[g * S + a: (g + 1) * S].zero_()
                    sb.ya[g * S + a: (g + 1) * S].zero_()
            n_a = torch.tensor([float(c) for c in counts_a])
            sb.n_a.copy_(n_a)
            sb.inv_a.copy_(torch.where(n_a > 0, 1.0 / n_a.clamp_min(1.0), torch.zeros_like(n_a)))
            w_v, k_v, w_a, k_a = interleaved_ema_weights(counts_v, counts_a, self.bn_momentum)
            sb.ema_a.copy_(torch.tensor(w_a))
            sb.keep_a.fill_(k_a)
        else:
            w_v, k_v = single_ema_weights(counts_v, self.bn_momentum)
        for g, n in enumerate(counts_v):  # zero pad view rows
            if n < S:
                sb.xv[g * S + n: (g + 1) * S].zero_()
                sb.yv[g * S + n: (g + 1) * S].zero_()
        sb.ema_v.copy_(torch.tensor(w_v))
        sb.keep_v.fill_(k_v)
        sb.counts_v, sb.counts_a = list(counts_v), list(counts_a)
        sb.rows = int(sum(counts_v))

    def close(self) -> None:
        for p in self.procs:
            if p.is_alive():
                p.terminate()
        for p in self.procs:
            p.join(timeout=10)
