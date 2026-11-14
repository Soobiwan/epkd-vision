#!/usr/bin/env python3
"""Parallel, resumable ImageNet TGSC cache precompute for one teacher and several cache configs.

The per-image teacher forward/backward (Grad-CAM++ / saliency union map) is computed ONCE per image and
shared by every target config (e.g. the epochs 1-80 multi-stage cache and the 81-100 224px tail cache of
the same teacher).  Candidate generation, teacher scoring and top/low selection are run per target with
the unmodified ``ImageNetTGSCPreprocessor.emit_stage_rows`` (identical to ``_process_image``).

Work is split into chunks of ``cache_shard_size`` samples; process i of P handles chunks c with
c % P == i.  Each finished chunk writes its part files then a ``.done`` marker per target, so any process
can be killed and restarted.  ``--finalize`` concatenates the parts, stable-sorts them by sample_id
(bug-1 fix) and marks the cache complete so the trainer's ``ensure_cache()`` accepts it.

    precompute_cache.py --configs A1.json A2.json --prepare
    precompute_cache.py --configs A1.json A2.json --device cuda:0 --proc-index 0 --num-procs 3
    precompute_cache.py --configs A1.json A2.json --finalize
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch

from common import build_teacher, load_cfg  # type: ignore

from tgsc_kd.imagenet_parquet import (  # noqa: E402
    ImageNetTGSCPreprocessor,
    _candidate_part_table,
    _stage_part_table,
    _write_table,
)


def make_targets(config_paths: list[str], device: torch.device, cache_dir: Path | None):
    cfgs = [load_cfg(p, cache_dir=cache_dir) for p in config_paths]
    names = {c.teacher_model for c in cfgs}
    assert len(names) == 1, f"all configs must share one teacher, got {names}"
    teacher, ckpt = build_teacher(cfgs[0])
    teacher.to(device).eval()
    targets = [
        ImageNetTGSCPreprocessor(teacher, c.data, c.tgsc, teacher_name=c.teacher_model, teacher_checkpoint=ckpt)
        for c in cfgs
    ]
    sig = {t._union_signature() for t in targets}
    assert len(sig) == 1, f"configs differ in union-map settings: {sig}"
    assert len({t.data_cfg.analysis_resize_short for t in targets}) == 1
    assert len({t.train_split.total_rows for t in targets}) == 1
    assert len({int(t.tgsc_cfg.cache_shard_size) for t in targets}) == 1
    edge = {t.edge_align for t in targets}
    print(f"[targets] edge_align_windows={sorted(edge)} roots={[t.cache_root.name for t in targets]}", file=sys.stderr, flush=True)
    return cfgs, teacher, ckpt, targets


def chunk_marker(target: ImageNetTGSCPreprocessor, chunk: int) -> Path:
    return target.cache_root / "chunks_done" / f"c{chunk:05d}.done"


def n_chunks(target: ImageNetTGSCPreprocessor) -> int:
    cs = int(target.tgsc_cfg.cache_shard_size)
    return -(-target.train_split.total_rows // cs)


def prepare(targets: list[ImageNetTGSCPreprocessor]) -> None:
    for t in targets:
        state = t._load_state()
        t._ensure_manifests(state)
        t._checkpoint_state(state, phase="candidates", phase_detail="parallel precompute")
        print(f"[prepare] {t.cache_root} stages={t.stage_sizes} k={t.tgsc_cfg.k_per_img} "
              f"top={t.tgsc_cfg.top_k_crops} low={t.tgsc_cfg.low_k_crops} edge_align={t.edge_align}", flush=True)


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def claim_iter(targets, total_chunks: int):
    """Dynamic work queue: atomically claim the next unfinished chunk (O_EXCL lock file holding our pid);
    locks of dead processes are taken over.  Lets the launcher add/remove workers on any GPU at any time."""
    lock_dir = targets[0].cache_root / "chunk_claims"
    lock_dir.mkdir(parents=True, exist_ok=True)
    for c in range(total_chunks):
        if all(chunk_marker(t, c).exists() for t in targets):
            continue
        lock = lock_dir / f"c{c:05d}.lock"
        for _attempt in range(2):
            try:
                fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.write(fd, str(os.getpid()).encode())
                os.close(fd)
                break
            except FileExistsError:
                try:
                    owner = int(lock.read_text().strip() or 0)
                except (OSError, ValueError):
                    owner = 0
                if owner and _pid_alive(owner):
                    lock = None
                    break
                try:
                    lock.unlink()
                except FileNotFoundError:
                    pass
        else:
            lock = None
        if lock is None:
            continue
        if all(chunk_marker(t, c).exists() for t in targets):
            lock.unlink(missing_ok=True)
            continue
        yield c
        lock.unlink(missing_ok=True)


def run_chunks(targets, *, proc_index: int, num_procs: int, workers: int, max_chunks: int | None,
               log_every: int, claim: bool = False) -> None:
    lead = targets[0]
    total_chunks = n_chunks(lead)
    cs = int(lead.tgsc_cfg.cache_shard_size)
    if claim:
        todo = claim_iter(targets, total_chunks)
        print(f"[proc {proc_index} pid {os.getpid()}] claim mode, {total_chunks} chunks total", flush=True)
    else:
        mine = [c for c in range(total_chunks) if c % num_procs == proc_index]
        todo = [c for c in mine if not all(chunk_marker(t, c).exists() for t in targets)]
        if max_chunks is not None:
            todo = todo[:max_chunks]
        print(f"[proc {proc_index}/{num_procs}] {len(todo)} of {len(mine)} chunks to do (total {total_chunks})", flush=True)
    for t in targets:
        t.tgsc_cfg.precompute_workers = workers
    done_imgs = 0
    t0 = time.time()
    for chunk in todo:
        start = chunk * cs
        stop = min(start + cs, lead.train_split.total_rows)
        cand = [{s: [] for s in t.stage_sizes} for t in targets]
        stage = [{s: [] for s in t.stage_sizes} for t in targets]
        tc = time.time()
        n = 0
        for sample in lead._make_precompute_loader(start=start, stop=stop):
            analysis = lead.compute_union(image_tensor=sample["image_tensor"], label=int(sample["label"]))
            sid = int(sample["sample_id"])
            for i, t in enumerate(targets):
                t.emit_stage_rows(analysis, sample_id=sid, candidate_rows=cand[i], stage_rows=stage[i])
            n += 1
            done_imgs += 1
            if log_every and done_imgs % log_every == 0:
                el = time.time() - t0
                print(f"[proc {proc_index}] {done_imgs} imgs {done_imgs / el:.1f} img/s", flush=True)
        if n != stop - start:
            print(f"[proc {proc_index}] WARNING chunk {chunk}: {n} of {stop - start} samples processed", flush=True)
        for i, t in enumerate(targets):
            for s in t.stage_sizes:
                _write_table(t.candidate_dir(s) / f"part-c{chunk:05d}.parquet", _candidate_part_table(cand[i][s]))
                _write_table(t.stage_tmp_dir(s) / f"part-c{chunk:05d}.parquet", _stage_part_table(stage[i][s]))
            m = chunk_marker(t, chunk)
            m.parent.mkdir(parents=True, exist_ok=True)
            m.write_text(json.dumps({"start": start, "stop": stop, "processed": n, "secs": time.time() - tc,
                                     "pid": os.getpid(), "proc": proc_index}))
        print(f"[proc {proc_index}] chunk {chunk} [{start},{stop}) done in {time.time() - tc:.0f}s "
              f"({n / max(1e-6, time.time() - tc):.1f} img/s)", flush=True)


def bench(targets, *, n: int, workers: int, start: int = 0) -> None:
    lead = targets[0]
    for t in targets:
        t.tgsc_cfg.precompute_workers = workers
    cand = [{s: [] for s in t.stage_sizes} for t in targets]
    stage = [{s: [] for s in t.stage_sizes} for t in targets]
    t_union = t_emit = 0.0
    warm = 20
    t0 = None
    for i, sample in enumerate(lead._make_precompute_loader(start=start, stop=start + n)):
        if i == warm:
            torch.cuda.synchronize()
            t0 = time.time()
            t_union = t_emit = 0.0
        a = time.time()
        analysis = lead.compute_union(image_tensor=sample["image_tensor"], label=int(sample["label"]))
        torch.cuda.synchronize()
        b = time.time()
        for j, t in enumerate(targets):
            t.emit_stage_rows(analysis, sample_id=int(sample["sample_id"]), candidate_rows=cand[j], stage_rows=stage[j])
        c = time.time()
        t_union += b - a
        t_emit += c - b
    el = time.time() - t0
    m = n - warm
    print(f"[bench] {m} imgs in {el:.1f}s = {m / el:.2f} img/s | union {1000 * t_union / m:.1f} ms/img, "
          f"emit {1000 * t_emit / m:.1f} ms/img | rows/target {[sum(len(v) for v in st.values()) for st in stage]}",
          flush=True)


def finalize(targets) -> bool:
    ok = True
    for t in targets:
        total = n_chunks(t)
        done = [c for c in range(total) if chunk_marker(t, c).exists()]
        if len(done) != total:
            print(f"[finalize] {t.cache_root}: {len(done)}/{total} chunks done -> not finalizing", flush=True)
            ok = False
            continue
        processed = sum(json.loads(chunk_marker(t, c).read_text())["processed"] for c in range(total))
        state = t._load_state()
        t._ensure_manifests(state)
        state["candidate_generation_complete"] = True
        state["processed_train_samples"] = processed
        state["last_checkpointed_sample"] = processed
        state["active_chunk"] = None
        state["candidate_parts"] = {str(s): total for s in t.stage_sizes}
        state["stage_parts"] = {str(s): total for s in t.stage_sizes}
        t._checkpoint_state(state, phase="finalizing_stage_indexes")
        t._finalize_stage_indexes(state)  # sorts by sample_id (bug-1 fix)
        state["complete"] = True
        state["teacher_identity"] = t._teacher_identity()
        t._checkpoint_state(state, phase="complete")
        t._stage_summaries_printed = False
        t._print_stage_summaries()
        print(f"[finalize] {t.cache_root} complete, processed={processed}/{t.train_split.total_rows}", flush=True)
    return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="+", required=True)
    ap.add_argument("--cache-dir", type=Path, default=None)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--proc-index", type=int, default=0)
    ap.add_argument("--num-procs", type=int, default=1)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--max-chunks", type=int, default=None)
    ap.add_argument("--log-every", type=int, default=0)
    ap.add_argument("--prepare", action="store_true")
    ap.add_argument("--claim", action="store_true", help="dynamic chunk claiming (launcher mode)")
    ap.add_argument("--status", action="store_true", help="print done/total chunks and exit")
    ap.add_argument("--finalize", action="store_true")
    ap.add_argument("--print-roots", action="store_true")
    ap.add_argument("--bench", type=int, default=0, help="benchmark N images (no writes)")
    ap.add_argument("--bench-start", type=int, default=0)
    ap.add_argument("--cudnn-benchmark", type=int, default=0,
                    help="0 (default): per-image input shapes vary, autotuning every new shape is very slow")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    # original precompute ran with benchmark=True (set_seed); numerically equivalent up to fp rounding
    torch.backends.cudnn.benchmark = bool(args.cudnn_benchmark)
    torch.backends.cudnn.deterministic = False
    if device.type == "cuda":
        torch.cuda.set_device(device)
    _cfgs, _teacher, ckpt, targets = make_targets(args.configs, device, args.cache_dir)
    if args.print_roots:
        for t in targets:
            print(t.cache_root)
        return 0
    if args.bench:
        bench(targets, n=args.bench, workers=args.workers, start=args.bench_start)
        return 0
    if args.status:
        tot = n_chunks(targets[0])
        done = sum(all(chunk_marker(t, c).exists() for t in targets) for c in range(tot))
        complete = all(t._cache_is_complete(t._load_state()) for t in targets)
        print(json.dumps({"done": done, "total": tot, "complete": complete,
                          "roots": [str(t.cache_root) for t in targets]}))
        return 0
    if args.prepare:
        prepare(targets)
        return 0
    if args.finalize:
        return 0 if finalize(targets) else 2
    run_chunks(targets, proc_index=args.proc_index, num_procs=args.num_procs, workers=args.workers,
               max_chunks=args.max_chunks, log_every=args.log_every, claim=args.claim)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
