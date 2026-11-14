#!/usr/bin/env python3
"""Loader-only throughput (no accelerator): rows/s and images/s of StepLoader for one stage, i.e. the CPU ceiling of
decode + crops.  python bench/bench_loader.py --config CFG --crop 112 --steps 60 --decode-procs 6"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import epkd_colab  # noqa: F401,E402
from epkd_colab.config import build_teacher, load_cfg  # noqa: E402
from epkd_colab.loader import StepLoader  # noqa: E402
from tgsc_kd.imagenet_parquet import ImageNetTGSCPreprocessor, StageIndexArrays  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--crop", type=int, default=112)
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--decode-procs", type=int, default=6)
    ap.add_argument("--stage-index", default=None, help="override stage index parquet path")
    ap.add_argument("--epoch", type=int, default=1)
    a = ap.parse_args()
    cfg = load_cfg(a.config)
    teacher, ckpt = build_teacher(cfg)
    pre = ImageNetTGSCPreprocessor(teacher, cfg.data, cfg.tgsc, teacher_name=cfg.teacher_model, teacher_checkpoint=ckpt)
    path = Path(a.stage_index) if a.stage_index else pre.stage_index_path(a.crop)
    split = pre.train_split
    arrays = StageIndexArrays.from_parquet(path, split.total_rows)
    file_rows = [int(arrays.offsets[min(off + cnt, split.total_rows)] - arrays.offsets[off]) for off, cnt in zip(split.offsets, split.row_counts)]
    anchor = 224 if a.crop < 224 else None
    loader = StepLoader(paths=split.paths, offsets=split.offsets, row_counts=split.row_counts, total_rows=split.total_rows,
                        stage_index_path=str(path), file_rows=file_rows, seed=27, epoch=a.epoch, crop=a.crop, anchor=anchor, short=256,
                        jitter=10, flip_prob=0.5, kd_weight_fn=lambda s, q: torch.ones_like(s), decode_procs_per_worker=a.decode_procs,
                        pin=False, max_steps=a.steps)
    t0 = time.time()
    n_rows = n_steps = 0
    t_first = None
    for sb in loader:
        if t_first is None:
            t_first = time.time()
            print(f"first step after {t_first - t0:.1f}s (workers start + buffers fill)", flush=True)
        n_rows += sb.rows
        n_steps += 1
        if n_steps % 20 == 0:
            el = time.time() - t_first
            print(f"{n_steps} steps {n_rows} rows: {n_rows / el:.0f} rows/s = {n_rows / 3 / el:.0f} img/s (3 rows/img) | gather {loader.t_gather:.1f}s wait {loader.t_wait:.1f}s", flush=True)
    el = time.time() - t_first
    print(f"DONE {n_steps} steps {n_rows} rows in {el:.1f}s: {n_rows / el:.0f} rows/s, {n_rows / 3 / el:.0f} img/s", flush=True)
    loader.close()


if __name__ == "__main__":
    main()
