#!/usr/bin/env python3
"""Download the ILSVRC/imagenet-1k parquet shards (train + validation) from Hugging Face into the data dir, in
shard order, resumable (hf_hub_download skips complete files).  The token must be at ~/.cache/huggingface/token
(uploaded as a file; never passed on the command line).

    python download_hf.py --out /media/e12056/ASi/imagenet/data [--train-shards 294] [--val-shards 14] [--workers 4]
Writes <out>/download_status.json ({"train_done": n, "val_done": m, "complete": bool}) after every file, so a
precompute process can start on shards that are already present.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
import time
from pathlib import Path

REPO = "ILSVRC/imagenet-1k"
N_TRAIN, N_VAL = 294, 14


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--train-shards", type=int, default=N_TRAIN)
    ap.add_argument("--val-shards", type=int, default=N_VAL)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--val-first", type=int, default=1)
    a = ap.parse_args()
    from huggingface_hub import hf_hub_download

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    files = []
    val = [f"data/validation-{i:05d}-of-{N_VAL:05d}.parquet" for i in range(a.val_shards)]
    train = [f"data/train-{i:05d}-of-{N_TRAIN:05d}.parquet" for i in range(a.train_shards)]
    files = (val + train) if a.val_first else (train + val)
    status_path = out / "download_status.json"
    done: set[str] = set()
    t0 = time.time()
    total_bytes = 0

    def status() -> None:
        tr = sum(1 for f in train if f in done)
        va = sum(1 for f in val if f in done)
        tmp = status_path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"train_done": tr, "val_done": va, "train_total": a.train_shards, "val_total": a.val_shards,
                                   "complete": tr == a.train_shards and va == a.val_shards, "secs": time.time() - t0,
                                   "gb": total_bytes / 1e9}))
        tmp.replace(status_path)

    def fetch(f: str) -> str:
        # final location: <out>/<basename>; hf_hub_download places it under <out>/data/ -> move (same filesystem)
        p = hf_hub_download(REPO, f, repo_type="dataset", local_dir=str(out / "_hf"))
        dst = out / Path(f).name
        if not dst.exists():
            os.replace(p, dst)
        return f

    for f in files:  # already present
        if (out / Path(f).name).exists():
            done.add(f)
    todo = [f for f in files if f not in done]
    status()
    print(f"[download] {len(done)} present, {len(todo)} to fetch", flush=True)
    with ThreadPoolExecutor(a.workers) as ex:
        futs = {ex.submit(fetch, f): f for f in todo}
        for fut in as_completed(futs):
            f = fut.result()
            done.add(f)
            total_bytes += (out / Path(f).name).stat().st_size
            status()
            el = time.time() - t0
            print(f"[download] {Path(f).name} | {len(done)}/{len(files)} | {total_bytes / 1e9:.1f} GB in {el:.0f}s = {total_bytes / 1e6 / max(el, 1e-6):.0f} MB/s", flush=True)
    status()
    print("[download] complete", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
