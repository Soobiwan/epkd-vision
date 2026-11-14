#!/usr/bin/env python3
"""Simulate the reference data order (1 rank, 6 workers, 512-row buffers, 32-row micro-batches, 3 rows per image)
from the split layout only (no image bytes) and count, per epoch, rows, micro-batches, distinct images per
micro-batch (= anchors) and partial micro-batches.  Used for the cost model."""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import epkd_colab  # noqa: F401,E402
from tgsc_kd.ddp_trainer import balanced_file_assignment  # noqa: E402


def stream(shard_indices, offsets, counts, rows_per_img, rng, buf_size):
    if len(shard_indices) > 1:
        rng.shuffle(shard_indices)
    buffer = []
    for sh in shard_indices:
        for r in range(counts[sh]):
            sid = offsets[sh] + r
            for _ in range(rows_per_img):
                buffer.append(sid)
                if len(buffer) >= buf_size:
                    yield buffer.pop(rng.randrange(len(buffer)))
    while buffer:
        yield buffer.pop(rng.randrange(len(buffer)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", default=str(Path(__file__).resolve().parents[1] / "meta" / "splits.json"))
    ap.add_argument("--epochs", default="1,2")
    ap.add_argument("--seed", type=int, default=27)
    ap.add_argument("--rows-per-img", type=int, default=3)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--mb", type=int, default=32)
    ap.add_argument("--buf", type=int, default=512)
    a = ap.parse_args()
    lay = json.load(open(a.splits))["train"]
    counts, offsets = lay["row_counts"], lay["offsets"]
    file_rows = [c * a.rows_per_img for c in counts]
    for epoch in [int(e) for e in a.epochs.split(",")]:
        assignment = balanced_file_assignment(file_rows, 1, a.seed, epoch)[0]
        its = []
        for w in range(a.workers):
            rng = random.Random(a.seed + epoch * 997 + w)
            its.append(stream(assignment[w::a.workers], offsets, counts, a.rows_per_img, rng, a.buf))
        active = list(range(a.workers))
        pos = 0
        n_rows = n_mb = n_anchor = n_partial = 0
        hist = {}
        while active:
            pos %= len(active)
            it = its[active[pos]]
            rows = []
            for _ in range(a.mb):
                try:
                    rows.append(next(it))
                except StopIteration:
                    break
            if rows:
                n_rows += len(rows)
                n_mb += 1
                d = len(set(rows))
                n_anchor += d
                hist[d] = hist.get(d, 0) + 1
            if len(rows) < a.mb:
                active.pop(pos)
                n_partial += 1 if rows else 0
                continue
            pos += 1
        steps = -(-n_mb // 16)
        print(json.dumps({"epoch": epoch, "rows": n_rows, "microbatches": n_mb, "anchors": n_anchor,
                          "anchors_per_row": round(n_anchor / n_rows, 4), "mean_anchors_per_mb": round(n_anchor / n_mb, 2),
                          "partial_microbatches": n_partial, "steps": steps,
                          "distinct_hist": dict(sorted(hist.items()))}), flush=True)


if __name__ == "__main__":
    main()
