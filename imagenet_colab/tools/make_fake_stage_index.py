#!/usr/bin/env python3
"""Synthetic stage indexes (3 boxes per image at fixed grid positions inside a 256x256 region) for loader /
throughput tests that do not need real proposals.  Writes stage_res{res}.parquet files in --out."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import epkd_colab  # noqa: F401,E402
from tgsc_kd.imagenet_parquet import _stage_index_schema, write_sorted_stage_index  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, required=True, help="number of samples (ids 0..n-1)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--res", default="112,140,168,196,224")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for res in [int(r) for r in a.res.split(",")]:
        ids = np.repeat(np.arange(a.n, dtype=np.int64), 3)
        rank = np.tile(np.arange(3, dtype=np.int16), a.n)
        off = 256 - res
        y0 = np.where(rank == 0, 0, np.where(rank == 1, off, off // 2)).astype(np.int32)
        x0 = np.where(rank == 0, 0, np.where(rank == 1, off, off // 2)).astype(np.int32)
        table = pa.table({
            "sample_id": ids, "resolution": np.full(len(ids), res, dtype=np.int16), "y0": y0, "x0": x0,
            "y1": (y0 + res).astype(np.int32), "x1": (x0 + res).astype(np.int32),
            "score": np.full(len(ids), 0.9, dtype=np.float32), "quality_flag": np.ones(len(ids), dtype=np.int8), "rank": rank,
        }, schema=_stage_index_schema())
        write_sorted_stage_index(table, out / f"stage_res{res}.parquet")
        print(out / f"stage_res{res}.parquet", len(ids))


if __name__ == "__main__":
    main()
