#!/usr/bin/env python3
"""Scan the HF ImageNet parquet shards (bytes only, no decode) and record per-image JPEG header facts:
magic (JPEG / PNG / other), SOF type (baseline / progressive / other), number of components (1 gray, 3 YCbCr,
4 CMYK), original width/height, byte size.  Also writes the split layout (paths, row counts, offsets) so the
Colab pipeline can reproduce ParquetSplitInfo before every shard has been downloaded.

Output: <out>/<split>_headers.npz (int32 arrays: sample_id, w, h, comps, sof, magic, nbytes) and <out>/splits.json.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import struct
import sys
from multiprocessing import Pool

import numpy as np
import pyarrow.parquet as pq

MAGIC_JPEG, MAGIC_PNG, MAGIC_OTHER = 0, 1, 2
SOF_BASELINE, SOF_PROGRESSIVE, SOF_OTHER, SOF_NONE = 0, 1, 2, 3


def parse(b: bytes):
    if b[:8] == b"\x89PNG\r\n\x1a\n":
        w, h = struct.unpack(">II", b[16:24])
        return MAGIC_PNG, SOF_NONE, 0, w, h
    if b[:2] != b"\xff\xd8":
        return MAGIC_OTHER, SOF_NONE, 0, 0, 0
    i, n = 2, len(b)
    sof, comps, w, h = SOF_NONE, 0, 0, 0
    while i + 4 <= n:
        if b[i] != 0xFF:
            i += 1
            continue
        m = b[i + 1]
        if m in (0xD8, 0x01) or 0xD0 <= m <= 0xD7:
            i += 2
            continue
        if m == 0xFF:
            i += 1
            continue
        seg = struct.unpack(">H", b[i + 2:i + 4])[0]
        if m in (0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF):
            if i + 10 <= n:
                h, w = struct.unpack(">HH", b[i + 5:i + 9])
                comps = b[i + 9]
            sof = SOF_BASELINE if m in (0xC0, 0xC1) else SOF_PROGRESSIVE if m == 0xC2 else SOF_OTHER
            break
        if m == 0xDA:  # start of scan without SOF
            break
        i += 2 + seg
    return MAGIC_JPEG, sof, comps, w, h


def scan(args):
    path, offset = args
    pf = pq.ParquetFile(path)
    rows = []
    r = 0
    for batch in pf.iter_batches(columns=["image.bytes"], batch_size=256):
        for b in batch.column(0).field("bytes").to_pylist():
            if b is None:
                rows.append((offset + r, 0, 0, 0, MAGIC_OTHER, SOF_NONE, 0))
            else:
                magic, sof, comps, w, h = parse(b)
                rows.append((offset + r, w, h, comps, magic, sof, len(b)))
            r += 1
    return path, np.asarray(rows, dtype=np.int64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/media/e12056/ASi/imagenet/data")
    ap.add_argument("--out", required=True)
    ap.add_argument("--procs", type=int, default=3)
    ap.add_argument("--splits", default="train,validation")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    layout = {}
    for split in a.splits.split(","):
        paths = sorted(glob.glob(os.path.join(a.data, f"{split}-*.parquet")))
        counts = [int(pq.ParquetFile(p).metadata.num_rows) for p in paths]
        offs = list(np.cumsum([0] + counts[:-1]))
        layout[split] = {"paths": [os.path.basename(p) for p in paths], "row_counts": counts,
                         "offsets": [int(o) for o in offs], "total_rows": int(sum(counts)),
                         "sizes": [os.path.getsize(p) for p in paths]}
        with Pool(a.procs) as pool:
            parts = []
            for k, (p, arr) in enumerate(pool.imap(scan, list(zip(paths, offs)))):
                parts.append(arr)
                print(f"[{split}] {k + 1}/{len(paths)} {os.path.basename(p)} rows={len(arr)}", flush=True)
        allr = np.concatenate(parts)
        np.savez_compressed(os.path.join(a.out, f"{split}_headers.npz"), sample_id=allr[:, 0], w=allr[:, 1], h=allr[:, 2],
                            comps=allr[:, 3], magic=allr[:, 4], sof=allr[:, 5], nbytes=allr[:, 6])
        print(f"[{split}] total {len(allr)} | png {int((allr[:, 4] == MAGIC_PNG).sum())} other {int((allr[:, 4] == MAGIC_OTHER).sum())} "
              f"| cmyk {int((allr[:, 3] == 4).sum())} gray {int((allr[:, 3] == 1).sum())} | progressive {int((allr[:, 5] == SOF_PROGRESSIVE).sum())} "
              f"sof_other {int((allr[:, 5] == SOF_OTHER).sum())} | bytes {allr[:, 6].sum() / 1e9:.1f} GB", flush=True)
    json.dump(layout, open(os.path.join(a.out, "splits.json"), "w"))
    print("done", flush=True)


if __name__ == "__main__":
    main()
