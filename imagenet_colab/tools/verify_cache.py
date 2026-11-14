#!/usr/bin/env python3
"""[imagenet_colab copy of the reference verify_cache.py; only change: device-agnostic (cpu/tpu) + config loading via epkd_colab.config]
Alignment check of finished caches: recompute boxes/scores for N random sample_ids FROM SCRATCH with the
unmodified single-image path (``ImageNetTGSCPreprocessor._process_image``) and compare with the rows the
training loader would read for that id (``StageIndexArrays.rows_for``).  Requires 100% match.

    verify_cache.py --configs A1.json A2.json --n 60 --device cuda:0 --out logs/verify_rn50.json
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

import sys as _s; from pathlib import Path as _P; _s.path.insert(0, str(_P(__file__).resolve().parents[1]))
import epkd_colab  # noqa
from epkd_colab.config import build_teacher, load_cfg

from tgsc_kd.imagenet_parquet import ImageNetTGSCPreprocessor, StageIndexArrays, _image_bytes_to_tensor  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="+", required=True)
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--score-rtol", type=float, default=1e-4)
    ap.add_argument("--score-atol", type=float, default=1e-5)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--cache-dir", type=Path, default=None)
    ap.add_argument("--tf32", type=int, default=1)
    ap.add_argument("--min-box-rate", type=float, default=1.0,
                    help="tolerant mode (<1.0, e.g. 0.95 for a TPU-built cache checked on CPU): pass a stage if this fraction "
                         "of ids has identical boxes and the scores of every box-matched id agree within --score-rtol/atol")
    ap.add_argument("--score-gate", choices=("on", "off"), default="on",
                    help="off: scores are reported but do not gate (valid only for kd_gate=off + saliency view selection, "
                         "where the cached score never reaches training; e.g. a bf16-scored TPU cache)")
    args = ap.parse_args()
    torch.backends.cudnn.benchmark = False
    if args.device == 'tpu':
        import torch_tpu  # noqa
    dev = torch.device(args.device)
    if dev.type == 'cuda':
        torch.cuda.set_device(dev)
        torch.backends.cudnn.allow_tf32 = bool(args.tf32)
        torch.backends.cuda.matmul.allow_tf32 = bool(args.tf32)
    report: dict = {"configs": args.configs, "n": args.n, "seed": args.seed, "targets": []}
    all_ok = True
    for cfg_path in args.configs:
        cfg = load_cfg(cfg_path, cache=args.cache_dir)
        teacher, ckpt = build_teacher(cfg)
        teacher.to(dev).eval()
        pre = ImageNetTGSCPreprocessor(teacher, cfg.data, cfg.tgsc, teacher_name=cfg.teacher_model, teacher_checkpoint=ckpt)
        state = pre._load_state()
        assert pre._cache_is_complete(state), f"cache {pre.cache_root} incomplete"
        total = pre.train_split.total_rows
        ids = sorted(random.Random(args.seed).sample(range(total), args.n))
        arrays = {s: StageIndexArrays.from_parquet(pre.stage_index_path(s), total) for s in pre.stage_sizes}
        sorted_on_disk = {}
        import pyarrow.parquet as pq
        for s in pre.stage_sizes:
            sid = pq.read_table(pre.stage_index_path(s), columns=["sample_id"])["sample_id"].to_numpy()
            sorted_on_disk[s] = bool(np.all(np.diff(sid) >= 0))
        res = {"config": cfg_path, "cache_root": str(pre.cache_root), "teacher": cfg.teacher_model,
               "teacher_ckpt": str(ckpt), "sorted_on_disk": sorted_on_disk, "ids": ids, "stages": {}}
        per_stage = {s: {"box_match": 0, "score_match": 0, "flag_rank_match": 0, "n": 0, "bad": []} for s in pre.stage_sizes}
        t0 = time.time()
        for sid in ids:
            sample = next(pre.train_split.iter_samples(columns=("image.bytes", "label"), start=sid, stop=sid + 1, batch_size=1))
            assert sample["sample_id"] == sid
            img = _image_bytes_to_tensor(sample["image_bytes"], cfg.data.analysis_resize_short)
            cand = {s: [] for s in pre.stage_sizes}
            rows = {s: [] for s in pre.stage_sizes}
            pre._process_image(sample_id=sid, image_tensor=img, label=int(sample["label"]), candidate_rows=cand, stage_rows=rows)
            for s in pre.stage_sizes:
                a = arrays[s]
                st, en = a.rows_for(sid)  # asserts ownership
                cached = [(int(a.y0[i]), int(a.x0[i]), int(a.y1[i]), int(a.x1[i]), float(a.scores[i]),
                           int(a.quality_flags[i]), int(a.ranks[i])) for i in range(st, en)]
                fresh = [(r["y0"], r["x0"], r["y1"], r["x1"], r["score"], r["quality_flag"], r["rank"]) for r in rows[s]]
                ps = per_stage[s]
                ps["n"] += 1
                same_len = len(cached) == len(fresh)
                bm = same_len and all(c[:4] == f[:4] for c, f in zip(cached, fresh))
                sm = same_len and bool(np.allclose([c[4] for c in cached], [f[4] for f in fresh],
                                                   rtol=args.score_rtol, atol=args.score_atol))
                fm = same_len and all(c[5:] == f[5:] for c, f in zip(cached, fresh))
                ps["box_match"] += int(bm)
                ps["score_match"] += int(sm)
                ps.setdefault("score_match_given_box", 0)
                ps["score_match_given_box"] += int(sm and bm)
                if bm:
                    rel = [abs(c[4] - f[4]) / max(abs(f[4]), 1e-6) for c, f in zip(cached, fresh)]
                    ps["max_rel_score_diff_matched"] = max(ps.get("max_rel_score_diff_matched", 0.0), max(rel, default=0.0))
                ps["flag_rank_match"] += int(fm)
                if not (bm and sm and fm):
                    ps["bad"].append({"sample_id": sid, "cached": cached, "fresh": fresh})
        for s, ps in per_stage.items():
            if args.min_box_rate >= 1.0:
                ok = ps["box_match"] == ps["score_match"] == ps["flag_rank_match"] == ps["n"] and sorted_on_disk[s]
            else:  # tolerant: near-tie window swaps allowed, but matched boxes must carry the same scores
                ok = (ps["box_match"] >= args.min_box_rate * ps["n"] and ps["flag_rank_match"] == ps["n"]
                      and (args.score_gate == "off" or ps.get("score_match_given_box", 0) == ps["box_match"])
                      and sorted_on_disk[s])
            all_ok &= ok
            res["stages"][str(s)] = {**ps, "ok": ok}
            print(f"[verify] {Path(cfg_path).stem} res{s}: boxes {ps['box_match']}/{ps['n']} scores "
                  f"{ps['score_match']}/{ps['n']} (given box {ps.get('score_match_given_box', 0)}/{ps['box_match']}, max rel "
                  f"{ps.get('max_rel_score_diff_matched', 0.0):.1e}) flags/ranks {ps['flag_rank_match']}/{ps['n']} sorted={sorted_on_disk[s]} "
                  f"-> {'OK' if ok else 'FAIL'}", flush=True)
        res["secs"] = time.time() - t0
        report["targets"].append(res)
        del teacher, pre
        if dev.type == 'cuda':
            torch.cuda.empty_cache()
    report["all_ok"] = all_ok
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1))
    print(f"[verify] ALL_OK={all_ok} -> {args.out}", flush=True)
    return 0 if all_ok else 3


if __name__ == "__main__":
    raise SystemExit(main())
