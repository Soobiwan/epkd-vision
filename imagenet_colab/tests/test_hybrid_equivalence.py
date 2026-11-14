"""CPU check: epkd_colab.precompute_hybrid (hook-free union, per-window means, sort-based quantile, crop scoring)
against the unmodified reference per-image path (ImageNetTGSCPreprocessor._process_image) on a few real images.
Light on purpose (2 threads, a handful of images).
  EPKD_MAX_TRAIN=48 EPKD_CACHE_DIR=/tmp/x python tests/test_hybrid_equivalence.py --n 6
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import epkd_colab  # noqa: F401,E402
from epkd_colab.config import build_teacher, load_cfg  # noqa: E402
from epkd_colab import precompute_hybrid as H  # noqa: E402
from tgsc_kd.imagenet_parquet import ImageNetTGSCPreprocessor, _image_bytes_to_tensor  # noqa: E402
from tgsc_kd.tgsc import resolve_stage_stride  # noqa: E402

CFGS = ["ref/configs/F1_rn34_rn18_epkd_e001-080.json", "ref/configs/F2_rn34_rn18_epkd_e081-100_cosine.json"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--configs", nargs="+", default=CFGS)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    root = Path(__file__).resolve().parents[1]
    cfgs = [load_cfg(str(root / c)) for c in a.configs]
    T, ckpt = build_teacher(cfgs[0])
    T.eval()
    targets = [ImageNetTGSCPreprocessor(T, c.data, c.tgsc, teacher_name=c.teacher_model, teacher_checkpoint=ckpt) for c in cfgs]
    tc = targets[0].tgsc_cfg
    ucfg = H.UnionCfg(bool(tc.use_gt_target), float(tc.w_cam), float(tc.w_sal), float(tc.temp), float(tc.gamma),
                      float(tc.pct_lo), float(tc.pct_hi))
    specs = tuple(H.Spec(ti, int(c), int(resolve_stage_stride(t.tgsc_cfg, c, t.stage_sizes)), int(t.tgsc_cfg.k_per_img),
                         float(t.tgsc_cfg.min_frac), float(t.tgsc_cfg.min_mean), bool(t.edge_align))
                  for ti, t in enumerate(targets) for c in t.stage_sizes)

    # 1. quantile re-implementation == torch.quantile (bitwise)
    g = torch.Generator().manual_seed(0)
    for n in (87296, 65536, 98304, 7):
        x = torch.rand(3, n, generator=g)
        for q in (0.01, 0.99, 0.75):
            ref = torch.quantile(x, torch.tensor([q], dtype=x.dtype), dim=1)[0]
            assert torch.equal(H.quantile_linear(x, q), ref), (n, q, (H.quantile_linear(x, q) - ref).abs().max())
    print("quantile_linear == torch.quantile (bitwise)", flush=True)

    total = targets[0].train_split.total_rows
    ids = sorted(random.Random(1).sample(range(total), a.n))
    box_ok = box_n = 0
    max_rel_score = 0.0
    T_noggrad = T
    for sid in ids:
        sample = next(targets[0].train_split.iter_samples(columns=("image.bytes", "label"), start=sid, stop=sid + 1, batch_size=1))
        img = _image_bytes_to_tensor(sample["image_bytes"], cfgs[0].data.analysis_resize_short)
        label = int(sample["label"])
        ref = {}
        for ti, t in enumerate(targets):
            cand = {s: [] for s in t.stage_sizes}
            rows = {s: [] for s in t.stage_sizes}
            t._process_image(sample_id=sid, image_tensor=img, label=label, candidate_rows=cand, stage_rows=rows)
            for s in t.stage_sizes:
                ref[(ti, s)] = cand[s]
        u8 = (img * 255.0).round().to(torch.uint8)
        assert torch.equal(u8.float().div(255.0), img), "uint8 round trip not exact"
        logits, boxes = H.union_pass(lambda x: H.teacher_to_cam_layer(T_noggrad, x),
                                     lambda A, r: H.cam_layer_to_logits(T_noggrad, A, r),
                                     lambda A, dA, gx: H.candidates_from_union(H.union_from_grads(A, dA, gx, ucfg), specs),
                                     u8.unsqueeze(0).float().div(255.0), torch.tensor([label]), ucfg.use_gt)
        pfull = torch.softmax(logits.float(), dim=-1)
        for si, sp in enumerate(specs):
            mine = boxes[si][0].tolist()
            theirs = [[r["y0"], r["x0"], r["y1"], r["x1"]] for r in ref[(sp.target, sp.crop)]]
            box_n += 1
            same = mine == theirs
            box_ok += int(same)
            if not same:
                print(f"  sid {sid} spec {sp.crop}/t{sp.target}: mine {mine} ref {theirs}", flush=True)
                continue
            crops = torch.stack([u8[:, y0:y1, x0:x1] for y0, x0, y1, x1 in mine])
            with torch.no_grad():
                sc = H.score_crops(T, crops, pfull.expand(len(mine), -1), torch.full((len(mine),), label), None)
            rs = torch.tensor([r["score"] for r in ref[(sp.target, sp.crop)]])
            rel = ((sc - rs).abs() / rs.abs().clamp_min(1e-6)).max().item()
            max_rel_score = max(max_rel_score, rel)
        print(f"sid {sid} shape {tuple(img.shape[-2:])}: boxes {box_ok}/{box_n} so far, max rel score diff {max_rel_score:.2e}", flush=True)
    ok = box_ok == box_n and max_rel_score < 1e-4
    print(f"HYBRID EQUIVALENCE {'OK' if ok else 'FAIL'}: boxes {box_ok}/{box_n} (target,stage,image) lists identical, "
          f"max rel score diff {max_rel_score:.2e}", flush=True)
    return 0 if ok else 3


if __name__ == "__main__":
    raise SystemExit(main())
