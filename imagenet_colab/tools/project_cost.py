#!/usr/bin/env python3
"""Wall-clock / compute-unit projection from measured end-to-end rows/s per stage.
python tools/project_cost.py --rate-per-hour 5.09 --rows-s 112=10000,140=9000,168=8000,196=7000,224=15000 \
    --eval-s 20 --precompute-img-s 400 --download-min 25 --margin 0.15 [--epoch-overhead-s 30]"""
from __future__ import annotations

import argparse

ROWS_PER_EPOCH = 3_843_501  # 3 proposals per image, 1,281,167 images (verified by simulation of the stage index)
N_IMAGES = 1_281_167
STAGES = [(112, 20), (140, 20), (168, 20), (196, 20), (224, 20)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rate-per-hour", type=float, required=True)
    ap.add_argument("--rows-s", required=True, help="crop=rows/s,... (end-to-end, 3 rows/img; 224 = tail)")
    ap.add_argument("--eval-s", type=float, default=30.0, help="seconds per epoch for the 50k eval")
    ap.add_argument("--epoch-overhead-s", type=float, default=40.0, help="loader start, compile, checkpoint per epoch")
    ap.add_argument("--precompute-img-s", type=float, required=True)
    ap.add_argument("--download-min", type=float, default=25.0)
    ap.add_argument("--setup-min", type=float, default=10.0, help="session creation, upload, val-cache build, verify")
    ap.add_argument("--margin", type=float, default=0.15)
    a = ap.parse_args()
    rates = {int(k): float(v) for k, v in (kv.split("=") for kv in a.rows_s.split(","))}
    total_s = 0.0
    print(f"{'stage':>6} {'rows/s':>8} {'min/epoch':>10} {'epochs':>6} {'hours':>7}")
    for crop, ep in STAGES:
        per = ROWS_PER_EPOCH / rates[crop] + a.eval_s + a.epoch_overhead_s
        hrs = per * ep / 3600
        total_s += per * ep
        print(f"{crop:>6} {rates[crop]:>8.0f} {per / 60:>10.1f} {ep:>6} {hrs:>7.2f}")
    train_h = total_s / 3600
    pre_h = N_IMAGES / a.precompute_img_s / 3600
    fixed_h = (a.download_min + a.setup_min) / 60
    total_h = train_h + pre_h + fixed_h
    units = total_h * a.rate_per_hour
    print(f"training (F1 112-196 + F2 224, incl. eval+overhead): {train_h:.2f} h")
    print(f"precompute @ {a.precompute_img_s:.0f} img/s: {pre_h:.2f} h | download+setup: {fixed_h:.2f} h")
    print(f"TOTAL {total_h:.2f} h = {units:.0f} units @ {a.rate_per_hour}/h ; with {a.margin:.0%} margin: {total_h * (1 + a.margin):.2f} h = {units * (1 + a.margin):.0f} units")


if __name__ == "__main__":
    main()
