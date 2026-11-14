#!/usr/bin/env python3
"""Compare two per-step loss traces (train.py --loss-trace) from runs with the same data order (same seed/epoch):
mean |a-b|, smoothed curves, relative difference.  python bench/compare_trace.py A.json B.json [--window 20]"""
from __future__ import annotations

import argparse
import json

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--window", type=int, default=20)
    ap.add_argument("--names", default="A,B")
    x = ap.parse_args()
    a = np.array(json.load(open(x.a))["step_loss"])
    b = np.array(json.load(open(x.b))["step_loss"])
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    na, nb = x.names.split(",")
    w = x.window
    sa = np.convolve(a, np.ones(w) / w, mode="valid")
    sb = np.convolve(b, np.ones(w) / w, mode="valid")
    print(f"steps compared: {n}")
    print(f"per-step mean |{na}-{nb}| = {np.abs(a - b).mean():.4f}  (mean loss {a.mean():.4f} / {b.mean():.4f}; relative {np.abs(a - b).mean() / a.mean():.2%})")
    print(f"{w}-step smoothed mean |diff| = {np.abs(sa - sb).mean():.4f}  max {np.abs(sa - sb).max():.4f}")
    print(f"first 5 steps {na}: {np.round(a[:5], 4).tolist()}  {nb}: {np.round(b[:5], 4).tolist()}")
    print(f"last-{w} mean {na} {a[-w:].mean():.4f} {nb} {b[-w:].mean():.4f}")
    print(f"step-to-step noise (|a[i+1]-a[i]| mean) {np.abs(np.diff(a)).mean():.4f}")


if __name__ == "__main__":
    main()
