"""CPU micro-batch preparation reproducing ``DDPKDTrainer._prepare_imagenet_batch`` / ``_full_anchor_views``
with integer slicing instead of roi_align (identical for integer boxes whose size equals the output size; see
tests/test_prep_equivalence.py), on uint8 images.  Normalisation ((x/255 - mean) / std) is done on the device.

Reference semantics (per 32-row micro-batch):
  * base = zero-pad the rows' 256-short-side images bottom/right to the micro-batch max (H, W)   [pad_stage_batch]
  * anchors: for each distinct sample_id (sorted), the FIRST row's image; valid extent from non-zero pixels; one
    random 224x224 window (x0 = floor(u * (width-224+1)), likewise y0) of the zero-padded image, flipped with p;
  * views: replicate-pad base by j=10; box += j; dx, dy ~ U{-j..j}; x0 = min(max(x0+dx, 0), Wpad - w) (same for y);
    crop = padded[y0:y0+h, x0:x0+w]; flip with p.
Random draws (in the reference's order: anchor u_x, u_y, flip; view dx, dy, flip) come from a numpy Generator;
``draws`` can be given explicitly (tests).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np


@dataclass
class Draws:
    ax: np.ndarray  # (A,) uniform [0,1)
    ay: np.ndarray  # (A,)
    aflip: np.ndarray  # (A,) uniform [0,1)
    dx: np.ndarray  # (n,) int in [-j, j]
    dy: np.ndarray  # (n,) int
    vflip: np.ndarray  # (n,) uniform [0,1)


def make_draws(rng: np.random.Generator, n: int, n_anchor: int, jitter: int) -> Draws:
    return Draws(
        ax=rng.random(n_anchor), ay=rng.random(n_anchor), aflip=rng.random(n_anchor),
        dx=rng.integers(-jitter, jitter + 1, size=n), dy=rng.integers(-jitter, jitter + 1, size=n),
        vflip=rng.random(n),
    )


def anchor_first_rows(sample_ids: np.ndarray) -> np.ndarray:
    """Index of the first row of each distinct sample_id, in ascending sample_id order (torch.unique + amin)."""
    uniq, first = np.unique(np.asarray(sample_ids), return_index=True)
    return first


def valid_extent(img_padded: np.ndarray) -> tuple[int, int]:
    """(height, width) = last row/column with any non-zero pixel + 1 (reference quirk: an all-black last
    row/column of the real image shrinks the range)."""
    # scan inward from the last row / column (identical result to max over the whole image, but usually touches one
    # row and one column instead of every pixel)
    H, W = img_padded.shape[1], img_padded.shape[2]
    height = 1
    for y in range(H - 1, -1, -1):
        if img_padded[:, y, :].any():
            height = y + 1
            break
    width = 1
    for x in range(W - 1, -1, -1):
        if img_padded[:, :, x].any():
            width = x + 1
            break
    return height, width


def prepare_microbatch(images: Sequence[np.ndarray], boxes: np.ndarray, sample_ids: np.ndarray, *, crop: int,
                       jitter: int, flip_prob: float, anchor_size: Optional[int], draws: Draws
                       ) -> tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """images: list of uint8 (3, h, w) arrays (one per ROW; rows of one image share the array);
    boxes: (n, 4) float [x0, y0, x1, y1] (integers); anchor_size None at the 224 stage.
    Returns (views (n,3,crop,crop) uint8, anchors (A,3,anchor,anchor) uint8 or None, anchor row indices or None)."""
    n = len(images)
    max_h = max(im.shape[1] for im in images)
    max_w = max(im.shape[2] for im in images)
    base = np.zeros((n, 3, max_h, max_w), dtype=np.uint8)
    for i, im in enumerate(images):
        base[i, :, : im.shape[1], : im.shape[2]] = im
    anchors = first = None
    if anchor_size is not None:
        first = anchor_first_rows(sample_ids)
        A = len(first)
        anchors = np.empty((A, 3, anchor_size, anchor_size), dtype=np.uint8)
        for k, i in enumerate(first):
            height, width = valid_extent(images[i])  # == on base[i]: beyond the image base is zero
            x0 = int(np.floor(draws.ax[k] * max(width - anchor_size + 1, 1)))
            y0 = int(np.floor(draws.ay[k] * max(height - anchor_size + 1, 1)))
            a = base[i, :, y0: y0 + anchor_size, x0: x0 + anchor_size]
            if a.shape[1] != anchor_size or a.shape[2] != anchor_size:  # window beyond the padded image (never with short side >= 224)
                pad = np.zeros((3, anchor_size, anchor_size), dtype=np.uint8)
                pad[:, : a.shape[1], : a.shape[2]] = a
                a = pad
            if draws.aflip[k] < flip_prob:
                a = a[:, :, ::-1]
            anchors[k] = a
    j = int(jitter)
    # edge padding by j == index clamping into base, so each window is cut from base directly (no full-batch np.pad)
    Hb, Wb = base.shape[-2:]
    Hp, Wp = Hb + 2 * j, Wb + 2 * j
    views = np.empty((n, 3, crop, crop), dtype=np.uint8)
    for i in range(n):
        x0, y0, x1, y1 = (float(v) for v in boxes[i])
        w, h = x1 - x0, y1 - y0
        assert w == crop and h == crop and float(x0).is_integer() and float(y0).is_integer(), (boxes[i], crop)
        x0, y0 = int(x0) + j, int(y0) + j
        if j > 0:
            x0 = int(min(max(x0 + int(draws.dx[i]), 0), Wp - int(w)))
            y0 = int(min(max(y0 + int(draws.dy[i]), 0), Hp - int(h)))
        ys, xs = y0 - j, x0 - j  # window start in base coordinates
        if 0 <= ys and ys + crop <= Hb and 0 <= xs and xs + crop <= Wb:
            v = base[i, :, ys: ys + crop, xs: xs + crop]
        else:
            ry = np.clip(np.arange(ys, ys + crop), 0, Hb - 1)
            rx = np.clip(np.arange(xs, xs + crop), 0, Wb - 1)
            v = base[i][:, ry][:, :, rx]
        if draws.vflip[i] < flip_prob:
            v = v[:, :, ::-1]
        views[i] = v
    return views, anchors, first
