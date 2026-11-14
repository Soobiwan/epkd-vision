"""Edge-aligned candidate windows (``tgsc.edge_align_windows``).

A plain stride grid ``range(0, size - crop + 1, stride)`` can stop short of the image edge (ImageNet
analysis images have short side 256: 112px windows at stride 56 start at 0, 56, 112 and never cover
[224, 256)).  With the flag on, a final window flush with the edge is appended.  Flag off must reproduce
the old windows / boxes / cache key exactly.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from tgsc_kd.config import DataConfig, TGSCConfig, experiment_config_from_dict  # noqa: E402
from tgsc_kd.imagenet_parquet import ImageNetTGSCPreprocessor  # noqa: E402
from tgsc_kd.modeling import build_model  # noqa: E402
from tgsc_kd.tgsc import (  # noqa: E402
    _boxes_from_grid,
    _build_windows_by_pool,
    _gather_top_windows,
    _percentile_scale_t,
    resolve_stage_stride,
    window_starts,
)

RECIPES = sorted((ROOT / "configs" / "imagenet_fix").glob("*.json"))
# (H, W) of analysis images: short side 256, common aspect ratios, both orientations, plus odd sizes
SHAPES = [(256, 256), (256, 341), (341, 256), (256, 384), (256, 455), (256, 257), (300, 400), (113, 150)]


def _old_build_windows_by_pool(union, crop_h, crop_w, stride):
    """Verbatim copy of the pre-change implementation."""
    thr = _percentile_scale_t(union, lo=75, hi=75)
    mask = (union >= thr).float()
    mean_union_map = F.avg_pool2d(union, kernel_size=(crop_h, crop_w), stride=stride, padding=0)
    frac_above_map = F.avg_pool2d(mask, kernel_size=(crop_h, crop_w), stride=stride, padding=0)
    return mean_union_map, frac_above_map


def _old_boxes_from_grid(yh, xw, crop_h, crop_w, stride, height, width):
    y0 = yh * stride
    x0 = xw * stride
    y1 = (y0 + crop_h).clamp(max=height)
    x1 = (x0 + crop_w).clamp(max=width)
    return torch.stack([y0, x0, y1, x1], dim=-1)


def _recipe_stage_strides() -> list[tuple[str, int, int]]:
    out = []
    import json

    for path in RECIPES:
        cfg = experiment_config_from_dict(json.loads(path.read_text()))
        stages = tuple(int(r) for r, _e in cfg.data.stages)
        for crop in stages:
            out.append((path.stem, crop, resolve_stage_stride(cfg.tgsc, crop, stages)))
    return out


STAGE_STRIDES = _recipe_stage_strides() + [("synthetic", 84, 65), ("synthetic", 20, 8), ("synthetic", 112, 1)]


def test_recipes_enable_edge_align() -> None:
    import json

    assert len(RECIPES) == 4
    for path in RECIPES:
        cfg = experiment_config_from_dict(json.loads(path.read_text()))
        assert cfg.tgsc.edge_align_windows is True, path.name
    assert TGSCConfig().edge_align_windows is False  # default off


@pytest.mark.parametrize("size", [256, 257, 300, 341, 384, 455])
@pytest.mark.parametrize("name,crop,stride", STAGE_STRIDES)
def test_window_starts(size: int, name: str, crop: int, stride: int) -> None:
    old = list(range(0, size - crop + 1, stride))
    assert window_starts(size, crop, stride) == old
    assert window_starts(size, crop, stride, False) == old
    new = window_starts(size, crop, stride, True)
    assert new[: len(old)] == old
    assert new[-1] == size - crop
    assert len(new) == len(old) + (old[-1] != size - crop)
    assert new == sorted(set(new))
    # every pixel row/column is covered by at least one window
    covered = torch.zeros(size, dtype=torch.bool)
    for s in new:
        covered[s : s + crop] = True
    assert bool(covered.all())


def test_window_starts_crop_larger_than_image() -> None:
    assert window_starts(100, 112, 56, True) == [] == window_starts(100, 112, 56, False)


@pytest.mark.parametrize("hw", SHAPES)
@pytest.mark.parametrize("name,crop,stride", STAGE_STRIDES)
def test_pool_and_boxes(hw: tuple[int, int], name: str, crop: int, stride: int) -> None:
    height, width = hw
    if crop > min(height, width):
        pytest.skip("crop larger than image")
    g = torch.Generator().manual_seed(height * 1000 + width + crop)
    union = torch.rand(2, 1, height, width, generator=g)

    # ---- flag off: bit-identical to the old implementation
    mo, fo = _old_build_windows_by_pool(union, crop, crop, stride)
    m0, f0 = _build_windows_by_pool(union, crop, crop, stride)
    m0b, f0b = _build_windows_by_pool(union, crop, crop, stride, False)
    assert torch.equal(mo, m0) and torch.equal(fo, f0) and torch.equal(mo, m0b) and torch.equal(fo, f0b)
    gh, gw = mo.shape[-2:]
    yh_all = torch.arange(gh).repeat_interleave(gw).view(1, -1).expand(2, -1)
    xw_all = torch.arange(gw).repeat(gh).view(1, -1).expand(2, -1)
    assert torch.equal(
        _boxes_from_grid(yh_all, xw_all, crop, crop, stride, height, width),
        _old_boxes_from_grid(yh_all, xw_all, crop, crop, stride, height, width),
    )
    assert torch.equal(
        _boxes_from_grid(yh_all, xw_all, crop, crop, stride, height, width, False),
        _old_boxes_from_grid(yh_all, xw_all, crop, crop, stride, height, width),
    )
    kw = dict(k_per_img=6, min_frac=0.2, min_mean=0.2)
    yo, xo, vo = _gather_top_windows(mo, fo, **kw)
    y0_, x0_, v0_ = _gather_top_windows(m0, f0, **kw)
    assert torch.equal(yo, y0_) and torch.equal(xo, x0_) and torch.equal(vo, v0_)

    # ---- flag on
    ys = window_starts(height, crop, stride, True)
    xs = window_starts(width, crop, stride, True)
    m1, f1 = _build_windows_by_pool(union, crop, crop, stride, True)
    assert m1.shape[-2:] == (len(ys), len(xs))
    assert ys[-1] == height - crop and xs[-1] == width - crop
    # shared grid positions: bit-identical to the strided pool
    assert torch.equal(m1[..., :gh, :gw], mo) and torch.equal(f1[..., :gh, :gw], fo)
    # every window mean equals the direct mean of the crop
    thr = _percentile_scale_t(union, lo=75, hi=75)
    mask = (union >= thr).float()
    for i, y in enumerate(ys):
        for j, x in enumerate(xs):
            direct = union[..., y : y + crop, x : x + crop].mean(dim=(-1, -2)).squeeze(1)
            direct_f = mask[..., y : y + crop, x : x + crop].mean(dim=(-1, -2)).squeeze(1)
            assert torch.allclose(m1[:, 0, i, j], direct, atol=1e-5, rtol=1e-5)
            assert torch.allclose(f1[:, 0, i, j], direct_f, atol=1e-5, rtol=1e-5)
    # boxes: all grid cells, in bounds, full crop size, starts from the edge-aligned set
    gh1, gw1 = m1.shape[-2:]
    yh1 = torch.arange(gh1).repeat_interleave(gw1).view(1, -1).expand(2, -1)
    xw1 = torch.arange(gw1).repeat(gh1).view(1, -1).expand(2, -1)
    boxes = _boxes_from_grid(yh1, xw1, crop, crop, stride, height, width, True)
    assert boxes.dtype == yh1.dtype
    assert bool((boxes[..., 0] >= 0).all() and (boxes[..., 1] >= 0).all())
    assert bool((boxes[..., 2] <= height).all() and (boxes[..., 3] <= width).all())
    assert bool(((boxes[..., 2] - boxes[..., 0]) == crop).all() and ((boxes[..., 3] - boxes[..., 1]) == crop).all())
    assert set(boxes[..., 0].unique().tolist()) == set(ys)
    assert set(boxes[..., 1].unique().tolist()) == set(xs)
    # selected top windows: boxes index the same cells whose means were ranked
    yt, xt, vt = _gather_top_windows(m1, f1, **kw)
    bt = _boxes_from_grid(yt, xt, crop, crop, stride, height, width, True)
    for b in range(2):
        for k in range(yt.size(1)):
            y0, x0, y1, x1 = bt[b, k].tolist()
            assert y0 in ys and x0 in xs
            mean = m1[b, 0, yt[b, k], xt[b, k]]
            assert torch.allclose(mean, union[b, 0, y0:y1, x0:x1].mean(), atol=1e-5)


def _pre(tmp_path: Path, stages, **tgsc_overrides) -> ImageNetTGSCPreprocessor:
    data_cfg = DataConfig(
        dataset_type="imagenet_parquet", train_parquet_glob=str(tmp_path / "none-*.parquet"),
        validation_parquet_glob=str(tmp_path / "none-*.parquet"), test_parquet_glob=None, stages=stages,
        mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225),
    )
    tgsc = TGSCConfig(
        cache_dir=str(tmp_path / "cache"), top_k_crops=3, low_k_crops=2, stride=56, candidate_stride=56,
        dynamic_stride=True, k_per_img=6, **tgsc_overrides,
    )
    torch.manual_seed(0)
    teacher = build_model("ResNet18", num_classes=1000)
    return ImageNetTGSCPreprocessor(teacher, data_cfg, tgsc, teacher_name="ResNet18")


def test_cache_key_changes_only_when_on(tmp_path: Path) -> None:
    stages = ((112, 1), (140, 1))
    off = _pre(tmp_path, stages)
    off2 = _pre(tmp_path, stages, edge_align_windows=False)
    on = _pre(tmp_path, stages, edge_align_windows=True)
    assert off.cache_root == off2.cache_root
    assert on.cache_root != off.cache_root
    assert on.cache_root.parent == off.cache_root.parent
    assert on._default_state()["cache_key"] == on.cache_root.name


@torch.no_grad()
def _fake_image(h: int, w: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(h + w)
    return torch.rand(3, h, w, generator=g)


@pytest.mark.parametrize("hw", [(256, 341), (341, 256)])
def test_emit_stage_rows_edge_aligned(tmp_path: Path, hw: tuple[int, int]) -> None:
    """Real per-image precompute path (compute_union + emit_stage_rows) with a ResNet18 teacher on CPU."""
    stages = ((112, 1), (140, 1), (168, 1), (196, 1))
    height, width = hw
    img = _fake_image(height, width)
    out = {}
    for flag in (False, True):
        pre = _pre(tmp_path, stages, edge_align_windows=flag)
        analysis = pre.compute_union(image_tensor=img, label=3)
        cand = {s: [] for s in pre.stage_sizes}
        rows = {s: [] for s in pre.stage_sizes}
        pre.emit_stage_rows(analysis, sample_id=17, candidate_rows=cand, stage_rows=rows)
        out[flag] = (cand, rows, analysis["union"])
        for s in pre.stage_sizes:
            stride = resolve_stage_stride(pre.tgsc_cfg, s, pre.stage_sizes)
            ys = window_starts(height, s, stride, flag)
            xs = window_starts(width, s, stride, flag)
            assert cand[s] and rows[s]
            assert all(r["sample_id"] == 17 for r in cand[s] + rows[s])
            scores = [r["score"] for r in cand[s]]
            assert scores == sorted(scores, reverse=True)
            for r in cand[s]:
                assert r["y0"] in ys and r["x0"] in xs
                assert r["y1"] - r["y0"] == s and r["x1"] - r["x0"] == s
                assert 0 <= r["y0"] and r["y1"] <= height and 0 <= r["x0"] and r["x1"] <= width
            cand_boxes = [(r["y0"], r["x0"], r["y1"], r["x1"]) for r in cand[s]]
            for r in rows[s]:
                assert (r["y0"], r["x0"], r["y1"], r["x1"]) in cand_boxes
            assert [r["quality_flag"] for r in rows[s]].count(1) == min(3, len(cand[s]))
    # same union map regardless of the flag (the flag only affects window proposal)
    assert torch.equal(out[False][2], out[True][2])
