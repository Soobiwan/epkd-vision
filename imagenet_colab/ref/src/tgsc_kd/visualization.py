from __future__ import annotations

from io import BytesIO
import json
import math
from pathlib import Path
from typing import Any, Literal

from PIL import Image, ImageDraw, ImageFont
import pyarrow.dataset as ds
import torch
import torch.nn.functional as F
from torchvision.transforms import functional as TF

from .config import DataConfig, TGSCConfig
from .imagenet_parquet import ParquetSplitInfo, _select_stage_rows
from .tgsc import (
    _LayerTaps,
    _boxes_from_grid,
    _build_windows_by_pool,
    _gradcam_pp_from_taps,
    _gather_top_windows,
    _percentile_scale_t,
    _score_candidates_with_teacher,
    _soft_or_t,
    resolve_stage_stride,
)
from .utils import forward_logits, pick_target_layer


SplitName = Literal["train", "validation", "test"]


def _load_split(data_cfg: DataConfig, split: SplitName) -> ParquetSplitInfo:
    if split == "train":
        return ParquetSplitInfo.from_glob("train", data_cfg.train_parquet_glob, data_cfg.max_train_samples)
    if split == "validation":
        return ParquetSplitInfo.from_glob("validation", data_cfg.validation_parquet_glob, data_cfg.max_val_samples)
    return ParquetSplitInfo.from_glob("test", data_cfg.test_parquet_glob, None)


def _load_image_bytes(image_bytes: bytes) -> Image.Image:
    with Image.open(BytesIO(image_bytes)) as image:
        return image.convert("RGB")


def _resize_short_side(image: Image.Image, short_side: int) -> Image.Image:
    return TF.resize(image, short_side, antialias=True)


def load_imagenet_sample(data_cfg: DataConfig, *, split: SplitName = "train", sample_id: int = 0) -> dict[str, Any]:
    split_info = _load_split(data_cfg, split)
    if sample_id < 0 or sample_id >= split_info.total_rows:
        raise IndexError(f"sample_id={sample_id} is out of range for {split} split with {split_info.total_rows} rows")

    iterator = split_info.iter_samples(
        columns=("image.bytes", "label"),
        start=sample_id,
        stop=sample_id + 1,
        batch_size=1,
    )
    try:
        sample = next(iterator)
    except StopIteration as exc:
        raise RuntimeError(f"Could not load sample_id={sample_id} from {split} split") from exc

    image_bytes = sample["image_bytes"]
    if image_bytes is None:
        raise RuntimeError(f"Sample {sample_id} does not contain image bytes")

    original_image = _load_image_bytes(image_bytes)
    analysis_image = _resize_short_side(original_image, data_cfg.analysis_resize_short)
    analysis_tensor = TF.to_tensor(analysis_image)
    return {
        "split": split,
        "sample_id": int(sample["sample_id"]),
        "label": None if sample["label"] is None else int(sample["label"]),
        "image_path": sample["image_path"],
        "original_image": original_image,
        "analysis_image": analysis_image,
        "analysis_tensor": analysis_tensor,
    }


def list_imagenet_tgsc_caches(cache_dir: str | Path = ".cache") -> list[dict[str, Any]]:
    root = Path(cache_dir) / "imagenet_tgsc"
    if not root.exists():
        return []

    entries: list[dict[str, Any]] = []
    for state_path in sorted(root.glob("*/precompute_state.json")):
        state = json.loads(state_path.read_text())
        entries.append(
            {
                "cache_key": state_path.parent.name,
                "cache_root": state_path.parent,
                "state_path": state_path,
                "complete": bool(state.get("complete", False)),
                "phase": state.get("phase"),
                "phase_detail": state.get("phase_detail"),
                "processed_train_samples": int(state.get("processed_train_samples", 0)),
                "last_checkpoint_at": state.get("last_checkpoint_at"),
                "state": state,
            }
        )
    entries.sort(
        key=lambda item: (
            int(item["complete"]),
            int(item["processed_train_samples"]),
            item["last_checkpoint_at"] or "",
        ),
        reverse=True,
    )
    return entries


def resolve_imagenet_tgsc_cache(
    *,
    cache_dir: str | Path = ".cache",
    cache_key: str | None = None,
    prefer_complete: bool = True,
) -> dict[str, Any]:
    caches = list_imagenet_tgsc_caches(cache_dir)
    if not caches:
        raise FileNotFoundError(f"No ImageNet TGSC cache found under {Path(cache_dir) / 'imagenet_tgsc'}")

    if cache_key is not None:
        for entry in caches:
            if entry["cache_key"] == cache_key:
                return entry
        raise FileNotFoundError(f"Cache key '{cache_key}' was not found under {Path(cache_dir) / 'imagenet_tgsc'}")

    if prefer_complete:
        for entry in caches:
            if entry["complete"]:
                return entry
    return caches[0]


def _load_cache_rows(path: Path, *, sample_id: int, columns: list[str]) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    dataset = ds.dataset(str(path), format="parquet")
    table = dataset.to_table(columns=columns, filter=ds.field("sample_id") == int(sample_id))
    return table.to_pylist()


def _cache_stage_entries(
    *,
    cache_root: Path,
    crop_size: int,
    sample_id: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    candidate_dir = cache_root / "candidates" / f"res{crop_size}"
    candidate_rows = _load_cache_rows(
        candidate_dir,
        sample_id=sample_id,
        columns=["sample_id", "y0", "x0", "y1", "x1", "score"],
    )
    candidate_rows.sort(key=lambda row: float(row["score"]), reverse=True)

    stage_index_path = cache_root / "stage_indexes" / f"stage_res{crop_size}.parquet"
    if stage_index_path.exists():
        selected_rows = _load_cache_rows(
            stage_index_path,
            sample_id=sample_id,
            columns=["sample_id", "resolution", "y0", "x0", "y1", "x1", "score", "quality_flag", "rank"],
        )
    else:
        selected_rows = _load_cache_rows(
            cache_root / "stage_indexes" / "tmp" / f"res{crop_size}",
            sample_id=sample_id,
            columns=["sample_id", "resolution", "y0", "x0", "y1", "x1", "score", "quality_flag", "rank"],
        )
    selected_rows.sort(key=lambda row: (-int(row["quality_flag"]), int(row["rank"])))

    selected_lookup: dict[tuple[int, int, int, int], dict[str, Any]] = {}
    for row in selected_rows:
        box = (int(row["y0"]), int(row["x0"]), int(row["y1"]), int(row["x1"]))
        selected_lookup[box] = {
            "selected": True,
            "quality_flag": int(row["quality_flag"]),
            "rank": int(row["rank"]),
        }

    entries: list[dict[str, Any]] = []
    for row in candidate_rows:
        box = (int(row["y0"]), int(row["x0"]), int(row["y1"]), int(row["x1"]))
        selected = selected_lookup.get(box, {"selected": False, "quality_flag": 0, "rank": None})
        entries.append(
            {
                "box": box,
                "teacher_score": float(row["score"]),
                "hotmap_score": math.nan,
                "selected": bool(selected["selected"]),
                "quality_flag": int(selected["quality_flag"]),
                "rank": selected["rank"],
            }
        )
    selected_entries = [entry for entry in entries if entry["selected"]]
    return entries, selected_entries


def load_imagenet_tgsc_cached_sample(
    data_cfg: DataConfig,
    *,
    sample_id: int = 0,
    split: SplitName = "train",
    cache_dir: str | Path = ".cache",
    cache_key: str | None = None,
    prefer_complete: bool = True,
) -> dict[str, Any]:
    if split != "train":
        raise ValueError("The cached TGSC precompute is only defined for the train split.")

    cache_info = resolve_imagenet_tgsc_cache(cache_dir=cache_dir, cache_key=cache_key, prefer_complete=prefer_complete)
    sample = load_imagenet_sample(data_cfg, split=split, sample_id=sample_id)

    stages: dict[int, list[dict[str, Any]]] = {}
    selected: dict[int, list[dict[str, Any]]] = {}
    for crop_size, _epochs in data_cfg.stages:
        crop_size = int(crop_size)
        entries, selected_entries = _cache_stage_entries(
            cache_root=Path(cache_info["cache_root"]),
            crop_size=crop_size,
            sample_id=sample["sample_id"],
        )
        stages[crop_size] = entries
        selected[crop_size] = selected_entries

    return {
        **sample,
        "cache_key": cache_info["cache_key"],
        "cache_root": Path(cache_info["cache_root"]),
        "cache_state": cache_info["state"],
        "stages": stages,
        "selected": selected,
    }


def analyze_imagenet_tgsc_sample(
    *,
    teacher: torch.nn.Module,
    data_cfg: DataConfig,
    tgsc_cfg: TGSCConfig,
    split: SplitName = "train",
    sample_id: int = 0,
    device: str | torch.device = "cpu",
) -> dict[str, Any]:
    sample = load_imagenet_sample(data_cfg, split=split, sample_id=sample_id)
    tensor = sample["analysis_tensor"]
    label = sample["label"]
    if label is None or label < 0:
        raise RuntimeError("This visualization requires a labeled sample.")

    device = torch.device(device)
    teacher = teacher.to(device).eval()
    target_layer = pick_target_layer(teacher)

    xb = tensor.unsqueeze(0).to(device).requires_grad_(True)
    yb = torch.tensor([label], dtype=torch.long, device=device)
    taps = _LayerTaps(target_layer)
    teacher.zero_grad(set_to_none=True)

    try:
        logits = forward_logits(teacher, xb)
        probs_full = torch.softmax(logits.detach(), dim=-1)
        top1 = probs_full.argmax(dim=1)
        target_idx = yb if tgsc_cfg.use_gt_target else top1

        oh = torch.zeros_like(logits)
        oh.scatter_(1, target_idx.view(-1, 1), 1.0)
        logits.backward(gradient=oh, retain_graph=False)

        _, _, height, width = xb.shape
        g = xb.grad.detach().abs().amax(dim=1, keepdim=True)
        sal = (g - g.amin(dim=(2, 3), keepdim=True)) / (
            g.amax(dim=(2, 3), keepdim=True) - g.amin(dim=(2, 3), keepdim=True) + 1e-8
        )

        A, dY = taps.A, taps.dY
        if A is None or dY is None:
            raise RuntimeError("Target layer hooks did not capture activations or gradients.")

        cam_small = _gradcam_pp_from_taps(A, dY)
        cam = F.interpolate(cam_small, size=(height, width), mode="bilinear", align_corners=False)
        cam = _percentile_scale_t(cam, lo=tgsc_cfg.pct_lo, hi=tgsc_cfg.pct_hi)
        sal = _percentile_scale_t(sal, lo=tgsc_cfg.pct_lo, hi=tgsc_cfg.pct_hi)
        union = _soft_or_t(
            cam,
            sal,
            w_cam=tgsc_cfg.w_cam,
            w_sal=tgsc_cfg.w_sal,
            temp=tgsc_cfg.temp,
        )
        if tgsc_cfg.gamma != 1.0:
            union = union.clamp(0, 1).pow(1.0 / max(tgsc_cfg.gamma, 1e-6))

        stage_sizes = tuple(int(resolution) for resolution, _epochs in data_cfg.stages)
        stage_entries: dict[int, list[dict[str, Any]]] = {}
        stage_selected: dict[int, list[dict[str, Any]]] = {}
        for crop_size, _epochs in data_cfg.stages:
            crop_size = int(crop_size)
            stride = resolve_stage_stride(tgsc_cfg, crop_size, stage_sizes)
            edge_align = bool(getattr(tgsc_cfg, "edge_align_windows", False))
            mean_union_map, frac_map = _build_windows_by_pool(union, crop_size, crop_size, stride, edge_align)
            if mean_union_map.numel() == 0:
                stage_entries[crop_size] = []
                stage_selected[crop_size] = []
                continue

            yh, xw, hot_scores = _gather_top_windows(
                mean_union_map,
                frac_map,
                k_per_img=tgsc_cfg.k_per_img,
                min_frac=tgsc_cfg.min_frac,
                min_mean=tgsc_cfg.min_mean,
            )
            boxes = _boxes_from_grid(yh, xw, crop_size, crop_size, stride, height, width, edge_align)
            if boxes.numel() == 0:
                stage_entries[crop_size] = []
                stage_selected[crop_size] = []
                continue

            boxes_sorted, scores_sorted, _ = _score_candidates_with_teacher(
                teacher,
                logits.detach(),
                target_idx,
                xb,
                boxes,
                eval_chunk=tgsc_cfg.eval_chunk,
            )
            selection_rows = _select_stage_rows(
                sample_id=sample["sample_id"],
                resolution=crop_size,
                boxes_sorted=boxes_sorted[0].detach().cpu(),
                scores_sorted=scores_sorted[0].detach().cpu(),
                top_k_crops=tgsc_cfg.top_k_crops,
                low_k_crops=tgsc_cfg.low_k_crops,
            )

            hotmap_lookup: dict[tuple[int, int, int, int], float] = {}
            hot_scores_cpu = hot_scores[0].detach().cpu()
            boxes_cpu_unsorted = boxes[0].detach().cpu()
            for idx in range(hot_scores_cpu.numel()):
                box_key = tuple(int(value) for value in boxes_cpu_unsorted[idx].tolist())
                hotmap_lookup[box_key] = float(hot_scores_cpu[idx].item())

            selected_lookup: dict[tuple[int, int, int, int], dict[str, Any]] = {}
            for row in selection_rows:
                key = (int(row["y0"]), int(row["x0"]), int(row["y1"]), int(row["x1"]))
                selected_lookup[key] = {
                    "selected": True,
                    "quality_flag": int(row["quality_flag"]),
                    "rank": int(row["rank"]),
                }

            entries: list[dict[str, Any]] = []
            for idx in range(scores_sorted.size(1)):
                y0, x0, y1, x1 = (int(value) for value in boxes_sorted[0, idx].detach().cpu().tolist())
                key = (y0, x0, y1, x1)
                selected = selected_lookup.get(key, {"selected": False, "quality_flag": 0, "rank": None})
                entries.append(
                    {
                        "box": key,
                        "teacher_score": float(scores_sorted[0, idx].detach().cpu().item()),
                        "hotmap_score": hotmap_lookup.get(key, float("nan")),
                        "selected": bool(selected["selected"]),
                        "quality_flag": int(selected["quality_flag"]),
                        "rank": selected["rank"],
                    }
                )
            stage_entries[crop_size] = entries
            stage_selected[crop_size] = [entry for entry in entries if entry["selected"]]

        return {
            **sample,
            "prediction_class": int(top1.item()),
            "target_class": int(target_idx.item()),
            "logits": logits.detach().cpu(),
            "cam": cam.detach().cpu()[0, 0],
            "saliency": sal.detach().cpu()[0, 0],
            "union": union.detach().cpu()[0, 0],
            "stride": stride,
            "stages": stage_entries,
            "selected": stage_selected,
        }
    finally:
        taps.close()
        xb.grad = None
        teacher.zero_grad(set_to_none=True)


def _heatmap_to_pil(heatmap: torch.Tensor) -> Image.Image:
    heat = heatmap.float()
    heat = (heat - heat.min()) / (heat.max() - heat.min() + 1e-8)
    rgb = torch.stack([heat, torch.zeros_like(heat), 1.0 - heat], dim=0).clamp(0.0, 1.0)
    return TF.to_pil_image(rgb)


def make_heatmap_overlay(base_image: Image.Image | torch.Tensor, heatmap: torch.Tensor, alpha: float = 0.45) -> Image.Image:
    if isinstance(base_image, torch.Tensor):
        base = TF.to_pil_image(base_image)
    else:
        base = base_image.copy()
    heat = _heatmap_to_pil(heatmap).resize(base.size)
    return Image.blend(base.convert("RGB"), heat.convert("RGB"), alpha=alpha)


def draw_stage_boxes(
    base_image: Image.Image | torch.Tensor,
    entries: list[dict[str, Any]],
    *,
    limit: int | None = None,
) -> Image.Image:
    if isinstance(base_image, torch.Tensor):
        canvas = TF.to_pil_image(base_image).convert("RGB")
    else:
        canvas = base_image.copy().convert("RGB")

    font = ImageFont.load_default()
    draw = ImageDraw.Draw(canvas)
    visible = entries if limit is None else entries[:limit]
    for idx, entry in enumerate(visible):
        y0, x0, y1, x1 = entry["box"]
        quality = int(entry["quality_flag"])
        if entry["selected"] and quality > 0:
            color = (0, 200, 0)
        elif entry["selected"] and quality < 0:
            color = (220, 0, 0)
        else:
            color = (255, 196, 0)
        draw.rectangle([(x0, y0), (x1, y1)], outline=color, width=3)
        label = f"{idx}: {entry['teacher_score']:.3f}"
        draw.text((x0 + 2, max(0, y0 - 12)), label, fill=color, font=font)
    return canvas


def _crop_with_reflect_pad(image_tensor: torch.Tensor, box: tuple[int, int, int, int], crop_size: int) -> torch.Tensor:
    y0, x0, y1, x1 = box
    crop = image_tensor[:, y0:y1, x0:x1]
    height, width = crop.shape[-2:]
    if height != crop_size or width != crop_size:
        pad_t = max(0, crop_size - height)
        pad_l = max(0, crop_size - width)
        crop = F.pad(crop, (0, pad_l, 0, pad_t), mode="reflect")[:, :crop_size, :crop_size]
    return crop


def build_crop_gallery(
    image_tensor: torch.Tensor,
    entries: list[dict[str, Any]],
    *,
    crop_size: int,
    selected_only: bool = False,
    max_items: int | None = None,
) -> Image.Image:
    filtered = [entry for entry in entries if entry["selected"]] if selected_only else list(entries)
    if max_items is not None:
        filtered = filtered[:max_items]
    if not filtered:
        return Image.new("RGB", (crop_size, crop_size), "white")

    font = ImageFont.load_default()
    gap = 10
    caption_h = 34
    tiles: list[Image.Image] = []
    for entry in filtered:
        crop_tensor = _crop_with_reflect_pad(image_tensor, entry["box"], crop_size)
        crop = TF.to_pil_image(crop_tensor)
        tile = Image.new("RGB", (crop_size, crop_size + caption_h), "white")
        tile.paste(crop, (0, 0))
        draw = ImageDraw.Draw(tile)
        score_text = f"s={entry['teacher_score']:.3f}"
        draw.text((4, crop_size + 2), score_text, fill="black", font=font)
        if entry["selected"]:
            prefix = "top" if entry["quality_flag"] > 0 else "low"
            draw.text((4, crop_size + 18), f"{prefix}#{entry['rank']}", fill="black", font=font)
        tiles.append(tile)

    width = len(tiles) * crop_size + max(0, len(tiles) - 1) * gap
    gallery = Image.new("RGB", (width, crop_size + caption_h), "white")
    cursor = 0
    for tile in tiles:
        gallery.paste(tile, (cursor, 0))
        cursor += crop_size + gap
    return gallery


def stage_summary_lines(entries: list[dict[str, Any]]) -> list[str]:
    lines: list[str] = []
    for idx, entry in enumerate(entries):
        y0, x0, y1, x1 = entry["box"]
        marker = "-"
        if entry["selected"]:
            marker = "TOP" if entry["quality_flag"] > 0 else "LOW"
        rank_text = "" if entry["rank"] is None else f" rank={entry['rank']}"
        lines.append(
            f"{idx:02d} {marker:>3} score={entry['teacher_score']:.4f} hot={entry['hotmap_score']:.4f} "
            f"box=({y0},{x0},{y1},{x1}){rank_text}"
        )
    return lines
