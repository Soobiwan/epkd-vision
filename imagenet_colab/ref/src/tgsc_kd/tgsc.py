from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm.auto import tqdm

from .config import DataConfig, TGSCConfig
from .utils import forward_logits, pick_target_layer


@dataclass
class TeacherUnionCache:
    union_maps: torch.Tensor
    logits: torch.Tensor

    def to(self, device: torch.device | str) -> "TeacherUnionCache":
        return TeacherUnionCache(
            union_maps=self.union_maps.to(device),
            logits=self.logits.to(device),
        )


def resolve_stage_stride(
    tgsc_cfg: TGSCConfig,
    crop_size: int,
    stage_sizes: tuple[int, ...] | list[int] | None = None,
) -> int:
    base_stride = int(tgsc_cfg.candidate_stride or tgsc_cfg.stride)
    if not tgsc_cfg.dynamic_stride or not stage_sizes:
        return max(1, base_stride)

    smallest_stage = max(1, min(int(stage) for stage in stage_sizes))
    crop_size = max(1, int(crop_size))
    power = max(0.0, float(tgsc_cfg.dynamic_stride_power))
    scaled = base_stride * ((smallest_stage / float(crop_size)) ** power)
    return max(1, int(round(scaled)))


def _top2_batch(p: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    vals, idx = torch.sort(p, dim=-1, descending=True)
    return idx[:, 0], vals[:, 0], idx[:, 1], vals[:, 1]


def _kl_div_batch(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    eps = 1e-8
    p_ = p.clamp_min(eps)
    q_ = q.clamp_min(eps)
    return (p_ * (p_.log() - q_.log())).sum(dim=1)


class _LayerTaps:
    def __init__(self, target_layer: nn.Module):
        self.A: Optional[torch.Tensor] = None
        self.dY: Optional[torch.Tensor] = None
        self._h1 = target_layer.register_forward_hook(self._store_act)
        self._h2 = target_layer.register_full_backward_hook(self._store_grad)

    def _store_act(self, _module: nn.Module, _inp: tuple[torch.Tensor, ...], out: torch.Tensor) -> None:
        self.A = out

    def _store_grad(
        self,
        _module: nn.Module,
        _gin: tuple[Optional[torch.Tensor], ...],
        gout: tuple[Optional[torch.Tensor], ...],
    ) -> None:
        self.dY = gout[0]

    def close(self) -> None:
        self._h1.remove()
        self._h2.remove()


def _gradcam_pp_from_taps(A: torch.Tensor, dY: torch.Tensor) -> torch.Tensor:
    eps = 1e-8
    dY2 = dY * dY
    dY3 = dY2 * dY
    sumA = A.sum(dim=(2, 3), keepdim=True)
    sumdY3 = dY3.sum(dim=(2, 3), keepdim=True)
    alpha = dY2 / (2.0 * dY2 + sumA * sumdY3 + eps)
    w = (alpha * dY.clamp_min(0)).sum(dim=(2, 3), keepdim=True)
    cam = (w * A).sum(dim=1, keepdim=True)
    cam = F.relu(cam)
    return cam / (cam.amax(dim=(2, 3), keepdim=True) + 1e-8)


def _soft_or_t(cam: torch.Tensor, sal: torch.Tensor, *, w_cam: float, w_sal: float, temp: float) -> torch.Tensor:
    cam = cam.clamp(0, 1)
    sal = sal.clamp(0, 1)
    if temp != 1.0:
        cam = cam.clamp(1e-8, 1 - 1e-8).pow(1.0 / temp)
        sal = sal.clamp(1e-8, 1 - 1e-8).pow(1.0 / temp)
    return 1.0 - (1.0 - cam).pow(w_cam) * (1.0 - sal).pow(w_sal)


def _percentile_scale_t(x: torch.Tensor, *, lo: float, hi: float) -> torch.Tensor:
    batch = x.size(0)
    flat = x.view(batch, -1)
    q = torch.tensor([lo / 100.0, hi / 100.0], device=x.device, dtype=x.dtype)
    values = torch.quantile(flat, q, dim=1)
    lo_v = values[0].view(batch, 1, 1, 1)
    hi_v = values[1].view(batch, 1, 1, 1)
    denom = (hi_v - lo_v).clamp_min(1e-8)
    return ((x - lo_v).clamp_min(0.0) / denom).clamp(0.0, 1.0)


def window_starts(size: int, crop: int, stride: int, edge_align: bool = False) -> list[int]:
    """Top/left offsets of the sliding windows along one axis.

    Without ``edge_align`` this is the plain stride grid, which can stop short of the image edge
    (e.g. 112px windows at stride 56 on a 256px side start at 0, 56, 112 and never cover the last 32
    pixels).  With ``edge_align`` a final window flush with the edge is appended when the grid does not
    already end there.
    """
    starts = list(range(0, int(size) - int(crop) + 1, int(stride)))
    if edge_align and starts and starts[-1] != int(size) - int(crop):
        starts.append(int(size) - int(crop))
    return starts


def _build_windows_by_pool(
    union: torch.Tensor,
    crop_h: int,
    crop_w: int,
    stride: int,
    edge_align: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    thr = _percentile_scale_t(union, lo=75, hi=75)
    mask = (union >= thr).float()
    if not edge_align:
        mean_union_map = F.avg_pool2d(union, kernel_size=(crop_h, crop_w), stride=stride, padding=0)
        frac_above_map = F.avg_pool2d(mask, kernel_size=(crop_h, crop_w), stride=stride, padding=0)
        return mean_union_map, frac_above_map
    # Dense (stride 1) pooling, then pick the grid + edge-aligned start offsets: the values at the
    # shared grid positions are the same per-window reductions as the strided pool.
    height, width = union.shape[-2:]
    ys = torch.as_tensor(window_starts(height, crop_h, stride, True), dtype=torch.long, device=union.device)
    xs = torch.as_tensor(window_starts(width, crop_w, stride, True), dtype=torch.long, device=union.device)
    dense_mean = F.avg_pool2d(union, kernel_size=(crop_h, crop_w), stride=1, padding=0)
    dense_frac = F.avg_pool2d(mask, kernel_size=(crop_h, crop_w), stride=1, padding=0)
    mean_union_map = dense_mean.index_select(2, ys).index_select(3, xs)
    frac_above_map = dense_frac.index_select(2, ys).index_select(3, xs)
    return mean_union_map, frac_above_map


def _gather_top_windows(
    mean_union_map: torch.Tensor,
    frac_map: torch.Tensor,
    *,
    k_per_img: int,
    min_frac: float,
    min_mean: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch, _, grid_h, grid_w = mean_union_map.shape
    flat_score = mean_union_map.view(batch, -1)
    flat_frac = frac_map.view(batch, -1)
    valid = (flat_frac >= min_frac) & (flat_score >= min_mean)
    score_masked = torch.where(valid, flat_score, torch.full_like(flat_score, -1.0))
    topk = min(k_per_img, grid_h * grid_w)
    vals, idxs = torch.topk(score_masked, k=topk, dim=1)
    yh = idxs // grid_w
    xw = idxs % grid_w
    return yh, xw, vals


def _boxes_from_grid(
    yh: torch.Tensor,
    xw: torch.Tensor,
    crop_h: int,
    crop_w: int,
    stride: int,
    height: int,
    width: int,
    edge_align: bool = False,
) -> torch.Tensor:
    if edge_align:
        ys = torch.as_tensor(window_starts(height, crop_h, stride, True), device=yh.device, dtype=yh.dtype)
        xs = torch.as_tensor(window_starts(width, crop_w, stride, True), device=xw.device, dtype=xw.dtype)
        y0 = ys[yh]
        x0 = xs[xw]
    else:
        y0 = yh * stride
        x0 = xw * stride
    y1 = (y0 + crop_h).clamp(max=height)
    x1 = (x0 + crop_w).clamp(max=width)
    return torch.stack([y0, x0, y1, x1], dim=-1)


def _score_candidates_with_teacher(
    model: nn.Module,
    full_logits: torch.Tensor,
    class_idx: torch.Tensor,
    x: torch.Tensor,
    boxes: torch.Tensor,
    *,
    eval_chunk: int,
    sort: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Teacher composite score of every candidate crop. sort=True (default, original behaviour) returns the
    candidates sorted by score; sort=False keeps the input (saliency) order."""
    device = x.device
    batch, num_candidates, _ = boxes.shape
    if num_candidates == 0:
        return (
            boxes[:, :0],
            torch.empty((batch, 0), device=device),
            torch.empty((batch, 0, full_logits.size(1)), device=device),
        )

    p_full = torch.softmax(full_logits, dim=-1)
    _, pc1_f, _, pc2_f = _top2_batch(p_full)

    crops: list[torch.Tensor] = []
    owners: list[int] = []
    candidate_indices: list[int] = []
    for batch_idx in range(batch):
        for candidate_idx in range(num_candidates):
            y0, x0, y1, x1 = boxes[batch_idx, candidate_idx].tolist()
            crops.append(x[batch_idx : batch_idx + 1, :, y0:y1, x0:x1])
            owners.append(batch_idx)
            candidate_indices.append(candidate_idx)

    crops_cat = torch.cat(crops, dim=0)
    if device.type == "cuda":
        crops_cat = crops_cat.contiguous(memory_format=torch.channels_last)
    preds: list[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, crops_cat.size(0), eval_chunk):
            logits_k = forward_logits(model, crops_cat[start : start + eval_chunk])
            preds.append(torch.softmax(logits_k, dim=-1))
    p_k_all = torch.cat(preds, dim=0)

    alpha = math.log(p_k_all.size(1))
    owners_t = torch.tensor(owners, device=device, dtype=torch.long)
    candidate_t = torch.tensor(candidate_indices, device=device, dtype=torch.long)
    owner_full = p_full[owners_t]
    owner_class = class_idx[owners_t].view(-1, 1)

    pc_k = p_k_all.gather(1, owner_class).squeeze(1)
    pc_f = owner_full.gather(1, owner_class).squeeze(1)
    conf_ratio = pc_k / (pc_f + 1e-8)

    kl = _kl_div_batch(owner_full, p_k_all)
    kl_norm = 1.0 - torch.clamp(kl / alpha, max=1.0)

    top2_k = torch.topk(p_k_all, k=min(2, p_k_all.size(1)), dim=1).values
    if top2_k.size(1) == 1:
        margin_k = top2_k[:, 0]
    else:
        margin_k = top2_k[:, 0] - top2_k[:, 1]
    margin_f = pc1_f[owners_t] - pc2_f[owners_t]
    margin_rat = margin_k / (margin_f + 1e-8)

    scores = (conf_ratio + kl_norm + margin_rat) / 3.0

    score_matrix = torch.full((batch, num_candidates), -1.0, device=device)
    prob_matrix = torch.zeros((batch, num_candidates, p_k_all.size(1)), dtype=p_k_all.dtype, device=device)
    score_matrix[owners_t, candidate_t] = scores
    prob_matrix[owners_t, candidate_t] = p_k_all
    if not sort:
        return boxes, score_matrix, prob_matrix

    sorted_scores, order = torch.sort(score_matrix, dim=1, descending=True)
    order_boxes = order.unsqueeze(-1).expand(-1, -1, 4)
    boxes_sorted = boxes.gather(1, order_boxes)
    order_probs = order.unsqueeze(-1).expand(-1, -1, prob_matrix.size(-1))
    probs_sorted = prob_matrix.gather(1, order_probs)
    return boxes_sorted, sorted_scores, probs_sorted


@torch.inference_mode(False)
def build_teacher_union_cache(
    base_loader: DataLoader,
    teacher: nn.Module,
    target_layer: nn.Module,
    *,
    use_gt_target: bool,
    w_cam: float,
    w_sal: float,
    temp: float,
    gamma: float,
    pct_lo: float,
    pct_hi: float,
) -> TeacherUnionCache:
    device = next(teacher.parameters()).device
    teacher.eval()

    union_maps_all: list[torch.Tensor] = []
    logits_all: list[torch.Tensor] = []

    print("[TGSC] Building teacher union cache (one-time)...")

    for xb, yb in tqdm(base_loader, desc="Teacher cache", leave=False):
        xb = xb.to(device).requires_grad_(True)
        yb = yb.to(device)
        _, _, height, width = xb.shape

        taps = _LayerTaps(target_layer)
        teacher.zero_grad(set_to_none=True)

        logits = forward_logits(teacher, xb)
        probs_full = torch.softmax(logits.detach(), dim=-1)
        top1 = probs_full.argmax(dim=1)
        target_idx = yb if use_gt_target else top1

        oh = torch.zeros_like(logits)
        oh.scatter_(1, target_idx.view(-1, 1), 1.0)
        logits.backward(gradient=oh, retain_graph=False)

        g = xb.grad.detach().abs().amax(dim=1, keepdim=True)
        sal = (g - g.amin(dim=(2, 3), keepdim=True)) / (
            g.amax(dim=(2, 3), keepdim=True) - g.amin(dim=(2, 3), keepdim=True) + 1e-8
        )

        A, dY = taps.A, taps.dY
        if A is None or dY is None:
            taps.close()
            raise RuntimeError("Target layer hooks did not capture activations or gradients.")

        cam_small = _gradcam_pp_from_taps(A, dY)
        cam = F.interpolate(cam_small, size=(height, width), mode="bilinear", align_corners=False)
        cam = _percentile_scale_t(cam, lo=pct_lo, hi=pct_hi)
        sal = _percentile_scale_t(sal, lo=pct_lo, hi=pct_hi)

        union = _soft_or_t(cam, sal, w_cam=w_cam, w_sal=w_sal, temp=temp)
        if gamma != 1.0:
            union = union.clamp(0, 1).pow(1.0 / max(gamma, 1e-6))

        union_maps_all.append(union.detach().cpu())
        logits_all.append(logits.detach().cpu())

        taps.close()
        xb.grad = None
        teacher.zero_grad(set_to_none=True)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    cache = TeacherUnionCache(
        union_maps=torch.cat(union_maps_all, dim=0),
        logits=torch.cat(logits_all, dim=0),
    )

    print(
        f"[TGSC] Teacher cache built: {cache.union_maps.shape[0]} images, "
        f"~{(cache.union_maps.numel() + cache.logits.numel()) * 4 / 1024**2:.1f} MB"
    )
    return cache


@torch.inference_mode(False)
def build_bestcrop_dataset_batchwise(
    base_loader: DataLoader,
    teacher: nn.Module,
    target_layer: nn.Module,
    *,
    crop_size: int | tuple[int, int],
    use_gt_target: bool,
    w_cam: float,
    w_sal: float,
    temp: float,
    gamma: float,
    pct_lo: float,
    pct_hi: float,
    stride: int,
    k_per_img: int,
    min_frac: float,
    min_mean: float,
    eval_chunk: int,
    top_k_crops: int,
    low_k_crops: int,
    dynamic_stride: bool = False,
    dynamic_stride_power: float = 0.5,
    stage_sizes: Optional[tuple[int, ...]] = None,
    teacher_cache: Optional[TeacherUnionCache] = None,
    edge_align: bool = False,
) -> "CropTensorDataset":
    device = next(teacher.parameters()).device
    teacher.eval()

    if isinstance(crop_size, int):
        crop_h = crop_w = int(crop_size)
    else:
        crop_h, crop_w = int(crop_size[0]), int(crop_size[1])
    effective_stage_sizes = stage_sizes or (crop_h,)
    stride_cfg = TGSCConfig(
        stride=stride,
        dynamic_stride=dynamic_stride,
        dynamic_stride_power=dynamic_stride_power,
    )
    stage_stride = resolve_stage_stride(stride_cfg, crop_h, effective_stage_sizes)

    crops_all: list[torch.Tensor] = []
    labels_all: list[torch.Tensor] = []
    scores_all: list[torch.Tensor] = []
    sources_all: list[torch.Tensor] = []
    quality_all: list[torch.Tensor] = []

    progress_desc = f"TGSC precompute {crop_h}x{crop_w}"

    if teacher_cache is not None:
        print(f"[TGSC] Using cached teacher outputs for {crop_h}x{crop_w} crops")
        seen = 0
        for xb, yb in tqdm(base_loader, desc=progress_desc, leave=False):
            xb = xb.to(device)
            yb = yb.to(device)

            batch = xb.size(0)
            start_idx = seen
            end_idx = start_idx + batch
            seen = end_idx

            union = teacher_cache.union_maps[start_idx:end_idx].to(device)
            logits = teacher_cache.logits[start_idx:end_idx].to(device)

            _, _, height, width = xb.shape
            base_indices = torch.arange(start_idx, end_idx, dtype=torch.long)

            probs_full = torch.softmax(logits, dim=-1)
            top1 = probs_full.argmax(dim=1)
            target_idx = yb if use_gt_target else top1

            mean_union_map, frac_map = _build_windows_by_pool(union, crop_h, crop_w, stage_stride, edge_align)
            yh, xw, _ = _gather_top_windows(
                mean_union_map,
                frac_map,
                k_per_img=k_per_img,
                min_frac=min_frac,
                min_mean=min_mean,
            )
            boxes = _boxes_from_grid(yh, xw, crop_h, crop_w, stage_stride, height, width, edge_align)
            boxes_sorted, scores_sorted, _ = _score_candidates_with_teacher(
                teacher,
                logits,
                target_idx,
                xb,
                boxes,
                eval_chunk=eval_chunk,
            )

            for sample_idx in range(batch):
                _collect_crop_candidates(
                    xb=xb,
                    yb=yb,
                    sample_idx=sample_idx,
                    boxes_sorted=boxes_sorted,
                    scores_sorted=scores_sorted,
                    crop_h=crop_h,
                    crop_w=crop_w,
                    base_indices=base_indices,
                    top_k_crops=top_k_crops,
                    low_k_crops=low_k_crops,
                    crops_all=crops_all,
                    labels_all=labels_all,
                    scores_all=scores_all,
                    sources_all=sources_all,
                    quality_all=quality_all,
                )

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    else:
        print(f"[TGSC] Computing from scratch (no cache) for {crop_h}x{crop_w} crops")
        seen = 0
        for xb, yb in tqdm(base_loader, desc=progress_desc, leave=False):
            xb = xb.to(device).requires_grad_(True)
            yb = yb.to(device)

            batch, _, height, width = xb.shape
            base_indices = torch.arange(seen, seen + batch, dtype=torch.long)
            seen += batch

            taps = _LayerTaps(target_layer)
            teacher.zero_grad(set_to_none=True)

            logits = forward_logits(teacher, xb)
            probs_full = torch.softmax(logits.detach(), dim=-1)
            top1 = probs_full.argmax(dim=1)
            target_idx = yb if use_gt_target else top1

            oh = torch.zeros_like(logits)
            oh.scatter_(1, target_idx.view(-1, 1), 1.0)
            logits.backward(gradient=oh, retain_graph=False)

            g = xb.grad.detach().abs().amax(dim=1, keepdim=True)
            sal = (g - g.amin(dim=(2, 3), keepdim=True)) / (
                g.amax(dim=(2, 3), keepdim=True) - g.amin(dim=(2, 3), keepdim=True) + 1e-8
            )

            A, dY = taps.A, taps.dY
            if A is None or dY is None:
                taps.close()
                raise RuntimeError("Target layer hooks did not capture activations or gradients.")

            cam_small = _gradcam_pp_from_taps(A, dY)
            cam = F.interpolate(cam_small, size=(height, width), mode="bilinear", align_corners=False)
            cam = _percentile_scale_t(cam, lo=pct_lo, hi=pct_hi)
            sal = _percentile_scale_t(sal, lo=pct_lo, hi=pct_hi)
            union = _soft_or_t(cam, sal, w_cam=w_cam, w_sal=w_sal, temp=temp)
            if gamma != 1.0:
                union = union.clamp(0, 1).pow(1.0 / max(gamma, 1e-6))

            taps.close()
            xb.grad = None

            mean_union_map, frac_map = _build_windows_by_pool(union, crop_h, crop_w, stage_stride, edge_align)
            yh, xw, _ = _gather_top_windows(
                mean_union_map,
                frac_map,
                k_per_img=k_per_img,
                min_frac=min_frac,
                min_mean=min_mean,
            )
            boxes = _boxes_from_grid(yh, xw, crop_h, crop_w, stage_stride, height, width, edge_align)
            boxes_sorted, scores_sorted, _ = _score_candidates_with_teacher(
                teacher,
                logits.detach(),
                target_idx,
                xb,
                boxes,
                eval_chunk=eval_chunk,
            )

            for sample_idx in range(batch):
                _collect_crop_candidates(
                    xb=xb,
                    yb=yb,
                    sample_idx=sample_idx,
                    boxes_sorted=boxes_sorted,
                    scores_sorted=scores_sorted,
                    crop_h=crop_h,
                    crop_w=crop_w,
                    base_indices=base_indices,
                    top_k_crops=top_k_crops,
                    low_k_crops=low_k_crops,
                    crops_all=crops_all,
                    labels_all=labels_all,
                    scores_all=scores_all,
                    sources_all=sources_all,
                    quality_all=quality_all,
                )

            teacher.zero_grad(set_to_none=True)
            xb = xb.detach()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if not crops_all:
        raise RuntimeError("No TGSC crops were generated; please verify teacher checkpoints and settings.")

    X = torch.cat(crops_all, dim=0)
    Y = torch.cat(labels_all, dim=0).long()
    S = torch.cat(scores_all, dim=0).float().clamp(0.0, 1.0)
    src = torch.cat(sources_all, dim=0).long()
    qflag = torch.cat(quality_all, dim=0).long()
    print(f"[TGSC] Built best-crop tensor {tuple(X.shape)} (C,H,W) with {S.numel()} crops.")
    return CropTensorDataset(X, Y, S, src, qflag)


def _collect_crop_candidates(
    *,
    xb: torch.Tensor,
    yb: torch.Tensor,
    sample_idx: int,
    boxes_sorted: torch.Tensor,
    scores_sorted: torch.Tensor,
    crop_h: int,
    crop_w: int,
    base_indices: torch.Tensor,
    top_k_crops: int,
    low_k_crops: int,
    crops_all: list[torch.Tensor],
    labels_all: list[torch.Tensor],
    scores_all: list[torch.Tensor],
    sources_all: list[torch.Tensor],
    quality_all: list[torch.Tensor],
) -> None:
    sample_scores = scores_sorted[sample_idx]
    valid_mask = sample_scores >= 0
    valid_idx = torch.nonzero(valid_mask, as_tuple=False).squeeze(-1)
    if valid_idx.numel() == 0:
        return

    valid = valid_idx.tolist()
    top_sel = valid[: max(0, min(top_k_crops, len(valid)))]
    low_sel_candidates = valid[-max(0, min(low_k_crops, len(valid))) :] if low_k_crops > 0 else []
    used = set(top_sel)
    low_sel: list[int] = []
    for candidate_idx in reversed(low_sel_candidates):
        if candidate_idx not in used:
            low_sel.append(candidate_idx)
            used.add(candidate_idx)

    selections = [(candidate_idx, rank, 1) for rank, candidate_idx in enumerate(top_sel)]
    selections.extend((candidate_idx, rank, -1) for rank, candidate_idx in enumerate(low_sel))

    for candidate_idx, rank, quality_flag in selections:
        score_val = sample_scores[candidate_idx]
        if score_val < 0:
            continue
        y0, x0, y1, x1 = boxes_sorted[sample_idx, candidate_idx].long().tolist()
        crop = xb[sample_idx, :, y0:y1, x0:x1].detach()
        height, width = crop.shape[-2:]
        if height != crop_h or width != crop_w:
            pad_t = max(0, crop_h - height)
            pad_l = max(0, crop_w - width)
            crop = F.pad(crop, (0, pad_l, 0, pad_t), mode="reflect")[:, :crop_h, :crop_w]

        crops_all.append(crop.cpu().unsqueeze(0))
        labels_all.append(yb[sample_idx].detach().cpu().unsqueeze(0))
        scores_all.append(score_val.detach().cpu().unsqueeze(0))
        sources_all.append(torch.tensor([int(base_indices[sample_idx].item())], dtype=torch.long))
        quality_all.append(torch.tensor([quality_flag * (rank + 1)], dtype=torch.long))


class CropTensorDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        crops: torch.Tensor,
        labels: torch.Tensor,
        scores: torch.Tensor,
        source_indices: torch.Tensor,
        quality_flags: torch.Tensor,
    ):
        self.crops = crops
        self.labels = labels
        self.scores = scores
        self.source_indices = source_indices
        self.quality_flags = quality_flags
        self.transform = None

    def __len__(self) -> int:
        return self.crops.size(0)

    def __getitem__(self, idx: int):
        x = self.crops[idx]
        if self.transform is not None:
            x = self.transform(x.clone())
        return (
            x,
            self.labels[idx],
            self.scores[idx],
            self.quality_flags[idx],
            idx,
        )

    def set_transform(self, transform) -> None:
        self.transform = transform

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {
            "crops": self.crops,
            "labels": self.labels,
            "scores": self.scores,
            "source_indices": self.source_indices,
            "quality_flags": self.quality_flags,
        }

    @classmethod
    def from_state_dict(cls, payload: dict[str, torch.Tensor]) -> "CropTensorDataset":
        return cls(
            crops=payload["crops"],
            labels=payload["labels"],
            scores=payload["scores"],
            source_indices=payload["source_indices"],
            quality_flags=payload["quality_flags"],
        )


class TGSCPreprocessor:
    def __init__(self, teacher: nn.Module, data_cfg: DataConfig, tgsc_cfg: TGSCConfig):
        self.teacher = teacher
        self.data_cfg = data_cfg
        self.tgsc_cfg = tgsc_cfg
        self.target_layer = pick_target_layer(teacher)
        base_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(data_cfg.mean, data_cfg.std),
            ]
        )
        self.base_dataset = datasets.CIFAR100(
            data_cfg.root,
            train=True,
            download=True,
            transform=base_transform,
        )
        self._cache: dict[int, CropTensorDataset] = {}
        self.last_dataset: Optional[CropTensorDataset] = None
        self._teacher_cache: Optional[TeacherUnionCache] = None
        self._cache_dir = Path(tgsc_cfg.cache_dir)
        self._cache_key = self._build_cache_key()

    def _make_base_loader(self) -> DataLoader:
        return DataLoader(
            self.base_dataset,
            batch_size=self.tgsc_cfg.precompute_batch_size,
            shuffle=False,
            num_workers=self.tgsc_cfg.precompute_workers,
            pin_memory=self.data_cfg.pin_memory,
        )

    def _build_teacher_cache(self) -> TeacherUnionCache:
        base_loader = self._make_base_loader()
        return build_teacher_union_cache(
            base_loader,
            self.teacher,
            self.target_layer,
            use_gt_target=self.tgsc_cfg.use_gt_target,
            w_cam=self.tgsc_cfg.w_cam,
            w_sal=self.tgsc_cfg.w_sal,
            temp=self.tgsc_cfg.temp,
            gamma=self.tgsc_cfg.gamma,
            pct_lo=self.tgsc_cfg.pct_lo,
            pct_hi=self.tgsc_cfg.pct_hi,
        )

    def _build_cache_key(self) -> str:
        payload = {
            "teacher_class": type(self.teacher).__name__,
            "data_root": self.data_cfg.root,
            "mean": self.data_cfg.mean,
            "std": self.data_cfg.std,
            "tgsc": {
                "use_gt_target": self.tgsc_cfg.use_gt_target,
                "w_cam": self.tgsc_cfg.w_cam,
                "w_sal": self.tgsc_cfg.w_sal,
                "temp": self.tgsc_cfg.temp,
                "gamma": self.tgsc_cfg.gamma,
                "pct_lo": self.tgsc_cfg.pct_lo,
                "pct_hi": self.tgsc_cfg.pct_hi,
                "candidate_stride": self.tgsc_cfg.candidate_stride or self.tgsc_cfg.stride,
                "dynamic_stride": self.tgsc_cfg.dynamic_stride,
                "dynamic_stride_power": self.tgsc_cfg.dynamic_stride_power,
                "k_per_img": self.tgsc_cfg.k_per_img,
                "min_frac": self.tgsc_cfg.min_frac,
                "min_mean": self.tgsc_cfg.min_mean,
                "eval_chunk": self.tgsc_cfg.eval_chunk,
                "top_k_crops": self.tgsc_cfg.top_k_crops,
                "low_k_crops": self.tgsc_cfg.low_k_crops,
            },
        }
        if bool(getattr(self.tgsc_cfg, "edge_align_windows", False)):
            payload["tgsc"]["edge_align_windows"] = True
        digest = hashlib.sha1(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:12]
        return f"tgsc_{digest}"

    def _cache_path(self, crop_size: int) -> Path:
        return self._cache_dir / f"{self._cache_key}_res{crop_size}.pt"

    def _load_cached_dataset(self, crop_size: int) -> Optional[CropTensorDataset]:
        path = self._cache_path(crop_size)
        if not path.exists():
            return None
        payload = torch.load(path, map_location="cpu", weights_only=False)
        dataset = CropTensorDataset.from_state_dict(payload)
        print(f"[TGSC] Loaded cached dataset from {path}")
        return dataset

    def _save_cached_dataset(self, crop_size: int, dataset: CropTensorDataset) -> None:
        path = self._cache_path(crop_size)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(dataset.state_dict(), path)
        print(f"[TGSC] Saved cached dataset to {path}")

    def _ensure_dataset(self, crop_size: int) -> CropTensorDataset:
        if self.tgsc_cfg.cache and crop_size in self._cache:
            print(f"[TGSC] Reusing cached crops at {crop_size}x{crop_size}.")
            return self._cache[crop_size]

        if self._cache:
            print(f"[TGSC] Clearing old cached datasets: {list(self._cache.keys())}")
            self._cache.clear()

        if self.tgsc_cfg.cache_to_disk:
            dataset = self._load_cached_dataset(crop_size)
            if dataset is not None:
                self._cache[crop_size] = dataset
                return dataset

        if self._teacher_cache is None:
            self._teacher_cache = self._build_teacher_cache()

        print(f"[TGSC] Building new crops for {crop_size}x{crop_size}...")
        base_loader = self._make_base_loader()
        dataset = build_bestcrop_dataset_batchwise(
            base_loader,
            self.teacher,
            self.target_layer,
            crop_size=crop_size,
            use_gt_target=self.tgsc_cfg.use_gt_target,
            w_cam=self.tgsc_cfg.w_cam,
            w_sal=self.tgsc_cfg.w_sal,
            temp=self.tgsc_cfg.temp,
            gamma=self.tgsc_cfg.gamma,
            pct_lo=self.tgsc_cfg.pct_lo,
            pct_hi=self.tgsc_cfg.pct_hi,
            stride=int(self.tgsc_cfg.candidate_stride or self.tgsc_cfg.stride),
            k_per_img=self.tgsc_cfg.k_per_img,
            min_frac=self.tgsc_cfg.min_frac,
            min_mean=self.tgsc_cfg.min_mean,
            eval_chunk=self.tgsc_cfg.eval_chunk,
            top_k_crops=self.tgsc_cfg.top_k_crops,
            low_k_crops=self.tgsc_cfg.low_k_crops,
            dynamic_stride=self.tgsc_cfg.dynamic_stride,
            dynamic_stride_power=self.tgsc_cfg.dynamic_stride_power,
            stage_sizes=tuple(int(resolution) for resolution, _epochs in self.data_cfg.stages),
            teacher_cache=self._teacher_cache,
            edge_align=bool(getattr(self.tgsc_cfg, "edge_align_windows", False)),
        )
        if self.tgsc_cfg.cache_to_disk:
            self._save_cached_dataset(crop_size, dataset)
        if self.tgsc_cfg.cache:
            self._cache[crop_size] = dataset
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return dataset

    def get_dataset(self, crop_size: int) -> CropTensorDataset:
        return self._ensure_dataset(crop_size)

    def get_loader(
        self,
        crop_size: int,
        *,
        batch_size: int,
        num_workers: int,
        pin_memory: bool,
    ) -> DataLoader:
        dataset = self._ensure_dataset(crop_size)
        aug_ops = []
        if self.tgsc_cfg.aug_padding > 0:
            aug_ops.append(
                transforms.RandomCrop(
                    crop_size,
                    padding=self.tgsc_cfg.aug_padding,
                    padding_mode="reflect",
                )
            )
        if self.tgsc_cfg.flip_prob > 0.0:
            aug_ops.append(transforms.RandomHorizontalFlip(p=self.tgsc_cfg.flip_prob))
        aug_tfm = transforms.Compose(aug_ops) if aug_ops else None

        dataset.set_transform(aug_tfm)
        self.last_dataset = dataset
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
