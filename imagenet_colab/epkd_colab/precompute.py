"""Batched, shape-bucketed, resumable TGSC/EP-KD proposal-cache precompute (device-agnostic: cuda / tpu / cpu).

Same algorithm as the reference per-image path (``ImageNetTGSCPreprocessor.compute_union`` + ``emit_stage_rows``):
images decoded by the unmodified reference dataset (PIL, un-normalised [0,1] teacher input quirk kept), grouped by
exact (H, W) after the 256-short-side resize (no padding: padding would change the CAM/saliency), teacher
forward/backward once per image, Grad-CAM++ x input-gradient saliency union, edge-aligned candidate windows per
stage, teacher scoring of the candidates, top-k selection in saliency order -- all with the reference functions
called on a batch instead of a single image.  Chunk/part/marker layout identical to the reference
``scripts/imagenet_fix/precompute_cache.py`` so its ``--finalize`` and ``verify_cache.py`` apply unchanged.

    python -m epkd_colab.precompute --configs F1.json F2.json --device cuda:0 --prepare
    python -m epkd_colab.precompute --configs F1.json F2.json --device cuda:0 --claim --batch 64 --workers 10
    python -m epkd_colab.precompute --configs F1.json F2.json --device cpu --finalize
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import epkd_colab  # noqa: F401
from epkd_colab.config import build_teacher, load_cfg
from tgsc_kd.imagenet_parquet import (
    ImageNetPrecomputeDataset,
    ImageNetTGSCPreprocessor,
    _candidate_part_table,
    _select_stage_rows,
    _stage_part_table,
    _write_table,
)
from tgsc_kd.tgsc import (
    _LayerTaps,
    _boxes_from_grid,
    _build_windows_by_pool,
    _gather_top_windows,
    _gradcam_pp_from_taps,
    _percentile_scale_t,
    _score_candidates_with_teacher,
    _soft_or_t,
    resolve_stage_stride,
)
from tgsc_kd.tgsc import _kl_div_batch, _top2_batch
from tgsc_kd.utils import forward_logits

import precompute_cache as ref_pc


def score_candidates_batched(model, full_logits, class_idx, x, boxes, *, eval_chunk: int, sort: bool):
    """``tgsc._score_candidates_with_teacher`` with ONE host transfer of the boxes (the reference calls
    ``.tolist()`` per candidate, i.e. B*k device syncs); same operations in the same order otherwise."""
    device = x.device
    batch, num_candidates, _ = boxes.shape
    if num_candidates == 0:
        return boxes[:, :0], torch.empty((batch, 0), device=device), torch.empty((batch, 0, full_logits.size(1)), device=device)
    p_full = torch.softmax(full_logits, dim=-1)
    _, pc1_f, _, pc2_f = _top2_batch(p_full)
    blist = boxes.detach().cpu().tolist()
    crops, owners, candidate_indices = [], [], []
    for batch_idx in range(batch):
        for candidate_idx in range(num_candidates):
            y0, x0, y1, x1 = (int(v) for v in blist[batch_idx][candidate_idx])
            crops.append(x[batch_idx: batch_idx + 1, :, y0:y1, x0:x1])
            owners.append(batch_idx)
            candidate_indices.append(candidate_idx)
    crops_cat = torch.cat(crops, dim=0)
    if device.type == "cuda":
        crops_cat = crops_cat.contiguous(memory_format=torch.channels_last)
    preds = []
    with torch.no_grad():
        for start in range(0, crops_cat.size(0), eval_chunk):
            preds.append(torch.softmax(forward_logits(model, crops_cat[start: start + eval_chunk]), dim=-1))
    p_k_all = torch.cat(preds, dim=0)
    import math as _m

    alpha = _m.log(p_k_all.size(1))
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
    margin_k = top2_k[:, 0] if top2_k.size(1) == 1 else top2_k[:, 0] - top2_k[:, 1]
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
    boxes_sorted = boxes.gather(1, order.unsqueeze(-1).expand(-1, -1, 4))
    probs_sorted = prob_matrix.gather(1, order.unsqueeze(-1).expand(-1, -1, prob_matrix.size(-1)))
    return boxes_sorted, sorted_scores, probs_sorted  # reference script (ref/scripts): chunk markers, claim_iter, finalize


FAST_SCORER = os.environ.get("EPKD_FAST_SCORER", "1") != "0"


def make_targets(config_paths: list[str], device: torch.device):
    cfgs = [load_cfg(p) for p in config_paths]
    assert len({c.teacher_model for c in cfgs}) == 1
    teacher, ckpt = build_teacher(cfgs[0])
    teacher.to(device).eval()
    targets = [ImageNetTGSCPreprocessor(teacher, c.data, c.tgsc, teacher_name=c.teacher_model, teacher_checkpoint=ckpt) for c in cfgs]
    assert len({t._union_signature() for t in targets}) == 1
    assert len({t.train_split.total_rows for t in targets}) == 1
    assert len({int(t.tgsc_cfg.cache_shard_size) for t in targets}) == 1
    print(f"[targets] roots={[t.cache_root.name for t in targets]} stages={[t.stage_sizes for t in targets]} "
          f"total_rows={targets[0].train_split.total_rows}", file=sys.stderr, flush=True)
    return cfgs, teacher, ckpt, targets


# ------------------------------------------------------------------------------------------- batched reference
def compute_union_batched(pre: ImageNetTGSCPreprocessor, xb: torch.Tensor, labels: torch.Tensor) -> dict[str, Any]:
    """``compute_union`` for a batch of same-size images (all ops are per-sample)."""
    dev = pre.device
    xb = xb.to(dev, non_blocking=True)
    if dev.type == "cuda":
        xb = xb.contiguous(memory_format=torch.channels_last)
    xb.requires_grad_(True)
    yb = labels.to(dev)
    taps = _LayerTaps(pre.target_layer)
    pre.teacher.zero_grad(set_to_none=True)
    logits = forward_logits(pre.teacher, xb)
    probs_full = torch.softmax(logits.detach(), dim=-1)
    top1 = probs_full.argmax(dim=1)
    target_idx = yb if pre.tgsc_cfg.use_gt_target else top1
    oh = torch.zeros_like(logits)
    oh.scatter_(1, target_idx.view(-1, 1), 1.0)
    logits.backward(gradient=oh, retain_graph=False)
    _, _, height, width = xb.shape
    g = xb.grad.detach().abs().amax(dim=1, keepdim=True)
    sal = (g - g.amin(dim=(2, 3), keepdim=True)) / (g.amax(dim=(2, 3), keepdim=True) - g.amin(dim=(2, 3), keepdim=True) + 1e-8)
    A, dY = taps.A, taps.dY
    if A is None or dY is None:
        taps.close()
        raise RuntimeError("Target layer hooks did not capture activations or gradients.")
    cam_small = _gradcam_pp_from_taps(A, dY)
    cam = F.interpolate(cam_small, size=(height, width), mode="bilinear", align_corners=False)
    cam = _percentile_scale_t(cam, lo=pre.tgsc_cfg.pct_lo, hi=pre.tgsc_cfg.pct_hi)
    sal = _percentile_scale_t(sal, lo=pre.tgsc_cfg.pct_lo, hi=pre.tgsc_cfg.pct_hi)
    union = _soft_or_t(cam, sal, w_cam=pre.tgsc_cfg.w_cam, w_sal=pre.tgsc_cfg.w_sal, temp=pre.tgsc_cfg.temp)
    if pre.tgsc_cfg.gamma != 1.0:
        union = union.clamp(0, 1).pow(1.0 / max(pre.tgsc_cfg.gamma, 1e-6))
    taps.close()
    xb.grad = None
    pre.teacher.zero_grad(set_to_none=True)
    return {"xb": xb.detach(), "logits": logits.detach(), "target_idx": target_idx, "union": union.detach(), "height": height,
            "width": width, "union_cfg": pre._union_signature()}


def emit_stage_rows_batched(pre: ImageNetTGSCPreprocessor, analysis: dict[str, Any], sample_ids: list[int],
                            candidate_rows: dict[int, list], stage_rows: dict[int, list]) -> None:
    """``emit_stage_rows`` for a batch (the reference helpers are batch-native; selection per image)."""
    if analysis["union_cfg"] != pre._union_signature():
        raise ValueError("union map was computed with different TGSC union settings")
    xb, logits, target_idx, union = analysis["xb"], analysis["logits"], analysis["target_idx"], analysis["union"]
    height, width = analysis["height"], analysis["width"]
    sort = getattr(pre.tgsc_cfg, "view_selection", "score") != "saliency"
    for crop_size in pre.stage_sizes:
        stride = resolve_stage_stride(pre.tgsc_cfg, crop_size, pre.stage_sizes)
        mean_map, frac_map = _build_windows_by_pool(union, crop_size, crop_size, stride, pre.edge_align)
        if mean_map.numel() == 0:
            continue
        yh, xw, _ = _gather_top_windows(mean_map, frac_map, k_per_img=pre.tgsc_cfg.k_per_img, min_frac=pre.tgsc_cfg.min_frac,
                                        min_mean=pre.tgsc_cfg.min_mean)
        boxes = _boxes_from_grid(yh, xw, crop_size, crop_size, stride, height, width, pre.edge_align)
        if boxes.numel() == 0:
            continue
        scorer = score_candidates_batched if FAST_SCORER else _score_candidates_with_teacher
        boxes_s, scores_s, _ = scorer(pre.teacher, logits, target_idx, xb, boxes, eval_chunk=pre.tgsc_cfg.eval_chunk, sort=sort)
        boxes_cpu = boxes_s.detach().cpu()
        scores_cpu = scores_s.detach().cpu()
        blist = boxes_cpu.tolist()
        slist = scores_cpu.tolist()
        for i, sid in enumerate(sample_ids):
            b, s = boxes_cpu[i], scores_cpu[i]
            for k in range(s.numel()):
                y0, x0, y1, x1 = blist[i][k]
                candidate_rows[crop_size].append({"sample_id": sid, "y0": int(y0), "x0": int(x0), "y1": int(y1), "x1": int(x1), "score": float(slist[i][k])})
            stage_rows[crop_size].extend(_select_stage_rows(sample_id=sid, resolution=crop_size, boxes_sorted=b, scores_sorted=s,
                                                            top_k_crops=pre.tgsc_cfg.top_k_crops, low_k_crops=pre.tgsc_cfg.low_k_crops))


class Batcher:
    """Groups decoded samples by exact (H, W); yields (xb, labels, sample_ids) batches of ``batch`` images."""

    def __init__(self, batch: int) -> None:
        self.batch = int(batch)
        self.buckets: dict[tuple[int, int], list] = defaultdict(list)
        self.n_pending = 0

    def add(self, sample: dict[str, Any]):
        t = sample["image_tensor"]
        key = (int(t.shape[-2]), int(t.shape[-1]))
        self.buckets[key].append(sample)
        self.n_pending += 1
        if len(self.buckets[key]) >= self.batch:
            return self._pop(key)
        return None

    def _pop(self, key):
        items = self.buckets.pop(key)
        self.n_pending -= len(items)
        xb = torch.stack([s["image_tensor"] for s in items])
        return xb, torch.tensor([int(s["label"]) for s in items], dtype=torch.long), [int(s["sample_id"]) for s in items]

    def flush(self):
        for key in list(self.buckets.keys()):
            yield self._pop(key)


def process_batch(targets, xb, labels, sids, cand, stage, timing: dict) -> None:
    lead = targets[0]
    t0 = time.time()
    analysis = compute_union_batched(lead, xb, labels)
    timing["union"] += time.time() - t0
    t0 = time.time()
    for i, t in enumerate(targets):
        emit_stage_rows_batched(t, analysis, sids, cand[i], stage[i])
    timing["emit"] += time.time() - t0


def run_chunks(targets, *, device, batch: int, workers: int, claim: bool, proc_index: int, num_procs: int, max_chunks, log_every: int) -> None:
    lead = targets[0]
    total_chunks = ref_pc.n_chunks(lead)
    cs = int(lead.tgsc_cfg.cache_shard_size)
    if claim:
        todo = ref_pc.claim_iter(targets, total_chunks)
    else:
        mine = [c for c in range(total_chunks) if c % num_procs == proc_index]
        todo = [c for c in mine if not all(ref_pc.chunk_marker(t, c).exists() for t in targets)]
        if max_chunks is not None:
            todo = todo[:max_chunks]
    for t in targets:
        t.tgsc_cfg.precompute_workers = workers
    done_imgs, t_start = 0, time.time()
    for chunk in todo:
        start, stop = chunk * cs, min((chunk + 1) * cs, lead.train_split.total_rows)
        cand = [{s: [] for s in t.stage_sizes} for t in targets]
        stage = [{s: [] for s in t.stage_sizes} for t in targets]
        timing = {"union": 0.0, "emit": 0.0, "decode_wait": 0.0}
        tc = time.time()
        n = 0
        batcher = Batcher(batch)
        loader = lead._make_precompute_loader(start=start, stop=stop)
        tw = time.time()
        for sample in loader:
            timing["decode_wait"] += time.time() - tw
            n += 1
            b = batcher.add(sample)
            if b is not None:
                process_batch(targets, *b, cand, stage, timing)
            tw = time.time()
        for b in batcher.flush():
            process_batch(targets, *b, cand, stage, timing)
        done_imgs += n
        if n != stop - start:
            print(f"[proc {proc_index}] WARNING chunk {chunk}: {n} of {stop - start} samples processed", flush=True)
        for i, t in enumerate(targets):
            for s in t.stage_sizes:
                _write_table(t.candidate_dir(s) / f"part-c{chunk:05d}.parquet", _candidate_part_table(cand[i][s]))
                _write_table(t.stage_tmp_dir(s) / f"part-c{chunk:05d}.parquet", _stage_part_table(stage[i][s]))
            m = ref_pc.chunk_marker(t, chunk)
            m.parent.mkdir(parents=True, exist_ok=True)
            m.write_text(json.dumps({"start": start, "stop": stop, "processed": n, "secs": time.time() - tc, "pid": os.getpid(),
                                     "proc": proc_index, "batched": True, "device": str(device)}))
        el = time.time() - tc
        print(f"[proc {proc_index}] chunk {chunk} [{start},{stop}) {n} imgs in {el:.0f}s = {n / el:.1f} img/s "
              f"(union {timing['union']:.0f}s, emit {timing['emit']:.0f}s, decode-wait {timing['decode_wait']:.0f}s) | "
              f"total {done_imgs} imgs {done_imgs / (time.time() - t_start):.1f} img/s", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="+", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--workers", type=int, default=8, help="PIL decode workers (reference dataset)")
    ap.add_argument("--proc-index", type=int, default=0)
    ap.add_argument("--num-procs", type=int, default=1)
    ap.add_argument("--max-chunks", type=int, default=None)
    ap.add_argument("--log-every", type=int, default=0)
    ap.add_argument("--prepare", action="store_true")
    ap.add_argument("--claim", action="store_true")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--finalize", action="store_true")
    ap.add_argument("--print-roots", action="store_true")
    ap.add_argument("--tf32", type=int, default=1, help="CUDA: allow TF32 convolutions (PyTorch default); 0 = strict fp32")
    ap.add_argument("--cudnn-benchmark", type=int, default=0)
    ap.add_argument("--f32-precision", default="highest", choices=["highest", "high", "medium"],
                    help="torch.set_float32_matmul_precision for fp32 convs/matmuls (TPU default passes are reduced precision)")
    args = ap.parse_args()
    torch.set_float32_matmul_precision(args.f32_precision)
    if args.device == "tpu":
        import torch_tpu  # noqa: F401
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.backends.cudnn.benchmark = bool(args.cudnn_benchmark)
        torch.backends.cudnn.allow_tf32 = bool(args.tf32)
        torch.backends.cuda.matmul.allow_tf32 = bool(args.tf32)
    _cfgs, _teacher, _ckpt, targets = make_targets(args.configs, device)
    if args.print_roots:
        for t in targets:
            print(t.cache_root)
        return 0
    if args.status:
        tot = ref_pc.n_chunks(targets[0])
        done = sum(all(ref_pc.chunk_marker(t, c).exists() for t in targets) for c in range(tot))
        complete = all(t._cache_is_complete(t._load_state()) for t in targets)
        print(json.dumps({"done": done, "total": tot, "complete": complete, "roots": [str(t.cache_root) for t in targets]}))
        return 0
    if args.prepare:
        ref_pc.prepare(targets)
        return 0
    if args.finalize:
        return 0 if ref_pc.finalize(targets) else 2
    run_chunks(targets, device=device, batch=args.batch, workers=args.workers, claim=args.claim, proc_index=args.proc_index,
               num_procs=args.num_procs, max_chunks=args.max_chunks, log_every=args.log_every)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
