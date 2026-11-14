"""Hybrid TPU + CPU proposal-cache precompute for the EP-KD ImageNet recipe (same algorithm and on-disk layout as
``epkd_colab.precompute`` / the reference ``ImageNetTGSCPreprocessor``: chunk part files + markers, then ``--finalize``).

Work split (author's design, 2026-11-10):
  * HEAD shapes = the most common post-resize image shapes (H, W) (meta/train_headers.npz): teacher forward+backward
    union map, window means and top-k on the accelerator; one compiled executable per shape, fixed batch padded by
    repeating the last image (every op is per-sample, so padding the BATCH never changes a real image's result).
  * TAIL shapes (ImageNet has 1,727 shapes; compiling the rare ones costs more than it saves): the same union code,
    eager on the CPU, in a process pool (1 thread per process) that runs concurrently with the accelerator.
  * Candidate scoring (teacher forward on the fixed-size crops; most of the FLOPs) for ALL images on the
    accelerator, crops cut on the host from the uint8 image; one compiled executable per crop size.
    With view_selection=saliency and kd_gate=off (F1/F2) the selected rows are the first top_k candidates in saliency
    order and the KD weights are 1, so the score only fills the cache's score column; --score-dtype bf16 is allowed.

Numerics vs the reference (verify with tools/verify_cache.py --device cpu --min-box-rate 0.95):
  * hook-free teacher forward that also returns the CAM layer output (layer4[-1].conv2), gradients by
    torch.autograd.grad w.r.t. (input, CAM layer output): the same maths as the reference's forward/backward hooks;
  * torch.quantile re-implemented with the same fp32 rank/lerp arithmetic (sort-based, compiles on the TPU);
  * window means/fractions computed per window by slicing instead of a dense stride-1 avg_pool + index_select:
    same values up to fp32 summation order (near-tie windows can swap; that is what the tolerant verify allows);
  * union in fp32 with torch.set_float32_matmul_precision('highest') (TPU default passes are reduced precision).

Usage (on the VM; the CPU pool is forked BEFORE torch_tpu is imported):
  python -m epkd_colab.precompute_hybrid --configs F1.json F2.json --prepare
  python -m epkd_colab.precompute_hybrid --configs F1.json F2.json --device tpu --head-k 60 --cpu-procs 16 ...
  python -m epkd_colab.precompute_hybrid --configs F1.json F2.json --finalize
"""
from __future__ import annotations

import os
import argparse
from dataclasses import dataclass
import json
import math
import multiprocessing as mp
from pathlib import Path
import sys
import time
from typing import Any, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import epkd_colab  # noqa: F401  (puts ref/src, ref/scripts, ref/ on sys.path)
from epkd_colab.config import build_teacher, load_cfg
from tgsc_kd.imagenet_parquet import (
    ImageNetPrecomputeDataset,
    ImageNetTGSCPreprocessor,
    _candidate_part_table,
    _select_stage_rows,
    _stage_part_table,
    _write_table,
)
from tgsc_kd.tgsc import _boxes_from_grid, _gather_top_windows, _gradcam_pp_from_taps, _kl_div_batch, _soft_or_t, \
    resolve_stage_stride, window_starts
from tgsc_kd.utils import unwrap_logits

import precompute_cache as ref_pc

ROOT = Path(__file__).resolve().parents[1]


# ------------------------------------------------------------------------------------------------ shared maths
@dataclass(frozen=True)
class Spec:
    """One (target, stage): candidate windows of side ``crop`` on a ``stride`` grid (edge-aligned)."""
    target: int
    crop: int
    stride: int
    k: int
    min_frac: float
    min_mean: float
    edge: bool


@dataclass(frozen=True)
class UnionCfg:
    use_gt: bool
    w_cam: float
    w_sal: float
    temp: float
    gamma: float
    pct_lo: float
    pct_hi: float


def _f32(v: float) -> np.float32:
    return np.float32(v)


def quantile_linear(flat: torch.Tensor, q: float) -> torch.Tensor:
    """torch.quantile(flat, q, dim=1) with interpolation='linear', same fp32 arithmetic (rank = q*(n-1) in fp32,
    lerp between the neighbouring order statistics), but sort-based so it compiles on the TPU."""
    s, _ = torch.sort(flat, dim=1)
    n = s.size(1)
    rank = _f32(q) * _f32(n - 1)
    lo = int(math.floor(float(rank)))
    hi = int(math.ceil(float(rank)))
    w = float(np.float32(rank - np.float32(lo)))
    a = s[:, lo]
    if hi == lo or w == 0.0:
        return a
    return torch.lerp(a, s[:, hi], torch.full_like(a, w))


def percentile_scale(x: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
    """Reference ``tgsc._percentile_scale_t`` with ``quantile_linear``."""
    b = x.size(0)
    flat = x.reshape(b, -1)
    lo_v = quantile_linear(flat, lo / 100.0).view(b, 1, 1, 1)
    hi_v = quantile_linear(flat, hi / 100.0).view(b, 1, 1, 1)
    denom = (hi_v - lo_v).clamp_min(1e-8)
    return ((x - lo_v).clamp_min(0.0) / denom).clamp(0.0, 1.0)


def teacher_to_cam_layer(T: torch.nn.Module, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """First half of ``unwrap_logits(T(x))`` for model/python/resnet_imagenet.ResNet (BasicBlock): returns the output
    of the CAM target layer (``pick_target_layer`` = layer4[-1].conv2, captured by hooks in the reference) and the
    last block's residual.  Split at the CAM layer so that it is an intermediate of the autograd graph also when
    each half is torch.compiled (a single compiled function would return it as a sibling output of the logits)."""
    h = T.relu(T.bn1(T.conv1(x)))
    h = T.maxpool(h)
    h = T.layer3(T.layer2(T.layer1(h)))
    for blk in list(T.layer4)[:-1]:
        h = blk(h)
    b = T.layer4[-1]
    hin = F.relu(h)
    res = hin if b.downsample is None else b.downsample(hin)
    A = b.conv2(b.relu(b.bn1(b.conv1(hin))))
    if hasattr(b, "conv3"):  # Bottleneck (ResNet50): CAM layer = layer4[-1].conv3 (reference pick_target_layer)
        A = b.conv3(b.relu(b.bn2(A)))
    return A, res


def cam_layer_to_logits(T: torch.nn.Module, A: torch.Tensor, res: torch.Tensor) -> torch.Tensor:
    """Second half: bn2 (BasicBlock) / bn3 (Bottleneck) + residual of the last block, ReLU, global average pool, fc."""
    b = T.layer4[-1]
    feat4 = (b.bn3(A) if hasattr(b, "conv3") else b.bn2(A)) + res
    return T.fc(T.avgpool(F.relu(feat4)).flatten(1))


def check_teacher_layout(T: torch.nn.Module) -> None:
    b = T.layer4[-1]
    assert hasattr(b, "conv2") and hasattr(b, "bn1") and hasattr(b, "downsample"), \
        "expects a model/python/resnet_imagenet ResNet teacher (BasicBlock ResNet18/34 or Bottleneck ResNet50)"


def union_from_grads(A: torch.Tensor, dA: torch.Tensor, gx: torch.Tensor, ucfg: UnionCfg) -> torch.Tensor:
    """Reference ``compute_union`` after the backward pass (per-sample ops only)."""
    height, width = gx.shape[-2:]
    g = gx.abs().amax(dim=1, keepdim=True)
    sal = (g - g.amin(dim=(2, 3), keepdim=True)) / (g.amax(dim=(2, 3), keepdim=True) - g.amin(dim=(2, 3), keepdim=True) + 1e-8)
    cam = F.interpolate(_gradcam_pp_from_taps(A, dA), size=(height, width), mode="bilinear", align_corners=False)
    cam = percentile_scale(cam, ucfg.pct_lo, ucfg.pct_hi)
    sal = percentile_scale(sal, ucfg.pct_lo, ucfg.pct_hi)
    union = _soft_or_t(cam, sal, w_cam=ucfg.w_cam, w_sal=ucfg.w_sal, temp=ucfg.temp)
    if ucfg.gamma != 1.0:
        union = union.clamp(0, 1).pow(1.0 / max(ucfg.gamma, 1e-6))
    return union


def window_maps(union: torch.Tensor, crop: int, ys: list[int], xs: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference ``_build_windows_by_pool`` (edge-aligned) evaluated only at the window start offsets."""
    thr = percentile_scale(union, 75, 75)
    mask = (union >= thr).float()
    u, m = union[:, 0], mask[:, 0]
    means = torch.stack([torch.stack([u[:, y:y + crop, x:x + crop].mean(dim=(1, 2)) for x in xs], 1) for y in ys], 1)
    fracs = torch.stack([torch.stack([m[:, y:y + crop, x:x + crop].mean(dim=(1, 2)) for x in xs], 1) for y in ys], 1)
    return means.unsqueeze(1), fracs.unsqueeze(1)


def candidates_from_union(union: torch.Tensor, specs: tuple[Spec, ...]) -> list[torch.Tensor]:
    """Per spec: (B, k, 4) candidate boxes (y0, x0, y1, x1) in saliency order (reference emit_stage_rows, sort=False)."""
    height, width = union.shape[-2:]
    out = []
    for sp in specs:
        ys = window_starts(height, sp.crop, sp.stride, sp.edge)
        xs = window_starts(width, sp.crop, sp.stride, sp.edge)
        mean_map, frac_map = window_maps(union, sp.crop, ys, xs)
        yh, xw, _ = _gather_top_windows(mean_map, frac_map, k_per_img=sp.k, min_frac=sp.min_frac, min_mean=sp.min_mean)
        out.append(_boxes_from_grid(yh, xw, sp.crop, sp.crop, sp.stride, height, width, sp.edge))
    return out


def union_pass(fwd1, fwd2, post, x: torch.Tensor, y: torch.Tensor, use_gt: bool):
    """Teacher forward + backward (input and CAM layer) -> (logits, [boxes per spec])."""
    x = x.requires_grad_(True)
    A, res = fwd1(x)
    logits = fwd2(A, res)
    target = y if use_gt else logits.detach().argmax(dim=1)
    oh = torch.zeros_like(logits).scatter_(1, target.view(-1, 1), 1.0)
    gx, dA = torch.autograd.grad(logits, [x, A], grad_outputs=oh)
    return logits.detach(), post(A.detach(), dA, gx)


def score_crops(T: torch.nn.Module, crops_u8: torch.Tensor, pfull: torch.Tensor, cls: torch.Tensor,
                amp_dtype: Optional[torch.dtype]) -> torch.Tensor:
    """Reference ``_score_candidates_with_teacher`` score for each crop: (conf ratio + KL term + margin ratio) / 3."""
    x = crops_u8.float() / 255.0
    with torch.autocast(x.device.type, dtype=amp_dtype or torch.float32, enabled=amp_dtype is not None):
        logits = unwrap_logits(T(x))
    pk = torch.softmax(logits.float(), dim=-1)
    pc_k = pk.gather(1, cls.view(-1, 1)).squeeze(1)
    pc_f = pfull.gather(1, cls.view(-1, 1)).squeeze(1)
    conf_ratio = pc_k / (pc_f + 1e-8)
    kl_norm = 1.0 - torch.clamp(_kl_div_batch(pfull, pk) / math.log(pk.size(1)), max=1.0)
    t2k = torch.topk(pk, k=2, dim=1).values
    t2f = torch.topk(pfull, k=2, dim=1).values
    margin_rat = (t2k[:, 0] - t2k[:, 1]) / ((t2f[:, 0] - t2f[:, 1]) + 1e-8)
    return (conf_ratio + kl_norm + margin_rat) / 3.0


# ------------------------------------------------------------------------------------------------ CPU tail pool
_W: dict[str, Any] = {}


def _cpu_init(cfg_path: str, specs: tuple[Spec, ...], ucfg: UnionCfg) -> None:
    torch.set_num_threads(1)
    os.environ["OMP_NUM_THREADS"] = "1"
    T, _ = build_teacher(load_cfg(cfg_path))
    T.eval()
    for p in T.parameters():
        p.requires_grad_(False)
    _W.update(T=T, specs=specs, ucfg=ucfg)


def _cpu_union(job: tuple[int, int, np.ndarray]):
    sid, label, u8 = job
    T, specs, ucfg = _W["T"], _W["specs"], _W["ucfg"]
    x = torch.from_numpy(u8).unsqueeze(0).float().div(255.0)
    logits, boxes = union_pass(lambda t: teacher_to_cam_layer(T, t), lambda a, r: cam_layer_to_logits(T, a, r),
                               lambda A, dA, gx: candidates_from_union(union_from_grads(A, dA, gx, ucfg), specs),
                               x, torch.tensor([label]), ucfg.use_gt)
    return sid, logits[0].numpy(), [b[0].numpy() for b in boxes]


def _cpu_post(job):
    """Post-processing of one accelerator union batch on the CPU: (sids, logits, A, dA, gx) -> per image boxes."""
    sids, logits, A, dA, gx = job
    specs, ucfg = _W["specs"], _W["ucfg"]
    with torch.no_grad():
        boxes = candidates_from_union(union_from_grads(torch.from_numpy(A), torch.from_numpy(dA), torch.from_numpy(gx), ucfg), specs)
    return [(sid, logits[i], [b[i].numpy() for b in boxes]) for i, sid in enumerate(sids)]


# ------------------------------------------------------------------------------------------------ driver
def head_shapes(total_rows: int, head_k: int, head_min: int) -> dict[tuple[int, int], int]:
    """(H, W) after the reference resize (short side 256, long = int(256 * long / short)) -> image count, for the
    head set (top ``head_k`` shapes by count among ids < total_rows, each with at least ``head_min`` images)."""
    d = np.load(ROOT / "meta" / "train_headers.npz")
    order = np.argsort(d["sample_id"])
    w = d["w"][order][:total_rows].astype(np.int64)
    h = d["h"][order][:total_rows].astype(np.int64)
    s, L = np.minimum(w, h), np.maximum(w, h)
    Ln = 256 * L // s
    H = np.where(h <= w, 256, Ln)
    W = np.where(h <= w, Ln, 256)
    keys, counts = np.unique(np.stack([H, W], 1), axis=0, return_counts=True)
    idx = np.argsort(-counts)[: max(0, head_k)]
    head = {(int(keys[i][0]), int(keys[i][1])): int(counts[i]) for i in idx if counts[i] >= head_min}
    cov = sum(head.values()) / max(1, total_rows)
    print(f"[head] {len(head)} shapes cover {cov:.1%} of {total_rows} images (of {len(keys)} shapes)", flush=True)
    return head


class Chunk:
    """Book-keeping of one chunk: candidates, full-image probabilities, scores, pending score crops."""

    def __init__(self, specs: tuple[Spec, ...]):
        self.specs = specs
        self.label: dict[int, int] = {}
        self.pfull: dict[int, np.ndarray] = {}
        self.boxes: dict[int, list[np.ndarray]] = {}
        self.scores: dict[int, list[np.ndarray]] = {}
        self.queues: list[list[tuple[int, int, np.ndarray]]] = [[] for _ in specs]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="+", required=True)
    ap.add_argument("--device", default="tpu", help="tpu | cpu (cpu = test the head path eagerly)")
    ap.add_argument("--chunk", type=int, default=None, help="images per chunk (cache_shard_size override; same for run and finalize)")
    ap.add_argument("--head-k", type=int, default=60, help="max number of head shapes on the accelerator")
    ap.add_argument("--head-min", type=int, default=1, help="min images of a shape to be in the head set")
    ap.add_argument("--union-batch", type=int, default=32)
    ap.add_argument("--flush-min", type=int, default=4, help="partial head buckets smaller than this go to the CPU pool")
    ap.add_argument("--score-batch", type=int, default=256)
    ap.add_argument("--score-dtype", default="bf16", choices=["bf16", "fp32"])
    ap.add_argument("--cpu-procs", type=int, default=8)
    ap.add_argument("--max-pending", type=int, default=0, help="max queued tail jobs (0 = 8 per CPU proc)")
    ap.add_argument("--decode-workers", type=int, default=4)
    ap.add_argument("--compile", type=int, default=1)
    ap.add_argument("--f32-precision", default="highest", choices=["highest", "high", "medium"])
    ap.add_argument("--max-chunks", type=int, default=None)
    ap.add_argument("--log-every", type=int, default=500)
    ap.add_argument("--precompile", type=int, default=1, help="compile every head shape and score size before the loop")
    ap.add_argument("--claim", action="store_true", help="claim chunks via markers (several processes)")
    ap.add_argument("--prepare", action="store_true")
    ap.add_argument("--finalize", action="store_true")
    ap.add_argument("--status", action="store_true", help="print {done, total, complete, roots} as JSON")
    args = ap.parse_args()

    torch.set_float32_matmul_precision(args.f32_precision)
    cfgs = [load_cfg(p) for p in args.configs]
    assert len({c.teacher_model for c in cfgs}) == 1
    teacher_cpu, ckpt = build_teacher(cfgs[0])
    teacher_cpu.eval()
    targets = [ImageNetTGSCPreprocessor(teacher_cpu, c.data, c.tgsc, teacher_name=c.teacher_model, teacher_checkpoint=ckpt)
               for c in cfgs]
    assert len({t._union_signature() for t in targets}) == 1
    if args.chunk:
        for t in targets:
            t.tgsc_cfg.cache_shard_size = int(args.chunk)
    if args.prepare:
        ref_pc.prepare(targets)
        return 0
    if args.finalize:
        return 0 if ref_pc.finalize(targets) else 2
    if args.status:
        tot = ref_pc.n_chunks(targets[0])
        done = sum(all(ref_pc.chunk_marker(t, c).exists() for t in targets) for c in range(tot))
        complete = all(t._cache_is_complete(t._load_state()) for t in targets)
        print(json.dumps({"done": done, "total": tot, "complete": complete, "roots": [str(t.cache_root) for t in targets]}))
        return 0

    lead = targets[0]
    tc0 = lead.tgsc_cfg
    ucfg = UnionCfg(bool(tc0.use_gt_target), float(tc0.w_cam), float(tc0.w_sal), float(tc0.temp), float(tc0.gamma),
                    float(tc0.pct_lo), float(tc0.pct_hi))
    specs = tuple(Spec(ti, int(c), int(resolve_stage_stride(t.tgsc_cfg, c, t.stage_sizes)), int(t.tgsc_cfg.k_per_img),
                       float(t.tgsc_cfg.min_frac), float(t.tgsc_cfg.min_mean), bool(t.edge_align))
                  for ti, t in enumerate(targets) for c in t.stage_sizes)
    for t in targets:
        assert t.edge_align and getattr(t.tgsc_cfg, "view_selection", "score") == "saliency", "recipe assumption"
    total = lead.train_split.total_rows
    head = head_shapes(total, args.head_k, args.head_min) if args.head_k > 0 else {}
    print(f"[specs] {[(s.target, s.crop, s.stride, s.k) for s in specs]}", flush=True)

    # CPU tail pool: fork NOW, before any accelerator runtime exists in this process.
    pool = mp.get_context("fork").Pool(args.cpu_procs, initializer=_cpu_init, initargs=(args.configs[0], specs, ucfg)) \
        if args.cpu_procs > 0 else None
    max_pending = args.max_pending or 8 * max(1, args.cpu_procs)

    if args.device == "tpu":
        import torch_tpu  # noqa: F401
    dev = torch.device(args.device)
    # On the Colab TPU image `import torch` auto-loads the torch_tpu backend.  The decode workers (forkserver) must
    # never try to grab the TPU ("TPU is already in use by process ..."), so autoload is switched off for every
    # process started from here on (the CPU pool was forked above, before the TPU was first used).
    os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
    import copy
    T = copy.deepcopy(teacher_cpu).to(dev).eval()
    for p in T.parameters():
        p.requires_grad_(False)
    amp_dtype = torch.bfloat16 if args.score_dtype == "bf16" else None

    check_teacher_layout(T)

    def _fwd1(x):
        return teacher_to_cam_layer(T, x)

    def _fwd2(A, res):
        return cam_layer_to_logits(T, A, res)

    def _post(A, dA, gx):
        return candidates_from_union(union_from_grads(A, dA, gx, ucfg), specs)

    def _score(crops, pfull, cls):
        return score_crops(T, crops, pfull, cls, amp_dtype)

    if args.compile:
        import torch._dynamo as dynamo
        dynamo.config.cache_size_limit = 4096
        if hasattr(dynamo.config, "accumulated_cache_size_limit"):
            dynamo.config.accumulated_cache_size_limit = 1 << 16
        fwd1_d = torch.compile(_fwd1, dynamic=False)
        fwd2_d = torch.compile(_fwd2, dynamic=False)
        score_d = torch.compile(_score, dynamic=False)
    else:
        fwd1_d, fwd2_d, score_d = _fwd1, _fwd2, _score

    if args.precompile and dev.type != "cpu":
        t0 = time.time()
        for i, (H_, W_) in enumerate(sorted(head, key=lambda k: -head[k])):
            t1 = time.time()
            x = torch.zeros(args.union_batch, 3, H_, W_, device=dev).requires_grad_(True)
            A, res = fwd1_d(x)
            lg = fwd2_d(A, res)
            gx, dA = torch.autograd.grad(lg, [x, A], grad_outputs=torch.ones_like(lg))
            float(gx.sum().cpu()) + float(dA.sum().cpu())
            print(f"[precompile] shape {i + 1}/{len(head)} {(H_, W_)}: {time.time() - t1:.1f}s", flush=True)
        for sp in specs:
            t1 = time.time()
            z = torch.zeros(args.score_batch, 3, sp.crop, sp.crop, dtype=torch.uint8, device=dev)
            pf = torch.full((args.score_batch, 1000), 1e-3, device=dev)
            float(score_d(z, pf, torch.zeros(args.score_batch, dtype=torch.long, device=dev)).sum().cpu())
            print(f"[precompile] score crop {sp.crop}: {time.time() - t1:.1f}s", flush=True)
        print(f"[precompile] done in {time.time() - t0:.0f}s", flush=True)
    seen_shapes: set = set(head) if args.precompile else set()

    total_chunks = ref_pc.n_chunks(lead)
    cs = int(lead.tgsc_cfg.cache_shard_size)
    todo = ref_pc.claim_iter(targets, total_chunks) if args.claim else \
        [c for c in range(total_chunks) if not all(ref_pc.chunk_marker(t, c).exists() for t in targets)]
    if args.max_chunks is not None and not args.claim:
        todo = todo[: args.max_chunks]
    t_start, done_imgs = time.time(), 0

    for chunk in todo:
        start, stop = chunk * cs, min((chunk + 1) * cs, total)
        tc = time.time()
        st = Chunk(specs)
        tim = {"union": 0.0, "score": 0.0, "tail_wait": 0.0, "decode": 0.0, "first_calls": 0.0, "host": 0.0}
        nh = nt = 0
        buckets: dict[tuple[int, int], list[tuple[int, int, np.ndarray]]] = {}
        images: dict[int, np.ndarray] = {}
        pending: list = []

        def run_score(si: int, force: bool = False) -> None:
            q = st.queues[si]
            while len(q) >= args.score_batch or (force and q):
                take, st.queues[si] = q[: args.score_batch], q[args.score_batch:]
                q = st.queues[si]
                n = len(take)
                pad = take + [take[-1]] * (args.score_batch - n)
                crops = torch.from_numpy(np.stack([e[2] for e in pad]))
                pf = torch.from_numpy(np.stack([st.pfull[e[0]] for e in pad]))
                cl = torch.tensor([st.label[e[0]] for e in pad], dtype=torch.long)
                t0 = time.time()
                s = score_d(crops.to(dev), pf.to(dev), cl.to(dev)).float().cpu().numpy()
                tim["score"] += time.time() - t0
                for (sid, kidx, _), v in zip(take, s[:n]):
                    st.scores[sid][si][kidx] = v

        def add_candidates(sid: int, u8: np.ndarray, logits: np.ndarray, boxes: list[np.ndarray]) -> None:
            t_add = time.time()
            _add_candidates(sid, u8, logits, boxes)
            tim["host"] += time.time() - t_add  # includes the score batches it triggers

        def _add_candidates(sid: int, u8: np.ndarray, logits: np.ndarray, boxes: list[np.ndarray]) -> None:
            st.pfull[sid] = torch.softmax(torch.from_numpy(np.asarray(logits, dtype=np.float32)), dim=-1).numpy()
            st.boxes[sid] = [np.asarray(b, dtype=np.int64) for b in boxes]
            st.scores[sid] = [np.full(len(b), -1.0, dtype=np.float32) for b in boxes]
            for si, (sp, b) in enumerate(zip(specs, st.boxes[sid])):
                for kidx, (y0, x0, y1, x1) in enumerate(b.tolist()):
                    crop = u8[:, y0:y1, x0:x1]
                    assert crop.shape[1:] == (sp.crop, sp.crop), (sid, sp, crop.shape)
                    st.queues[si].append((sid, kidx, np.ascontiguousarray(crop)))
                run_score(si)

        def run_head(key: tuple[int, int], items: list[tuple[int, int, np.ndarray]]) -> None:
            n = len(items)
            pad = items + [items[-1]] * (args.union_batch - n)
            xb = torch.from_numpy(np.stack([e[2] for e in pad])).to(dev).float().div(255.0)
            yb = torch.tensor([e[1] for e in pad], dtype=torch.long).to(dev)
            t0 = time.time()
            x = xb.requires_grad_(True)
            A, res = fwd1_d(x)
            logits = fwd2_d(A, res)
            target = yb if ucfg.use_gt else logits.detach().argmax(dim=1)
            oh = torch.zeros_like(logits).scatter_(1, target.view(-1, 1), 1.0)
            gx, dA = torch.autograd.grad(logits, [x, A], grad_outputs=oh)
            lg = logits.detach().float().cpu().numpy()[:n]
            job = ([e[0] for e in items], lg, A.detach().float().cpu().numpy()[:n], dA.float().cpu().numpy()[:n],
                   gx.float().cpu().numpy()[:n])
            dt = time.time() - t0
            tim["union"] += dt
            if key not in seen_shapes:
                seen_shapes.add(key)
                tim["first_calls"] += dt
                print(f"[hybrid] first call of shape {key}: {dt:.1f}s (compile + run)", flush=True)
            for e in items:
                images[e[0]] = e[2]
            if pool is None:
                _W.update(specs=specs, ucfg=ucfg)
                for sid, lgi, bx in _cpu_post(job):
                    add_candidates(sid, images.pop(sid), lgi, bx)
            else:
                pending.append(pool.apply_async(_cpu_post, (job,)))
                if len(pending) > max_pending:
                    pending[0].wait()

        def drain(block: bool) -> None:
            nonlocal pending
            keep = []
            for r in pending:
                if block or r.ready():
                    t0 = time.time()
                    out = r.get()
                    tim["tail_wait"] += time.time() - t0
                    for sid, logits, boxes in (out if isinstance(out, list) else [out]):
                        add_candidates(sid, images.pop(sid), logits, boxes)
                else:
                    keep.append(r)
            pending = keep

        def to_tail(sid: int, label: int, u8: np.ndarray) -> None:
            nonlocal nt
            if pool is None:
                logits, boxes = union_pass(fwd1_d, fwd2_d, _post, torch.from_numpy(u8).unsqueeze(0).to(dev).float().div(255.0),
                                           torch.tensor([label]).to(dev), ucfg.use_gt)
                add_candidates(sid, u8, logits[0].float().cpu().numpy(), [b[0].cpu().numpy() for b in boxes])
            else:
                images[sid] = u8
                pending.append(pool.apply_async(_cpu_union, ((sid, label, u8),)))
                if len(pending) > max_pending:
                    pending[0].wait()
                drain(False)
            nt += 1

        ds = ImageNetPrecomputeDataset(split=lead.train_split, analysis_resize_short=lead.data_cfg.analysis_resize_short,
                                       start=start, stop=stop, read_batch_size=lead.tgsc_cfg.precompute_batch_size)
        kw: dict[str, Any] = dict(batch_size=None, num_workers=args.decode_workers)
        if args.decode_workers > 0:
            kw.update(prefetch_factor=4, multiprocessing_context="forkserver")
        n = 0
        tw = time.time()
        for sample in DataLoader(ds, **kw):
            tim["decode"] += time.time() - tw
            n += 1
            sid, label = int(sample["sample_id"]), int(sample["label"])
            st.label[sid] = label
            x = sample["image_tensor"]
            u8 = (x * 255.0).round_().to(torch.uint8).numpy()  # exact: the image is to_tensor(uint8) = u8 / 255
            key = (int(x.shape[-2]), int(x.shape[-1]))
            if key in head:
                b = buckets.setdefault(key, [])
                b.append((sid, label, u8))
                nh += 1
                if len(b) >= args.union_batch:
                    run_head(key, buckets.pop(key))
            else:
                to_tail(sid, label, u8)
            if pending:
                drain(False)
            if args.log_every and n % args.log_every == 0:
                el = time.time() - tc
                print(f"[hybrid] chunk {chunk}: {n} decoded in {el:.0f}s ({n / el:.1f} img/s) | head {nh} tail {nt} | "
                      f"done {len(st.boxes)} | pending {len(pending)} | union {tim['union']:.0f}s (first calls "
                      f"{tim['first_calls']:.0f}s, {len(seen_shapes)} shapes) score {tim['score']:.0f}s decode-wait {tim['decode']:.0f}s "
                      f"host(crops+score) {tim['host']:.0f}s tail-wait {tim['tail_wait']:.0f}s", flush=True)
            tw = time.time()
        for key, items in list(buckets.items()):
            if len(items) >= args.flush_min:
                run_head(key, items)
            else:
                for sid, label, u8 in items:
                    nh -= 1
                    to_tail(sid, label, u8)
        drain(True)
        for si in range(len(specs)):
            run_score(si, force=True)
        if n != stop - start:
            print(f"[hybrid] WARNING chunk {chunk}: {n} of {stop - start} samples", flush=True)

        for ti, t in enumerate(targets):
            cand_rows = {s: [] for s in t.stage_sizes}
            stage_rows = {s: [] for s in t.stage_sizes}
            for sid in sorted(st.boxes):
                for si, sp in enumerate(specs):
                    if sp.target != ti:
                        continue
                    b, sc = st.boxes[sid][si], st.scores[sid][si]
                    for kidx in range(len(b)):
                        y0, x0, y1, x1 = (int(v) for v in b[kidx])
                        cand_rows[sp.crop].append({"sample_id": sid, "y0": y0, "x0": x0, "y1": y1, "x1": x1,
                                                   "score": float(sc[kidx])})
                    stage_rows[sp.crop].extend(_select_stage_rows(
                        sample_id=sid, resolution=sp.crop, boxes_sorted=torch.from_numpy(b),
                        scores_sorted=torch.from_numpy(sc), top_k_crops=t.tgsc_cfg.top_k_crops,
                        low_k_crops=t.tgsc_cfg.low_k_crops))
            for s in t.stage_sizes:
                _write_table(t.candidate_dir(s) / f"part-c{chunk:05d}.parquet", _candidate_part_table(cand_rows[s]))
                _write_table(t.stage_tmp_dir(s) / f"part-c{chunk:05d}.parquet", _stage_part_table(stage_rows[s]))
            m = ref_pc.chunk_marker(t, chunk)
            m.parent.mkdir(parents=True, exist_ok=True)
            m.write_text(json.dumps({"start": start, "stop": stop, "processed": n, "secs": time.time() - tc,
                                     "pid": os.getpid(), "hybrid": True, "head": nh, "tail": nt, "device": str(dev)}))
        done_imgs += n
        el = time.time() - tc
        print(f"[hybrid] chunk {chunk} [{start},{stop}) {n} imgs in {el:.0f}s = {n / max(el, 1e-9):.1f} img/s | head {nh} "
              f"tail {nt} | union {tim['union']:.0f}s score {tim['score']:.0f}s tail-wait {tim['tail_wait']:.0f}s "
              f"decode-wait {tim['decode']:.0f}s | shapes compiled {len(seen_shapes)} | total {done_imgs} imgs "
              f"{done_imgs / (time.time() - t_start):.1f} img/s", flush=True)
    if pool is not None:
        pool.close()
        pool.join()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
