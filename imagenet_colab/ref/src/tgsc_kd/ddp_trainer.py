"""Multi-GPU (DDP) ImageNet TGSC-KD trainer that reproduces the single-GPU ``KDTrainer`` recipe.

Equivalence with the original single-GPU loop (``KDTrainer.train``, imagenet branch):
  * optimizer step every ``train_batch_size`` (512) crop rows, built from micro-batches of
    ``train_gpu_batch_size`` (32) rows; BatchNorm therefore still sees 32-row batches.  The 16
    micro-batches of one step are spread over the ranks (one rank gets 6, two get 5, rotating every
    step); each micro-batch loss is scaled by rows/512 * world_size so that DDP's gradient *average*
    equals the original accumulated gradient sum(rows_i * grad(mean_loss_i)) / 512 exactly.
  * same loss (KDLoss + score/quality KD weights), LR schedules, box jitter, flip, roi_align crops,
    stage plan, SGD; same checkpoint format (latest.pt / epoch_XXX.pt / student_*.pt).
  * BN running stats are averaged over ranks before every evaluation / checkpoint (broadcast_buffers
    is off), i.e. an EMA over all micro-batches as in the single-GPU run.
Speed changes (no numeric effect unless stated): sharded data + validation over ranks and workers,
no per-micro-batch host syncs (.item(), boolean-mask indexing, .any()), pinned memory, prefetching,
channels_last, cudnn.benchmark, optional AMP (fp16 autocast + GradScaler: numeric change, opt-in).
"""
from __future__ import annotations

from contextlib import nullcontext
import math
import os
from pathlib import Path
import random
import time
from typing import Any, Iterator, Optional, Sequence

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, get_worker_info
from torchvision.ops import roi_align

from .imagenet_parquet import (
    ImageNetStageDataset,
    ImageNetTGSCPreprocessor,
    ImageNetValidationDataset,
    ParquetSplitInfo,
    StageIndexArrays,
    StageScoreCalibration,
    pad_stage_batch,
)
from .trainer import KDTrainer
from .utils import forward_logits, set_seed


def is_main() -> bool:
    return not dist.is_initialized() or dist.get_rank() == 0


def balanced_file_assignment(file_rows: Sequence[int], world: int, seed: int, epoch: int) -> list[list[int]]:
    """Per-epoch shuffled, row-balanced assignment of parquet files to ranks (greedy LPT)."""
    rng = random.Random(seed * 1_000_003 + epoch)
    order = list(range(len(file_rows)))
    rng.shuffle(order)
    order.sort(key=lambda i: -file_rows[i])  # stable: shuffled among equal sizes
    loads = [0] * world
    out: list[list[int]] = [[] for _ in range(world)]
    for i in order:
        r = min(range(world), key=lambda k: (loads[k], k))
        out[r].append(i)
        loads[r] += int(file_rows[i])
    for r in range(world):
        rng.shuffle(out[r])
    return out


class DistributedStageDataset(ImageNetStageDataset):
    """``ImageNetStageDataset`` whose parquet files are split over (rank, worker)."""

    def __init__(self, *, rank_files: list[int], per_worker_buffer: int, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.rank_files = list(rank_files)
        self.per_worker_buffer = int(per_worker_buffer)
        self.rank = dist.get_rank() if dist.is_initialized() else 0

    def __iter__(self):
        worker = get_worker_info()
        wid, nw = (0, 1) if worker is None else (worker.id, worker.num_workers)
        shard_indices = self.rank_files[wid::nw]
        global_wid = self.rank * 1000 + wid
        rng = random.Random(self.seed + self.epoch * 997 + global_wid)
        if len(shard_indices) > 1:
            rng.shuffle(shard_indices)
        raw_iter = self._iter_records(shard_indices)
        buffer: list = []
        for item in raw_iter:
            buffer.append(item)
            if len(buffer) >= self.per_worker_buffer:
                yield buffer.pop(rng.randrange(len(buffer)))
        while buffer:
            yield buffer.pop(rng.randrange(len(buffer)))


class ShardedValidationDataset(ImageNetValidationDataset):
    def __init__(self, split: ParquetSplitInfo, data_cfg: Any, rank: int, world: int) -> None:
        super().__init__(split, data_cfg)
        self.rank_files = list(range(rank, len(split.paths), world))

    def __iter__(self):
        worker = get_worker_info()
        wid, nw = (0, 1) if worker is None else (worker.id, worker.num_workers)
        for sample in self.split.iter_samples(
            columns=("image.bytes", "label"), shard_indices=self.rank_files[wid::nw], batch_size=16
        ):
            image_bytes, label = sample["image_bytes"], sample["label"]
            if image_bytes is None or label is None or int(label) < 0:
                continue
            from .imagenet_parquet import _image_bytes_to_eval_tensor

            yield _image_bytes_to_eval_tensor(
                image_bytes,
                resize_short=self.data_cfg.eval_resize_short,
                crop_size=self.data_cfg.eval_crop_size,
                mean=self.data_cfg.mean,
                std=self.data_cfg.std,
            ), int(label)


class _AnchorPair(nn.Module):
    """Runs the restricted views and the full-image anchor views through the student in ONE forward, so
    that DDP (static_graph) sees one forward per backward."""

    def __init__(self, student: nn.Module) -> None:
        super().__init__()
        self.student = student

    def forward(self, x: torch.Tensor, xa: Optional[torch.Tensor] = None):
        if xa is None:
            return forward_logits(self.student, x)
        return forward_logits(self.student, x), forward_logits(self.student, xa)


class _NullLogger:
    path = None

    def log(self, *_a: Any, **_k: Any) -> None:
        pass

    def finalize(self, *_a: Any, **_k: Any) -> None:
        pass


class DDPKDTrainer(KDTrainer):
    def __init__(
        self,
        *args: Any,
        rank: int,
        world: int,
        amp: bool = False,
        teacher_fp16: bool = False,
        channels_last: bool = True,
        num_workers: int = 6,
        val_workers: int = 5,
        pin_memory: bool = True,
        prefetch_factor: int = 4,
        bucket_cap_mb: int = 25,
        static_graph: bool = True,
        init_student_from: Optional[str] = None,
        max_steps: Optional[int] = None,
        skip_eval: bool = False,
        no_save: bool = False,
        stage_index_override: Optional[str] = None,
        progress_every: int = 200,
        loss_trace: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        self.rank = rank
        self.world = world
        if rank != 0:
            # only rank 0 writes the JSON run log
            import tgsc_kd.trainer as trainer_mod

            orig = trainer_mod.JsonRunLogger
            trainer_mod.JsonRunLogger = lambda **_k: _NullLogger()  # type: ignore[assignment]
            try:
                super().__init__(*args, **kwargs)
            finally:
                trainer_mod.JsonRunLogger = orig  # type: ignore[assignment]
        else:
            super().__init__(*args, **kwargs)
        self.amp = bool(amp)
        self.teacher_fp16 = bool(teacher_fp16)
        self.channels_last = bool(channels_last)
        self.num_workers = int(num_workers)
        self.val_workers = int(val_workers)
        self.pin_memory = bool(pin_memory)
        self.prefetch_factor = int(prefetch_factor)
        self.bucket_cap_mb = int(bucket_cap_mb)
        self.static_graph = bool(static_graph)
        self.init_student_from = init_student_from
        self.max_steps = max_steps
        self.skip_eval = skip_eval
        self.no_save = no_save
        self.stage_index_override = stage_index_override
        self.progress_every = int(progress_every)
        self.loss_trace = loss_trace
        self._trace: list[torch.Tensor] = []
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp)
        self.gloo = dist.new_group(backend="gloo") if dist.is_initialized() else None
        self.ddp_student: Optional[DDP] = None
        self._stage_cache: dict[int, tuple[StageIndexArrays, StageScoreCalibration, list[int]]] = {}
        self.logger.log(
            "ddp_config",
            world=world,
            amp=self.amp,
            teacher_fp16=self.teacher_fp16,
            channels_last=self.channels_last,
            num_workers_per_rank=self.num_workers,
            val_workers_per_rank=self.val_workers,
            pin_memory=self.pin_memory,
            prefetch_factor=self.prefetch_factor,
            bucket_cap_mb=self.bucket_cap_mb,
            static_graph=self.static_graph,
            init_student_from=init_student_from,
            micro_batch=int(self.data_cfg.train_gpu_batch_size),
            step_rows=int(self.data_cfg.train_batch_size),
            nccl_env={k: v for k, v in os.environ.items() if k.startswith("NCCL_")},
        )

    # ------------------------------------------------------------------ models
    def _prepare_models(self) -> None:
        mf = torch.channels_last if self.channels_last else torch.contiguous_format
        self.teacher.to(self.device, memory_format=mf).eval()
        for p in self.teacher.parameters():
            p.requires_grad_(False)
        self.student.to(self.device, memory_format=mf).train()
        self._mean_dev = self.mean.to(self.device)
        self._std_dev = self.std.to(self.device)

    def _wrap_ddp(self) -> None:
        self.ddp_student = DDP(
            _AnchorPair(self.student) if float(self.tgsc_cfg.full_anchor) > 0.0 else self.student,
            device_ids=[self.device.index],
            broadcast_buffers=False,
            bucket_cap_mb=self.bucket_cap_mb,
            gradient_as_bucket_view=True,
            static_graph=self.static_graph,
        )

    def _sync_bn_buffers(self) -> None:
        if not dist.is_initialized():
            return
        for name, buf in self.student.named_buffers():
            if buf.dtype.is_floating_point:
                dist.all_reduce(buf, op=dist.ReduceOp.SUM)
                buf.div_(self.world)

    # ------------------------------------------------------------------ data
    def _preproc(self) -> ImageNetTGSCPreprocessor:
        if self.tgsc_preproc is None:
            self.tgsc_preproc = ImageNetTGSCPreprocessor(
                self.teacher,
                self.data_cfg,
                self.tgsc_cfg,
                teacher_name=self.teacher_name,
                teacher_checkpoint=self.teacher_checkpoint,
            )
        return self.tgsc_preproc  # type: ignore[return-value]

    def _stage(self, crop_size: int) -> tuple[StageIndexArrays, StageScoreCalibration, list[int]]:
        if crop_size not in self._stage_cache:
            pre = self._preproc()
            if self.stage_index_override:
                path = Path(self.stage_index_override.format(res=crop_size))
            else:
                state = pre._load_state()
                if not pre._cache_is_complete(state):
                    raise RuntimeError(
                        f"TGSC cache {pre.cache_root} is not complete; run scripts/imagenet_fix/precompute_cache.py first"
                    )
                path = pre.stage_index_path(crop_size)
            arrays = StageIndexArrays.from_parquet(path, pre.train_split.total_rows)
            calib = StageScoreCalibration.from_stage_index(arrays)
            if is_main():
                pre._log_stage_summary(crop_size, arrays, calib)
            split = pre.train_split
            file_rows = [
                int(arrays.offsets[min(off + cnt, split.total_rows)] - arrays.offsets[off])
                for off, cnt in zip(split.offsets, split.row_counts)
            ]
            self._stage_cache[crop_size] = (arrays, calib, file_rows)
        return self._stage_cache[crop_size]

    def _train_loader(self, crop_size: int, epoch: int) -> DataLoader:
        arrays, calib, file_rows = self._stage(crop_size)
        pre = self._preproc()
        assignment = balanced_file_assignment(file_rows, self.world, self.train_cfg.seed, epoch)
        mb = int(self.data_cfg.train_gpu_batch_size)
        ds = DistributedStageDataset(
            rank_files=assignment[self.rank],
            # original: shuffle_buffer=max(32*bs, 1024) split over 2 workers -> 512 rows per worker
            per_worker_buffer=max(mb * 32, 1024) // 2,
            split=pre.train_split,
            stage_index=arrays,
            score_calibration=calib,
            crop_size=crop_size,
            analysis_resize_short=self.data_cfg.analysis_resize_short,
            shuffle_buffer=0,
            seed=self.train_cfg.seed,
            epoch=epoch,
        )
        kw: dict[str, Any] = dict(
            batch_size=mb, shuffle=False, num_workers=self.num_workers, pin_memory=self.pin_memory,
            collate_fn=pad_stage_batch,
        )
        if self.num_workers > 0:
            kw["prefetch_factor"] = self.prefetch_factor
        self._epoch_rows = sum(file_rows[i] for i in assignment[self.rank])
        return DataLoader(ds, **kw)

    def _val_loader(self) -> DataLoader:
        pre = self._preproc()
        ds = ShardedValidationDataset(pre.validation_split, self.data_cfg, self.rank, self.world)
        nw = min(self.val_workers, max(1, len(ds.rank_files)))
        kw: dict[str, Any] = dict(batch_size=self.data_cfg.val_batch_size, num_workers=nw, pin_memory=self.pin_memory)
        if nw > 0:
            kw["prefetch_factor"] = self.prefetch_factor
        return DataLoader(ds, **kw)

    # ------------------------------------------------------------------ step pieces (sync-free)
    def _score_to_kd_weights(self, scores: torch.Tensor, qflags: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Element-wise identical to KDTrainer._score_to_kd_weights but without boolean-mask indexing
        (which forces a host sync). kd_gate='off' gives unit weights (EP-KD for vision)."""
        if getattr(self.tgsc_cfg, "kd_gate", "on") == "off":
            return torch.ones_like(scores)
        scores = scores.clamp(0.0, 1.0)
        lo = float(self.tgsc_cfg.score_low)
        hi = float(self.tgsc_cfg.score_high)
        if hi <= lo + 1e-6:
            weights = 0.25 + 0.75 * scores
        else:
            w_low = 0.25 + 0.25 * (scores / max(lo, 1e-6))
            w_mid = 0.5 + 0.5 * ((scores - lo) / max(hi - lo, 1e-6))
            w_high = 1.0 + 0.15 * ((scores - hi) / max(1.0 - hi, 1e-6))
            weights = torch.where(scores <= lo, w_low, torch.where(scores >= hi, w_high, w_mid))
        if qflags is not None:
            weights = torch.where(qflags < 0, torch.minimum(weights, torch.full_like(weights, 0.75)), weights)
        return weights.clamp(min=0.25, max=1.15)

    def _prepare_imagenet_batch(self, batch, *, crop_size: int):
        base_images, labels, scores, qflags, boxes, sample_ids = batch
        dev = self.device
        base_images = base_images.to(dev, non_blocking=True).float().div_(255.0)
        labels = labels.to(dev, non_blocking=True)
        anchor = None
        if float(self.tgsc_cfg.full_anchor) > 0.0 and crop_size < int(self.data_cfg.eval_crop_size):
            anchor = self._full_anchor_views(base_images, labels, sample_ids.to(dev, non_blocking=True))
        scores = scores.to(dev, non_blocking=True)
        qflags = qflags.to(dev, non_blocking=True)
        boxes = boxes.to(dev, non_blocking=True).float()
        jitter_px = max(0, int(self.tgsc_cfg.box_jitter_px))
        if jitter_px > 0:
            base_images = F.pad(base_images, (jitter_px, jitter_px, jitter_px, jitter_px), mode="replicate")
            widths = boxes[:, 2] - boxes[:, 0]
            heights = boxes[:, 3] - boxes[:, 1]
            boxes[:, [0, 2]] += jitter_px
            boxes[:, [1, 3]] += jitter_px
            dx = torch.randint(-jitter_px, jitter_px + 1, (boxes.size(0),), device=dev).to(boxes.dtype)
            dy = torch.randint(-jitter_px, jitter_px + 1, (boxes.size(0),), device=dev).to(boxes.dtype)
            max_x0 = (base_images.size(-1) - widths).clamp_min(0.0)
            max_y0 = (base_images.size(-2) - heights).clamp_min(0.0)
            x0 = torch.minimum((boxes[:, 0] + dx).clamp_min(0.0), max_x0)
            y0 = torch.minimum((boxes[:, 1] + dy).clamp_min(0.0), max_y0)
            boxes = torch.stack([x0, y0, x0 + widths, y0 + heights], dim=1)
        batch_indices = torch.arange(base_images.size(0), device=dev, dtype=torch.float32).unsqueeze(1)
        rois = torch.cat([batch_indices, boxes.float()], dim=1)
        crops = roi_align(base_images, rois, output_size=(crop_size, crop_size), spatial_scale=1.0, aligned=True)
        if self.tgsc_cfg.flip_prob > 0.0:
            flip_mask = torch.rand(crops.size(0), device=dev) < self.tgsc_cfg.flip_prob
            crops = torch.where(flip_mask.view(-1, 1, 1, 1), torch.flip(crops, dims=[3]), crops)
        crops = (crops - self._mean_dev) / self._std_dev
        if self.channels_last:
            crops = crops.contiguous(memory_format=torch.channels_last)
        return crops, labels, scores, qflags, anchor

    def _full_anchor_views(self, base_images: torch.Tensor, labels: torch.Tensor, sample_ids: torch.Tensor):
        """Full-evidence views for the EP-KD full-input objective at restricted stages: one random
        eval_crop_size x eval_crop_size crop (+ flip) of each distinct source image in the micro-batch, the
        ImageNet counterpart of CIFAR pad-and-crop.  base_images are float [0,1], zero-padded bottom/right
        by pad_stage_batch; the valid extent is recovered from the non-zero pixels (an under-estimate only
        shrinks the offset range)."""
        dev = base_images.device
        size = int(self.data_cfg.eval_crop_size)
        _, inverse = torch.unique(sample_ids, return_inverse=True)
        first = torch.full((int(inverse.max()) + 1,), sample_ids.numel(), device=dev, dtype=torch.long)
        first = first.scatter_reduce(0, inverse, torch.arange(sample_ids.numel(), device=dev), reduce="amin")
        frac = float(self.tgsc_cfg.full_anchor_frac)
        if frac < 1.0:
            keep = max(1, int(round(frac * first.numel())))
            first = first[torch.randperm(first.numel(), device=dev)[:keep]]
        images = base_images[first]
        valid = images.amax(dim=1) > 0
        width = valid.any(dim=1).float().cumsum(-1).argmax(-1) + 1
        height = valid.any(dim=2).float().cumsum(-1).argmax(-1) + 1
        x0 = (torch.rand(first.numel(), device=dev) * (width - size + 1).clamp_min(1)).floor()
        y0 = (torch.rand(first.numel(), device=dev) * (height - size + 1).clamp_min(1)).floor()
        idx = torch.arange(first.numel(), device=dev, dtype=torch.float32)
        rois = torch.stack([idx, x0, y0, x0 + size, y0 + size], dim=1)
        views = roi_align(images, rois, output_size=(size, size), spatial_scale=1.0, aligned=True)
        if self.tgsc_cfg.flip_prob > 0.0:
            flip_mask = torch.rand(views.size(0), device=dev) < self.tgsc_cfg.flip_prob
            views = torch.where(flip_mask.view(-1, 1, 1, 1), torch.flip(views, dims=[3]), views)
        views = (views - self._mean_dev) / self._std_dev
        if self.channels_last:
            views = views.contiguous(memory_format=torch.channels_last)
        return views, labels[first]

    def _teacher_logits(self, x: torch.Tensor) -> torch.Tensor:
        with torch.inference_mode():
            with torch.autocast("cuda", dtype=torch.float16, enabled=self.teacher_fp16):
                t = forward_logits(self.teacher, x)
        return t.float().clone()  # leave inference mode

    # ------------------------------------------------------------------ eval / ckpt
    @torch.no_grad()
    def _evaluate(self) -> tuple[float, float, float]:
        self._sync_bn_buffers()
        model = self.student
        model.eval()
        sums = torch.zeros(4, device=self.device, dtype=torch.float64)  # top1, top5, loss*n, n
        for x, y in self._val_loader():
            x = x.to(self.device, non_blocking=True)
            if self.channels_last:
                x = x.contiguous(memory_format=torch.channels_last)
            y = y.to(self.device, non_blocking=True)
            logits = forward_logits(model, x).float()
            loss = F.cross_entropy(logits, y)
            top5 = logits.topk(5, dim=1).indices
            correct = top5.eq(y.view(-1, 1))
            n = y.size(0)
            sums[0] += correct[:, :1].sum()
            sums[1] += correct.sum()
            sums[2] += loss.double() * n
            sums[3] += n
        model.train()
        if dist.is_initialized():
            dist.all_reduce(sums)
        top1, top5, loss_n, n = sums.tolist()
        n = max(1.0, n)
        return 100.0 * top1 / n, 100.0 * top5 / n, loss_n / n

    def _save_checkpoint_files(self, *, epoch: int, resolution: int, val_top1: float, val_top5: float) -> None:
        if not is_main() or self.no_save:
            return
        super()._save_checkpoint_files(epoch=epoch, resolution=resolution, val_top1=val_top1, val_top5=val_top5)
        # add AMP scaler state to the resumable files (atomic rewrite of latest.pt last)
        extra = {"scaler": self.scaler.state_dict(), "ddp_world": self.world, "amp": self.amp}
        for name in (f"epoch_{epoch:03d}.pt", "latest.pt"):
            path = self.checkpoint_dir / name
            state = torch.load(path, map_location="cpu", weights_only=False)
            state.update(extra)
            tmp = path.with_name(f".{path.name}.tmp")
            torch.save(state, tmp)
            tmp.replace(path)

    def _auto_resume(self) -> int:
        latest = self.checkpoint_dir / "latest.pt"
        if latest.exists():
            state = torch.load(latest, map_location="cpu", weights_only=False)
            self.student.load_state_dict(state["student"], strict=True)
            self.optimizer.load_state_dict(state["optimizer"])
            if "scaler" in state and self.amp:
                self.scaler.load_state_dict(state["scaler"])
            self.best_top1 = float(state.get("best_top1", -1.0))
            nxt = int(state["epoch"]) + 1
            self.logger.log("resume", path=str(latest), next_epoch=nxt, best_top1=self.best_top1)
            return nxt
        if self.init_student_from:
            state = torch.load(self.init_student_from, map_location="cpu", weights_only=False)
            sd = state.get("student") or state.get("model") or state
            self.student.load_state_dict(sd, strict=True)
            self.logger.log(
                "init_student", path=str(self.init_student_from), source_epoch=state.get("epoch"),
                source_best_top1=state.get("best_top1"), fresh_optimizer=True,
            )
        return int(self.train_cfg.start_epoch or 1)

    # ------------------------------------------------------------------ main loop
    def train(self) -> None:
        set_seed(self.train_cfg.seed)
        torch.manual_seed(self.train_cfg.seed * 100 + self.rank)  # per-rank jitter/flip streams
        torch.cuda.manual_seed(self.train_cfg.seed * 100 + self.rank)
        total_epochs = len(self.per_epoch_res)
        if total_epochs != self.train_cfg.epochs:
            raise ValueError(f"epochs ({self.train_cfg.epochs}) must equal sum of stage epochs ({total_epochs})")
        self._prepare_models()
        start_epoch = self._auto_resume()  # every rank loads the same file; DDP also broadcasts rank 0
        self._wrap_ddp()
        assert self.ddp_student is not None
        mb = int(self.data_cfg.train_gpu_batch_size)
        step_rows = int(self.data_cfg.train_batch_size)
        if step_rows % mb:
            raise ValueError("train_batch_size must be a multiple of train_gpu_batch_size")
        mbs_per_step = step_rows // mb
        self.logger.log("train_start", total_epochs=total_epochs, checkpoint_dir=str(self.checkpoint_dir),
                        stage_plan=self.per_epoch_res, resume_epoch=start_epoch)

        for epoch in range(start_epoch, total_epochs + 1):
            resolution = self.per_epoch_res[epoch - 1]
            lr = self._adjust_learning_rate(epoch)
            loader = self._train_loader(resolution, epoch)
            it: Iterator = iter(loader)
            stats = torch.zeros(6, device=self.device, dtype=torch.float64)  # loss, ce, kd, top1, top5, n
            step = 0
            t_epoch = time.time()
            t_win, rows_win = time.time(), 0
            exhausted = False
            dropped_rows = 0
            self.optimizer.zero_grad(set_to_none=True)
            while True:
                if self.max_steps is not None and step >= self.max_steps:
                    break
                want = sum(1 for i in range(mbs_per_step) if (i + step) % self.world == self.rank)
                batches = []
                if not exhausted:
                    for _ in range(want):
                        try:
                            batches.append(next(it))
                        except StopIteration:
                            exhausted = True
                            break
                local_rows = sum(int(b[1].size(0)) for b in batches)
                # CPU (gloo) all-reduce: [rows this step, number of ranks without data] -- no GPU sync
                flag = torch.tensor([local_rows, 0 if batches else 1], dtype=torch.int64)
                if self.gloo is not None:
                    dist.all_reduce(flag, group=self.gloo)
                global_rows, empty_ranks = int(flag[0]), int(flag[1])
                if empty_ranks > 0:
                    # a rank ran out of data (row-balanced file split -> only the last few hundred rows of
                    # the other ranks are left); end the epoch on all ranks together
                    dropped_rows += global_rows
                    break
                for i, batch in enumerate(batches):
                    sync = i == len(batches) - 1
                    ctx = nullcontext() if sync else self.ddp_student.no_sync()
                    with ctx:
                        x, y, scores, qflags, anchor = self._prepare_imagenet_batch(batch, crop_size=resolution)
                        t_logits = self._teacher_logits(x)
                        if anchor is None:
                            with torch.autocast("cuda", dtype=torch.float16, enabled=self.amp):
                                s_logits = forward_logits(self.ddp_student, x)
                        else:
                            xa, ya = anchor
                            ta = self._teacher_logits(xa)
                            with torch.autocast("cuda", dtype=torch.float16, enabled=self.amp):
                                s_logits, sa = self.ddp_student(x, xa)
                        s_logits = s_logits.float()
                        kd_w = self._score_to_kd_weights(scores, qflags)
                        out = self.kd_loss(s_logits, t_logits, y, weights=kd_w)
                        loss = out["loss"]
                        if anchor is not None:
                            loss = loss + float(self.tgsc_cfg.full_anchor) * self.kd_loss(sa.float(), ta, ya, weights=None)["loss"]
                        n = y.size(0)
                        scale = float(n) / float(step_rows) * float(self.world)
                        self.scaler.scale(loss * scale).backward()
                    with torch.no_grad():
                        top5 = s_logits.topk(5, dim=1).indices.eq(y.view(-1, 1))
                        stats[0] += out["loss"].detach().double() * n
                        stats[1] += out["ce"].double() * n
                        stats[2] += out["kd"].double() * n
                        stats[3] += top5[:, :1].sum() * 100.0
                        stats[4] += top5.sum() * 100.0
                        stats[5] += n
                if self.loss_trace:
                    self._trace.append(stats[0].clone())  # cumulative loss*n (local); diffed at the end
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
                step += 1
                rows_win += global_rows
                if self.progress_every and step % self.progress_every == 0 and is_main():
                    torch.cuda.synchronize(self.device)
                    dt = time.time() - t_win
                    print(f"[e{epoch} s{step}] {rows_win / dt:.0f} rows/s (global) | "
                          f"{(time.time() - t_epoch) / 60:.1f} min | scale {self.scaler.get_scale() if self.amp else 1.0:g}",
                          flush=True)
                    t_win, rows_win = time.time(), 0
            del it, loader
            if dist.is_initialized():
                dist.all_reduce(stats)
            loss, ce, kd, top1, top5, n = stats.tolist()
            if n == 0:
                raise RuntimeError("Training loop processed zero samples for the current epoch.")
            train_secs = time.time() - t_epoch
            if self.skip_eval:
                self._sync_bn_buffers()
                val_top1 = val_top5 = val_loss = float("nan")
            else:
                val_top1, val_top5, val_loss = self._evaluate()
            if is_main():
                print(
                    f"Epoch {epoch:3d}/{total_epochs} | res {resolution:3d} | loss {loss / n:.4f} (CE {ce / n:.4f}, "
                    f"KD {kd / n:.4f}) | train top1 {top1 / n:5.2f}% | val top1 {val_top1:5.2f}% top5 {val_top5:5.2f}% "
                    f"| lr {lr:.6f} | {n / train_secs:.0f} rows/s | train {train_secs / 60:.1f} min "
                    f"+ eval {(time.time() - t_epoch - train_secs) / 60:.1f} min",
                    flush=True,
                )
            self.logger.log(
                "epoch", epoch=epoch, total_epochs=total_epochs, resolution=resolution, train_loss=loss / n,
                train_ce=ce / n, train_kd=kd / n, train_top1=top1 / n, train_top5=top5 / n, val_top1=val_top1,
                val_top5=val_top5, val_loss=val_loss, lr=lr, rows=int(n), steps=step, train_secs=train_secs,
                epoch_secs=time.time() - t_epoch, rows_per_sec=n / train_secs,
                amp_scale=float(self.scaler.get_scale()) if self.amp else None,
                dropped_rows_end_of_epoch=dropped_rows,
                rank0_assigned_rows=getattr(self, "_epoch_rows", None),
            )
            if not self.skip_eval:
                self._save_checkpoint_files(epoch=epoch, resolution=resolution, val_top1=val_top1, val_top5=val_top5)
            if dist.is_initialized():
                dist.barrier()
            torch.cuda.empty_cache()
            if self.max_steps is not None:
                break
        if self.loss_trace and is_main():
            import json as _json
            cum = torch.stack(self._trace).tolist()
            per_step = [cum[0]] + [b - a for a, b in zip(cum[:-1], cum[1:])]
            Path(self.loss_trace).write_text(_json.dumps({"cum_loss_x_rows_rank0": cum, "step_loss_x_rows_rank0": per_step}))
        if is_main():
            print(f"Finished. Best Val Top-1: {self.best_top1:.2f}%", flush=True)
        self.logger.finalize(status="completed", best_top1=self.best_top1, checkpoint_dir=str(self.checkpoint_dir))
