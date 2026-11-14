from __future__ import annotations

from dataclasses import asdict, replace
import math
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.ops import roi_align
from tqdm.auto import tqdm

from .config import DataConfig, KDConfig, OptimConfig, TGSCConfig, TrainConfig
from .imagenet_parquet import ImageNetTGSCPreprocessor, make_imagenet_validation_loader
from .logging_utils import JsonRunLogger
from .tgsc import TGSCPreprocessor
from .utils import (
    accuracy,
    build_stage_plan,
    evaluate_topk,
    forward_logits,
    resolve_checkpoint_dir,
    set_seed,
)

ROOT = Path(__file__).resolve().parents[2]


class KDLoss(nn.Module):
    def __init__(self, temperature: float = 4.0, ce_weight: float = 0.1, kd_weight: float = 0.9):
        super().__init__()
        self.temperature = float(temperature)
        self.ce_weight = float(ce_weight)
        self.kd_weight = float(kd_weight)

    def forward(
        self,
        s_logits: torch.Tensor,
        t_logits: torch.Tensor,
        y: torch.Tensor,
        *,
        weights: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        temperature = self.temperature
        ce_each = F.cross_entropy(s_logits, y, reduction="none")
        log_s = F.log_softmax(s_logits / temperature, dim=1)
        p_t = F.softmax(t_logits.detach() / temperature, dim=1)
        log_t = torch.log(p_t + 1e-8)
        kd_each = (p_t * (log_t - log_s)).sum(dim=1) * (temperature * temperature)
        if weights is not None:
            kd_each = kd_each * weights
        ce = ce_each.mean()
        kd = kd_each.mean()
        loss = self.ce_weight * ce + self.kd_weight * kd
        return {
            "loss": loss,
            "ce": ce.detach(),
            "kd": kd.detach(),
        }


def make_eval_loader(data_cfg: DataConfig) -> DataLoader:
    if data_cfg.dataset_type.startswith("imagenet"):
        return make_imagenet_validation_loader(data_cfg)

    tfm = transforms.Compose([transforms.ToTensor(), transforms.Normalize(data_cfg.mean, data_cfg.std)])
    ds = datasets.CIFAR100(data_cfg.root, train=False, download=True, transform=tfm)
    return DataLoader(
        ds,
        batch_size=data_cfg.val_batch_size,
        shuffle=False,
        num_workers=data_cfg.num_workers,
        pin_memory=data_cfg.pin_memory,
    )


class KDTrainer:
    def __init__(
        self,
        student: nn.Module,
        teacher: nn.Module,
        data_cfg: DataConfig,
        kd_cfg: KDConfig,
        opt_cfg: OptimConfig,
        train_cfg: TrainConfig,
        tgsc_cfg: TGSCConfig,
        device: Optional[torch.device] = None,
        *,
        teacher_name: Optional[str] = None,
        teacher_checkpoint: Optional[str | Path] = None,
    ):
        self.student = student
        self.teacher = teacher
        self.teacher_name = teacher_name
        self.teacher_checkpoint = teacher_checkpoint
        self.data_cfg = data_cfg
        self.kd_cfg = kd_cfg
        self.opt_cfg = opt_cfg
        self.train_cfg = train_cfg
        self.tgsc_cfg = tgsc_cfg
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.is_imagenet = data_cfg.dataset_type.startswith("imagenet")

        self.kd_loss = KDLoss(kd_cfg.temperature, kd_cfg.ce_weight, kd_cfg.kd_weight)
        self.optimizer = optim.SGD(
            list(self.student.parameters()),
            lr=opt_cfg.base_lr,
            momentum=opt_cfg.momentum,
            weight_decay=opt_cfg.weight_decay,
        )
        self.scheduler = None
        self.lr_schedule = opt_cfg.lr_schedule.lower()
        supported_lr_schedules = {"auto", "cosine", "epoch_halving", "multistep"}
        if self.lr_schedule not in supported_lr_schedules:
            raise ValueError(
                f"Unsupported lr_schedule={opt_cfg.lr_schedule!r}; "
                f"expected one of {sorted(supported_lr_schedules)}."
            )
        if (
            not self.is_imagenet
            and not opt_cfg.lr_half_on_stage_change
            and self.lr_schedule in {"auto", "multistep"}
        ):
            self.scheduler = optim.lr_scheduler.MultiStepLR(
                self.optimizer,
                milestones=list(opt_cfg.milestones),
                gamma=opt_cfg.milestone_gamma,
            )

        self.eval_loader = make_eval_loader(data_cfg)
        self.per_epoch_res = build_stage_plan(data_cfg.stages)
        self.tgsc_preproc: Optional[object] = None
        self.best_top1 = -1.0
        self.mean = torch.tensor(data_cfg.mean, dtype=torch.float32).view(1, 3, 1, 1)
        self.std = torch.tensor(data_cfg.std, dtype=torch.float32).view(1, 3, 1, 1)
        self.checkpoint_dir = resolve_checkpoint_dir(train_cfg.save_path, train_cfg.checkpoint_dir)
        self.save_path = Path(train_cfg.save_path)

        run_name = train_cfg.run_name or self.checkpoint_dir.name
        self.logger = JsonRunLogger(
            log_dir=train_cfg.log_dir,
            run_name=run_name,
            metadata={
                "student_model": type(student).__name__,
                "teacher_model": type(teacher).__name__,
                "device": str(self.device),
                "data": asdict(data_cfg),
                "kd": asdict(kd_cfg),
                "optim": asdict(opt_cfg),
                "train": asdict(train_cfg),
                "tgsc": asdict(tgsc_cfg),
            },
        )

    def _prepare_models(self) -> None:
        self.teacher.to(self.device).eval()
        self.student.to(self.device).train()

    def _remaining_stage_data_cfg(self, *, start_epoch: int) -> DataConfig:
        future_resolutions = self.per_epoch_res[max(0, start_epoch - 1):]
        if not future_resolutions:
            future_resolutions = self.per_epoch_res[-1:]

        stage_sizes = tuple(dict.fromkeys(int(resolution) for resolution in future_resolutions))
        return replace(
            self.data_cfg,
            stages=tuple((resolution, 1) for resolution in stage_sizes),
        )

    def _build_preprocessor(self, *, start_epoch: int) -> object:
        if self.is_imagenet:
            # FIX: always use the full configured stage list.  Restricting it to the remaining stages
            # (previous behaviour, `_remaining_stage_data_cfg`) changed the cache key AND the dynamic
            # stride (it depends on the smallest stage) whenever a multi-stage run was resumed.
            # Identical for the single-stage 80->100 tails and for fresh multi-stage starts.
            del start_epoch
            return ImageNetTGSCPreprocessor(
                self.teacher,
                self.data_cfg,
                self.tgsc_cfg,
                teacher_name=self.teacher_name,
                teacher_checkpoint=self.teacher_checkpoint,
            )
        return TGSCPreprocessor(self.teacher, self.data_cfg, self.tgsc_cfg)

    def _get_train_loader(self, *, crop_size: int, epoch: int) -> DataLoader:
        if self.tgsc_preproc is None:
            self.tgsc_preproc = self._build_preprocessor(start_epoch=epoch)
        if self.is_imagenet:
            loader_batch_size = min(
                int(self.data_cfg.train_batch_size),
                int(self.data_cfg.train_gpu_batch_size or self.data_cfg.train_batch_size),
            )
            return self.tgsc_preproc.get_loader(  # type: ignore[return-value]
                crop_size=crop_size,
                batch_size=loader_batch_size,
                num_workers=self.data_cfg.num_workers,
                pin_memory=self.data_cfg.pin_memory,
                epoch=epoch,
                seed=self.train_cfg.seed,
            )
        return self.tgsc_preproc.get_loader(  # type: ignore[return-value]
            crop_size=crop_size,
            batch_size=self.data_cfg.train_batch_size,
            num_workers=self.data_cfg.num_workers,
            pin_memory=self.data_cfg.pin_memory,
        )

    def _set_learning_rate(self, lr: float) -> float:
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        return lr

    def _cosine_learning_rate(self, epoch: int) -> float:
        start_epoch = self.opt_cfg.cosine_start_epoch
        if start_epoch is None:
            start_epoch = self.train_cfg.start_epoch or 1
        end_epoch = self.opt_cfg.cosine_end_epoch or self.train_cfg.epochs

        if end_epoch <= start_epoch:
            progress = 1.0
        else:
            progress = (epoch - start_epoch) / float(end_epoch - start_epoch)
            progress = min(1.0, max(0.0, progress))

        max_lr = float(self.opt_cfg.base_lr)
        min_lr = float(self.opt_cfg.min_lr)
        return min_lr + 0.5 * (max_lr - min_lr) * (1.0 + math.cos(math.pi * progress))

    def _adjust_learning_rate(self, epoch: int) -> float:
        if self.lr_schedule == "cosine":
            return self._set_learning_rate(self._cosine_learning_rate(epoch))

        if self.is_imagenet:
            use_epoch_halving = (
                self.lr_schedule == "epoch_halving"
                or (self.lr_schedule == "auto" and self.opt_cfg.epoch_halving_interval)
            )
            if use_epoch_halving and self.opt_cfg.epoch_halving_interval and self.opt_cfg.epoch_halving_interval > 0:
                steps = epoch // self.opt_cfg.epoch_halving_interval
                lr = self.opt_cfg.base_lr * (0.5 ** steps)
            else:
                steps = sum(epoch > milestone for milestone in self.opt_cfg.milestones)
                lr = self.opt_cfg.base_lr * (self.opt_cfg.milestone_gamma ** steps)
            for group in self.optimizer.param_groups:
                group["lr"] = lr
            return lr

        if self.opt_cfg.lr_half_on_stage_change and epoch > 1:
            prev_res = self.per_epoch_res[epoch - 2]
            curr_res = self.per_epoch_res[epoch - 1]
            if curr_res != prev_res:
                for group in self.optimizer.param_groups:
                    group["lr"] *= 0.5
        return float(self.optimizer.param_groups[0]["lr"])

    def _score_to_kd_weights(
        self,
        scores: torch.Tensor,
        qflags: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        scores = scores.clamp(0.0, 1.0)
        score_low = float(self.tgsc_cfg.score_low)
        score_high = float(self.tgsc_cfg.score_high)

        if score_high <= score_low + 1e-6:
            weights = 0.25 + 0.75 * scores
        else:
            weights = torch.empty_like(scores)
            low_mask = scores <= score_low
            high_mask = scores >= score_high
            mid_mask = (~low_mask) & (~high_mask)

            weights[low_mask] = 0.25 + 0.25 * (scores[low_mask] / max(score_low, 1e-6))
            weights[mid_mask] = 0.5 + 0.5 * (
                (scores[mid_mask] - score_low) / max(score_high - score_low, 1e-6)
            )
            weights[high_mask] = 1.0 + 0.15 * (
                (scores[high_mask] - score_high) / max(1.0 - score_high, 1e-6)
            )

        if qflags is not None:
            low_quality = qflags < 0
            if low_quality.any():
                weights[low_quality] = torch.minimum(
                    weights[low_quality],
                    torch.full_like(weights[low_quality], 0.75),
                )

        return weights.clamp(min=0.25, max=1.15)

    def _prepare_imagenet_batch(
        self,
        batch: tuple[torch.Tensor, ...],
        *,
        crop_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        base_images, labels, scores, qflags, boxes, _sample_ids = batch
        base_images = base_images.to(self.device, non_blocking=True).float().div_(255.0)
        labels = labels.to(self.device, non_blocking=True)
        scores = scores.to(self.device, non_blocking=True)
        qflags = qflags.to(self.device, non_blocking=True)
        boxes = boxes.to(self.device, non_blocking=True).float()

        jitter_px = max(0, int(self.tgsc_cfg.box_jitter_px))
        if jitter_px > 0:
            base_images = F.pad(base_images, (jitter_px, jitter_px, jitter_px, jitter_px), mode="replicate")
            widths = boxes[:, 2] - boxes[:, 0]
            heights = boxes[:, 3] - boxes[:, 1]
            boxes[:, [0, 2]] += jitter_px
            boxes[:, [1, 3]] += jitter_px

            dx = torch.randint(
                -jitter_px,
                jitter_px + 1,
                (boxes.size(0),),
                device=self.device,
            ).to(boxes.dtype)
            dy = torch.randint(
                -jitter_px,
                jitter_px + 1,
                (boxes.size(0),),
                device=self.device,
            ).to(boxes.dtype)

            max_x0 = (base_images.size(-1) - widths).clamp_min(0.0)
            max_y0 = (base_images.size(-2) - heights).clamp_min(0.0)
            x0 = (boxes[:, 0] + dx).clamp_min(0.0)
            y0 = (boxes[:, 1] + dy).clamp_min(0.0)
            x0 = torch.minimum(x0, max_x0)
            y0 = torch.minimum(y0, max_y0)
            x1 = x0 + widths
            y1 = y0 + heights
            boxes = torch.stack([x0, y0, x1, y1], dim=1)

        batch_indices = torch.arange(base_images.size(0), device=self.device, dtype=torch.float32).unsqueeze(1)
        rois = torch.cat([batch_indices, boxes.float()], dim=1)
        crops = roi_align(
            base_images,
            rois,
            output_size=(crop_size, crop_size),
            spatial_scale=1.0,
            aligned=True,
        )
        if self.tgsc_cfg.flip_prob > 0.0:
            flip_mask = torch.rand(crops.size(0), device=self.device) < self.tgsc_cfg.flip_prob
            if flip_mask.any():
                crops[flip_mask] = torch.flip(crops[flip_mask], dims=[3])

        mean = self.mean.to(self.device)
        std = self.std.to(self.device)
        crops = (crops - mean) / std
        return crops, labels, scores, qflags

    def _iter_imagenet_microbatches(
        self,
        batch: tuple[torch.Tensor, ...],
        *,
        crop_size: int,
    ):
        base_images, labels, scores, qflags, boxes, sample_ids = batch
        total = int(labels.size(0))
        gpu_batch_size = max(1, int(self.data_cfg.train_gpu_batch_size or total))
        for start in range(0, total, gpu_batch_size):
            stop = min(start + gpu_batch_size, total)
            sl = slice(start, stop)
            yield self._prepare_imagenet_batch(
                (
                    base_images[sl],
                    labels[sl],
                    scores[sl],
                    qflags[sl],
                    boxes[sl],
                    sample_ids[sl],
                ),
                crop_size=crop_size,
            )

    def _save_checkpoint_files(self, *, epoch: int, resolution: int, val_top1: float, val_top5: float) -> None:
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.save_path.parent.mkdir(parents=True, exist_ok=True)

        student_state = (
            self.student.module.state_dict()
            if isinstance(self.student, nn.DataParallel)
            else self.student.state_dict()
        )

        is_best = val_top1 >= self.best_top1
        if is_best:
            self.best_top1 = val_top1

        state: dict[str, object] = {
            "epoch": epoch,
            "resolution": resolution,
            "student": student_state,
            "optimizer": self.optimizer.state_dict(),
            "best_top1": self.best_top1,
            "best_top5": val_top5 if is_best else None,
        }
        if self.scheduler is not None:
            state["scheduler"] = self.scheduler.state_dict()

        latest_path = self.checkpoint_dir / "latest.pt"
        latest_student_path = self.checkpoint_dir / "student_latest.pt"
        epoch_path = self.checkpoint_dir / f"epoch_{epoch:03d}.pt"
        epoch_student_path = self.checkpoint_dir / f"student_epoch_{epoch:03d}.pt"

        torch.save(state, latest_path)
        torch.save({"model": student_state}, latest_student_path)
        torch.save(state, epoch_path)
        torch.save({"model": student_state}, epoch_student_path)

        self.logger.log(
            "checkpoint",
            epoch=epoch,
            resolution=resolution,
            latest=str(latest_path),
            latest_student=str(latest_student_path),
            epoch_checkpoint=str(epoch_path),
            epoch_student=str(epoch_student_path),
        )

        if is_best:
            best_path = self.checkpoint_dir / "best.pt"
            best_student_path = self.checkpoint_dir / "student_best.pt"
            torch.save(state, best_path)
            torch.save({"model": student_state}, best_student_path)
            torch.save(student_state, self.save_path)
            self.logger.log(
                "best_checkpoint",
                epoch=epoch,
                resolution=resolution,
                best_top1=self.best_top1,
                best=str(best_path),
                best_student=str(best_student_path),
                save_path=str(self.save_path),
            )

    def _resolve_resume_path(self) -> Optional[Path]:
        resume_path = self.train_cfg.resume_path
        if not resume_path:
            return None
        if resume_path == "latest":
            return self.checkpoint_dir / "latest.pt"
        path = Path(resume_path)
        if not path.is_absolute():
            path = ROOT / path
        return path.resolve()

    def _load_resume_checkpoint(self) -> int:
        path = self._resolve_resume_path()
        if path is None:
            return self.train_cfg.start_epoch or 1
        if not path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {path}")

        state = torch.load(path, map_location="cpu", weights_only=False)
        student_state = state.get("student")
        if not isinstance(student_state, dict):
            raise RuntimeError(f"Resume checkpoint does not contain a student state: {path}")

        model = self.student.module if isinstance(self.student, nn.DataParallel) else self.student
        missing, unexpected = model.load_state_dict(student_state, strict=False)
        if missing or unexpected:
            raise RuntimeError(
                f"Resume checkpoint load mismatch for {path}: missing={len(missing)}, unexpected={len(unexpected)}"
            )
        self.optimizer.load_state_dict(state["optimizer"])
        if self.scheduler is not None and "scheduler" in state:
            self.scheduler.load_state_dict(state["scheduler"])

        self.best_top1 = float(state.get("best_top1", -1.0))
        start_epoch = int(state.get("epoch", 0)) + 1
        self.logger.log(
            "resume",
            path=str(path),
            next_epoch=start_epoch,
            configured_start_epoch=self.train_cfg.start_epoch,
            best_top1=self.best_top1,
        )
        return start_epoch

    def train(self) -> None:
        set_seed(self.train_cfg.seed)
        self._prepare_models()
        total_epochs = len(self.per_epoch_res)
        if total_epochs != self.train_cfg.epochs:
            raise ValueError(
                f"epochs ({self.train_cfg.epochs}) must equal sum of stage epochs ({total_epochs})"
            )

        start_epoch = self._load_resume_checkpoint()
        if start_epoch < 1 or start_epoch > total_epochs + 1:
            raise ValueError(f"start epoch ({start_epoch}) must be in [1, {total_epochs + 1}]")
        self.logger.log(
            "train_start",
            total_epochs=total_epochs,
            checkpoint_dir=str(self.checkpoint_dir),
            save_path=str(self.save_path),
            stage_plan=self.per_epoch_res,
            resume_epoch=start_epoch,
        )

        for epoch in range(start_epoch, total_epochs + 1):
            resolution = self.per_epoch_res[epoch - 1]
            lr = self._adjust_learning_rate(epoch)
            train_loader = self._get_train_loader(crop_size=resolution, epoch=epoch)
            accum_target = max(1, int(self.data_cfg.train_batch_size))
            accum_samples = 0
            has_pending_step = False

            running = {
                "loss": 0.0,
                "ce": 0.0,
                "kd": 0.0,
                "top1": 0.0,
                "top5": 0.0,
                "n": 0,
            }

            iterator = tqdm(train_loader, desc=f"Epoch {epoch}/{total_epochs} | res={resolution}", leave=False)
            self.optimizer.zero_grad(set_to_none=True)
            for batch in iterator:
                if self.is_imagenet:
                    for x, y, scores, qflags in self._iter_imagenet_microbatches(batch, crop_size=resolution):
                        with torch.no_grad():
                            t_logits = forward_logits(self.teacher, x)

                        s_logits = forward_logits(self.student, x)
                        kd_weights = self._score_to_kd_weights(scores, qflags)
                        kd_out = self.kd_loss(s_logits, t_logits, y, weights=kd_weights)
                        loss_scale = float(y.size(0)) / float(accum_target)
                        (kd_out["loss"] * loss_scale).backward()
                        accum_samples += int(y.size(0))
                        has_pending_step = True

                        batch_size = y.size(0)
                        acc1, acc5 = accuracy(s_logits.detach(), y, topk=(1, min(5, s_logits.size(1))))
                        running["loss"] += float(kd_out["loss"].item()) * batch_size
                        running["ce"] += float(kd_out["ce"].item()) * batch_size
                        running["kd"] += float(kd_out["kd"].item()) * batch_size
                        running["top1"] += float(acc1.item()) * batch_size
                        running["top5"] += float(acc5.item()) * batch_size
                        running["n"] += batch_size

                    if accum_samples >= accum_target:
                        self.optimizer.step()
                        self.optimizer.zero_grad(set_to_none=True)
                        accum_samples = 0
                        has_pending_step = False
                    continue
                else:
                    x, y, scores, _qflags, _idx = batch
                    x = x.to(self.device, non_blocking=True)
                    y = y.to(self.device, non_blocking=True)
                    scores = scores.to(self.device, non_blocking=True).float()
                    kd_weights = self._score_to_kd_weights(scores, _qflags.to(self.device, non_blocking=True))

                self.optimizer.zero_grad(set_to_none=True)
                with torch.no_grad():
                    t_logits = forward_logits(self.teacher, x)

                s_logits = forward_logits(self.student, x)
                kd_out = self.kd_loss(s_logits, t_logits, y, weights=kd_weights)

                kd_out["loss"].backward()
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)

                batch_size = y.size(0)
                acc1, acc5 = accuracy(s_logits.detach(), y, topk=(1, min(5, s_logits.size(1))))
                running["loss"] += float(kd_out["loss"].item()) * batch_size
                running["ce"] += float(kd_out["ce"].item()) * batch_size
                running["kd"] += float(kd_out["kd"].item()) * batch_size
                running["top1"] += float(acc1.item()) * batch_size
                running["top5"] += float(acc5.item()) * batch_size
                running["n"] += batch_size

            if self.is_imagenet and has_pending_step:
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)

            if self.scheduler is not None:
                self.scheduler.step()

            if running["n"] == 0:
                raise RuntimeError("Training loop processed zero samples for the current epoch.")

            train_loss = running["loss"] / running["n"]
            train_ce = running["ce"] / running["n"]
            train_kd = running["kd"] / running["n"]
            train_top1 = running["top1"] / running["n"]
            train_top5 = running["top5"] / running["n"]
            val_top1, val_top5, val_loss = evaluate_topk(self.student, self.eval_loader, self.device)

            print(
                f"Epoch {epoch:3d}/{total_epochs} | res {resolution:3d} | "
                f"loss {train_loss:.4f} (CE {train_ce:.4f}, KD {train_kd:.4f}) | "
                f"train top1 {train_top1:5.2f}% top5 {train_top5:5.2f}% | "
                f"val top1 {val_top1:5.2f}% top5 {val_top5:5.2f}% | "
                f"lr {lr:.5f}"
            )
            self.logger.log(
                "epoch",
                epoch=epoch,
                total_epochs=total_epochs,
                resolution=resolution,
                train_loss=train_loss,
                train_ce=train_ce,
                train_kd=train_kd,
                train_top1=train_top1,
                train_top5=train_top5,
                val_top1=val_top1,
                val_top5=val_top5,
                val_loss=val_loss,
                lr=lr,
            )

            self._save_checkpoint_files(epoch=epoch, resolution=resolution, val_top1=val_top1, val_top5=val_top5)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        print(f"Finished. Best Val Top-1: {self.best_top1:.2f}%")
        self.logger.finalize(
            status="completed",
            best_top1=self.best_top1,
            checkpoint_dir=str(self.checkpoint_dir),
            save_path=str(self.save_path),
        )
