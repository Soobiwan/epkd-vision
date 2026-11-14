"""Single-accelerator EP-KD ImageNet trainer (TPU v6e via torch_tpu, or one CUDA GPU) reproducing the reference recipe.

Inherits the reference ``KDTrainer`` for config handling, LR schedules, KD loss, checkpoint files and the JSON run
log; the data order, micro-batch preparation and BN semantics are reproduced by loader.py / prep.py / ghost_bn.py,
the step by step.py.
    python -m epkd_colab.train --config CFG.json --device tpu [--init-student-from epoch_080.pt] ...
Auto-resumes from <checkpoint_dir>/latest.pt (reference file format + scaler state).
"""
from __future__ import annotations

import argparse
import json
import math
import os as _os_cf

# 11-14: default ON (E1 e5: ~2,480 -> ~2,950 rows/s). Dispatch the step before gathering the next one; EPKD_COMPUTE_FIRST=0 = old order
COMPUTE_FIRST = _os_cf.environ.get("EPKD_COMPUTE_FIRST", "1") == "1"
from pathlib import Path
import sys
import time
from typing import Any, Optional

import torch
import torch.nn.functional as F

import epkd_colab  # noqa: F401
from epkd_colab.config import build_teacher, load_cfg
from epkd_colab.ghost_bn import GhostHolder, convert_to_ghost_bn
from epkd_colab.loader import StepLoader
from epkd_colab.step import LogitsOnly, StepFn
from epkd_colab.val_cache import ValCache
from tgsc_kd.imagenet_parquet import ImageNetTGSCPreprocessor, StageIndexArrays, StageScoreCalibration
from tgsc_kd.modeling import build_model
from tgsc_kd.trainer import KDTrainer
from tgsc_kd.utils import forward_logits, set_seed


def get_device(name: str) -> torch.device:
    if name == "tpu":
        import torch_tpu  # noqa: F401  (registers the 'tpu' device)

        return torch.device("tpu")
    if name.startswith("cuda"):
        d = torch.device(name)
        torch.cuda.set_device(d)
        return d
    return torch.device("cpu")


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "tpu":
        torch.accelerator.synchronize()


class ColabTrainer(KDTrainer):
    def __init__(self, *args: Any, opts: argparse.Namespace, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.opts = opts
        self.dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[opts.dtype]
        self.amp = opts.dtype != "fp32"
        self.channels_last = self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=True) if (self.dtype == torch.float16 and self.device.type == "cuda") else None
        self.holder = GhostHolder()
        self._stage_cache: dict[int, tuple[Path, StageScoreCalibration, list[int]]] = {}
        self.logger.log("colab_config", device=str(self.device), dtype=opts.dtype, grad_scaler=self.scaler is not None,
                        compile_scope=opts.compile_scope, compile_mode=opts.compile_mode, decode_procs_per_worker=opts.decode_procs,
                        micro_batch=int(self.data_cfg.train_gpu_batch_size), step_rows=int(self.data_cfg.train_batch_size),
                        num_workers_logical=6, per_worker_buffer=512, torch=torch.__version__)

    # ------------------------------------------------------------------ models / optimizer
    def _prepare_models(self) -> None:
        mf = torch.channels_last if self.channels_last else torch.contiguous_format
        self.teacher.to(self.device, memory_format=mf).eval()
        for p in self.teacher.parameters():
            p.requires_grad_(False)
        self.student.to(self.device, memory_format=mf).train()
        self._mean_dev = self.mean.to(self.device)
        self._std_dev = self.std.to(self.device)
        n_bn = convert_to_ghost_bn(self.student, self.holder)
        fused = self.device.type == "cuda"
        # TPU: foreach SGD (one multi-tensor update instead of 62 per-parameter ones; bit-identical in the 11-12 probe)
        self.optimizer = torch.optim.SGD(list(self.student.parameters()), lr=self.opt_cfg.base_lr, momentum=self.opt_cfg.momentum,
                                         weight_decay=self.opt_cfg.weight_decay, fused=fused,
                                         foreach=True if self.device.type == "tpu" else None)
        if fused:  # fused kernel leaves the buffer uninitialised if the scaler skips step 1
            for p in self.student.parameters():
                self.optimizer.state[p]["momentum_buffer"] = torch.zeros_like(p)
        self.logger.log("models_prepared", ghost_bn_layers=n_bn, student_params=sum(p.numel() for p in self.student.parameters()))

    def _load_optimizer_state(self, sd: dict) -> None:
        self.optimizer.load_state_dict(sd)
        fused = self.device.type == "cuda"
        for g in self.optimizer.param_groups:
            g["fused"], g["foreach"] = fused, (True if self.device.type == "tpu" else None)
        if fused:
            for p in self.student.parameters():
                buf = self.optimizer.state.get(p, {}).get("momentum_buffer")
                self.optimizer.state[p]["momentum_buffer"] = torch.zeros_like(p) if buf is None else torch.empty_like(p).copy_(buf)

    def _auto_resume(self) -> int:
        latest = self.checkpoint_dir / "latest.pt"
        if latest.exists():
            state = torch.load(latest, map_location="cpu", weights_only=False)
            self.student.load_state_dict(state["student"], strict=True)
            self._load_optimizer_state(state["optimizer"])
            if self.scaler is not None and state.get("scaler"):
                self.scaler.load_state_dict(state["scaler"])
            self.best_top1 = float(state.get("best_top1", -1.0))
            nxt = int(state["epoch"]) + 1
            self.logger.log("resume", path=str(latest), next_epoch=nxt, best_top1=self.best_top1)
            print(f"[resume] {latest} -> epoch {nxt} (best {self.best_top1:.2f})", flush=True)
            return nxt
        if self.opts.init_student_from:
            state = torch.load(self.opts.init_student_from, map_location="cpu", weights_only=False)
            sd = state.get("student") or state.get("model") or state
            self.student.load_state_dict(sd, strict=True)
            self.logger.log("init_student", path=str(self.opts.init_student_from), source_epoch=state.get("epoch"),
                            source_best_top1=state.get("best_top1"), fresh_optimizer=True)
            print(f"[init] student weights from {self.opts.init_student_from} (epoch {state.get('epoch')})", flush=True)
        return int(self.train_cfg.start_epoch or 1)

    def _save_checkpoint_files(self, *, epoch: int, resolution: int, val_top1: float, val_top5: float) -> None:
        if self.opts.no_save:
            return
        super()._save_checkpoint_files(epoch=epoch, resolution=resolution, val_top1=val_top1, val_top5=val_top5)
        extra = {"scaler": self.scaler.state_dict() if self.scaler is not None else None, "dtype": self.opts.dtype,
                 "pipeline": "imagenet_colab", "device": str(self.device)}
        for name in (f"epoch_{epoch:03d}.pt", "latest.pt"):
            path = self.checkpoint_dir / name
            state = torch.load(path, map_location="cpu", weights_only=False)
            state.update(extra)
            tmp = path.with_name(f".{path.name}.tmp")
            torch.save(state, tmp)
            tmp.replace(path)
        (self.checkpoint_dir / "DONE_EPOCH").write_text(str(epoch))

    # ------------------------------------------------------------------ data
    def _preproc(self) -> ImageNetTGSCPreprocessor:
        if self.tgsc_preproc is None:
            self.tgsc_preproc = ImageNetTGSCPreprocessor(self.teacher, self.data_cfg, self.tgsc_cfg, teacher_name=self.teacher_name,
                                                         teacher_checkpoint=self.teacher_checkpoint)
        return self.tgsc_preproc  # type: ignore[return-value]

    def _stage(self, crop: int):
        if crop not in self._stage_cache:
            pre = self._preproc()
            if self.opts.stage_index_override:
                path = Path(self.opts.stage_index_override.format(res=crop))
            else:
                if not pre._cache_is_complete(pre._load_state()):
                    raise RuntimeError(f"TGSC cache {pre.cache_root} is not complete")
                path = pre.stage_index_path(crop)
            arrays = StageIndexArrays.from_parquet(path, pre.train_split.total_rows)
            calib = StageScoreCalibration.from_stage_index(arrays)
            pre._log_stage_summary(crop, arrays, calib)
            split = pre.train_split
            file_rows = [int(arrays.offsets[min(off + cnt, split.total_rows)] - arrays.offsets[off]) for off, cnt in zip(split.offsets, split.row_counts)]
            self._stage_cache[crop] = (path, calib, file_rows)
        return self._stage_cache[crop]

    def _loader(self, crop: int, epoch: int) -> StepLoader:
        path, _calib, file_rows = self._stage(crop)
        split = self._preproc().train_split
        o = self.opts
        anchor = int(self.data_cfg.eval_crop_size) if (float(self.tgsc_cfg.full_anchor) > 0 and crop < int(self.data_cfg.eval_crop_size)) else None
        return StepLoader(
            paths=split.paths, offsets=split.offsets, row_counts=split.row_counts, total_rows=split.total_rows, stage_index_path=str(path),
            file_rows=file_rows, seed=self.train_cfg.seed, epoch=epoch, crop=crop, anchor=anchor, short=int(self.data_cfg.analysis_resize_short),
            jitter=int(self.tgsc_cfg.box_jitter_px), flip_prob=float(self.tgsc_cfg.flip_prob), kd_weight_fn=self._kd_weights_cpu,
            num_workers=6, micro_batch=int(self.data_cfg.train_gpu_batch_size), step_rows=int(self.data_cfg.train_batch_size),
            buffer_rows=max(int(self.data_cfg.train_gpu_batch_size) * 32, 1024) // 2, decode_procs_per_worker=o.decode_procs,
            ring_slots=o.ring_slots, pin=self.device.type == "cuda", max_steps=o.max_steps,
            n_step_buffers=max(4 if o.gather_thread else 3, o.prefetch + 2), gather_thread=bool(o.gather_thread),
        )

    def _kd_weights_cpu(self, scores: torch.Tensor, qflags: torch.Tensor) -> torch.Tensor:
        """Reference DDPKDTrainer._score_to_kd_weights (sync-free form) on CPU tensors."""
        if getattr(self.tgsc_cfg, "kd_gate", "on") == "off":
            return torch.ones_like(scores)
        scores = scores.clamp(0.0, 1.0)
        lo, hi = float(self.tgsc_cfg.score_low), float(self.tgsc_cfg.score_high)
        if hi <= lo + 1e-6:
            w = 0.25 + 0.75 * scores
        else:
            w_low = 0.25 + 0.25 * (scores / max(lo, 1e-6))
            w_mid = 0.5 + 0.5 * ((scores - lo) / max(hi - lo, 1e-6))
            w_high = 1.0 + 0.15 * ((scores - hi) / max(1.0 - hi, 1e-6))
            w = torch.where(scores <= lo, w_low, torch.where(scores >= hi, w_high, w_mid))
        w = torch.where(qflags < 0, torch.minimum(w, torch.full_like(w, 0.75)), w)
        return w.clamp(min=0.25, max=1.15)

    # ------------------------------------------------------------------ eval
    @torch.no_grad()
    def _evaluate(self) -> tuple[float, float, float]:
        vc: ValCache = self._val_cache
        model = self.student
        model.eval()
        sums = torch.zeros(4, device=self.device, dtype=torch.float64 if self.device.type != "tpu" else torch.float32)
        mean, std = self._mean_dev, self._std_dev
        for x_u8, y in vc.batches(self.opts.val_batch, self.device):
            x = x_u8.float().div_(255.0).sub_(mean).div_(std)
            if self.channels_last:
                x = x.contiguous(memory_format=torch.channels_last)
            logits = self._eval_fwd(x).float()
            loss = F.cross_entropy(logits, y)
            top5 = logits.topk(5, dim=1).indices.eq(y.view(-1, 1))
            n = y.size(0)
            sums[0] += top5[:, :1].sum()
            sums[1] += top5.sum()
            sums[2] += loss.to(sums.dtype) * n
            sums[3] += n
        model.train()
        top1, top5, loss_n, n = sums.cpu().tolist()
        n = max(1.0, n)
        return 100.0 * top1 / n, 100.0 * top5 / n, loss_n / n

    # ------------------------------------------------------------------ main loop
    def train(self) -> None:
        o = self.opts
        set_seed(self.train_cfg.seed)
        total_epochs = len(self.per_epoch_res)
        if total_epochs != self.train_cfg.epochs:
            raise ValueError(f"epochs ({self.train_cfg.epochs}) must equal sum of stage epochs ({total_epochs})")
        self._prepare_models()
        start_epoch = self._auto_resume()
        if start_epoch > total_epochs:
            print(f"[done] all {total_epochs} epochs already trained", flush=True)
            return
        self._val_cache = None if o.skip_eval else ValCache(o.val_cache, self._preproc().validation_split, self.data_cfg, procs=o.val_procs)
        eval_net = self.student
        self._eval_fwd = (lambda x: forward_logits(eval_net, x))
        if o.compile_scope != "none" and o.compile_eval:
            self._eval_fwd = torch.compile(LogitsOnly(self.student), dynamic=False)
        mb = int(self.data_cfg.train_gpu_batch_size)
        step_rows = int(self.data_cfg.train_batch_size)
        G = step_rows // mb
        fn = StepFn(student=self.student, teacher=self.teacher, holder=self.holder, optimizer=self.optimizer, device=self.device,
                    mean=self.mean, std=self.std, temperature=self.kd_cfg.temperature, ce_weight=self.kd_cfg.ce_weight,
                    kd_weight=self.kd_cfg.kd_weight, full_anchor=self.tgsc_cfg.full_anchor, groups=G, slot=mb, dtype=self.dtype,
                    scaler=self.scaler, channels_last=self.channels_last, compile_scope=o.compile_scope, compile_mode=o.compile_mode, amp=self.amp)
        self.logger.log("train_start", total_epochs=total_epochs, checkpoint_dir=str(self.checkpoint_dir), stage_plan=self.per_epoch_res,
                        resume_epoch=start_epoch)
        for epoch in range(start_epoch, total_epochs + 1):
            resolution = self.per_epoch_res[epoch - 1]
            lr = self._adjust_learning_rate(epoch)
            loader = self._loader(resolution, epoch)
            t_epoch = time.time()
            t_win, rows_win, step = time.time(), 0, 0
            t_step = 0.0
            loss_t = None
            trace = []
            it = iter(loader)
            if o.prefetch > 1 and o.async_h2d:
                # K uploads in flight ahead of the step being computed (fresh-tensor H2D modes only: to / i32).  Each
                # in-flight batch keeps its own loader step buffer (the loader has prefetch + 2 of them) until its
                # upload thread is joined, so no buffer is refilled while it is still being read.
                from collections import deque
                nslots, q, sub = o.prefetch + 1, deque(), [0]

                def submit() -> bool:
                    sb = next(it, None)
                    if sb is None:
                        return False
                    th_, d_ = fn.upload_async(sb, sub[0] % nslots)
                    sub[0] += 1
                    q.append((sb.rows, sb.anchor is not None, th_, d_, sb))
                    return True

                for _ in range(o.prefetch):
                    if not submit():
                        break
                while q:
                    t0 = time.time()
                    rows_cur, has_anchor, th, d_cur, sb_done = q.popleft()
                    th.join()
                    loader.release(sb_done)
                    submit()
                    loss_t = fn.compute(d_cur, has_anchor)
                    if o.loss_trace:
                        trace.append(float(loss_t))
                    elif o.sync_every and step % o.sync_every == 0:
                        float(loss_t)
                    t_step += time.time() - t0
                    step += 1
                    rows_win += rows_cur
                    if o.progress_every and step % o.progress_every == 0:
                        lv = float(loss_t)
                        if not math.isfinite(lv):
                            raise FloatingPointError(f"non-finite loss {lv} at epoch {epoch} step {step}")
                        dt = time.time() - t_win
                        print(f"[e{epoch} s{step}] {rows_win / dt:.0f} rows/s | loss {lv:.4f} | {(time.time() - t_epoch) / 60:.1f} min | "
                              f"host: gather {loader.t_gather:.1f}s wait {loader.t_wait:.1f}s step-call {t_step:.1f}s", flush=True)
                        t_win, rows_win = time.time(), 0
                        loader.t_gather = loader.t_wait = t_step = 0.0
                    if o.max_steps and step >= o.max_steps:
                        break
                for _, _, th_, _, _ in q:  # max_steps break: let pending uploads finish before the loader closes
                    th_.join()
                sb_cur = None
            else:
                sb_cur = next(it, None)
            d_cur = fn.upload(sb_cur, 0) if sb_cur is not None else None
            meta_cur = (sb_cur.rows, sb_cur.anchor is not None) if sb_cur is not None else None
            if sb_cur is not None:
                loader.release(sb_cur)  # uploaded synchronously above
            slot = 0
            while sb_cur is not None:
                t0 = time.time()
                if COMPUTE_FIRST:
                    # 11-13 (py-spy on the TPU trainer): fn.compute only DISPATCHES the step (returns in ~10 ms), but the
                    # next batch's upload cannot finish before that step finishes on the device, so the old order
                    # [gather N+1, start upload, dispatch N, join upload] left the device idle while the host gathered
                    # (~50 ms/step of a ~195 ms step).  Dispatch first, then gather the next step while the device
                    # computes, then upload it.  Same batches, same order, same ops: only host timing changes.
                    loss_t = fn.compute(d_cur, meta_cur[1])
                    if o.loss_trace:
                        trace.append(float(loss_t))
                    elif o.sync_every and step % o.sync_every == 0:
                        float(loss_t)
                    sb_next = next(it, None)
                    meta_next = (sb_next.rows, sb_next.anchor is not None) if sb_next is not None else None
                    if sb_next is not None:
                        d_next = fn.upload(sb_next, 1 - slot)  # fresh device tensors (i32/to) or the other slot (copy)
                        loader.release(sb_next)
                else:
                    sb_next = next(it, None)
                    meta_next = (sb_next.rows, sb_next.anchor is not None) if sb_next is not None else None
                    th = None
                    if o.async_h2d and sb_next is not None:  # upload the next step while this one computes
                        th, d_next = fn.upload_async(sb_next, 1 - slot)
                    loss_t = fn.compute(d_cur, meta_cur[1])
                    if o.loss_trace:
                        trace.append(float(loss_t))
                    elif o.sync_every and step % o.sync_every == 0:
                        float(loss_t)  # host sync (bounds host run-ahead; surfaces device errors early)
                    if th is not None:
                        th.join()
                    elif sb_next is not None:
                        d_next = fn.upload(sb_next, 1 - slot)
                    if sb_next is not None:
                        loader.release(sb_next)  # its upload has finished
                t_step += time.time() - t0
                step += 1
                rows_win += meta_cur[0]
                if o.progress_every and step % o.progress_every == 0:
                    lv = float(loss_t)
                    if not math.isfinite(lv):  # never train on (or checkpoint) a NaN model: restart from latest.pt
                        raise FloatingPointError(f"non-finite loss {lv} at epoch {epoch} step {step}")
                    dt = time.time() - t_win
                    print(f"[e{epoch} s{step}] {rows_win / dt:.0f} rows/s | loss {lv:.4f} | {(time.time() - t_epoch) / 60:.1f} min | "
                          f"host: gather {loader.t_gather:.1f}s wait {loader.t_wait:.1f}s step-call {t_step:.1f}s", flush=True)
                    t_win, rows_win = time.time(), 0
                    loader.t_gather = loader.t_wait = t_step = 0.0
                if o.max_steps and step >= o.max_steps:
                    break
                sb_cur, d_cur, meta_cur, slot = sb_next, (d_next if sb_next is not None else None), meta_next, 1 - slot
            sync(self.device)
            if loss_t is not None:
                float(loss_t)
            if o.loss_trace:
                Path(o.loss_trace).write_text(json.dumps({"epoch": epoch, "resolution": resolution, "step_loss": trace}))
            train_secs = time.time() - t_epoch
            loss, ce, kd, top1, top5, n = fn.pop_stats()
            if not math.isfinite(loss):
                raise FloatingPointError(f"non-finite epoch loss at epoch {epoch} -- not saving this epoch")
            loader.close()
            if n == 0:
                raise RuntimeError("zero rows in epoch")
            if o.skip_eval:
                val_top1 = val_top5 = val_loss = float("nan")
            else:
                val_top1, val_top5, val_loss = self._evaluate()
            eval_secs = time.time() - t_epoch - train_secs
            print(f"Epoch {epoch:3d}/{total_epochs} | res {resolution:3d} | loss {loss / n:.4f} (CE {ce / n:.4f}, KD {kd / n:.4f}) | "
                  f"train top1 {top1 / n:5.2f}% | val top1 {val_top1:5.2f}% top5 {val_top5:5.2f}% | lr {lr:.6f} | "
                  f"{n / train_secs:.0f} rows/s | train {train_secs / 60:.1f} min + eval {eval_secs / 60:.1f} min", flush=True)
            self.logger.log("epoch", epoch=epoch, total_epochs=total_epochs, resolution=resolution, train_loss=loss / n, train_ce=ce / n,
                            train_kd=kd / n, train_top1=top1 / n, train_top5=top5 / n, val_top1=val_top1, val_top5=val_top5, val_loss=val_loss,
                            lr=lr, rows=int(n), steps=step, train_secs=train_secs, eval_secs=eval_secs, rows_per_sec=n / train_secs,
                            amp_scale=float(self.scaler.get_scale()) if self.scaler is not None else None, loader_rows=loader.epoch_rows)
            if not o.skip_eval:
                self._save_checkpoint_files(epoch=epoch, resolution=resolution, val_top1=val_top1, val_top5=val_top5)
            if o.max_steps or (o.max_epochs and epoch - start_epoch + 1 >= o.max_epochs):
                break
        print(f"Finished. Best Val Top-1: {self.best_top1:.2f}%", flush=True)
        self.logger.finalize(status="completed", best_top1=self.best_top1, checkpoint_dir=str(self.checkpoint_dir))


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--device", default="tpu", help="tpu | cuda:0 | cpu")
    ap.add_argument("--init-student-from", default=None)
    ap.add_argument("--run-dir", default=None, help="checkpoint/log dir override")
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"], help="autocast dtype (fp16 -> GradScaler on CUDA)")
    ap.add_argument("--compile-scope", default="step", choices=["none", "model", "step", "fwdbwd"])
    ap.add_argument("--compile-mode", default=None)
    ap.add_argument("--compile-eval", type=int, default=1)
    ap.add_argument("--decode-procs", type=int, default=6, help="PIL decode processes per logical worker (6 workers)")
    ap.add_argument("--val-procs", type=int, default=16)
    ap.add_argument("--ring-slots", type=int, default=12)
    ap.add_argument("--val-cache", default=None)
    ap.add_argument("--val-batch", type=int, default=500)
    ap.add_argument("--progress-every", type=int, default=100)
    ap.add_argument("--sync-every", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--max-epochs", type=int, default=None)
    ap.add_argument("--skip-eval", action="store_true")
    ap.add_argument("--no-save", action="store_true")
    ap.add_argument("--stage-index-override", default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--start-epoch", type=int, default=None, help="bench: start at this epoch (selects the stage)")
    ap.add_argument("--gather-thread", type=int, default=0, help="assemble step buffers in a background thread (TPU input path)")
    ap.add_argument("--prefetch", type=int, default=1, help="uploads in flight ahead of the computing step (>1 needs --async-h2d 1)")
    ap.add_argument("--async-h2d", type=int, default=0, help="upload the next step's buffers in a thread during compute")
    ap.add_argument("--loss-trace", default=None, help="write per-step loss (host sync every step) to this JSON")
    return ap.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    cfg = load_cfg(args.config)
    if args.run_dir:
        cfg.train.checkpoint_dir = args.run_dir
        cfg.train.save_path = str(Path(args.run_dir) / "student_best.pth")
        cfg.train.log_dir = str(Path(args.run_dir) / "logs")
    if args.seed is not None:
        cfg.train.seed = args.seed
    if args.start_epoch is not None:
        cfg.train.start_epoch = args.start_epoch
    if args.val_cache is None:
        args.val_cache = str(Path(cfg.tgsc.cache_dir) / "val_cache_256_224.npy")
    device = get_device(args.device)
    teacher, ckpt = build_teacher(cfg)
    student = build_model(cfg.student_model, num_classes=cfg.num_classes)
    trainer = ColabTrainer(student=student, teacher=teacher, data_cfg=cfg.data, kd_cfg=cfg.kd, opt_cfg=cfg.optim, train_cfg=cfg.train,
                           tgsc_cfg=cfg.tgsc, device=device, teacher_name=cfg.teacher_model, teacher_checkpoint=ckpt, opts=args)
    print(json.dumps({"argv": sys.argv, "config": args.config, "log": str(getattr(trainer.logger, "path", None)),
                      "cache_root": str(trainer._preproc().cache_root)}), flush=True)
    trainer.train()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
