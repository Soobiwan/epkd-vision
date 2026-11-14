"""Device-agnostic batched EP-KD training step (TPU via torch_tpu, or CUDA), numerically equivalent to the
reference per-micro-batch loop (``DDPKDTrainer.train``, world 1):

reference, per 32-row micro-batch g of a 512-row step:
    x, y, .., (xa, ya) = _prepare_imagenet_batch(batch)     -> here: loader.py / prep.py on the CPU (uint8 crops)
    t = teacher(x); s = student(x)                         -> BN statistics over the 32 rows   (ghost BN group g)
    ta = teacher(xa); sa = student(xa)                     -> anchors: their own BN batch       (ghost BN group g)
    loss_g = KDLoss(s, t, y) + full_anchor * KDLoss(sa, ta, ya)        (means over rows)
    (loss_g * n_g / 512).backward()
then one SGD step.  Here all 16 micro-batches are forwarded at once (views: 512 rows in 16 slots of 32; anchors:
16 slots of 32 padded with zero rows and masked), and the loss is the identical sum of per-row terms:
    sum_views  (1/512) * (ce_w*CE_i + kd_w*kdw_i*KD_i)  +  full_anchor * sum_anchors (n_g/(512*A_g)) * (ce_w*CE_a + kd_w*KD_a)
The whole step (normalisation, teacher, student, loss, backward, SGD) is one function that can be torch.compiled
(TPU: required for speed; static shapes) or run eagerly.
"""
from __future__ import annotations

import os

from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from tgsc_kd.utils import forward_logits

from .ghost_bn import GhostHolder, GhostSpec
from .loader import StepBatch


def kd_terms_per_row(s_logits: torch.Tensor, t_logits: torch.Tensor, y: torch.Tensor, temperature: float
                     ) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-row CE and KD terms exactly as ``tgsc_kd.trainer.KDLoss`` computes them before the mean."""
    T = float(temperature)
    ce_each = F.cross_entropy(s_logits, y, reduction="none")
    log_s = F.log_softmax(s_logits / T, dim=1)
    p_t = F.softmax(t_logits.detach() / T, dim=1)
    log_t = torch.log(p_t + 1e-8)
    kd_each = (p_t * (log_t - log_s)).sum(dim=1) * (T * T)
    return ce_each, kd_each


class LogitsOnly(nn.Module):
    def __init__(self, net: nn.Module) -> None:
        super().__init__()
        self.net = net

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return forward_logits(self.net, x)


class StepFn:
    def __init__(self, *, student: nn.Module, teacher: nn.Module, holder: GhostHolder, optimizer: torch.optim.Optimizer,
                 device: torch.device, mean: torch.Tensor, std: torch.Tensor, temperature: float, ce_weight: float,
                 kd_weight: float, full_anchor: float, groups: int, slot: int, dtype: torch.dtype = torch.bfloat16,
                 scaler: Optional[torch.amp.GradScaler] = None, channels_last: bool = False, compile_scope: str = "none",
                 compile_mode: Optional[str] = None, amp: bool = True) -> None:
        self.student_raw, self.teacher_raw = student, teacher
        self.student: Any = LogitsOnly(student)
        self.teacher: Any = LogitsOnly(teacher)
        self.holder = holder
        self.opt = optimizer
        self.scaler = scaler
        self.device = device
        self.dtype = dtype
        self.amp = bool(amp)
        self.channels_last = bool(channels_last)
        self.mean = mean.to(device).view(1, 3, 1, 1)
        self.std = std.to(device).view(1, 3, 1, 1)
        self.T, self.ce_w, self.kd_w, self.lam = float(temperature), float(ce_weight), float(kd_weight), float(full_anchor)
        self.G, self.slot = int(groups), int(slot)
        self.N = self.G * self.slot
        self.spec_v = GhostSpec(self.G, self.slot, device)
        self.spec_a = GhostSpec(self.G, self.slot, device)
        self.dev: dict[tuple, dict[str, torch.Tensor]] = {}
        self._bt = None
        self._bt_bad: set[str] = set()
        # Host->device upload method (EPKD_H2D):
        #   copy : copy_ into a persistent device buffer per (crop, slot)          -- correct (11-11 A/B), slow
        #   to   : a FRESH device tensor per key per step (src.to(dev))            -- no buffer reuse -> no aliasing
        #   i32  : like `to`, image bytes packed as int32 (4x fewer elements), viewed back as uint8 on the device
        #   bt   : torch_tpu raw batch_transfer into the persistent buffer          -- CORRUPTS training data (11-11)
        self.h2d = os.environ.get("EPKD_H2D", "copy") if device.type == "tpu" else "copy"
        if self.h2d == "bt":
            try:
                from torch_tpu._internal.batch_transfer import batch_transfer_impl as _bt

                self._bt = _bt
            except Exception:  # noqa: BLE001
                self._bt = None
        self.stats = torch.zeros(6, device=device, dtype=torch.float32)
        self.compile_scope = compile_scope
        kw = {"mode": compile_mode} if compile_mode and compile_mode != "none" else {}
        if compile_scope == "model":
            self.student = torch.compile(self.student, dynamic=False, **kw)
            self.teacher = torch.compile(self.teacher, dynamic=False, **kw)
            self._run = self._step_impl
        elif compile_scope == "step":
            self._run = torch.compile(self._step_impl, dynamic=False, **kw)
        elif compile_scope == "fwdbwd":  # forward + loss + backward compiled, optimizer step eager
            self._fb = torch.compile(self._fwd_bwd, dynamic=False, **kw)

            def _run(d, has_anchor):
                loss = self._fb(d, has_anchor)
                self._opt_step()
                return loss

            self._run = _run
        else:
            self._run = self._step_impl

    # ------------------------------------------------------------------ buffers / upload
    _IMG_KEYS = ("xv", "xa")  # image buffers live FLAT (1-D uint8) on the device; reshaped inside the step

    def _dev(self, sb: StepBatch, slot: int = 0) -> dict[str, torch.Tensor]:
        key = (sb.crop, slot)
        if key not in self.dev:
            self.dev[key] = {k: (torch.zeros(v.numel(), dtype=v.dtype, device=self.device) if k in self._IMG_KEYS
                                 else torch.zeros_like(v, device=self.device)) for k, v in sb.tensors().items()}
        return self.dev[key]

    def _img(self, d: dict[str, torch.Tensor], k: str) -> torch.Tensor:
        """Flat uint8 device buffer -> (N, 3, c, c) (static shape; c from the numel)."""
        flat = d[k]
        c = int(round((flat.numel() // (3 * self.N)) ** 0.5))
        return flat.view(self.N, 3, c, c)

    def _h2d(self, d: dict[str, torch.Tensor], sb: StepBatch) -> None:
        # torch_tpu's raw-copy path assumes the tiled layout (second-minor dim padded to 8) and fails for crops % 8 != 0,
        # so images go over as flat 1-D buffers (no padding, no layout assumption); per-key fallback to copy_.
        for k, v in sb.tensors().items():
            src = v.view(-1) if k in self._IMG_KEYS else v
            if self.h2d in ("to", "i32"):
                if self.h2d == "i32" and k in self._IMG_KEYS and src.numel() % 4 == 0:
                    d[k] = src.view(torch.int32).to(self.device).view(torch.uint8)
                else:
                    d[k] = src.to(self.device)
                continue
            dst = d[k]
            if self._bt is not None and k not in self._bt_bad and src.numel() * src.element_size() >= (1 << 20):
                try:
                    self._bt.batch_transfer_h2d_sync([src], [dst])  # ~2x faster than copy_ on torch_tpu
                    continue
                except Exception as e:  # noqa: BLE001
                    self._bt_bad.add(k)  # remember: no repeated exceptions
                    print(f"[step] batch_transfer failed for '{k}' ({str(e)[:80]}); using copy_ for it from now on", flush=True)
            dst.copy_(src, non_blocking=self._bt is None)

    def upload(self, sb: StepBatch, slot: int = 0) -> dict[str, torch.Tensor]:
        d = self._dev(sb, slot)
        self._h2d(d, sb)
        return d

    def upload_async(self, sb: StepBatch, slot: int):
        """Start the host->device copy of ``sb`` into buffer ``slot`` in a thread (overlaps the current compute)."""
        import threading

        d = self._dev(sb, slot)
        th = threading.Thread(target=self._h2d, args=(d, sb), daemon=True)
        th.start()
        return th, d

    def _bind_specs(self, d: dict[str, torch.Tensor]) -> None:
        sv, sa = self.spec_v, self.spec_a
        sv.mask, sv.inv_n, sv.n_rows, sv.ema_w, sv.keep = d["mv"], d["inv_v"], d["n_v"], d["ema_v"], d["keep_v"]
        if "xa" in d:
            sa.mask, sa.inv_n, sa.n_rows, sa.ema_w, sa.keep = d["ma"], d["inv_a"], d["n_a"], d["ema_a"], d["keep_a"]

    # ------------------------------------------------------------------ the step
    def _norm(self, x_u8: torch.Tensor) -> torch.Tensor:
        x = (x_u8.float().div(255.0) - self.mean) / self.std
        if self.channels_last:
            x = x.contiguous(memory_format=torch.channels_last)
        return x

    def _teacher(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad(), torch.autocast(self.device.type, dtype=self.dtype, enabled=True):
            t = self.teacher(x)
        return t.float().detach()

    def _fwd_bwd(self, d: dict[str, torch.Tensor], has_anchor: bool) -> torch.Tensor:
        xv = self._norm(self._img(d, "xv"))
        t_v = self._teacher(xv)
        self.holder.spec = self.spec_v
        with torch.autocast(self.device.type, dtype=self.dtype, enabled=self.amp):
            s_v = self.student(xv)
        s_v = s_v.float()
        ce_v, kd_v = kd_terms_per_row(s_v, t_v, d["yv"], self.T)
        row_loss = self.ce_w * ce_v + self.kd_w * kd_v * d["kdw"]
        loss = (d["wv"] * row_loss).sum()
        if has_anchor:
            xa = self._norm(self._img(d, "xa"))
            t_a = self._teacher(xa)
            self.holder.spec = self.spec_a
            with torch.autocast(self.device.type, dtype=self.dtype, enabled=self.amp):
                s_a = self.student(xa)
            ce_a, kd_a = kd_terms_per_row(s_a.float(), t_a, d["ya"], self.T)
            loss = loss + self.lam * (d["wa"] * (self.ce_w * ce_a + self.kd_w * kd_a)).sum()
        self.opt.zero_grad(set_to_none=True)
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
        with torch.no_grad():  # reference epoch statistics (views only)
            mv = d["mv"]
            top5 = s_v.detach().topk(5, dim=1).indices.eq(d["yv"].view(-1, 1)).float() * mv.view(-1, 1)
            st = torch.stack([
                (row_loss.detach() * mv).sum(), (ce_v.detach() * mv).sum(), (kd_v.detach() * d["kdw"] * mv).sum(),
                top5[:, :1].sum() * 100.0, top5.sum() * 100.0, mv.sum(),
            ])
            self.stats.add_(st)
        return loss.detach()

    def _opt_step(self) -> None:
        if self.scaler is not None:
            self.scaler.step(self.opt)
            self.scaler.update()
        else:
            self.opt.step()

    def _step_impl(self, d: dict[str, torch.Tensor], has_anchor: bool) -> torch.Tensor:
        loss = self._fwd_bwd(d, has_anchor)
        self._opt_step()
        return loss

    def run(self, sb: StepBatch) -> torch.Tensor:
        d = self.upload(sb)
        self._bind_specs(d)
        return self._run(d, sb.anchor is not None)

    def compute(self, d: dict[str, torch.Tensor], has_anchor: bool) -> torch.Tensor:
        self._bind_specs(d)
        return self._run(d, has_anchor)

    def pop_stats(self) -> list[float]:
        v = self.stats.cpu().tolist()
        self.stats.zero_()
        return v
