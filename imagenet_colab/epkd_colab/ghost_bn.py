"""Ghost BatchNorm: several 32-row micro-batches in ONE student forward, with BatchNorm statistics computed per
micro-batch group exactly as if the groups had been forwarded one after the other (the reference loop in
``tgsc_kd.ddp_trainer.DDPKDTrainer.train`` forwards each 32-row micro-batch separately, and the anchors of a
micro-batch as their own BN batch).

Layout: the N rows of a forward are G contiguous slots of ``slot`` rows (N = G * slot); slot g holds the rows of
micro-batch g (padded with zero rows to ``slot`` when a micro-batch has fewer rows, e.g. anchors: one per distinct
source image, or the last partial micro-batch of an epoch).  A per-row mask excludes pad rows from the statistics
(and the caller gives them zero loss weight).

Running statistics: PyTorch's BatchNorm updates ``running = (1-m) running + m stat`` once per forward (unbiased
variance).  The reference performs 2G such updates per chunk of G micro-batches in the order v1, a1, v2, a2, ...
(views of micro-batch 1, anchors of micro-batch 1, ...).  Here the view forward and the anchor forward each fold
their own groups with closed-form weights ``w_g = m (1-m)^(number of later updates)`` (the anchor forward comes
second, so the view weights are pre-divided by the anchor forward's keep factor); the result equals the
sequential EMA exactly in exact arithmetic (fp32 rounding only).  See tests/test_ghost_bn.py.
"""
from __future__ import annotations

from typing import Optional, Sequence

import torch
import torch.nn as nn


class GhostSpec:
    """Device buffers describing the group layout of one forward; updated in place (static shapes)."""

    def __init__(self, groups: int, slot: int, device: torch.device) -> None:
        self.G = int(groups)
        self.slot = int(slot)
        self.N = self.G * self.slot
        self.mask = torch.ones(self.N, device=device, dtype=torch.float32)  # 1 = real row
        self.inv_n = torch.full((self.G,), 1.0 / self.slot, device=device, dtype=torch.float32)  # 1 / rows in group
        self.n_rows = torch.full((self.G,), float(self.slot), device=device, dtype=torch.float32)
        self.ema_w = torch.zeros(self.G, device=device, dtype=torch.float32)  # running-stat weight per group
        self.keep = torch.ones((), device=device, dtype=torch.float32)  # factor on the old running value
        self.update_running = True

    def set(self, counts: Sequence[int], ema_w: Sequence[float], keep: float, update_running: bool = True) -> None:
        assert len(counts) == self.G and len(ema_w) == self.G
        mask = torch.zeros(self.N, dtype=torch.float32)
        for g, c in enumerate(counts):
            assert 0 <= c <= self.slot
            mask[g * self.slot: g * self.slot + c] = 1.0
        n = torch.tensor([float(c) for c in counts])
        self.mask.copy_(mask, non_blocking=True)
        self.n_rows.copy_(n, non_blocking=True)
        self.inv_n.copy_(torch.where(n > 0, 1.0 / n.clamp_min(1.0), torch.zeros_like(n)), non_blocking=True)
        self.ema_w.copy_(torch.tensor([float(w) for w in ema_w]), non_blocking=True)
        self.keep.fill_(float(keep))
        self.update_running = bool(update_running)


def interleaved_ema_weights(view_counts: Sequence[int], anchor_counts: Sequence[int], momentum: float
                            ) -> tuple[list[float], float, list[float], float]:
    """Closed-form weights reproducing the sequential EMA for the update order v1, a1, v2, a2, ..., vG, aG
    (a group with 0 rows produces no update).  Returns (w_view, keep_view, w_anchor, keep_anchor) such that
    applying ``running = keep_view * running + sum_g w_view[g] * stat_v[g]`` and then
    ``running = keep_anchor * running + sum_g w_anchor[g] * stat_a[g]`` equals the sequential result."""
    m = float(momentum)
    G = len(view_counts)
    assert len(anchor_counts) == G
    fv = [1 if c > 0 else 0 for c in view_counts]
    fa = [1 if c > 0 else 0 for c in anchor_counts]
    n_v, n_a = sum(fv), sum(fa)
    keep_v, keep_a = (1.0 - m) ** n_v, (1.0 - m) ** n_a
    w_v, w_a = [0.0] * G, [0.0] * G
    for g in range(G):
        later_views = sum(fv[g + 1:])
        later_anchors_incl = sum(fa[g:])  # a_g comes after v_g
        later_anchors = sum(fa[g + 1:])
        if fv[g]:
            # true exponent: later_views + later_anchors_incl; the anchor forward multiplies by keep_a afterwards
            w_v[g] = m * (1.0 - m) ** (later_views + later_anchors_incl - n_a)
        if fa[g]:
            w_a[g] = m * (1.0 - m) ** (later_views + later_anchors)
    return w_v, keep_v, w_a, keep_a


def single_ema_weights(counts: Sequence[int], momentum: float) -> tuple[list[float], float]:
    """Sequential EMA weights when only one forward (no anchors) is folded: order g = 0..G-1."""
    m = float(momentum)
    f = [1 if c > 0 else 0 for c in counts]
    w = [m * (1.0 - m) ** sum(f[g + 1:]) if f[g] else 0.0 for g in range(len(f))]
    return w, (1.0 - m) ** sum(f)


class GhostHolder:
    """Shared, mutable pointer to the active GhostSpec (set by the step before each student forward)."""

    def __init__(self) -> None:
        self.spec: Optional[GhostSpec] = None


class GhostBatchNorm2d(nn.Module):
    """Drop-in replacement of an ``nn.BatchNorm2d`` (shares its parameters and buffers) computing per-group
    statistics over the groups described by ``holder.spec``.  In eval mode it is the standard BN."""

    def __init__(self, bn: nn.BatchNorm2d, holder: GhostHolder) -> None:
        super().__init__()
        self.num_features = bn.num_features
        self.eps = float(bn.eps)
        self.momentum = float(bn.momentum if bn.momentum is not None else 0.1)
        self.weight = bn.weight
        self.bias = bn.bias
        self.register_buffer("running_mean", bn.running_mean)
        self.register_buffer("running_var", bn.running_var)
        self.register_buffer("num_batches_tracked", bn.num_batches_tracked)
        self.holder = holder

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return nn.functional.batch_norm(
                x, self.running_mean, self.running_var, self.weight, self.bias, False, 0.0, self.eps
            )
        spec = self.holder.spec
        assert spec is not None, "GhostBatchNorm2d: no GhostSpec set"
        N, C, H, W = x.shape
        G, S = spec.G, spec.slot
        assert N == G * S, f"ghost BN expects {G}x{S} rows, got {N}"
        hw = float(H * W)
        if _FAST:
            return self._forward_fast(x, spec, N, C, G, S, hw)
        xf = x.float()
        mask = spec.mask
        inv = (spec.inv_n / hw)  # (G,)
        s1 = xf.sum(dim=(2, 3)) * mask[:, None]  # (N, C)
        mean = s1.view(G, S, C).sum(1) * inv[:, None]  # (G, C) biased mean per group
        mean_rows = mean.view(G, 1, C).expand(G, S, C).reshape(N, C)
        d = xf - mean_rows[:, :, None, None]
        s2 = (d * d).sum(dim=(2, 3)) * mask[:, None]
        var = s2.view(G, S, C).sum(1) * inv[:, None]  # (G, C) biased variance per group
        rstd = torch.rsqrt(var + self.eps)
        scale = (rstd * self.weight[None, :]).view(G, 1, C).expand(G, S, C).reshape(N, C)
        y = d * scale[:, :, None, None] + self.bias[None, :, None, None]
        # pad rows -> exactly 0.  A group with NO valid rows (last step of an epoch) has var 0 -> rstd = eps^-0.5 ~ 316,
        # which compounds over the BN layers to inf/NaN in its pad rows; their loss weight is 0 but 0 * NaN = NaN
        # poisons the loss and every gradient (11-11 F1: NaN from the end of epoch 1).  Valid rows are unchanged.
        y = torch.where(mask[:, None, None, None] > 0, y, torch.zeros((), dtype=y.dtype, device=y.device))
        if spec.update_running:
            with torch.no_grad():
                n = spec.n_rows * hw  # elements per group
                unbiased = var * (n / (n - 1.0).clamp_min(1.0))[:, None]
                w = spec.ema_w
                self.running_mean.mul_(spec.keep).add_((w[:, None] * mean).sum(0))
                self.running_var.mul_(spec.keep).add_((w[:, None] * unbiased).sum(0))
                self.num_batches_tracked.add_(int((w > 0).sum()) if not torch.is_tensor(w) else (w > 0).sum().to(self.num_batches_tracked.dtype))
        return y.to(x.dtype)

    def _forward_fast(self, x: torch.Tensor, spec: "GhostSpec", N: int, C: int, G: int, S: int, hw: float) -> torch.Tensor:
        """Same statistics and output as ``forward`` (two-pass mean/variance in fp32), written so that no full-size fp32
        tensor is materialised: x stays in its dtype (bf16 under autocast) and is up-cast inside the reductions, and
        the normalisation, affine, pad-row mask and down-cast are ONE per-row multiply-add  y = x * A + B  with
        A = mask * rstd * weight and B = mask * (bias - mean * rstd * weight)  (pad rows: A = B = 0 -> y = 0 exactly).
        Differs from ``forward`` only by fp32 rounding order.  Default since 11-13; EPKD_GHOST_IMPL=orig for the old path."""
        mask = spec.mask
        inv = (spec.inv_n / hw)  # (G,)
        s1 = x.sum(dim=(2, 3), dtype=torch.float32) * mask[:, None]  # (N, C)
        mean = s1.view(G, S, C).sum(1) * inv[:, None]  # (G, C)
        mean_rows = mean.view(G, 1, C).expand(G, S, C).reshape(N, C)
        s2 = (x.float() - mean_rows[:, :, None, None]).square().sum(dim=(2, 3)) * mask[:, None]  # fused: no fp32 copy kept
        var = s2.view(G, S, C).sum(1) * inv[:, None]
        rstd = torch.rsqrt(var + self.eps)
        sc = rstd * self.weight[None, :]  # (G, C)
        sh = self.bias[None, :] - mean * sc  # (G, C)
        A = (sc.view(G, 1, C).expand(G, S, C).reshape(N, C) * mask[:, None])[:, :, None, None]
        B = (sh.view(G, 1, C).expand(G, S, C).reshape(N, C) * mask[:, None])[:, :, None, None]
        y = (x.float() * A + B).to(x.dtype)
        if spec.update_running:
            with torch.no_grad():
                n = spec.n_rows * hw
                unbiased = var * (n / (n - 1.0).clamp_min(1.0))[:, None]
                w = spec.ema_w
                self.running_mean.mul_(spec.keep).add_((w[:, None] * mean).sum(0))
                self.running_var.mul_(spec.keep).add_((w[:, None] * unbiased).sum(0))
                self.num_batches_tracked.add_((w > 0).sum().to(self.num_batches_tracked.dtype))
        return y


import os as _os  # noqa: E402

# 11-13: default "fast" (4% faster TPU step on R50->MNv1, same maths; CPU check vs the original: rel 1e-6). "orig" restores the old path.
_FAST = _os.environ.get("EPKD_GHOST_IMPL", "fast") == "fast"


def convert_to_ghost_bn(model: nn.Module, holder: GhostHolder) -> int:
    """Replace every nn.BatchNorm2d in ``model`` (in place) by a GhostBatchNorm2d sharing its tensors."""
    n = 0
    for name, child in list(model.named_children()):
        if isinstance(child, nn.BatchNorm2d) and not isinstance(child, GhostBatchNorm2d):
            setattr(model, name, GhostBatchNorm2d(child, holder))
            n += 1
        else:
            n += convert_to_ghost_bn(child, holder)
    return n


def ghost_state_dict_to_plain(sd: dict) -> dict:
    """GhostBatchNorm2d keeps the original parameter/buffer names, so the state dict is already plain."""
    return sd
