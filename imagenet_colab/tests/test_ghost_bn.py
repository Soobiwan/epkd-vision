"""Ghost-BN equivalence: G micro-batches (views) + their anchor groups in ONE forward each == the reference's
sequential per-micro-batch forwards (outputs, parameter gradients of the accumulated step loss, running stats).
CPU, fp32, reference ResNet18 class.  Run: python -m pytest tests/test_ghost_bn.py -q  (from imagenet_colab)."""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import epkd_colab  # noqa: F401,E402  (sys.path for the reference code)
from epkd_colab.ghost_bn import GhostHolder, GhostSpec, convert_to_ghost_bn, interleaved_ema_weights  # noqa: E402
from model.python.resnet_imagenet import ResNet18  # noqa: E402
from tgsc_kd.utils import forward_logits  # noqa: E402


def _loss_rows(logits: torch.Tensor, t_logits: torch.Tensor, y: torch.Tensor, T: float = 1.0):
    ce = F.cross_entropy(logits, y, reduction="none")
    log_s = F.log_softmax(logits / T, dim=1)
    p_t = F.softmax(t_logits / T, dim=1)
    kd = (p_t * (torch.log(p_t + 1e-8) - log_s)).sum(1) * T * T
    return 0.5 * ce + 0.5 * kd


def test_ghost_bn_matches_sequential_microbatches():
    torch.manual_seed(0)
    G, slot, side, side_a, step_rows = 3, 8, 48, 64, 24  # "32 rows / 512 per step" scaled down
    view_counts = [8, 8, 5]  # last micro-batch partial
    anchor_counts = [7, 8, 3]  # one anchor per distinct image, variable
    ref = ResNet18(num_classes=20).train()
    ghost = copy.deepcopy(ref)
    holder = GhostHolder()
    assert convert_to_ghost_bn(ghost, holder) == 20
    xs_v = [torch.randn(c, 3, side, side) for c in view_counts]
    xs_a = [torch.randn(c, 3, side_a, side_a) for c in anchor_counts]
    ys_v = [torch.randint(0, 20, (c,)) for c in view_counts]
    ys_a = [torch.randint(0, 20, (c,)) for c in anchor_counts]
    ts_v = [torch.randn(c, 20) * 3 for c in view_counts]
    ts_a = [torch.randn(c, 20) * 3 for c in anchor_counts]

    # ---- reference: sequential micro-batches, loss_g scaled by n_g/step_rows, anchors as their own BN batch
    outs_v_ref, outs_a_ref = [], []
    for g in range(G):
        o_v = forward_logits(ref, xs_v[g])
        o_a = forward_logits(ref, xs_a[g])
        n = view_counts[g]
        loss = _loss_rows(o_v, ts_v[g], ys_v[g]).mean() + 1.0 * _loss_rows(o_a, ts_a[g], ys_a[g]).mean()
        (loss * n / step_rows).backward()
        outs_v_ref.append(o_v.detach())
        outs_a_ref.append(o_a.detach())

    # ---- ghost: one view forward + one anchor forward with padded slots
    def pad(xs, counts, s):
        out = torch.zeros(G * slot, 3, s, s)
        for g, x in enumerate(xs):
            out[g * slot: g * slot + counts[g]] = x
        return out

    xv, xa = pad(xs_v, view_counts, side), pad(xs_a, anchor_counts, side_a)
    w_v, keep_v, w_a, keep_a = interleaved_ema_weights(view_counts, anchor_counts, 0.1)
    spec_v, spec_a = GhostSpec(G, slot, torch.device("cpu")), GhostSpec(G, slot, torch.device("cpu"))
    spec_v.set(view_counts, w_v, keep_v)
    spec_a.set(anchor_counts, w_a, keep_a)
    holder.spec = spec_v
    o_v = forward_logits(ghost, xv)
    holder.spec = spec_a
    o_a = forward_logits(ghost, xa)
    # per-row weights: view row -> 1/step_rows ; anchor row -> n_g / (step_rows * A_g)
    wv = torch.zeros(G * slot)
    wa = torch.zeros(G * slot)
    yv = torch.zeros(G * slot, dtype=torch.long)
    ya = torch.zeros(G * slot, dtype=torch.long)
    tv = torch.zeros(G * slot, 20)
    ta = torch.zeros(G * slot, 20)
    for g in range(G):
        sl_v = slice(g * slot, g * slot + view_counts[g])
        sl_a = slice(g * slot, g * slot + anchor_counts[g])
        wv[sl_v] = 1.0 / step_rows
        wa[sl_a] = view_counts[g] / (step_rows * anchor_counts[g])
        yv[sl_v], ya[sl_a], tv[sl_v], ta[sl_a] = ys_v[g], ys_a[g], ts_v[g], ts_a[g]
    loss = (wv * _loss_rows(o_v, tv, yv)).sum() + (wa * _loss_rows(o_a, ta, ya)).sum()
    loss.backward()

    # ---- compare
    for g in range(G):
        assert torch.allclose(o_v[g * slot: g * slot + view_counts[g]], outs_v_ref[g], atol=2e-4, rtol=1e-4)
        assert torch.allclose(o_a[g * slot: g * slot + anchor_counts[g]], outs_a_ref[g], atol=2e-4, rtol=1e-4)
    max_rel = 0.0
    for (n1, p1), (n2, p2) in zip(ref.named_parameters(), ghost.named_parameters()):
        assert n1 == n2
        diff = (p1.grad - p2.grad).abs().max().item()
        scale = p1.grad.abs().max().item() + 1e-12
        max_rel = max(max_rel, diff / scale)
    assert max_rel < 1e-3, max_rel
    for (n1, b1), (n2, b2) in zip(ref.named_buffers(), ghost.named_buffers()):
        assert n1 == n2
        if b1.dtype.is_floating_point:
            assert torch.allclose(b1, b2, atol=1e-5, rtol=1e-4), (n1, (b1 - b2).abs().max())
        else:
            assert int(b1) == int(b2) == 2 * G, (n1, b1, b2)
    print(f"ghost-BN test OK: max rel grad diff {max_rel:.2e}")


if __name__ == "__main__":
    test_ghost_bn_matches_sequential_microbatches()
