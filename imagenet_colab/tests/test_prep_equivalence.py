"""CPU slicing preparation (prep.py) == reference roi_align preparation (DDPKDTrainer._prepare_imagenet_batch +
_full_anchor_views) for identical random draws.  Run from imagenet_colab: python tests/test_prep_equivalence.py"""
from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import epkd_colab  # noqa: F401,E402
from epkd_colab.prep import Draws, prepare_microbatch  # noqa: E402
from epkd_colab.step import make_prepare_shim  # noqa: E402
from tgsc_kd.imagenet_parquet import pad_stage_batch  # noqa: E402


def run_case(seed: int, crop: int, n: int = 32, anchor: bool = True, jitter: int = 10) -> float:
    g = np.random.default_rng(seed)
    sizes = [(256, int(256 * 4 / 3)), (int(256 * 4 / 3), 256), (256, 256), (256, 500), (300, 256)]
    n_img = 12
    imgs = [g.integers(0, 256, size=(3,) + sizes[g.integers(len(sizes))], dtype=np.uint8) for _ in range(n_img)]
    imgs[3][:, :, -1] = 0  # black last column (valid-extent quirk)
    imgs[5][:, -2:, :] = 0
    ids = np.sort(g.integers(100, 100 + n_img, size=n))
    rows = []
    for sid in ids:
        im = imgs[sid - 100]
        h, w = im.shape[1:]
        y0 = int(g.integers(0, h - crop + 1))
        x0 = int(g.integers(0, w - crop + 1))
        rows.append((torch.from_numpy(im.copy()), int(sid % 7), 0.5, 1, torch.tensor([x0, y0, x0 + crop, y0 + crop], dtype=torch.float32), int(sid)))
    batch = pad_stage_batch(rows)
    boxes_np = batch[4].numpy().copy()  # the reference modifies its (CPU-aliased) boxes in place
    mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    tr = types.SimpleNamespace(
        device=torch.device("cpu"), channels_last=False,
        tgsc_cfg=types.SimpleNamespace(full_anchor=1.0 if anchor else 0.0, full_anchor_frac=1.0, flip_prob=0.5, box_jitter_px=jitter, kd_gate="off"),
        data_cfg=types.SimpleNamespace(eval_crop_size=224),
        _mean_dev=torch.tensor(mean).view(1, 3, 1, 1), _std_dev=torch.tensor(std).view(1, 3, 1, 1),
    )
    prepare = make_prepare_shim(tr)
    torch.manual_seed(seed)
    x_ref, y_ref, _s, _q, anchor_ref = prepare(batch, crop_size=crop)
    # replay the reference's random draws in its order
    torch.manual_seed(seed)
    A = len(np.unique(ids)) if anchor and crop < 224 else 0
    if A:
        ax, ay, af = torch.rand(A), torch.rand(A), torch.rand(A)
    else:
        ax = ay = af = torch.zeros(0)
    dx = torch.randint(-jitter, jitter + 1, (n,))
    dy = torch.randint(-jitter, jitter + 1, (n,))
    vf = torch.rand(n)
    draws = Draws(ax=ax.numpy(), ay=ay.numpy(), aflip=af.numpy(), dx=dx.numpy(), dy=dy.numpy(), vflip=vf.numpy())
    views, anchors, first = prepare_microbatch([imgs[s - 100] for s in ids], boxes_np, ids, crop=crop, jitter=jitter,
                                              flip_prob=0.5, anchor_size=224 if A else None, draws=draws)
    m = torch.tensor(mean).view(1, 3, 1, 1)
    s = torch.tensor(std).view(1, 3, 1, 1)
    mine = (torch.from_numpy(views).float() / 255.0 - m) / s
    err = (mine - x_ref).abs().max().item()
    assert err < 1e-5, err
    if A:
        xa_ref, ya_ref = anchor_ref
        mine_a = (torch.from_numpy(anchors).float() / 255.0 - m) / s
        err_a = (mine_a - xa_ref).abs().max().item()
        assert err_a < 1e-5, err_a
        assert torch.equal(ya_ref, y_ref[torch.from_numpy(first)])
        err = max(err, err_a)
    return err


if __name__ == "__main__":
    worst = 0.0
    for seed in range(6):
        for crop in (112, 140, 168, 196):
            worst = max(worst, run_case(seed, crop))
        worst = max(worst, run_case(seed, 224, anchor=False))
    print(f"prep equivalence OK: max |mine - reference| = {worst:.2e} (normalised units)")
