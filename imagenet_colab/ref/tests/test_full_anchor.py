"""Full-image anchor views (EP-KD full-input objective at restricted stages)."""
import types

import torch

from tgsc_kd.ddp_trainer import DDPKDTrainer


def _fake_trainer(frac=1.0):
    return types.SimpleNamespace(
        device="cpu",
        data_cfg=types.SimpleNamespace(eval_crop_size=224),
        tgsc_cfg=types.SimpleNamespace(full_anchor_frac=frac, flip_prob=0.0),
        channels_last=False,
        _mean_dev=torch.zeros(1, 3, 1, 1),
        _std_dev=torch.ones(1, 3, 1, 1),
    )


def _batch():
    torch.manual_seed(0)
    sizes = {7: (256, 300), 3: (280, 256), 9: (256, 256)}
    images = {k: torch.rand(3, h, w) * 0.9 + 0.05 for k, (h, w) in sizes.items()}
    ids = torch.tensor([7, 3, 7, 9, 3])
    base = torch.zeros(5, 3, 280, 300)  # zero-padded bottom/right, as pad_stage_batch does
    for i, k in enumerate(ids.tolist()):
        h, w = sizes[k]
        base[i, :, :h, :w] = images[k]
    return sizes, images, ids, base, ids * 10


def test_one_exact_full_res_window_per_distinct_image():
    sizes, images, ids, base, labels = _batch()
    views, y = DDPKDTrainer._full_anchor_views(_fake_trainer(), base, labels, ids)
    assert views.shape == (3, 3, 224, 224)
    assert sorted(y.tolist()) == [30, 70, 90]
    for view, label in zip(views, y.tolist()):
        img = images[label // 10]
        h, w = sizes[label // 10]
        assert any(
            torch.allclose(img[:, y0:y0 + 224, x0:x0 + 224], view, atol=1e-5)
            for y0 in range(h - 223)
            for x0 in range(w - 223)
        )


def test_fraction_subsamples_images():
    _, _, ids, base, labels = _batch()
    views, _ = DDPKDTrainer._full_anchor_views(_fake_trainer(0.34), base, labels, ids)
    assert views.shape[0] == 1
