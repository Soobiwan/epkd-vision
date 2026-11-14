"""The sync-free DDP helpers must be element-wise identical to the original single-GPU code."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from tgsc_kd.config import TGSCConfig  # noqa: E402
from tgsc_kd.ddp_trainer import DDPKDTrainer, balanced_file_assignment  # noqa: E402
from tgsc_kd.trainer import KDTrainer  # noqa: E402


class _Stub:
    tgsc_cfg = TGSCConfig(score_high=0.75, score_low=0.35)


def test_kd_weights_identical() -> None:
    g = torch.Generator().manual_seed(0)
    scores = torch.cat([torch.rand(5000, generator=g) * 1.4 - 0.2, torch.tensor([0.0, 0.35, 0.75, 1.0])])
    qflags = torch.where(torch.rand(scores.numel(), generator=g) < 0.4, -1, 1).to(torch.int16)
    for q in (None, qflags):
        ref = KDTrainer._score_to_kd_weights(_Stub(), scores.clone(), q)  # type: ignore[arg-type]
        new = DDPKDTrainer._score_to_kd_weights(_Stub(), scores.clone(), q)  # type: ignore[arg-type]
        assert torch.equal(ref, new)


def test_balanced_assignment_partitions_files() -> None:
    rows = [4300 + (i * 37) % 120 for i in range(294)]
    for epoch in (1, 2):
        a = balanced_file_assignment(rows, 3, 27, epoch)
        flat = sorted(i for r in a for i in r)
        assert flat == list(range(294))
        loads = [sum(rows[i] for i in r) for r in a]
        assert max(loads) - min(loads) <= max(rows)
    assert balanced_file_assignment(rows, 3, 27, 1) != balanced_file_assignment(rows, 3, 27, 2)
