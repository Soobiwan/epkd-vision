"""Shared helpers for the imagenet-fix reruns (paths, config loading, teacher build)."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT / "src", ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from tgsc_kd.config import ExperimentConfig, experiment_config_from_dict  # noqa: E402
from tgsc_kd.modeling import build_model, load_checkpoint, resolve_teacher_checkpoint  # noqa: E402

FIX = Path("/media/e12056/ASi/TGSC/imagenet_fix")
IMAGENET = Path("/media/e12056/ASi/imagenet/data")
CACHE_DIR = Path(os.environ.get("IMAGENET_FIX_CACHE", str(FIX / "cache")))




def load_cfg(path: str | Path, *, cache_dir: Path | None = None) -> ExperimentConfig:
    raw: dict[str, Any] = json.loads(Path(path).read_text())
    data = raw.setdefault("data", {})
    data["train_parquet_glob"] = str(IMAGENET / "train-*.parquet")
    data["validation_parquet_glob"] = str(IMAGENET / "validation-*.parquet")
    data["test_parquet_glob"] = str(IMAGENET / "test-*.parquet")
    cfg = experiment_config_from_dict(raw)
    cfg.tgsc.cache_dir = str(cache_dir or CACHE_DIR)
    # smoke-test overrides only (change the cache key through train_paths, never used for real runs)
    if os.environ.get("IMAGENET_FIX_MAX_TRAIN"):
        cfg.data.max_train_samples = int(os.environ["IMAGENET_FIX_MAX_TRAIN"])
        cfg.data.max_val_samples = int(os.environ.get("IMAGENET_FIX_MAX_VAL", "2000"))
        cfg.tgsc.cache_shard_size = int(os.environ.get("IMAGENET_FIX_CHUNK", "100"))
    return cfg


def build_teacher(cfg: ExperimentConfig):
    teacher = build_model(cfg.teacher_model, num_classes=cfg.num_classes)
    ckpt = resolve_teacher_checkpoint(cfg.teacher_model, cfg.teacher_checkpoint)
    load_checkpoint(teacher, ckpt)
    return teacher, Path(ckpt)
