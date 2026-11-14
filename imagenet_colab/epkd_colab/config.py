"""Config loading for the Colab pipeline: the reference configs with the data / cache locations taken from the
environment (defaults = the author's PC layout, which the Colab VM reproduces so that cache keys are identical).

    EPKD_IMAGENET_DIR   parquet shards dir   (default /media/e12056/ASi/imagenet/data)
    EPKD_CACHE_DIR      TGSC cache root      (default <imagenet_colab>/cache)
    EPKD_MAX_TRAIN / EPKD_MAX_VAL / EPKD_CHUNK   smoke-test subsets (change the cache key)
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from tgsc_kd.config import ExperimentConfig, experiment_config_from_dict
from tgsc_kd.modeling import build_model, load_checkpoint, resolve_teacher_checkpoint

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IMAGENET = "/media/e12056/ASi/imagenet/data"


def imagenet_dir() -> Path:
    return Path(os.environ.get("EPKD_IMAGENET_DIR", DEFAULT_IMAGENET))


def cache_dir() -> Path:
    return Path(os.environ.get("EPKD_CACHE_DIR", str(ROOT / "cache")))


def load_cfg(path: str | Path, *, cache: Path | None = None) -> ExperimentConfig:
    raw: dict[str, Any] = json.loads(Path(path).read_text())
    data = raw.setdefault("data", {})
    d = imagenet_dir()
    data["train_parquet_glob"] = str(d / "train-*.parquet")
    data["validation_parquet_glob"] = str(d / "validation-*.parquet")
    data["test_parquet_glob"] = str(d / "test-*.parquet")
    cfg = experiment_config_from_dict(raw)
    cfg.tgsc.cache_dir = str(cache or cache_dir())
    if os.environ.get("EPKD_MAX_TRAIN"):
        cfg.data.max_train_samples = int(os.environ["EPKD_MAX_TRAIN"])
        cfg.data.max_val_samples = int(os.environ.get("EPKD_MAX_VAL", "2000"))
        cfg.tgsc.cache_shard_size = int(os.environ.get("EPKD_CHUNK", "1024"))
    return cfg


def build_teacher(cfg: ExperimentConfig):
    teacher = build_model(cfg.teacher_model, num_classes=cfg.num_classes)
    ckpt = resolve_teacher_checkpoint(cfg.teacher_model, cfg.teacher_checkpoint)
    load_checkpoint(teacher, ckpt)
    return teacher, Path(ckpt)
