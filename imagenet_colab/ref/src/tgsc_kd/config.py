from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any, Optional


@dataclass
class DataConfig:
    dataset_type: str = "cifar100"
    root: str = "./data"
    train_parquet_glob: str = "data/data/train-*.parquet"
    validation_parquet_glob: str = "data/data/validation-*.parquet"
    test_parquet_glob: Optional[str] = "data/data/test-*.parquet"
    mean: tuple[float, float, float] = (0.5071, 0.4867, 0.4408)
    std: tuple[float, float, float] = (0.2675, 0.2565, 0.2761)
    batch_size: int = 128
    train_batch_size: Optional[int] = None
    train_gpu_batch_size: Optional[int] = None
    val_batch_size: Optional[int] = None
    num_workers: int = 4
    pin_memory: bool = True
    analysis_resize_short: int = 256
    eval_resize_short: int = 256
    eval_crop_size: int = 224
    max_train_samples: Optional[int] = None
    max_val_samples: Optional[int] = None
    stages: tuple[tuple[int, int], ...] = (
        (16, 48),
        (20, 48),
        (24, 48),
        (28, 48),
        (32, 48),
    )

    def __post_init__(self) -> None:
        self.mean = tuple(self.mean)
        self.std = tuple(self.std)
        if self.train_batch_size is None:
            self.train_batch_size = int(self.batch_size)
        if self.train_gpu_batch_size is None:
            self.train_gpu_batch_size = int(self.train_batch_size)
        if self.val_batch_size is None:
            self.val_batch_size = int(self.batch_size)
        if self.max_train_samples is not None:
            self.max_train_samples = int(self.max_train_samples)
        if self.max_val_samples is not None:
            self.max_val_samples = int(self.max_val_samples)
        self.stages = tuple((int(res), int(epochs)) for res, epochs in self.stages)


@dataclass
class KDConfig:
    temperature: float = 4.0
    ce_weight: float = 0.1
    kd_weight: float = 0.9


@dataclass
class OptimConfig:
    base_lr: float = 0.05
    min_lr: float = 0.0
    momentum: float = 0.9
    weight_decay: float = 5e-4
    lr_schedule: str = "auto"
    lr_half_on_stage_change: bool = True
    milestones: tuple[int, int, int] = (150, 180, 210)
    milestone_gamma: float = 0.1
    epoch_halving_interval: Optional[int] = None
    cosine_start_epoch: Optional[int] = None
    cosine_end_epoch: Optional[int] = None

    def __post_init__(self) -> None:
        self.lr_schedule = str(self.lr_schedule).lower()
        self.milestones = tuple(int(value) for value in self.milestones)
        if self.epoch_halving_interval is not None:
            self.epoch_halving_interval = int(self.epoch_halving_interval)
        if self.cosine_start_epoch is not None:
            self.cosine_start_epoch = int(self.cosine_start_epoch)
        if self.cosine_end_epoch is not None:
            self.cosine_end_epoch = int(self.cosine_end_epoch)


@dataclass
class TrainConfig:
    epochs: int = 240
    start_epoch: Optional[int] = None
    seed: int = 27
    save_path: str = "outputs/student_kd.pth"
    checkpoint_dir: Optional[str] = None
    resume_path: Optional[str] = None
    log_every: int = 0
    log_dir: str = "logs"
    run_name: Optional[str] = None

    def __post_init__(self) -> None:
        self.epochs = int(self.epochs)
        if self.start_epoch is not None:
            self.start_epoch = int(self.start_epoch)


@dataclass
class TGSCConfig:
    precompute_batch_size: int = 256
    precompute_workers: int = 2
    use_gt_target: bool = True
    w_cam: float = 0.7
    w_sal: float = 0.3
    temp: float = 1.5
    gamma: float = 1.2
    pct_lo: float = 1.0
    pct_hi: float = 99.0
    stride: int = 8
    k_per_img: int = 6
    min_frac: float = 0.20
    min_mean: float = 0.20
    eval_chunk: int = 256
    cache: bool = True
    cache_in_memory: bool = True
    cache_to_disk: bool = False
    cache_dir: str = ".cache"
    cache_shard_size: int = 1024
    candidate_stride: Optional[int] = None
    dynamic_stride: bool = False
    dynamic_stride_power: float = 0.5
    flip_prob: float = 0.5
    aug_padding: int = 2
    box_jitter_px: int = 0
    top_k_crops: int = 2
    low_k_crops: int = 1
    score_high: float = 0.7
    score_low: float = 0.3
    easy_threshold: float = 0.75
    hard_threshold: float = 0.25
    difficulty_step: float = 0.05
    edge_align_windows: bool = False  # add a final window flush with the image edge when the stride grid stops short
    full_anchor: float = 0.0  # weight of the KD term on a full-resolution view of each source image (0 = off)
    full_anchor_frac: float = 1.0  # fraction of the micro-batch's source images that get the anchor term
    view_selection: str = "score"  # "score": rank candidates by the composite teacher score (TGSC); "saliency": keep the
    #                                teacher-saliency order of the proposals (EP-KD for vision)
    kd_gate: str = "on"  # "on": score-dependent KD weight (TGSC); "off": unit weight on every view (EP-KD for vision)


@dataclass
class ExperimentConfig:
    teacher_model: str = "vgg13"
    teacher_checkpoint: Optional[str] = None
    student_model: str = "vgg8"
    student_checkpoint: Optional[str] = None
    num_classes: int = 100
    device: Optional[str] = None
    data: DataConfig = field(default_factory=DataConfig)
    kd: KDConfig = field(default_factory=KDConfig)
    optim: OptimConfig = field(
        default_factory=lambda: OptimConfig(
            lr_half_on_stage_change=False,
            milestones=(48, 144, 192),
            milestone_gamma=0.1,
        )
    )
    train: TrainConfig = field(
        default_factory=lambda: TrainConfig(
            epochs=240,
            seed=27,
            save_path="outputs/vgg13_to_vgg8_tgsc_kd.pth",
            log_every=0,
        )
    )
    tgsc: TGSCConfig = field(
        default_factory=lambda: TGSCConfig(
            score_high=0.75,
            score_low=0.35,
        )
    )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _merge_section(defaults: dict[str, Any], updates: Optional[dict[str, Any]]) -> dict[str, Any]:
    merged = dict(defaults)
    if updates:
        merged.update(updates)
    return merged


def experiment_config_from_dict(raw: dict[str, Any], base: Optional[ExperimentConfig] = None) -> ExperimentConfig:
    base_cfg = base or ExperimentConfig()
    base_dict = base_cfg.to_dict()

    merged: dict[str, Any] = {}
    for key in ("teacher_model", "teacher_checkpoint", "student_model", "student_checkpoint", "num_classes", "device"):
        merged[key] = raw.get(key, base_dict[key])

    merged["data"] = DataConfig(**_merge_section(base_dict["data"], raw.get("data")))
    kd_raw = _merge_section(base_dict["kd"], raw.get("kd"))
    if "T" in kd_raw and "temperature" not in kd_raw:
        kd_raw["temperature"] = kd_raw.pop("T")
    if "alpha" in kd_raw:
        alpha = float(kd_raw.pop("alpha"))
        kd_raw.setdefault("ce_weight", 1.0 - alpha)
        kd_raw.setdefault("kd_weight", alpha)
    merged["kd"] = KDConfig(**kd_raw)
    merged["optim"] = OptimConfig(**_merge_section(base_dict["optim"], raw.get("optim")))
    merged["train"] = TrainConfig(**_merge_section(base_dict["train"], raw.get("train")))
    merged["tgsc"] = TGSCConfig(**_merge_section(base_dict["tgsc"], raw.get("tgsc")))
    return ExperimentConfig(**merged)


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    config_path = Path(path)
    raw = json.loads(config_path.read_text())
    return experiment_config_from_dict(raw)
