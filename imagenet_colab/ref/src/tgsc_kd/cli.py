from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import torch

from .config import ExperimentConfig, load_experiment_config
from .modeling import available_models, build_model, load_checkpoint, resolve_teacher_checkpoint
from .trainer import KDTrainer
from .utils import forward_logits


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs" / "imagenet_resnet34_to_resnet18.json"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TGSC KD trainer")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Path to JSON experiment config.")
    parser.add_argument("--teacher-model", help="Override teacher model name.")
    parser.add_argument("--teacher-checkpoint", help="Override teacher checkpoint path.")
    parser.add_argument("--student-model", help="Override student model name.")
    parser.add_argument("--student-checkpoint", help="Optional student checkpoint to resume from.")
    parser.add_argument("--resume", help="Resume full training state from a checkpoint path or use 'latest'.")
    parser.add_argument("--save-path", help="Override output checkpoint path.")
    parser.add_argument("--device", help="Override device, for example cpu or cuda:0.")
    parser.add_argument("--dry-run", action="store_true", help="Build models and load checkpoints without training.")
    parser.add_argument("--list-models", action="store_true", help="Print supported architectures and exit.")
    return parser.parse_args(argv)


def _apply_overrides(cfg: ExperimentConfig, args: argparse.Namespace) -> ExperimentConfig:
    if args.teacher_model:
        cfg.teacher_model = args.teacher_model
    if args.teacher_checkpoint is not None:
        cfg.teacher_checkpoint = args.teacher_checkpoint
    if args.student_model:
        cfg.student_model = args.student_model
    if args.student_checkpoint is not None:
        cfg.student_checkpoint = args.student_checkpoint
    if args.resume is not None:
        cfg.train.resume_path = args.resume
    if args.save_path:
        cfg.train.save_path = args.save_path
    if args.device:
        cfg.device = args.device
    return cfg


def _build_models(cfg: ExperimentConfig) -> tuple[torch.nn.Module, torch.nn.Module, Path]:
    teacher = build_model(cfg.teacher_model, num_classes=cfg.num_classes)
    student = build_model(cfg.student_model, num_classes=cfg.num_classes)

    teacher_ckpt = resolve_teacher_checkpoint(cfg.teacher_model, cfg.teacher_checkpoint)
    load_checkpoint(teacher, teacher_ckpt)

    if cfg.student_checkpoint:
        load_checkpoint(student, cfg.student_checkpoint)

    return teacher, student, teacher_ckpt


def _dry_run(cfg: ExperimentConfig, teacher: torch.nn.Module, student: torch.nn.Module, teacher_ckpt: Path) -> int:
    device_name = cfg.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_name)
    teacher.to(device).eval()
    student.to(device).eval()

    dummy_size = cfg.data.eval_crop_size if cfg.data.dataset_type.startswith("imagenet") else 32
    dummy = torch.randn(2, 3, dummy_size, dummy_size, device=device)
    with torch.no_grad():
        teacher_logits = forward_logits(teacher, dummy)
        student_logits = forward_logits(student, dummy)

    print(f"Config: {Path(cfg.train.save_path)}")
    print(f"Teacher: {cfg.teacher_model} <- {teacher_ckpt}")
    print(f"Student: {cfg.student_model}")
    print(f"Device: {device}")
    print(f"Teacher logits: {tuple(teacher_logits.shape)}")
    print(f"Student logits: {tuple(student_logits.shape)}")
    if cfg.train.start_epoch:
        print(f"Start epoch: {cfg.train.start_epoch}")
    if cfg.student_checkpoint:
        print(f"Student checkpoint: {cfg.student_checkpoint}")
    if cfg.train.resume_path:
        print(f"Resume path: {cfg.train.resume_path}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.list_models:
        for name in available_models():
            print(name)
        return 0

    cfg = load_experiment_config(args.config)
    cfg = _apply_overrides(cfg, args)

    teacher, student, teacher_ckpt = _build_models(cfg)
    if args.dry_run:
        return _dry_run(cfg, teacher, student, teacher_ckpt)

    device = torch.device(cfg.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    trainer = KDTrainer(
        student=student,
        teacher=teacher,
        data_cfg=cfg.data,
        kd_cfg=cfg.kd,
        opt_cfg=cfg.optim,
        train_cfg=cfg.train,
        tgsc_cfg=cfg.tgsc,
        device=device,
        teacher_name=cfg.teacher_model,
        teacher_checkpoint=teacher_ckpt,
    )
    trainer.train()
    return 0
