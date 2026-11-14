#!/usr/bin/env python3
"""torchrun entry point for the DDP ImageNet TGSC-KD reruns.

    torchrun --standalone --nproc_per_node 3 scripts/imagenet_fix/train_ddp.py --config CFG.json [opts]

Auto-resumes from <checkpoint_dir>/latest.pt.  For the 81->100 tails pass --init-student-from
<phase-1 run>/epoch_080.pt (student weights only, fresh optimizer, as in the reported runs).
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist

from common import build_teacher, load_cfg  # type: ignore

from tgsc_kd.ddp_trainer import DDPKDTrainer  # noqa: E402
from tgsc_kd.modeling import build_model  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--init-student-from", default=None)
    ap.add_argument("--amp", type=int, default=0)
    ap.add_argument("--teacher-fp16", type=int, default=0)
    ap.add_argument("--channels-last", type=int, default=1)
    ap.add_argument("--num-workers", type=int, default=6)
    ap.add_argument("--val-workers", type=int, default=5)
    ap.add_argument("--pin-memory", type=int, default=1)
    ap.add_argument("--prefetch-factor", type=int, default=4)
    ap.add_argument("--bucket-cap-mb", type=int, default=25)
    ap.add_argument("--static-graph", type=int, default=0)
    # benchmarking helpers
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--skip-eval", action="store_true")
    ap.add_argument("--no-save", action="store_true")
    ap.add_argument("--start-epoch", type=int, default=None)
    ap.add_argument("--stage-index-override", default=None, help="bench only: path with {res}")
    ap.add_argument("--run-dir", default=None, help="override checkpoint/log dir (bench)")
    ap.add_argument("--progress-every", type=int, default=200)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--loss-trace", default=None)
    args = ap.parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", timeout=dt.timedelta(minutes=60), device_id=torch.device("cuda", local_rank))
    rank, world = dist.get_rank(), dist.get_world_size()

    cfg = load_cfg(args.config)
    if args.run_dir:
        cfg.train.checkpoint_dir = args.run_dir
        cfg.train.save_path = str(Path(args.run_dir) / "student_best.pth")
        cfg.train.log_dir = str(Path(args.run_dir) / "logs")
    if args.start_epoch is not None:
        cfg.train.start_epoch = args.start_epoch
    if args.seed is not None:
        cfg.train.seed = args.seed
    teacher, ckpt = build_teacher(cfg)
    student = build_model(cfg.student_model, num_classes=cfg.num_classes)
    device = torch.device("cuda", local_rank)

    trainer = DDPKDTrainer(
        student=student, teacher=teacher, data_cfg=cfg.data, kd_cfg=cfg.kd, opt_cfg=cfg.optim,
        train_cfg=cfg.train, tgsc_cfg=cfg.tgsc, device=device,
        teacher_name=cfg.teacher_model, teacher_checkpoint=ckpt,
        rank=rank, world=world, amp=bool(args.amp), teacher_fp16=bool(args.teacher_fp16),
        channels_last=bool(args.channels_last), num_workers=args.num_workers, val_workers=args.val_workers,
        pin_memory=bool(args.pin_memory), prefetch_factor=args.prefetch_factor, bucket_cap_mb=args.bucket_cap_mb,
        static_graph=bool(args.static_graph), init_student_from=args.init_student_from,
        max_steps=args.max_steps, skip_eval=args.skip_eval, no_save=args.no_save,
        stage_index_override=args.stage_index_override, progress_every=args.progress_every,
        loss_trace=args.loss_trace,
    )
    if rank == 0:
        print(json.dumps({"argv": sys.argv, "world": world, "config": args.config,
                          "log": str(getattr(trainer.logger, "path", None))}), flush=True)
    trainer.train()  # set_seed() inside sets cudnn.benchmark=True exactly like the original trainer
    dist.barrier()
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
