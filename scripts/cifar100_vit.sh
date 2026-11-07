#!/bin/bash
# CIFAR-100 at 224 px, ViT-S/16 (OFA-KD teacher) -> DeiT-T/16 from scratch, OFA-KD recipe. Usage: scripts/cifar100_vit.sh [seed]
# Teacher: scripts/get_vit_teacher.sh. Each run trains 300 epochs (global batch 1024 = 8 x 128 accumulated).
set -e
cd "$(dirname "$0")/.."
SEED=${1:-0}
# EP-KD (token-window views ranked by the teacher's CLS attention, 5 stages of 60 epochs, passes 3,3,3,1,1)
python train_vit.py --run-dir runs/cifar100_vit_epkd_s$SEED --seed $SEED --views 1 --passes 3,3,3,1,1
# KD baselines: standard (1 pass per epoch), and the same optimizer-step budget as EP-KD (KD_sm)
python train_vit.py --run-dir runs/cifar100_vit_kd_s$SEED --seed $SEED --views 0 --passes 1
python train_vit.py --run-dir runs/cifar100_vit_kd_steps_s$SEED --seed $SEED --views 0 --passes 3,3,3,1,1
