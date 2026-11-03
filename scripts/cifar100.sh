#!/bin/bash
# CIFAR-100, WRN-40-2 -> WRN-40-1. Usage: scripts/cifar100.sh [seed]
set -e
cd "$(dirname "$0")/.."
SEED=${1:-0}
T="--teacher-ckpt teachers/wrn_40_2_cifar100.pth --seed $SEED"
# EP-KD for vision (paper method)
python train.py $T --run-dir runs/cifar100_epkd_s$SEED \
  --views 1 --view-cache 1 --view-set sal1,sal2,sal3 --view-rotate 1 \
  --passes 3,3,3,1,1 --anchor 1 --milestones 48,144,192,210
# KD baselines: standard schedule, and the same optimizer-step budget as EP-KD
python train.py $T --run-dir runs/cifar100_kd_s$SEED --views 0 --milestones 150,180,210
python train.py $T --run-dir runs/cifar100_kd_steps_s$SEED --views 0 --passes 3,3,3,1,1 --milestones 48,144,192,210
