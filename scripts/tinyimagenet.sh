#!/bin/bash
# Tiny-ImageNet (64 px), WRN-40-2 -> WRN-40-1, 120 epochs. Usage: scripts/tinyimagenet.sh [seed]
set -e
cd "$(dirname "$0")/.."
SEED=${1:-0}
C="--dataset tinyimagenet --teacher-ckpt teachers/wrn_40_2_tinyimagenet.pth --bs 128 --epochs 120 --seed $SEED"
python train.py $C --run-dir runs/tinyimagenet_epkd_s$SEED \
  --views 1 --view-cache 1 --view-set sal1,sal2,sal3 --view-rotate 1 --passes 3,3,3,1,1 --anchor 1 \
  --view-sizes 32,40,48,56,64 --stage-epochs 24 --grid-stride 16 --view-pad 4 --milestones 24,72,96,105
python train.py $C --run-dir runs/tinyimagenet_kd_s$SEED --views 0 --milestones 75,90,105
python train.py $C --run-dir runs/tinyimagenet_kd_steps_s$SEED --views 0 --passes 3,3,3,1,1 \
  --view-sizes 32,40,48,56,64 --stage-epochs 24 --milestones 24,72,96,105
