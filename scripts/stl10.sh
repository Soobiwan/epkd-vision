#!/bin/bash
# STL-10 (96 px), WRN-40-2 -> WRN-40-1. Usage: scripts/stl10.sh [seed]
set -e
cd "$(dirname "$0")/.."
SEED=${1:-0}
C="--dataset stl10 --teacher-ckpt teachers/wrn_40_2_stl10.pth --bs 64 --seed $SEED"
python train.py $C --run-dir runs/stl10_epkd_s$SEED \
  --views 1 --view-cache 1 --view-set sal1,sal2,sal3 --view-rotate 1 --passes 3,3,3,1,1 --anchor 1 \
  --view-sizes 48,60,72,84,96 --grid-stride 24 --view-pad 6 --milestones 48,144,192,210
python train.py $C --run-dir runs/stl10_kd_s$SEED --views 0 --milestones 150,180,210
python train.py $C --run-dir runs/stl10_kd_steps_s$SEED --views 0 --passes 3,3,3,1,1 \
  --view-sizes 48,60,72,84,96 --milestones 48,144,192,210
