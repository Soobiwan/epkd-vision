#!/bin/bash
# CIFAR-100 with 10 / 25 / 50% of the training set (class-balanced, subsets/lowdata_f*.json) and with label skew
# (Dirichlet class proportions, subsets/skew_a*.json): EP-KD vs KD with the same steps.
# Usage: scripts/low_data.sh [seed]   (the label-skew runs in the paper use seed 27)
set -e
cd "$(dirname "$0")/.."
SEED=${1:-0}
T="--teacher-ckpt teachers/wrn_40_2_cifar100.pth --seed $SEED --milestones 48,144,192,210 --passes 3,3,3,1,1"
EP="--views 1 --view-cache 1 --view-rotate 1 --view-set sal1,sal2,sal3 --anchor 1"
for S in lowdata_f10 lowdata_f25 lowdata_f50 skew_a0p1 skew_a0p3 skew_a1p0 skew_a10p0 skew_a100p0; do
  python train.py $T --subset-manifest subsets/$S.json --run-dir runs/${S}_epkd_s$SEED $EP
  python train.py $T --subset-manifest subsets/$S.json --run-dir runs/${S}_kd_steps_s$SEED --views 0
done
