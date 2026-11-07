#!/bin/bash
# Off-centre CIFAR-100 (64 x 64 canvas, object = 1/4 of the image): teacher, EP-KD, random windows, downsampled
# whole image, KD with the same steps. Usage: scripts/canvas.sh [seed]   (data: scripts/prepare_canvas.py data data)
set -e
cd "$(dirname "$0")/.."
SEED=${1:-0}
if [ ! -f teachers/wrn_40_2_canvas.pth ]; then
  python train.py --dataset cifar100canvas --teacher none --student wrn_40_2 --bs 128 --epochs 240 \
    --milestones 150,180,210 --run-dir runs/teacher_canvas_wrn402
  mkdir -p teachers && cp runs/teacher_canvas_wrn402/final.pth teachers/wrn_40_2_canvas.pth
fi
C="--dataset cifar100canvas --teacher-ckpt teachers/wrn_40_2_canvas.pth --bs 128 --epochs 120 --seed $SEED
   --view-sizes 32,40,48,56,64 --stage-epochs 24 --milestones 24,72,96,105"
EP="--views 1 --view-cache 1 --view-rotate 1 --passes 3,3,3,1,1 --anchor 1 --grid-stride 16 --view-pad 4"
python train.py $C --run-dir runs/canvas_epkd_s$SEED $EP --view-set sal1,sal2,sal3
python train.py $C --run-dir runs/canvas_redrawn_random_s$SEED $EP --view-set sal1,sal2,sal3 --free-random 1
python train.py $C --run-dir runs/canvas_downsampled_s$SEED $EP --view-set down1,down2,down3
python train.py $C --run-dir runs/canvas_kd_steps_s$SEED --views 0 --passes 3,3,3,1,1
