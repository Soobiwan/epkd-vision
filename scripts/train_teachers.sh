#!/bin/bash
# WRN-40-2 teachers for STL-10 and Tiny-ImageNet (cross-entropy). CIFAR-100 uses the standard pretrained
# WRN-40-2 teacher from CRD / RepDistiller (75.61%), saved as teachers/wrn_40_2_cifar100.pth.
set -e
cd "$(dirname "$0")/.."
python train.py --dataset stl10 --teacher none --student wrn_40_2 --bs 64 --epochs 240 \
  --milestones 150,180,210 --run-dir runs/teacher_stl10_wrn402
python train.py --dataset tinyimagenet --teacher none --student wrn_40_2 --bs 128 --epochs 120 \
  --milestones 75,90,105 --run-dir runs/teacher_tinyimagenet_wrn402
mkdir -p teachers
cp runs/teacher_stl10_wrn402/final.pth teachers/wrn_40_2_stl10.pth
cp runs/teacher_tinyimagenet_wrn402/final.pth teachers/wrn_40_2_tinyimagenet.pth
