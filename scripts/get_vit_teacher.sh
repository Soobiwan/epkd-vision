#!/bin/bash
# Downloads the OFA-KD CIFAR-100 ViT-S/16 teacher (Hao et al., NeurIPS 2023; github.com/Hao840/OFAKD, release
# checkpoint-cifar100) and saves its weights as a plain state dict, teachers/vit_small_patch16_224_cifar100_weights.pt.
set -e
cd "$(dirname "$0")/.."
mkdir -p teachers
F=teachers/vit_small_patch16_224_cifar100.pth
[ -f $F ] || curl -L -o $F https://github.com/Hao840/OFAKD/releases/download/checkpoint-cifar100/vit_small_patch16_224_cifar100.pth
python scripts/convert_ofa_checkpoint.py $F teachers/vit_small_patch16_224_cifar100_weights.pt
