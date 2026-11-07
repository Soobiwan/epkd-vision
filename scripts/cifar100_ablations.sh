#!/bin/bash
# CIFAR-100 ablations of EP-KD (WRN-40-2 -> WRN-40-1), one seed. Usage: scripts/cifar100_ablations.sh [seed]
# Each line changes one component of the EP-KD command in scripts/cifar100.sh.
set -e
cd "$(dirname "$0")/.."
SEED=${1:-0}
T="--teacher-ckpt teachers/wrn_40_2_cifar100.pth --seed $SEED"
EP="--views 1 --view-cache 1 --view-rotate 1 --passes 3,3,3,1,1 --milestones 48,144,192,210"
run() { NAME=$1; shift; python train.py $T --run-dir runs/cifar100_abl_${NAME}_s$SEED "$@"; }
# objective
run no_anchor      $EP --view-set sal1,sal2,sal3 --anchor 0                  # views only at restricted stages
run full_target    $EP --view-set sal1,sal2,sal3 --anchor 1 --view-target full # transported target p_T(.|x) on views
run soft_weight    $EP --view-set sal1,sal2,sal3 --anchor 1 --gate soft --sigma 0.5  # view weight exp(-KL/sigma)
# budget
run reversed       $EP --view-set sal1,sal2,sal3 --anchor 1 --stage-order reverse   # full image first, smallest views last
run shuffled       $EP --view-set sal1,sal2,sal3 --anchor 1 --stage-order shuffled  # same stages in random order
# which views
run top1           $EP --view-set sal1,sal1,sal1 --anchor 1                  # the top proposal three times
run grid_random    $EP --view-set rnd1,rnd2,rnd3 --anchor 1                  # random grid windows, fixed per stage
run free_random    $EP --view-set frn1,frn2,frn3 --anchor 1                  # random windows anywhere, fixed per stage
run redrawn_random $EP --view-set sal1,sal2,sal3 --anchor 1 --free-random 1  # random windows, redrawn every pass
run least_salient  $EP --view-set low1,low2,low3 --anchor 1                  # least CAM mass on the proposal grid
run pool_agree     $EP --view-set agr1,agr2,agr3 --anchor 1                  # of 16 random windows: most agreeing with p_T(.|x)
run pool_disagree  $EP --view-set dis1,dis2,dis3 --anchor 1                  #   ... least agreeing
run pool_random    $EP --view-set pol1,pol2,pol3 --anchor 1                  #   ... 3 at random
# view form
run upsampled      $EP --view-set sal1,sal2,sal3 --anchor 1 --view-resize 32 # crop resized to 32 px
run masked         $EP --view-set sal1,sal2,sal3 --anchor 1 --view-type mask # window kept in place, rest = mean
run downsampled    $EP --view-set down1,down2,down3 --anchor 1               # whole image resized to the view size
# KD controls
run kd_1pass       --views 0 --milestones 48,144,192,210                     # EP-KD's lr schedule, 1 pass
run kd_more_steps  --views 0 --passes 4,4,4,2,2 --milestones 48,144,192,210  # more steps than EP-KD
run kd_fm          --views 0 --passes 3,3,3,1,1 --milestones 48,144,192,210 --fm-aug 1  # function matching (Beyer et al.)
