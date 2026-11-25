# `cubpets/`: the code of the paper's CUB-200 / Oxford-Pets runs (96 px)

This folder holds the trainer that produced the paper's CUB-200-2011 and Oxford-IIIT Pets results (the "do teacher
proposals matter?" table: EP-KD with the teacher's proposals vs. random windows fixed per stage vs. least-salient
windows vs. KD). It is a byte copy of that code. It is a separate trainer from the CIFAR / STL / Tiny-ImageNet
`train.py` at the top of this repository, and its options and defaults differ from that one.

| file | what |
|---|---|
| `train.py` | the trainer: GPU-resident data and augmentation, AMP; resumes from `<run-dir>/latest.pt`, which is saved after every epoch. Copy of `view_rules/code_cubpets/train.py`, md5 `42499d258673a8bcdcb9657ba5183a82`. That file is the WRN CUB / Pets trainer (`epkd2/cubucf/code/train.py`, sha256 `401b8805202d…`) plus a 13-line patch: models outside `wrn.py` come from `models/`, `teacher_forward` handles those models, and `fc_of()` supplies the CAM weights. WRN paths are unchanged. |
| `wrn.py` | WRN definitions, unchanged (md5 `d45dfc6947f1a89af49184e2979ec5f7`) |
| `models/` | the CRD CIFAR zoo with adaptive pooling (`resnet32x4`, `resnet8x4`, `ShuffleV2`, …), unchanged. Any input or crop size works: a 96-px image gives a 24×24 map (ResNet) or a 12×12 map (ShuffleV2). |
| `prep_cub_ucf.py` | the script that built `cub_uint8.pt` / `pets_uint8.pt` (whole image squashed to 96×96, official splits), unchanged |
| `get_data.py` | downloads the raw files the paper tensors were built from (Hugging Face mirrors at pinned revisions, sha256 checked), runs `prep_cub_ucf.py` with Pillow 13.0.0, and checks the content sha256 of `train_x/train_y/test_x/test_y` against the paper tensors |
| `plan.py` | the 48 runs (dataset × pair × arm × seed), their exact `train.py` arguments, and their split over 16 Kaggle notebooks |
| `kaggle_run.py` | runs one notebook's share on a Kaggle T4 x2 session: data, teachers, one job per GPU, resume, 11.5-h guard, results, zip |

Checks done on the copy (11-25): `diff` against the source files shows no differences; 4-arm smoke tests ran through
all five stages; the data rebuilt from the downloads has the same tensor hashes as the paper data (CUB and Pets,
Pillow 13.0.0). Pillow 11.3.0 and 12.0.0 decode or resize to different pixels, so `get_data.py` pins Pillow 13.0.0.

## Runs (ResNet32x4 teacher, students ResNet8x4 and ShuffleNetV2)

These are the WRN-40-2 → WRN-40-1 command lines (`epkd2/cubucf/make_jobs.sh`) with the teacher, student and
learning rate changed:

    P="--dataset <cub|pets> --data-root DATA --teacher resnet32x4 --teacher-ckpt tea_<ds>_r32x4.pth --bs 64"
    S="--student resnet8x4 --lr 0.05"  |  "--student ShuffleV2 --lr 0.01"
    EP="--view-sizes 48,60,72,84,96 --grid-stride 24 --view-pad 6 --view-cache 1 --view-rotate 1 --passes 3,3,3,1,1 --anchor 1 --milestones 48,144,192,210 --views 1"
    prop: $P $S $EP --view-set sal1,sal2,sal3 --seed k
    rfix: $P $S $EP --view-set rnd1,rnd2,rnd3 --seed k
    low:  $P $S $EP --view-set low1,low2,low3 --seed k
    kd:   $P $S --views 0 --seed k                      (240 epochs, LR x0.1 at 150/180/210)

    python cubpets/train.py <args> --run-dir runs/<ds>_<r8x4|sv2>_<arm>_s<k>

The reported number is the best test accuracy (`done.json` "best"), as in the WRN table. The `final` value is also
stored.

Teachers: ResNet32x4 trained with cross-entropy from scratch, using the recipe of the WRN-40-2 teachers:
`--teacher none --student resnet32x4 --bs 64 --epochs 240 --milestones 150,180,210`. Students load the last epoch
(`final.pth`). The checkpoints are assets of the GitHub release
[`cubpets-teachers`](https://github.com/Soobiwan/epkd-vision/releases/tag/cubpets-teachers): `tea_<ds>_r32x4.pth`,
with `tea_<ds>_r32x4.json` holding the sha256 and the test accuracy.

Data: `python cubpets/get_data.py --out data cub pets` writes `data/cub_uint8.pt` and `data/pets_uint8.pt`. These hold
only the tensors that training reads; the box and trimap fields used by the analysis scripts are left out.

Kaggle: see `kaggle/CUBPETS_README.md`.
