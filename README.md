# EP-KD for Vision Classification

Evidence-Preserving Knowledge Distillation (EP-KD) controls the evidence a student is trained on.
For image classification:

1. **Teacher-proposed evidence.** The frozen teacher's class-activation map (CAM) for the ground-truth class
   ranks square windows of the image. At each budget, the top-3 windows become the restricted views.
2. **Progressive evidence budget.** Views grow over five stages (16, 20, 24, 28, 32 px on 32 px images,
   i.e. 25%, 39%, 56%, 77% and 100% of the image). The last stage is the full image only.
3. **Consistent distillation on every view.** The teacher is re-evaluated on exactly the crop the student
   sees. Crops are fed at native resolution, without resizing.
4. **Full-input anchor.** Every step also trains the student on the full image with the standard KD
   objective, so the student is never trained on restricted evidence alone.

Loss at stage *t* (α = 0.9, τ = 4, λ = 1):

```
L = ℓ(x, y) + λ · mean_{v ∈ V_t(x)} ℓ(v, y),
ℓ(u, y) = (1 − α) · CE(S(u), y) + α · τ² · KL(p_T^τ(·|u) || p_S^τ(·|u))
```

## Setup

```bash
pip install -r requirements.txt        # PyTorch with CUDA, numpy, pyarrow, pillow
scripts/download_data.sh               # CIFAR-100, STL-10, Tiny-ImageNet -> ./data
```

Teachers:
* **CIFAR-100.** The standard pretrained WRN-40-2 (75.61%) from
  [RepDistiller / CRD](https://github.com/HobbitLong/RepDistiller). Save it as `teachers/wrn_40_2_cifar100.pth`.
  A plain state dict and a dict with key `model` are both accepted.
* **STL-10 and Tiny-ImageNet.** Train WRN-40-2 teachers with `scripts/train_teachers.sh`.

All data is held on the GPU. Augmentation (reflection-padded random crop and horizontal flip) runs on the GPU too.

## Running

```bash
scripts/cifar100.sh 0        # EP-KD, KD (standard schedule), KD with the same optimizer-step budget
scripts/stl10.sh 0
scripts/tinyimagenet.sh 0
```

Each run writes `config.json`, `train.jsonl` (one line per epoch), `best_test.pth`, `final.pth`, `latest.pt`
(resumable) and `done.json` into its `--run-dir`.

### EP-KD flags

| flag | meaning (paper value) |
|---|---|
| `--views 1` | enable restricted views (`--views 0` is plain KD) |
| `--view-cache 1` | proposals computed on the clean image at the start of each stage and fixed for the stage |
| `--view-set sal1,sal2,sal3` | the 3 highest-CAM windows on the grid |
| `--grid-stride 8` | window grid stride (scale with the image side: 24 for STL-10, 16 for Tiny-ImageNet) |
| `--view-sizes 16,20,24,28,32` | stage crop sizes; the last entry is the full image |
| `--stage-epochs 48` | epochs per stage |
| `--passes 3,3,3,1,1` | passes over the data per epoch, per stage (one proposed view per pass) |
| `--view-rotate 1` | pass *r* uses the *r*-th ranked view |
| `--view-pad 2` | random-crop padding applied to each view |
| `--anchor 1` | weight of the full-image objective at restricted stages |
| `--milestones 48,144,192,210` | learning-rate divided by 10 after these epochs |

Normalisation statistics are updated only by the full-image passes when the anchor is on.

### Compute

The 3 passes at the 16, 20 and 24 px stages give **2.2× the optimizer steps of standard KD**. Each step processes
a full image plus a crop. Always compare against `--views 0 --passes 3,3,3,1,1` (KD with the same step budget)
as well as standard KD.

### Ablation options used during development

* `--view-set gt1,gt2,conf` (online, or with `--view-cache 1`): distinct label-evidence regions and the region
  behind the teacher's runner-up class. `confK` is the K-th ranked non-label class.
* `--stage-view-sets "gt1,gt2,conf;gt1,gt2,conf;sal1,sal2,sal3;sal1"`: a different view family per stage.
* `--pos random`: random windows instead of teacher proposals (online mode).
* `--gate on --sigma 1 --tau-g 0.75`: reliability gate w = min(1, exp(−KL/σ)/τ_g) on the view KD term.
* `--view-ce-mode valid`: CE on a view only if the teacher still predicts the label on it.
* `--xview-w 1 --xview-eps 0.5`: additional full-image target on views with KL(p_T(x) || p_T(v)) ≤ ε.
* `--size-mode verified --eps 0.5`: per-image budget from the smallest teacher-consistent view.
* `--view-type mask`: keep the window in place and set the rest to the mean (instead of cropping).
* `--amp 0`: fp32 instead of mixed precision.

## Vision Transformers (CIFAR-100 at 224 px)

`train_vit.py` applies the same method to a ViT student. Only the view family and the teacher relevance change:

* **Pair.** ViT-S/16 teacher → DeiT-T/16 student trained from scratch.
  * The teacher is the OFA-KD CIFAR-100 checkpoint ([Hao et al., NeurIPS 2023](https://github.com/Hao840/OFAKD)).
    It scores 92.22% under the eval transform below.
  * Teacher and student both see 14 × 14 = 196 patch tokens.
* **View.** A square window of *s* × *s* patch tokens.
  * The tokens keep their original position embeddings, and the CLS token is kept.
  * The other tokens are dropped. They are not masked, and nothing is resized.
  * Window sides per stage: 7, 9, 10, 12, then the full image (25, 41, 51, 73, 100% of the tokens).
* **Relevance.** The teacher's last-layer CLS→patch attention, averaged over heads.
  * It is computed on the full augmented image during the anchor forward, so the proposals are online and need no cache.
  * The 3 windows with the most attention mass are the proposals. Selection is greedy and suppresses IoU > 0.3.
  * Pass *p* uses the (*p* mod 3)-th proposal.
* **Targets.** Each view gets the teacher's response on the same token window (own-view target). The full-image
  objective stays at every step.
* **Budget.** 5 stages of 60 epochs over 300 epochs, with passes 3,3,3,1,1. That is 2.2× the optimizer steps of
  standard KD, so always also report KD with the same steps (`--views 0 --passes 3,3,3,1,1`).

Training recipe: OFA-KD `configs/cifar/vit_mlp.yaml`, reproduced on one GPU.
* **Batch and optimizer.**
  * OFA used 8 GPUs × 128. Here the global batch of 1024 comes from 8 × 128 gradient accumulation (`--accum 8`).
  * AdamW, lr 5e-4, weight decay 0.05.
  * Cosine schedule to 1e-5, stepped per epoch, after 20 warmup epochs from 5e-7. Gradient clipping at 5.
* **Augmentation.** RandomResizedCrop 224 (0.08–1) from the 32-px image, horizontal flip, RandAugment
  `rand-m9-mstd0.5-inc1`, random erasing 0.25, mixup 0.8 / cutmix 1.0, label smoothing 0.1.
  * Views are cut from the same augmented, mixed image as the anchor.
* **Loss.** OFA's KD, applied to the full image and to every view: soft-target CE (weight 1) + KL at τ = 1 (weight 1).
* **Evaluation.** Bicubic resize to 256, center crop 224 (the timm eval transform). The resized test set is cached as
  uint8 on first use (`--test-cache`).
* **Pillow compatibility.** Pillow ≥ 11 rejects posterize with 0 bits, which `inc1` reaches at magnitude 10.
  `train_vit.py` restores the old-Pillow behaviour (a black image) that OFA ran with.

```bash
scripts/get_vit_teacher.sh     # download + weights-only conversion -> teachers/vit_small_patch16_224_cifar100_weights.pt
scripts/cifar100_vit.sh 0      # EP-KD, KD, KD with the same optimizer-step budget (seed 0)
```

On an RTX 2080 Ti, KD takes ~4.6 h, KD with the same steps ~9.8 h and EP-KD ~14 h. Peak memory is ~3.2 GB, and data
loading uses 6 workers.
The outputs are the same files as `train.py`. `train.jsonl` also records `side`, `view_D` (KL(p_T(x) ‖ p_T(v))),
`img_s` and `data_wait`.

## Results so far (WRN-40-2 → WRN-40-1, best test epoch)

Preliminary numbers from development. They are not final paper numbers.

| dataset | teacher | KD (standard) | KD (same steps) | EP-KD |
|---|---|---|---|---|
| CIFAR-100 | 75.61 | 74.36 (seed 0) | 74.80 (seed 0) | **76.13 ± 0.25** (3 seeds: 76.42 / 76.01 / 75.97) |
| STL-10 | 86.56 | 87.34 (seed 0) | 87.50 (seed 0) | **87.88** (seed 0) |
| Tiny-ImageNet | 62.05 | 56.91 (seed 0) | running | running |

Ablations, CIFAR-100, seed 0:

| variant | best |
|---|---|
| no anchor (views only, otherwise identical) | 75.04 |
| anchor + 1 sampled view per step, 1 pass | 75.31 |
| same, random windows | 75.06 |
| same, + reliability gate | 74.95 |
| same, + label-validity CE and verified full-image target | 75.19 |
| anchor + reasoning views (gt1, gt2, conf), 3 passes | 76.04 |
| anchor + 5 reasoning views, 1 pass, views until the last epoch | 75.89 |

## Files

* `train.py`: training script (EP-KD, KD baselines, teacher training, ablations).
* `train_vit.py`: EP-KD and KD baselines for ViT students (token-window views, CLS-attention proposals).
* `wrn.py`: Wide-ResNet (CRD / RepDistiller definition).
* `prepare_data.py`: converts the STL-10 and Tiny-ImageNet parquet files to uint8 tensors.
* `scripts/`: data download, teacher training, per-dataset run scripts, and the ViT teacher download
  (`get_vit_teacher.sh`, `convert_ofa_checkpoint.py`) and runs (`cifar100_vit.sh`).
