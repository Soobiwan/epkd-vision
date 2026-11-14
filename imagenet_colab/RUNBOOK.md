# EP-KD ImageNet on Colab TPU -- runbook

## Running from another PC (start here)
The local PC only drives Colab: it creates the TPU VM, uploads this directory, and backs up the checkpoints and the
proposal cache. All training runs on the VM. Two teacher-student pairs: `PAIR=F` ResNet34->ResNet18 (finished, best
72.55 % at epoch 95) and `PAIR=E` ResNet50->MobileNetV1 (paused; local backup at epoch 30, val top-1 67.34 %).

1. **Tools.** Install the Colab CLI (version 0.7.5 was used) and sign in once: running `colab usage` opens the Google login.
   Python 3 and GNU `tar`/`split`/`md5sum` must be on the PATH.
2. **Hugging Face token.** Save it to `~/.cache/huggingface/token` (ImageNet comes from the gated `ILSVRC/imagenet-1k`
   parquet shards). `colab/ctl.sh push` uploads that file to the VM; it is never stored in the repository.
3. **Teacher weights (not in git; copy them by hand).** These exact files, NOT freshly downloaded torchvision files: the
   proposal-cache key contains the file's sha256, and a different file gives a different key, which forces a new
   precompute (about 6-7 h on a v6e1).

   | Path (inside `imagenet_colab/`) | Size | sha256 |
   |---|---|---|
   | `ref/model/imagenet_teachers/ResNet34_vanilla/resnet34-333f7ec4.pth` | 87.3 MB | `0f1c17db…6e321` |
   | `ref/model/imagenet_teachers/ResNet50_vanilla/resnet50-19c8e357.pth` | 102.5 MB | `14b7379e…df0d66` |

4. **Backups to resume from (not in git; copy them by hand).** `backup_full_E/` (778 MB, MobileNetV1 run) holds
   `cache_inc_001..028.tgz` (proposal cache), `cache_final`, `SMOKE_OK`, `incarnation` and `state.tgz` (checkpoint after epoch 30, and
   logs). Copy the whole directory to `imagenet_colab/backup_full_E/`. `backup_full/` (3.4 GB) is the finished ResNet18 run
   (`state_FINAL_R18.tgz` = final, best and epoch-80 checkpoints); it is only needed as an archive.
5. **Do not edit the configs.** Their absolute data path (`/media/e12056/ASi/imagenet/data`) is a path on the VM, which
   `ctl.sh push` creates on every VM. It is also part of the cache key, so changing it invalidates the backed-up cache.
6. **Resume the MobileNetV1 run** (new VM, upload the code, restore cache and checkpoint, download ImageNet, continue
   training; about 1-1.5 h of setup):
   ```bash
   cd imagenet_colab
   BK=$PWD/backup_full_E SESSION_ENV="PAIR=E CPUP=16" setsid nohup colab/drive_full.sh > drive_full_E.log 2>&1 < /dev/null &
   ```
   The driver polls every 10 min, backs up after every epoch, moves to a new VM when the old one is lost (Colab's limit is
   about 24 h), and backs up and stops the VM when the balance falls below 6 units. Control files in the backup directory:
   `QUIT` (back up and stop), `HOLD`, `RESTART`. The VM also runs `colab/stall_watchdog.sh`, which restarts a trainer
   that hangs after an error.
7. **Speed settings, already the defaults:** compute-first training loop (`EPKD_COMPUTE_FIRST=1`), fast ghost BatchNorm
   (`EPKD_GHOST_IMPL=fast`), int32 uploads (`H2D_METHOD=i32`; the plain `to` upload slows down as VM memory fragments).
   ResNet50->MobileNetV1 on a v6e1 runs at about 3,600 rows/s at 112 px and about 3,300 at 140 px. See `UTIL_PLAN.md`.

## Original notes (written on the first PC; paths are that PC's)

Code: `/media/e12056/ASi/TGSC/imagenet_colab` (this dir).  `ref/` = unmodified copies of the reference code
(`src/tgsc_kd`, `model/python`, the F1/F2 configs, `scripts/imagenet_fix/{common,precompute_cache,verify_cache,train_ddp}.py`)
plus the teacher file; `epkd_colab/` = the single-accelerator pipeline (TPU v6e via `torch_tpu`, or one CUDA GPU);
`colab/` = VM/CLI helpers; `tests/`, `bench/`, `tools/`.  Measured numbers: see the final report (`REPORT.md`).

## 0. One-time local preparation
```bash
cd /media/e12056/ASi/TGSC/imagenet_colab
colab/ctl.sh new fable-run tpu          # retries every 4 min up to 2 h ("Service Unavailable" is common); A100: ... a100
colab/ctl.sh push fable-run             # code bundle (incl. teacher .pth), HF token file -> /root/.cache/huggingface/token, VM dirs
```
The data live at `/media/e12056/ASi/imagenet/data` on the VM too (real directory on the /content disk), so the cache keys
are the same as on this PC: F1 `imagenet_aa509b015195320b`, F2 `imagenet_b880798ac8919627`.

## 1. Launch the full run (detached on the VM; idempotent, resumable)
```bash
colab/ctl.sh run fable-run pipeline 'python3 colab/pipeline.py --device tpu --work /content/work'
#  A100 fallback:  ... --device cuda:0 --dtype fp16 --compile-scope model --decode-procs 2 --pre-workers 10
```
Phases (each skipped when its marker exists): download (294+14 shards, `download_status.json`) -> precompute
(`epkd_colab.precompute --claim`, chunk markers `chunks_done/cXXXXX.done`) -> `--finalize` -> `tools/verify_cache.py --n 60`
(must be ALL_OK, else the pipeline stops) -> F1 (`runs/F1_.../DONE_EPOCH` = 80) -> F2 (`--init-student-from runs/F1/epoch_080.pt`,
DONE_EPOCH = 100).  Training auto-resumes from `latest.pt`; a crash restarts the trainer after 60 s (max 20 times).

## 2. Monitor
```bash
colab/ctl.sh log fable-run pipeline 30                    # phase log
colab/ctl.sh sh fable-run 'cat /content/work/logs/pipeline_status.json | head -40'
colab/ctl.sh log fable-run F1_rn34_rn18_epkd_e001-080 20   # "[e3 s1200] 10345 rows/s | loss ..." and "Epoch N/80 | ... | val top1 ..."
colab/ctl.sh sh fable-run 'cat /content/work/runs/F1_rn34_rn18_epkd_e001-080/DONE_EPOCH; tail -c 2000 /content/work/logs/precompute.log'
colab/ctl.sh ps fable-run
timeout 60 colab usage
```
ETA: epochs 1-20 at 112 px, 21-40 at 140, 41-60 at 168, 61-80 at 196, 81-100 at 224; per-epoch times are printed and logged
(`runs/<run>/logs/*.json`, event "epoch": train_secs, eval_secs, rows_per_sec).

## 3. Persist (every ~hour, and before stopping): latest checkpoint + run logs + cache indexes -> this PC
```bash
colab/ctl.sh sync fable-run F1_rn34_rn18_epkd_e001-080 /media/e12056/ASi/TGSC/imagenet_colab/pulled
colab/ctl.sh sync fable-run F2_rn34_rn18_epkd_e081-100_cosine /media/e12056/ASi/TGSC/imagenet_colab/pulled
colab/ctl.sh pull fable-run /content/work/runs/F1_rn34_rn18_epkd_e001-080/epoch_080.pt pulled/runs/F1_rn34_rn18_epkd_e001-080/epoch_080.pt
```
`sync` pulls `latest.pt`, `best.pt`, `student_best.pt`, `DONE_EPOCH`, `logs/` of the run and a tarball of the cache
(`stage_res*.parquet`, `precompute_state.json`, manifests, chunk markers -- everything the trainer needs, ~0.5 GB), so a lost
VM costs at most one epoch and no re-precompute.

## 4. Resume on a fresh VM (TPU or A100)
```bash
colab/ctl.sh new fable-run2 tpu && colab/ctl.sh push fable-run2
# restore the cache and the latest checkpoint(s)
timeout 1800 colab upload -s fable-run2 pulled/cache/cache_indexes.tgz /content/cache_indexes.tgz
colab/ctl.sh sh fable-run2 'mkdir -p /content/work/cache && cd /content/work/cache && tar xzf /content/cache_indexes.tgz && ls imagenet_tgsc'
colab/ctl.sh sh fable-run2 'mkdir -p /content/work/runs/F1_rn34_rn18_epkd_e001-080'
timeout 1800 colab upload -s fable-run2 pulled/runs/F1_rn34_rn18_epkd_e001-080/latest.pt /content/work/runs/F1_rn34_rn18_epkd_e001-080/latest.pt
# (for F2 also upload F1/epoch_080.pt and F2/latest.pt)
colab/ctl.sh run fable-run2 pipeline 'python3 colab/pipeline.py --device tpu --work /content/work'   # re-downloads the shards, skips precompute (markers + verified json)
```
The verify json (`/content/work/logs/verify_F.json`) is part of the pulled logs; copy it back to `/content/work/logs/` to skip
re-verification, or let the pipeline re-verify (~2 min).

## 5. Stop
```bash
colab/ctl.sh sh fable-run 'touch /content/work/STOP'     # trainer finishes the current epoch (checkpoint), pipeline exits
colab/ctl.sh sync ... ; colab/ctl.sh stop fable-run
```
