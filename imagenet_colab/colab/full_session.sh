#!/bin/bash
# Full R34->R18 EP-KD run, DETACHED on the VM (started by colab/drive_full.sh).  State in /content/work/logs/SESSION_STATE.
#  A. (first VM only, while the full ImageNet download runs in the background and the TPU would otherwise idle)
#     smoke of the real pipeline on 4,096 images in a separate work/data dir: precompute -> verify -> F1 epoch 1
#     (eval + checkpoint) -> restart -> resume into epoch 2.  Pass = DONE_EPOCH 2.
#  B. the full pipeline (download -> precompute -> verify -> F1 -> F2); every phase resumes from what is on disk
#     (a restored cache / latest.pt from the local backup is picked up as-is).
cd /content/imagenet_colab || exit 1
export PYTHONUNBUFFERED=1
# torch_tpu warns "tcmalloc is not loaded; TorchTPU host performance may be degraded due to glibc ptmalloc
# contention": preload it for every process of the session (trainer, loader workers, precompute)
TCM=/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4
[ -f $TCM ] && export LD_PRELOAD=$TCM
W=/content/work; L=$W/logs; mkdir -p $L
# teacher/student pair (colab/pipeline.py PAIRS): F = ResNet34->ResNet18 (default), E = ResNet50->MobileNetV1
export EPKD_PAIR=${PAIR:-F}
case "$EPKD_PAIR" in F) N1=F1_rn34_rn18_epkd_e001-080;; E) N1=E1_rn50_mnv1_epkd_e001-080;; *) echo "unknown PAIR $EPKD_PAIR"; exit 1;; esac
# SMOKE_OK is per pair (the default pair keeps the original marker name)
SOK=$W/SMOKE_OK; [ "$EPKD_PAIR" != F ] && SOK=$W/SMOKE_OK_$EPKD_PAIR
D=/media/e12056/ASi/imagenet/data
NP=$(nproc); DP=${DP:-4}; CPUP=${CPUP:-$((NP - 14))}
state() { echo "$1" > $L/SESSION_STATE; echo "== $1 $(date)"; }
state "START nproc=$NP DP=$DP cpu-procs=$CPUP"
pgrep -f "[v]m_watch.sh" >/dev/null || { nohup bash colab/vm_watch.sh > /dev/null 2>&1 < /dev/null & }
# kills a trainer left hung after a traceback (torch_tpu OOM at stage 196, 11-13) so pipeline.py auto-resumes
pgrep -f "[s]tall_watchdog.sh" >/dev/null || { nohup bash colab/stall_watchdog.sh > /dev/null 2>&1 < /dev/null & }

nice -n 10 python3 colab/download_hf.py --out $D --workers 6 > $L/download_full.log 2>&1 &
DLPID=$!

if [ ! -f $SOK ] && [ "${SKIP_SMOKE:-0}" != 1 ]; then
  state SMOKE
  rm -rf /content/smoke /content/smoke_data
  EPKD_MAX_TRAIN=4096 EPKD_MAX_VAL=3571 timeout 3600 nice -n 10 python3 colab/pipeline.py --device tpu --work /content/smoke \
    --data /content/smoke_data --train-shards 1 --val-shards 1 --pre-chunk 4096 --head-k 4 --cpu-procs $CPUP \
    --decode-procs $DP --train-args "--max-steps 40" --max-failures 1 > $L/smoke_pipeline.log 2>&1
  R=/content/smoke/runs/$N1
  grep -hE "^Epoch|\[resume\]|Traceback|Error" /content/smoke/logs/$N1.log 2>/dev/null | tail -6
  grep -h "\[verify\]" /content/smoke/logs/verify.log 2>/dev/null | tail -6
  if [ "$(cat $R/DONE_EPOCH 2>/dev/null)" = 2 ] && grep -q "\[resume\]" /content/smoke/logs/$N1.log; then
    touch $SOK; state SMOKE_OK
  else
    state SMOKE_FAIL; kill $DLPID 2>/dev/null; exit 1
  fi
fi

# upload-method benchmark while the download runs (the TPU is idle now): speed + correctness per method.  Unless
# $W/H2D_METHOD pins one, pick the fastest CORRECT fresh-tensor method at crop 196 (to / i32; fresh tensors cannot
# alias a buffer a queued step still reads, so the next batch may upload during compute: --async-h2d 1);
# fallback = copy (validated 11-11) without async.
if [ ! -f $W/H2D_METHOD ]; then
  state H2D_BENCH
  timeout 900 python3 bench/h2d_bench.py > $L/h2d_bench.log 2>&1
  grep RESULT $L/h2d_bench.log
  # 11-14: prefer i32 whenever it is correct -- on a fresh VM `to` and i32 tie (16.6 vs 17.2 ms), but `to` degrades as host
  # memory fragments over hours (E1 epochs 8-9: TPU duty 97% -> 65%, ~3,000 -> ~2,050 rows/s); i32 held up on 11-12.
  best=$(grep -qE "^RESULT crop 196 i32 +correct True" $L/h2d_bench.log && echo i32)
  [ -z "$best" ] && best=$(grep -E "^RESULT crop 196 (to|i32) +correct True" $L/h2d_bench.log | sort -k7 -n | head -1 | awk '{print $4}')
  echo "${best:-copy}" > $W/H2D_METHOD
fi
H2D=$(cat $W/H2D_METHOD); ASYNC=0; [ "$H2D" = to ] || [ "$H2D" = i32 ] && ASYNC=1
[ -f $W/H2D_ASYNC ] && ASYNC=$(cat $W/H2D_ASYNC)
[ -f $W/PREFETCH ] && PREFETCH=$(cat $W/PREFETCH)
[ -f $W/GATHER_THREAD ] && GATHER_THREAD=$(cat $W/GATHER_THREAD)
echo "== upload method $H2D async $ASYNC prefetch ${PREFETCH:-1}"
state DOWNLOAD_WAIT; wait $DLPID; tail -1 $L/download_full.log
state "PIPELINE h2d=$H2D async=$ASYNC"
EPKD_H2D=$H2D nice -n 10 python3 colab/pipeline.py --device tpu --work $W --data $D --cpu-procs $CPUP --decode-procs $DP \
  --max-failures ${MAX_FAILURES:-4} --train-args "--async-h2d $ASYNC --prefetch ${PREFETCH:-1} --gather-thread ${GATHER_THREAD:-0}" > $L/pipeline_main.log 2>&1
rc=$?
if grep -q '"phase": "done"' $L/pipeline_status.json 2>/dev/null; then state ALL_DONE; else state "PIPELINE_EXIT rc=$rc"; fi
