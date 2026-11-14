#!/bin/bash
# Dev/benchmark sequence run DETACHED on the Colab VM (TPU or CUDA).  usage: dev_session_vm.sh tpu|cuda:0
# Logs in /content/work/logs/dev_*.log; "DEV_DONE" marker at the end.
DEV=${1:-tpu}
cd /content/imagenet_colab || exit 1
export PYTHONUNBUFFERED=1 EPKD_IMAGENET_DIR=/media/e12056/ASi/imagenet/data EPKD_CACHE_DIR=/content/work/cache
L=/content/work/logs
PY=python3
if [ "$DEV" = tpu ]; then CS=step; DT=bf16; DP=6; else CS=model; DT=fp16; DP=2; fi
echo "== dev session $DEV start $(date)"

# 1. data: 4 train shards (17,432 images) + 1 val shard (3,571) in the background
$PY colab/download_hf.py --out $EPKD_IMAGENET_DIR --train-shards 4 --val-shards 1 --workers 4 > $L/dev_download.log 2>&1 &
DL=$!

# 2. synthetic probe of THIS pipeline's step (compile, correctness of weight update, rows/s) + precompute ops
$PY bench/tpu_probe3.py --device $DEV --dtype $DT --compile-scope $CS --crops 112,140,168,196,224 --steps 20 > $L/dev_probe3.log 2>&1
echo "== probe3 done $(date)"
# 3. one-step equivalence vs the reference loop (CPU fp32 reference)
$PY tests/test_step_equivalence.py --device $DEV --dtype $DT --compile-scope $CS --crop 112 --anchor 224 --groups 4 --slot 8 > $L/dev_step_eq_$DT.log 2>&1
$PY tests/test_step_equivalence.py --device $DEV --dtype fp32 --compile-scope none --crop 112 --anchor 224 --groups 4 --slot 8 > $L/dev_step_eq_fp32.log 2>&1
echo "== step equivalence done $(date)"
wait $DL
echo "== download done $(date)"; tail -2 $L/dev_download.log

# 4. real dev cache on 1,000 images (chunk 500): batched precompute on the device, then the reference verify (60 ids)
export EPKD_MAX_TRAIN=1000 EPKD_MAX_VAL=3571 EPKD_CHUNK=500
CF="ref/configs/F1_rn34_rn18_epkd_e001-080.json ref/configs/F2_rn34_rn18_epkd_e081-100_cosine.json"
$PY -m epkd_colab.precompute --configs $CF --device $DEV --prepare > $L/dev_precompute.log 2>&1
( time $PY -m epkd_colab.precompute --configs $CF --device $DEV --batch 32 --workers 16 ) >> $L/dev_precompute.log 2>&1
if ! grep -q "chunk 1 " $L/dev_precompute.log; then
  echo "== device precompute failed -> CPU fallback" >> $L/dev_precompute.log
  ( time $PY -m epkd_colab.precompute --configs $CF --device cpu --batch 32 --workers 16 ) >> $L/dev_precompute.log 2>&1
fi
$PY -m epkd_colab.precompute --configs $CF --device cpu --finalize >> $L/dev_precompute.log 2>&1
( time $PY tools/verify_cache.py --configs $CF --n 60 --device $DEV --out $L/dev_verify_$DEV.json ) > $L/dev_verify.log 2>&1 || \
( time $PY tools/verify_cache.py --configs $CF --n 60 --device cpu --out $L/dev_verify_cpu.json ) >> $L/dev_verify.log 2>&1
echo "== precompute+verify done $(date)"; tail -8 $L/dev_verify.log

# 5. throughput on 17,432 images with a synthetic stage index (box positions do not affect speed)
export EPKD_MAX_TRAIN=17432 EPKD_CHUNK=1024
$PY tools/make_fake_stage_index.py --n 17432 --out /content/work/cache/fake > $L/dev_fake.log 2>&1
F1=ref/configs/F1_rn34_rn18_epkd_e001-080.json; F2=ref/configs/F2_rn34_rn18_epkd_e081-100_cosine.json
# 5a. loader-only ceiling (CPU decode + crops), 112 (3 rows/img + anchors) and 224
$PY bench/bench_loader.py --config $F1 --crop 112 --steps 60 --decode-procs $DP --stage-index /content/work/cache/fake/stage_res112.parquet > $L/dev_loader_112.log 2>&1
$PY bench/bench_loader.py --config $F2 --crop 224 --steps 60 --decode-procs $DP --stage-index /content/work/cache/fake/stage_res224.parquet > $L/dev_loader_224.log 2>&1
echo "== loader bench done $(date)"; tail -1 $L/dev_loader_112.log; tail -1 $L/dev_loader_224.log
# 5b. 200-step loss trace on CPU fp32 (background, reference numerics) for the device-vs-CPU curve comparison
OMP_NUM_THREADS=24 $PY -m epkd_colab.train --config $F1 --device cpu --dtype fp32 --compile-scope none --max-steps 200 --skip-eval --no-save \
  --stage-index-override /content/work/cache/fake/stage_res{res}.parquet --run-dir /content/work/runs/dev_cpu_trace --decode-procs 2 \
  --progress-every 20 --loss-trace $L/trace_cpu_fp32.json > $L/dev_trace_cpu.log 2>&1 &
TR=$!
# 5c. end-to-end rows/s per stage (real loader + compiled step), 60 steps each
for S in 112 140 168 196; do
  $PY -m epkd_colab.train --config $F1 --device $DEV --dtype $DT --compile-scope $CS --max-steps 60 --skip-eval --no-save --decode-procs $DP \
    --stage-index-override /content/work/cache/fake/stage_res{res}.parquet --run-dir /content/work/runs/dev_e2e_$S --progress-every 20 \
    --seed $((27 + S)) > $L/dev_e2e_$S.log 2>&1
  grep -E "^\[e|Epoch|Traceback|Error" $L/dev_e2e_$S.log | tail -4
done
$PY -m epkd_colab.train --config $F2 --device $DEV --dtype $DT --compile-scope $CS --max-steps 60 --skip-eval --no-save --decode-procs $DP \
  --stage-index-override /content/work/cache/fake/stage_res{res}.parquet --run-dir /content/work/runs/dev_e2e_224 --progress-every 20 > $L/dev_e2e_224.log 2>&1
grep -E "^\[e|Epoch|Traceback|Error" $L/dev_e2e_224.log | tail -4
echo "== e2e done $(date)"
# 6. device 200-step loss trace (same seed / data order as 5b) + eval timing with the cached val crops (1 shard)
$PY -m epkd_colab.train --config $F1 --device $DEV --dtype $DT --compile-scope $CS --max-steps 200 --no-save --decode-procs $DP \
  --stage-index-override /content/work/cache/fake/stage_res{res}.parquet --run-dir /content/work/runs/dev_trace_$DEV --progress-every 50 \
  --loss-trace $L/trace_${DEV}_$DT.json > $L/dev_trace_$DEV.log 2>&1
grep -E "^\[e|Epoch|Traceback|Error|val-cache" $L/dev_trace_$DEV.log | tail -5
wait $TR
$PY bench/compare_trace.py $L/trace_cpu_fp32.json $L/trace_${DEV}_$DT.json --names cpu_fp32,${DEV}_$DT > $L/dev_trace_compare.log 2>&1
cat $L/dev_trace_compare.log
echo "== dev session done $(date)"; echo DEV_DONE > $L/DEV_DONE
