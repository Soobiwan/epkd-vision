#!/bin/bash
# TPU validation + throughput session, run DETACHED on a Colab TPU v6e1 VM.  Logs: /content/work/logs/tpu_*.log,
# marker TPU_DONE at the end.  CPU-side checks (verify vs the reference, CPU fp32 loss trace) run on the local PC afterwards,
# so nothing CPU-heavy runs here.  Hard wall-clock cap per phase via `timeout` so the session cannot run away.
cd /content/imagenet_colab || exit 1
export PYTHONUNBUFFERED=1 EPKD_IMAGENET_DIR=/media/e12056/ASi/imagenet/data EPKD_CACHE_DIR=/content/work/cache
L=/content/work/logs; mkdir -p $L
PY=python3; DEV=tpu; CS=step; DT=bf16; DP=${DP:-6}
F1=ref/configs/F1_rn34_rn18_epkd_e001-080.json; F2=ref/configs/F2_rn34_rn18_epkd_e081-100_cosine.json
echo "== tpu session start $(date) nproc=$(nproc)"

# 1. data in the background: 4 train shards (17,432 images) + 1 val shard
$PY colab/download_hf.py --out $EPKD_IMAGENET_DIR --train-shards 4 --val-shards 1 --workers 4 > $L/tpu_download.log 2>&1 &
DL=$!

# 2. synthetic probe of the compiled step at every stage (+ weight-update / NaN checks) and precompute ops on the device
timeout 900 $PY bench/tpu_probe3.py --device $DEV --dtype $DT --compile-scope $CS --crops 112,140,168,196,224 --steps 20 > $L/tpu_probe3.log 2>&1
echo "== probe3 rc=$? $(date)"
# 3. one-step equivalence vs the reference loop (CPU fp32 reference, one step only)
timeout 600 $PY tests/test_step_equivalence.py --device $DEV --dtype $DT --compile-scope $CS --crop 112 --anchor 224 --groups 4 --slot 8 > $L/tpu_step_eq_bf16.log 2>&1
timeout 600 $PY tests/test_step_equivalence.py --device $DEV --dtype fp32 --compile-scope none --crop 112 --anchor 224 --groups 4 --slot 8 > $L/tpu_step_eq_fp32.log 2>&1
echo "== step equivalence done $(date)"; tail -1 $L/tpu_step_eq_bf16.log; tail -1 $L/tpu_step_eq_fp32.log
wait $DL; echo "== download done $(date)"; tail -2 $L/tpu_download.log

# 4. precompute on the TPU (eager, fp32 'highest') over 3,000 real images (many distinct shapes), 1,000-image chunks
#    -> per-chunk img/s shows the warm-up cost of new shapes; the cache is pulled and verified on the local CPU.
export EPKD_MAX_TRAIN=3000 EPKD_MAX_VAL=3571 EPKD_CHUNK=1000
CF="$F1 $F2"
$PY -m epkd_colab.precompute --configs $CF --device $DEV --prepare > $L/tpu_precompute.log 2>&1
( time timeout 1500 $PY -m epkd_colab.precompute --configs $CF --device $DEV --batch 32 --workers 16 --log-every 200 ) >> $L/tpu_precompute.log 2>&1
$PY -m epkd_colab.precompute --configs $CF --device cpu --finalize >> $L/tpu_precompute.log 2>&1
$PY -m epkd_colab.precompute --configs $CF --device cpu --print-roots > $L/tpu_cache_roots.txt 2>&1
echo "== precompute done $(date)"; grep -E "chunk|real" $L/tpu_precompute.log | tail -5

# 5. throughput on 17,432 images with a synthetic stage index (box positions do not affect speed)
export EPKD_MAX_TRAIN=17432 EPKD_CHUNK=1024
$PY tools/make_fake_stage_index.py --n 17432 --out /content/work/cache/fake > $L/tpu_fake.log 2>&1
timeout 300 $PY bench/bench_loader.py --config $F1 --crop 112 --steps 100 --decode-procs $DP --stage-index /content/work/cache/fake/stage_res112.parquet > $L/tpu_loader_112.log 2>&1
timeout 300 $PY bench/bench_loader.py --config $F2 --crop 224 --steps 100 --decode-procs $DP --stage-index /content/work/cache/fake/stage_res224.parquet > $L/tpu_loader_224.log 2>&1
echo "== loader bench done $(date)"; tail -1 $L/tpu_loader_112.log; tail -1 $L/tpu_loader_224.log
for S in 112 140 168 196; do
  E=$(( (S - 112) / 28 * 20 + 1 ))
  timeout 900 $PY -m epkd_colab.train --config $F1 --device $DEV --dtype $DT --compile-scope $CS --max-steps 150 --skip-eval --no-save \
    --decode-procs $DP --start-epoch $E --stage-index-override /content/work/cache/fake/stage_res{res}.parquet \
    --run-dir /content/work/runs/tpu_e2e_$S --progress-every 25 --seed 27 > $L/tpu_e2e_$S.log 2>&1
  echo "-- e2e $S"; grep -E "^\[e|Traceback|Error" $L/tpu_e2e_$S.log | tail -3
done
timeout 900 $PY -m epkd_colab.train --config $F2 --device $DEV --dtype $DT --compile-scope $CS --max-steps 150 --skip-eval --no-save \
  --decode-procs $DP --stage-index-override /content/work/cache/fake/stage_res{res}.parquet --run-dir /content/work/runs/tpu_e2e_224 \
  --progress-every 25 > $L/tpu_e2e_224.log 2>&1
echo "-- e2e 224"; grep -E "^\[e|Traceback|Error" $L/tpu_e2e_224.log | tail -3
echo "== e2e done $(date)"

# 6. 100-step loss trace on the TPU (seed 27, same data order as the local CPU fp32 trace) + eval timing on 1 val shard
timeout 1200 $PY -m epkd_colab.train --config $F1 --device $DEV --dtype $DT --compile-scope $CS --max-steps 100 --no-save --decode-procs $DP \
  --stage-index-override /content/work/cache/fake/stage_res{res}.parquet --run-dir /content/work/runs/tpu_trace --progress-every 25 \
  --seed 27 --val-cache /content/work/cache/val_u8.npy --loss-trace $L/trace_tpu_bf16.json > $L/tpu_trace.log 2>&1
grep -E "^\[e|Epoch|Traceback|Error|val-cache" $L/tpu_trace.log | tail -5
echo "== tpu session done $(date)"; echo TPU_DONE > $L/TPU_DONE
