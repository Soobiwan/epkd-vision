#!/bin/bash
# TPU session 5 (DETACHED on the VM): (1) end-to-end train speed with --compile-scope model (the scope whose updates are
# correct on torch_tpu), (2) hybrid precompute on 8,192 real images with progress logs.  Logs s5_*, marker TPU_DONE.
cd /content/imagenet_colab || exit 1
export PYTHONUNBUFFERED=1 EPKD_IMAGENET_DIR=/media/e12056/ASi/imagenet/data EPKD_CACHE_DIR=/content/work/cache
L=/content/work/logs; mkdir -p $L; rm -f $L/TPU_DONE
PY=python3; NP=$(nproc); CPUP=${CPUP:-$((NP - 8))}; DP=${DP:-4}
F1=ref/configs/F1_rn34_rn18_epkd_e001-080.json; F2=ref/configs/F2_rn34_rn18_epkd_e081-100_cosine.json
echo "== s5 start $(date) nproc=$NP cpu-procs=$CPUP"
$PY colab/download_hf.py --out $EPKD_IMAGENET_DIR --train-shards 4 --val-shards 1 --workers 4 > $L/s5_download.log 2>&1
echo "== download done $(date)"; tail -1 $L/s5_download.log

# 1. train speed, compile scope model, real loader, fake stage index, 120 steps at 112 and 196 and the 224 tail
export EPKD_MAX_TRAIN=17432 EPKD_CHUNK=1024
$PY tools/make_fake_stage_index.py --n 17432 --out /content/work/cache/fake > $L/s5_fake.log 2>&1
for S in 112 196; do
  E=$(( (S - 112) / 28 * 20 + 1 ))
  timeout 600 $PY -m epkd_colab.train --config $F1 --device tpu --dtype bf16 --compile-scope model --max-steps 120 --skip-eval --no-save \
    --decode-procs $DP --start-epoch $E --stage-index-override /content/work/cache/fake/stage_res{res}.parquet \
    --run-dir /content/work/runs/s5_e2e_$S --progress-every 20 --seed 27 > $L/s5_e2e_$S.log 2>&1
  echo "-- e2e $S (model scope)"; grep -E "^\[e|Traceback|Error" $L/s5_e2e_$S.log | tail -3
done
timeout 600 $PY -m epkd_colab.train --config $F2 --device tpu --dtype bf16 --compile-scope model --max-steps 120 --skip-eval --no-save \
  --decode-procs $DP --stage-index-override /content/work/cache/fake/stage_res{res}.parquet --run-dir /content/work/runs/s5_e2e_224 \
  --progress-every 20 > $L/s5_e2e_224.log 2>&1
echo "-- e2e 224 (model scope)"; grep -E "^\[e|Traceback|Error" $L/s5_e2e_224.log | tail -3

# 2. hybrid precompute on 8,192 images (40 head shapes on the TPU; post-processing + tail on the CPU pool)
export EPKD_MAX_TRAIN=8192 EPKD_MAX_VAL=3571
CF="$F1 $F2"
$PY -m epkd_colab.precompute_hybrid --configs $CF --chunk 8192 --prepare > $L/s5_hybrid.log 2>&1
( time timeout 1500 $PY -m epkd_colab.precompute_hybrid --configs $CF --chunk 8192 --device tpu --head-k 40 --union-batch 32 \
    --score-batch 256 --score-dtype bf16 --cpu-procs $CPUP --decode-workers 4 --log-every 500 ) >> $L/s5_hybrid.log 2>&1
$PY -m epkd_colab.precompute_hybrid --configs $CF --chunk 8192 --finalize >> $L/s5_hybrid.log 2>&1
echo "== hybrid done $(date)"; grep -E "\[head\]|\[hybrid\]|real|Traceback|Error" $L/s5_hybrid.log | tail -12
OMP_NUM_THREADS=$NP timeout 600 $PY tools/verify_cache.py --configs $CF --n 60 --device cpu --min-box-rate 0.95 \
  --score-rtol 2e-2 --score-atol 1e-3 --out $L/s5_verify.json > $L/s5_verify.log 2>&1
echo "== verify done $(date)"; grep "\[verify\]" $L/s5_verify.log
echo "== s5 done $(date)"; echo TPU_DONE > $L/TPU_DONE
