#!/bin/bash
# TPU session 2 (run DETACHED on the VM): (1) Sonnet's H2D/probe fixes, (2) hybrid precompute speed + verify,
# (3) end-to-end train at 140/196.  Logs /content/work/logs/s2_*.log, marker TPU_DONE.  Every phase has a timeout.
cd /content/imagenet_colab || exit 1
export PYTHONUNBUFFERED=1 EPKD_IMAGENET_DIR=/media/e12056/ASi/imagenet/data EPKD_CACHE_DIR=/content/work/cache
L=/content/work/logs; mkdir -p $L; rm -f $L/TPU_DONE
PY=python3; NP=$(nproc); DP=${DP:-4}; CPUP=${CPUP:-$((NP - 8))}
F1=ref/configs/F1_rn34_rn18_epkd_e001-080.json; F2=ref/configs/F2_rn34_rn18_epkd_e081-100_cosine.json
echo "== s2 start $(date) nproc=$NP cpu-procs=$CPUP"
$PY colab/download_hf.py --out $EPKD_IMAGENET_DIR --train-shards 4 --val-shards 1 --workers 4 > $L/s2_download.log 2>&1 &
DL=$!

# 1. H2D / weight-update fixes (Sonnet): probe at all crops, scopes step + fwdbwd; step equivalence at crop 140
timeout 1200 bash bench/tpu_check_fixes.sh > $L/s2_check_fixes.log 2>&1
echo "== check_fixes rc=$? $(date)"; grep -E "UPDATES_|ERROR|EQUIVALENCE" $L/s2_check_fixes.log $L/check_*.log 2>/dev/null | tail -16
wait $DL; echo "== download done $(date)"; tail -1 $L/s2_download.log

# 2. hybrid precompute on 8,192 real images (one chunk), 40 head shapes on the TPU, the rest on the CPU pool
export EPKD_MAX_TRAIN=8192 EPKD_MAX_VAL=3571
CF="$F1 $F2"
$PY -m epkd_colab.precompute_hybrid --configs $CF --chunk 8192 --prepare > $L/s2_hybrid.log 2>&1
( time timeout 2400 $PY -m epkd_colab.precompute_hybrid --configs $CF --chunk 8192 --device tpu --head-k 40 --union-batch 32 \
    --score-batch 256 --score-dtype bf16 --cpu-procs $CPUP --decode-workers 4 ) >> $L/s2_hybrid.log 2>&1
$PY -m epkd_colab.precompute_hybrid --configs $CF --chunk 8192 --finalize >> $L/s2_hybrid.log 2>&1
echo "== hybrid done $(date)"; grep -E "\[head\]|\[hybrid\]|real|Traceback|Error" $L/s2_hybrid.log | tail -6
# CPU reference check of the TPU-built cache (tolerant: near-tie swaps allowed, matched boxes must keep their scores
# within 2e-3; bf16 scoring -> scores are expected to differ more, so the score tolerance is reported, not gated)
OMP_NUM_THREADS=$NP timeout 1200 $PY tools/verify_cache.py --configs $CF --n 60 --device cpu --min-box-rate 0.95 \
  --score-rtol 2e-2 --score-atol 1e-3 --out $L/s2_verify.json > $L/s2_verify.log 2>&1
echo "== verify done $(date)"; grep "\[verify\]" $L/s2_verify.log

# 3. end-to-end train at the two crops that crashed before (fake stage index, 100 steps)
export EPKD_MAX_TRAIN=17432 EPKD_CHUNK=1024
$PY tools/make_fake_stage_index.py --n 17432 --out /content/work/cache/fake > $L/s2_fake.log 2>&1
for S in 140 196; do
  E=$(( (S - 112) / 28 * 20 + 1 ))
  timeout 900 $PY -m epkd_colab.train --config $F1 --device tpu --dtype bf16 --compile-scope step --max-steps 100 --skip-eval --no-save \
    --decode-procs $DP --start-epoch $E --stage-index-override /content/work/cache/fake/stage_res{res}.parquet \
    --run-dir /content/work/runs/s2_e2e_$S --progress-every 25 --seed 27 > $L/s2_e2e_$S.log 2>&1
  echo "-- e2e $S"; grep -E "^\[e|Traceback|Error" $L/s2_e2e_$S.log | tail -3
done
echo "== s2 done $(date)"; echo TPU_DONE > $L/TPU_DONE
