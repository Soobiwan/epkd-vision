#!/bin/bash
# TPU session 4 (DETACHED on the VM): (1) minimal in-place/optimizer repro on torch_tpu, (2) hybrid precompute on
# 8,192 real images + CPU-reference verify.  Logs /content/work/logs/s4_*.log, marker TPU_DONE.
cd /content/imagenet_colab || exit 1
export PYTHONUNBUFFERED=1 EPKD_IMAGENET_DIR=/media/e12056/ASi/imagenet/data EPKD_CACHE_DIR=/content/work/cache
L=/content/work/logs; mkdir -p $L; rm -f $L/TPU_DONE
PY=python3; NP=$(nproc); CPUP=${CPUP:-$((NP - 8))}
F1=ref/configs/F1_rn34_rn18_epkd_e001-080.json; F2=ref/configs/F2_rn34_rn18_epkd_e081-100_cosine.json
echo "== s4 start $(date) nproc=$NP cpu-procs=$CPUP"
$PY colab/download_hf.py --out $EPKD_IMAGENET_DIR --train-shards 2 --val-shards 1 --workers 3 > $L/s4_download.log 2>&1 &
DL=$!
timeout 900 $PY bench/tpu_update_probe.py --device tpu > $L/s4_update.log 2>&1
echo "== update probe rc=$? $(date)"; grep -E "BN native|scope=" $L/s4_update.log
wait $DL; echo "== download done $(date)"; tail -1 $L/s4_download.log

export EPKD_MAX_TRAIN=8192 EPKD_MAX_VAL=3571
CF="$F1 $F2"
$PY -m epkd_colab.precompute_hybrid --configs $CF --chunk 8192 --prepare > $L/s4_hybrid.log 2>&1
( time timeout 2100 $PY -m epkd_colab.precompute_hybrid --configs $CF --chunk 8192 --device tpu --head-k 40 --union-batch 32 \
    --score-batch 256 --score-dtype bf16 --cpu-procs $CPUP --decode-workers 4 ) >> $L/s4_hybrid.log 2>&1
$PY -m epkd_colab.precompute_hybrid --configs $CF --chunk 8192 --finalize >> $L/s4_hybrid.log 2>&1
echo "== hybrid done $(date)"; grep -E "\[head\]|\[hybrid\]|real|Traceback|Error" $L/s4_hybrid.log | tail -8
OMP_NUM_THREADS=$NP timeout 900 $PY tools/verify_cache.py --configs $CF --n 60 --device cpu --min-box-rate 0.95 \
  --score-rtol 2e-2 --score-atol 1e-3 --out $L/s4_verify.json > $L/s4_verify.log 2>&1
echo "== verify done $(date)"; grep "\[verify\]" $L/s4_verify.log
echo "== s4 done $(date)"; echo TPU_DONE > $L/TPU_DONE
