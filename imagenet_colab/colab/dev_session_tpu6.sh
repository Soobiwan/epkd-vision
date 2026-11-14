#!/bin/bash
# TPU session 6 (DETACHED on the VM): smoke test of the full pipeline on a subset (2 train shards, 8,192 images in two
# 4,096-image chunks): download -> hybrid precompute (+precompile, finalize, tolerant verify) -> F1 start (60 steps, val cache).
cd /content/imagenet_colab || exit 1
export PYTHONUNBUFFERED=1 EPKD_MAX_TRAIN=8192 EPKD_MAX_VAL=3571
L=/content/work/logs; mkdir -p $L; rm -f $L/TPU_DONE
echo "== s6 start $(date) nproc=$(nproc)"
timeout 2700 python3 colab/pipeline.py --device tpu --work /content/work --train-shards 2 --val-shards 1 --pre-chunk 4096 \
  --head-k 20 --train-args "--max-steps 60" --max-failures 0 > $L/s6_pipeline.log 2>&1
echo "== pipeline rc=$? $(date)"; grep -v "^\s*$" $L/s6_pipeline.log | tail -12
grep -E "\[precompile\] done|\[hybrid\] chunk [0-9]+ \[|Traceback|Error" $L/precompute.log | tail -6
grep "\[verify\]" $L/verify.log | tail -6
grep -E "^\[e|Epoch|val|Traceback|Error" $L/F1_rn34_rn18_epkd_e001-080.log | tail -6
echo "== s6 done $(date)"; echo TPU_DONE > $L/TPU_DONE
