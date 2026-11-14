#!/bin/bash
# A/B of the TPU training loop on real data (F1 stage 112, 300 steps each).  Logs /content/work/logs/ab_*.log
cd /content/imagenet_colab || exit 1
export PYTHONUNBUFFERED=1 EPKD_CACHE_DIR=/content/work/cache
L=/content/work/logs; F1=ref/configs/F1_rn34_rn18_epkd_e001-080.json
run() {  # tag, env, extra args
  local tag=$1 envs=$2; shift 2
  env $envs timeout 900 python3 -m epkd_colab.train --config $F1 --device tpu --compile-scope model --max-steps 300 --skip-eval \
    --no-save --decode-procs 6 --progress-every 20 --run-dir /content/ab_$tag "$@" > $L/ab_$tag.log 2>&1
  echo "== $tag rc=$? $(grep -E '^\[e1 s(20|100|200|300)\]' $L/ab_$tag.log | awk '{print $2, $6, $7}' | tr '\n' ' ')"
}
run A    "EPKD_H2D=bt"   --dtype bf16
run Async "EPKD_H2D=bt"  --dtype bf16 --sync-every 1
run B    "EPKD_H2D=copy" --dtype bf16 --async-h2d 0
run C    "EPKD_H2D=copy" --dtype fp32 --async-h2d 0
echo AB_DONE
