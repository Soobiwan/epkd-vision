#!/bin/bash
# Real-data training speed/robustness bench on the TPU (training must be stopped).  Logs /content/work/logs/pb_*.log
#  part 1: prefetch depth at the current stage (140 + 224 anchors, epoch 26), 500 steps each, loss checked
#  part 2: 200-step no-crash checks of every later stage (140/168/196 + anchors, F2 224 without anchors)
#  then: writes the best prefetch to /content/work/PREFETCH and prints PB_DONE
cd /content/imagenet_colab || exit 1
export PYTHONUNBUFFERED=1 EPKD_CACHE_DIR=/content/work/cache EPKD_H2D=$(cat /content/work/H2D_METHOD)
TCM=/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4; [ -f $TCM ] && export LD_PRELOAD=$TCM
L=/content/work/logs; F1=ref/configs/F1_rn34_rn18_epkd_e001-080.json; F2=ref/configs/F2_rn34_rn18_epkd_e081-100_cosine.json
( while true; do echo "$(date +%T) $(timeout 20 tpu-info 2>&1 | tr '│┃' '||' | grep -E '^\| 0 ' | tr -s ' ' | tr '\n' ' ')"; sleep 15; done ) > $L/pb_tpuinfo.log 2>&1 &
TI=$!
run() {  # tag steps cfg extra...
  local tag=$1 steps=$2 cfg=$3; shift 3
  rm -rf /content/pb_$tag
  timeout 1200 python3 -m epkd_colab.train --config $cfg --device tpu --dtype bf16 --compile-scope model --max-steps $steps --skip-eval \
    --no-save --decode-procs 6 --progress-every 100 --run-dir /content/pb_$tag "$@" > $L/pb_$tag.log 2>&1
  local rc=$?
  local r=$(grep -E "^\[e[0-9]+ s[0-9]+\]" $L/pb_$tag.log | awk 'NR>2{print $3}' | tr '\n' ' ')
  echo "== $tag rc=$rc rows/s(after warmup): $r | loss: $(grep -E '^\[e[0-9]+ s[0-9]+\]' $L/pb_$tag.log | awk '{print $7}' | tr '\n' ' ') $(grep -m1 -E 'Error|Traceback' $L/pb_$tag.log)"
}
echo "== bench start $(date)"
run p1   500 $F1 --async-h2d 1 --prefetch 1 --start-epoch 26
run p2   500 $F1 --async-h2d 1 --prefetch 2 --start-epoch 26
run p3   500 $F1 --async-h2d 1 --prefetch 3 --start-epoch 26
run sync 300 $F1 --async-h2d 0 --start-epoch 26
best=$(for t in p1 p2 p3; do v=$(grep -E "^\[e26 s(300|400|500)\]" $L/pb_$t.log | awk '{s+=$3;n++} END{if(n) printf "%d", s/n; else print 0}'); l2=$(grep -E "^\[e26 s200\]" $L/pb_$t.log | awk '{print $7}'); ok=$(awk -v l="$l2" 'BEGIN{print (l>9.0 && l<14.0)?1:0}'); [ "$ok" = 1 ] && echo "$v ${t#p}"; done | sort -n | tail -1 | awk '{print $2}')
echo "${best:-1}" > /content/work/PREFETCH; echo "== best prefetch ${best:-1}"
P=$(cat /content/work/PREFETCH)
run s168 200 $F1 --async-h2d 1 --prefetch $P --start-epoch 41
run s196 200 $F1 --async-h2d 1 --prefetch $P --start-epoch 61
run f224 200 $F2 --async-h2d 1 --prefetch $P
kill $TI 2>/dev/null
echo "PB_DONE $(date)"
