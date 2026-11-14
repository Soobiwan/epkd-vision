#!/bin/bash
# Trainer stall watchdog (11-13): torch_tpu OOM at the 196 px stage left the trainer hung after its traceback
# (loader workers alive -> no exit -> pipeline never retried). If the newest F1/F2 log shows no progress line for
# STALL_MIN minutes AND has a Traceback after its last progress line, kill the trainer's children + trainer by PID;
# pipeline.py then auto-resumes from latest.pt.  Pattern held in variables so this script's cmdline never matches.
STALL_MIN=${STALL_MIN:-15}; L=/content/work/logs; OUT=$L/stall_watchdog.log
P1="epkd_colab"; PAT="$P1.train"
last=""; since=$(date +%s)
while true; do
  sleep 60
  F=$(ls -t $L/[EF][12]_*.log 2>/dev/null | head -1); [ -z "$F" ] && continue
  tailtxt=$(tail -c 4000000 "$F" | grep -a -n -E '^\[e[0-9]+ s|^Epoch |^\[resume\]|^Traceback')
  prog=$(echo "$tailtxt" | grep -E ':\[e[0-9]+ s|:Epoch |:\[resume\]' | tail -1 | cut -d: -f2-)
  if [ "$prog" != "$last" ]; then last=$prog; since=$(date +%s); continue; fi
  idle=$(( ($(date +%s) - since) / 60 ))
  [ $idle -lt $STALL_MIN ] && continue
  lastline=$(echo "$tailtxt" | tail -1 | cut -d: -f2-)
  case "$lastline" in Traceback*) ;; *) echo "$(date) stalled ${idle} min, no traceback after last progress (not acting)" >> $OUT; since=$(date +%s); continue;; esac
  tp=$(pgrep -f "$PAT" | head -1); [ -z "$tp" ] && continue
  kids=$(pgrep -P $tp | tr '\n' ' ')
  echo "$(date) STALL ${idle} min after traceback in $(basename $F); killing trainer $tp children $kids" >> $OUT
  kill $kids 2>/dev/null; sleep 3; kill $tp 2>/dev/null; sleep 30
  kill -0 $tp 2>/dev/null && { kill -9 $kids $tp 2>/dev/null; echo "$(date) SIGKILL $tp" >> $OUT; }
  since=$(date +%s)
done
