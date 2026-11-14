#!/bin/bash
# VM-side telemetry + memory guard (started by full_session.sh).  Every 30 s one line in /content/work/logs/mem.log:
#   time | avail/total MB | /content disk use | load | top-3 RSS processes.
# Guard: when MemAvailable < GUARD_PCT % of RAM, SIGKILL the largest-RSS python process that is not the
# pipeline itself (precompute_hybrid / train / its pool workers) -> pipeline.py retries from the on-disk markers
# instead of Colab killing the whole runtime.  A MEMGUARD line is logged.
L=/content/work/logs; mkdir -p $L; GUARD_PCT=${GUARD_PCT:-6}; INT=${WATCH_INT:-10}
while true; do
  read -r tot avail < <(awk '/MemTotal/{t=$2}/MemAvailable/{a=$2}END{print int(t/1024), int(a/1024)}' /proc/meminfo)
  top=$(ps -eo rss=,pid=,args= --sort=-rss | head -3 | awk '{printf "%dM:%s:%s %s | ", $1/1024, $2, $3, $5}')
  echo "$(date +%H:%M:%S) mem ${avail}/${tot}MB disk $(df -h /content | awk 'NR==2{print $5}') load $(cut -d' ' -f1-4 /proc/loadavg) procs $(ps -e --no-headers | wc -l) swap $(awk '/SwapFree/{s=$2}END{print int(s/1024)}' /proc/meminfo) $(dmesg 2>/dev/null | grep -iE 'oom|killed process' | tail -1 | cut -c1-80) | $top" >> $L/mem.log
  if [ "$avail" -lt $(( tot * GUARD_PCT / 100 )) ]; then
    victim=$(ps -eo rss=,pid=,args= --sort=-rss | awk '/python/ && !/pipeline\.py/ && !/vm_watch/ {print $2; exit}')
    echo "$(date +%H:%M:%S) MEMGUARD avail ${avail}MB < ${GUARD_PCT}% -> kill $victim ($(ps -o args= -p $victim | cut -c1-120))" >> $L/mem.log
    [ -n "$victim" ] && kill -9 "$victim"
  fi
  sleep $INT
done
