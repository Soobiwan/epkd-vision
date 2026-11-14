#!/bin/bash
# Local driver for the TPU validation session with a HARD cost cap: new (retry) -> push -> run dev_session_tpu.sh
# detached -> poll -> pull logs + dev cache -> ALWAYS stop the VM.  Status: logs/drive_tpu_dev.log (local).
#   setsid nohup colab/drive_tpu_dev.sh > logs/drive_tpu_dev.log 2>&1 < /dev/null &
NAME=${NAME:-epkd-tpu-dev}
CAP_MIN=${CAP_MIN:-75}
ACCEL=${ACCEL:-tpu}            # tpu (= v6e1) | v5e
DP=${DP:-6}                      # PIL decode procs per logical worker (6 workers): v6e 44 vCPU -> 6, v5e 24 vCPU -> 4
SCRIPT=${SCRIPT:-colab/dev_session_tpu.sh}  # session script run detached on the VM
HERE=$(cd "$(dirname "$0")" && pwd); ROOT=$(cd "$HERE/.." && pwd)
CTL="$HERE/ctl.sh"; OUT="$ROOT/tpu_dev_out"; mkdir -p "$OUT"
cleanup() {
  echo "== pulling results $(date)"
  "$CTL" sh "$NAME" 'cd /content/work && tar czf /content/tpu_out.tgz logs' >/dev/null 2>&1
  "$CTL" pull "$NAME" /content/tpu_out.tgz "$OUT/${NAME}_out.tgz"
  echo "== stopping $NAME $(date)"; "$CTL" stop "$NAME"
}
"$CTL" new "$NAME" "$ACCEL" || { echo "no TPU after retries"; exit 1; }
trap cleanup EXIT
"$CTL" push "$NAME"
"$CTL" run "$NAME" tpu_session "DP=$DP bash $SCRIPT"
T0=$(date +%s)
while true; do
  sleep 180
  el=$(( ($(date +%s) - T0) / 60 ))
  st=$("$CTL" sh "$NAME" 'test -f /content/work/logs/TPU_DONE && echo DONE; grep "^==" /content/work/logs/tpu_session.log | tail -1')
  echo "[$el min] $st"
  echo "$st" | grep -q DONE && break
  [ "$el" -ge "$CAP_MIN" ] && { echo "== CAP reached ($CAP_MIN min)"; break; }
done
