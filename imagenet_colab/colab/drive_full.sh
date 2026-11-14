#!/bin/bash
# Local driver for the FULL R34->R18 EP-KD run (F1 e1-80 + F2 e81-100) on a Colab TPU, across VM losses / the ~24 h cap.
#   setsid nohup colab/drive_full.sh > logs/drive_full.log 2>&1 < /dev/null &
# Per VM incarnation k: new (v6e1 for ~1 h, else v5e1) -> push code -> restore local backup (cache.tgz, state.tgz)
# -> run colab/full_session.sh detached -> every POLL s: status; after each new epoch fetch state.tgz (runs without
# epoch_*.pt except epoch_080.pt, verify json, logs); once training has started fetch the precompute cache once.
# VM unreachable 3 polls in a row -> next incarnation (auto-resume from the backup).
# Session failure (SMOKE_FAIL / PIPELINE_EXIT) -> "ALERT" line, VM kept GRACE_MIN for a fix, then stopped.
# Local controls in $BK: HOLD (never stop on failure), RESTART (re-push code and rerun the session on the same VM),
# QUIT (back up and stop).  Hard floor: stop when the balance drops below MIN_UNITS.
HERE=$(cd "$(dirname "$0")" && pwd); ROOT=$(cd "$HERE/.." && pwd); CTL="$HERE/ctl.sh"
BK=${BK:-$ROOT/backup_full}; mkdir -p "$BK"
POLL=${POLL:-600}; GRACE_MIN=${GRACE_MIN:-40}; MIN_UNITS=${MIN_UNITS:-6}; V6E_TRIES=${V6E_TRIES:-15}
W=/content/work
log() { echo "[$(date '+%m-%d %H:%M')] $*"; }
balance() { timeout 60 colab usage 2>/dev/null | awk '/Current balance/{print $3}'; }

backup_state() {  # $1 = name
  # all epoch_NNN.pt except epoch_080.pt (the F2 init) are left on the VM (GNU tar excludes match named members too)
  "$CTL" fetch "$1" $W "--exclude='*epoch_0[0-7]?.pt' --exclude='*epoch_08[1-9].pt' --exclude='*epoch_09?.pt' --exclude='*epoch_1??.pt' runs \$(ls -d SMOKE_OK* logs/verify_*.json logs/pipeline_status.json logs/[EF][12]_*.log logs/pipeline_main.log logs/precompute.log logs/verify.log 2>/dev/null)" "$BK/state.tgz.new" \
    && mv "$BK/state.tgz.new" "$BK/state.tgz" && log "state backup ok ($(du -h "$BK/state.tgz" | cut -f1))"
}
backup_cache() {  # incremental: cache files changed since the last successful increment -> $BK/cache_inc_NNN.tgz
  local n next; n=$(ls "$BK"/cache_inc_*.tgz 2>/dev/null | wc -l); next=$(printf %03d $((n + 1)))
  "$CTL" fetch "$1" $W/cache "\$(touch $W/.bk_new; if [ -f $W/.bk_last ]; then find imagenet_tgsc -type f -newer $W/.bk_last; else find imagenet_tgsc -type f; fi)" "$BK/cache_inc_$next.tgz.new" >/dev/null 2>&1 \
    && mv "$BK/cache_inc_$next.tgz.new" "$BK/cache_inc_$next.tgz" && "$CTL" sh "$1" "mv $W/.bk_new $W/.bk_last" >/dev/null \
    && log "cache increment $next ok ($(du -h "$BK/cache_inc_$next.tgz" | cut -f1))"
  rm -f "$BK"/cache_inc_*.tgz.new
}
start_session() {  # $1 name $2 accel
  local dp=4; [ "$2" = tpu ] && dp=10
  local skip=0; [ -f "$BK/SMOKE_OK" ] && skip=1
  "$CTL" run "$1" full_session "DP=$dp SKIP_SMOKE=$skip ${SESSION_ENV:-} bash colab/full_session.sh"
}

MISS_MAX=${MISS_MAX:-6}
k=$(cat "$BK/incarnation" 2>/dev/null || echo 0)
ATTACH=${ATTACH:-}   # ATTACH=tpu|v5e: re-attach to the running incarnation's VM (no new/push/restore/start)
while true; do
  if [ -n "$ATTACH" ]; then
    NAME=epkd-full-$k; ACCEL=$ATTACH; ATTACH=; log "attached to $NAME ($ACCEL)"
  else
  k=$((k + 1)); echo $k > "$BK/incarnation"; NAME=epkd-full-$k
  b=$(balance); log "incarnation $k, balance $b"
  if [ -n "$b" ] && awk "BEGIN{exit !($b < $MIN_UNITS)}"; then log "balance below $MIN_UNITS -- not starting"; exit 4; fi
  ACCEL=tpu
  TRIES=$V6E_TRIES "$CTL" new "$NAME" tpu || { ACCEL=v5e; TRIES=60 "$CTL" new "$NAME" v5e || { log "no TPU at all"; exit 1; }; }
  log "VM $NAME ($ACCEL) READY"
  out=$("$CTL" push "$NAME" 2>&1); rc=$?; echo "$out" | tail -3
  [ $rc = 0 ] && echo "$out" | grep -q "^epkd_colab" || { log "push failed"; "$CTL" stop "$NAME"; continue; }
  for f in $(ls "$BK"/cache_inc_*.tgz 2>/dev/null | sort); do
    log "restoring $(basename $f)"; "$CTL" put "$NAME" "$f" $W/cache | tail -1 || log "restore of $f FAILED"
  done
  "$CTL" sh "$NAME" "touch $W/.bk_last" >/dev/null
  [ -f "$BK/state.tgz" ] && { log "restoring state"; "$CTL" put "$NAME" "$BK/state.tgz" $W || log "state restore FAILED"; }
  start_session "$NAME" "$ACCEL"
  fi
  miss=0; last_ep=""; fail_t=0
  while true; do
    sleep "$POLL"
    st=$("$CTL" sh "$NAME" "echo ALIVE; cat $W/logs/SESSION_STATE; for f in $W/runs/*/DONE_EPOCH; do [ -f \$f ] && echo EP \$(basename \$(dirname \$f)) \$(cat \$f); done; python3 -c \"import json;s=json.load(open('$W/logs/pipeline_status.json'));print('PHASE',s.get('phase'),json.dumps(s.get('precompute',{}))[:160])\" 2>/dev/null; grep -a '\[hybrid\]' $W/logs/precompute.log 2>/dev/null | tail -1; ls -t $W/logs/[EF][12]_*.log 2>/dev/null | head -1 | xargs -r tail -n 2; echo NANS \$(cat $W/logs/[EF][12]_*.log 2>/dev/null | grep -c FloatingPointError); tail -n ${MEM_LINES:-1} $W/logs/mem.log | cut -c1-330; grep MEMGUARD $W/logs/mem.log | tail -1" 2>&1)
    if ! echo "$st" | grep -q ALIVE; then
      miss=$((miss + 1)); log "VM unreachable ($miss/$MISS_MAX): $(echo "$st" | tail -1)"
      [ $miss -ge $MISS_MAX ] && { log "VM $NAME lost -> new incarnation"; timeout 120 colab stop -s "$NAME" >/dev/null 2>&1; break; }
      continue
    fi
    miss=0
    log "$(echo "$st" | grep -v ALIVE | tr '\n' ' ' | cut -c1-${LOG_CHARS:-600})"
    nans=$(echo "$st" | awk '/^NANS /{print $2}'); [ "${nans:-0}" -gt 0 ] && log "ALERT non-finite loss seen $nans times (training restarts from latest.pt)"
    ep=$(echo "$st" | awk '/^EP /{print $2":"$3}' | sort | tail -1)
    if [ -n "$ep" ] && [ "$ep" != "$last_ep" ]; then backup_state "$NAME" && last_ep=$ep; fi
    echo "$st" | grep -qE "^(DOWNLOAD_WAIT|PIPELINE)" && touch "$BK/SMOKE_OK"
    if [ ! -f "$BK/cache_final" ]; then
      backup_cache "$NAME"
      echo "$st" | grep -qE "^PHASE [EF][12]_" && touch "$BK/cache_final"   # finalize + verify are done once F1 runs
    fi
    if echo "$st" | grep -q "^ALL_DONE"; then backup_state "$NAME"; log "ALL DONE"; "$CTL" stop "$NAME"; exit 0; fi
    if [ -f "$BK/QUIT" ]; then rm -f "$BK/QUIT"; backup_state "$NAME"; log "QUIT requested"; "$CTL" stop "$NAME"; exit 0; fi
    if [ -f "$BK/RESTART" ]; then
      rm -f "$BK/RESTART"; log "RESTART requested: re-push + rerun session"; "$CTL" push "$NAME" | tail -2
      start_session "$NAME" "$ACCEL"; fail_t=0; continue
    fi
    if echo "$st" | grep -qE "^(SMOKE_FAIL|PIPELINE_EXIT)"; then
      [ $fail_t = 0 ] && { fail_t=$(date +%s); log "ALERT session failed: $(echo "$st" | grep -E '^(SMOKE_FAIL|PIPELINE_EXIT)')"; backup_state "$NAME"; }
      if [ ! -f "$BK/HOLD" ] && [ $(( ($(date +%s) - fail_t) / 60 )) -ge "$GRACE_MIN" ]; then
        log "grace over -> stopping $NAME"; "$CTL" stop "$NAME"; exit 2
      fi
    fi
    b=$(balance)
    if [ -n "$b" ] && awk "BEGIN{exit !($b < $MIN_UNITS)}"; then log "balance $b < $MIN_UNITS -> backup + stop"; backup_state "$NAME"; "$CTL" stop "$NAME"; exit 4; fi
  done
done
