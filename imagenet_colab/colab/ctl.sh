#!/bin/bash
# Local helper around the `colab` CLI for the EP-KD ImageNet pipeline.  Every long VM-side job is started
# DETACHED (nohup) and polled with short execs (long `colab exec` websockets drop and can wedge the CLI).
#   ctl.sh new NAME tpu|v5e|a100      create a session (tpu = v6e1; retries every 4 min up to 2 h)
#   ctl.sh push NAME                  upload code bundle (+ HF token, teacher) and set the VM up
#   ctl.sh run NAME TAG 'shell cmd'   run a shell command detached on the VM, log /content/work/logs/TAG.log
#   ctl.sh log NAME TAG [N]           tail the log
#   ctl.sh ps NAME                    running python processes on the VM
#   ctl.sh sh NAME 'shell cmd'        short synchronous shell command (<= 60 s)
#   ctl.sh pull NAME REMOTE LOCAL     download a file
#   ctl.sh sync NAME RUN LOCALDIR     pull latest.pt, best, logs of a run + the cache indexes
#   ctl.sh fetch NAME PARENT 'RELPATHS' LOCAL.tgz   tar RELPATHS (relative to PARENT on the VM), download in 40 MB parts, md5 check
#   ctl.sh put NAME LOCAL.tgz PARENT               upload a local tarball in 15 MB parts, md5 check, untar into PARENT on the VM
#   ctl.sh stop NAME
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/.." && pwd)
BUNDLE=${BUNDLE:-${TMPDIR:-/tmp}/imagenet_colab_bundle.tgz}  # local temp file for the code bundle
VMROOT=/content/imagenet_colab
WORK=/content/work
cmd=$1; shift
name=${1:-}; [ -n "$name" ] && shift

cx() {  # run python code on the VM with a short timeout; code from stdin
  timeout 90 colab exec -s "$name" --timeout 75 2>&1
}
vsh() {  # shell command on the VM (synchronous, short)
  local c=$1
  printf 'import subprocess,sys\nr=subprocess.run(%s, shell=True, capture_output=True, text=True)\nsys.stdout.write(r.stdout[-6000:]); sys.stderr.write(r.stderr[-3000:])\n' "$(python3 -c 'import json,sys; print(json.dumps(sys.argv[1]))' "$c")" | cx
}

case "$cmd" in
  new)
    accel=${1:-tpu}
    case "$accel" in tpu|v6e) flags="--tpu v6e1";; v5e) flags="--tpu v5e1";; *) flags="--gpu A100";; esac
    for i in $(seq 1 ${TRIES:-30}); do
      out=$(timeout 900 colab new -s "$name" $flags 2>&1); echo "$out" | tail -2
      if echo "$out" | grep -q READY; then echo "SESSION $name READY (try $i) $(date)"; timeout 60 colab usage; exit 0; fi
      echo "try $i failed $(date)"; sleep 240
    done; exit 1;;
  push)
    (cd "$ROOT/.." && tar czf "$BUNDLE" --exclude='cache*' --exclude='logs' --exclude='__pycache__' --exclude='runs' --exclude='*.tgz' imagenet_colab)
    ls -la "$BUNDLE"
    # single large uploads fail ("Max retries exceeded"); upload 15 MB parts with retries and reassemble on the VM
    rm -f "$BUNDLE".part_*; split -b 15M -d "$BUNDLE" "$BUNDLE".part_
    for f in "$BUNDLE".part_*; do
      ok=0; for t in 1 2 3 4; do
        timeout 180 colab upload -s "$name" "$f" "/content/$(basename "$f")" 2>&1 | tail -1 | grep -q Uploaded && { ok=1; break; }
      done; [ $ok = 1 ] || { echo "upload of $f failed"; exit 3; }
    done
    vsh "cd /content && cat $(basename "$BUNDLE").part_* > bundle.tgz && rm -f $(basename "$BUNDLE").part_* && md5sum bundle.tgz"; md5sum "$BUNDLE"
    if [ -f ~/.cache/huggingface/token ]; then
      vsh 'mkdir -p /root/.cache/huggingface'
      timeout 60 colab upload -s "$name" ~/.cache/huggingface/token /root/.cache/huggingface/token 2>&1 | tail -1
    fi
    vsh "mkdir -p $WORK/logs $WORK/cache $WORK/runs /media/e12056/ASi/imagenet/data && cd /content && rm -rf $VMROOT && tar xzf bundle.tgz && ls $VMROOT && df -h /content / | tail -2 && nproc && free -g | head -2";;
  run)
    tag=$1; c=$2
    vsh "cd $VMROOT && nohup bash -c $(python3 -c 'import shlex,sys; print(shlex.quote(sys.argv[1]))' "export PYTHONUNBUFFERED=1 EPKD_CACHE_DIR=$WORK/cache; $c") > $WORK/logs/$tag.log 2>&1 < /dev/null & echo started pid \$!";;
  log) tag=$1; n=${2:-40}; vsh "tail -n $n $WORK/logs/$tag.log";;
  ps) vsh "ps aux | grep -E 'python|download' | grep -v grep | awk '{print \$2, \$3, \$4, \$11, \$12, \$13, \$14, \$15}' | head -30";;
  sh) vsh "$1";;
  pull) timeout 1800 colab download -s "$name" "$1" "$2" 2>&1 | tail -1;;
  sync)
    run=$1; dst=$2; mkdir -p "$dst/runs/$run" "$dst/cache"
    for f in latest.pt best.pt student_best.pt DONE_EPOCH; do timeout 1800 colab download -s "$name" "$WORK/runs/$run/$f" "$dst/runs/$run/$f" 2>&1 | tail -1; done
    vsh "cd $WORK/runs/$run && tar czf /content/logs_$run.tgz logs" && timeout 600 colab download -s "$name" "/content/logs_$run.tgz" "$dst/runs/$run/logs.tgz" 2>&1 | tail -1
    vsh "cd $WORK/cache && tar czf /content/cache_indexes.tgz \$(find imagenet_tgsc -name 'stage_res*.parquet' -o -name 'precompute_state.json' -o -name 'manifest_*.parquet' -o -name '*.done' 2>/dev/null)" && timeout 1800 colab download -s "$name" /content/cache_indexes.tgz "$dst/cache/cache_indexes.tgz" 2>&1 | tail -1;;
  fetch)
    parent=$1; rel=$2; dst=$3; tmpd=$(mktemp -d "$(dirname "$dst")/.fetch.XXXX")
    r=$(vsh "cd $parent && rm -f /content/xfer.tgz /content/xfer.part_* && tar czf /content/xfer.tgz $rel && cd /content && split -b 40M -d xfer.tgz xfer.part_ && md5sum xfer.tgz && ls xfer.part_*")
    echo "$r" | tail -3; md5=$(echo "$r" | awk '/xfer.tgz/{print $1}' | head -1); parts=$(echo "$r" | grep -o 'xfer.part_[0-9]*')
    [ -n "$md5" ] && [ -n "$parts" ] || { echo "fetch: remote tar failed"; rm -rf "$tmpd"; exit 3; }
    for f in $parts; do
      ok=0; for t in 1 2 3 4; do
        timeout 900 colab download -s "$name" "/content/$f" "$tmpd/$f" 2>&1 | tail -1 | grep -q Downloaded && { ok=1; break; }
      done; [ $ok = 1 ] || { echo "fetch: download of $f failed"; rm -rf "$tmpd"; exit 3; }
    done
    cat "$tmpd"/xfer.part_* > "$tmpd/x.tgz"; got=$(md5sum "$tmpd/x.tgz" | awk '{print $1}')
    [ "$got" = "$md5" ] || { echo "fetch: md5 mismatch $got != $md5"; rm -rf "$tmpd"; exit 3; }
    mv "$tmpd/x.tgz" "$dst"; rm -rf "$tmpd"; ls -la "$dst"; vsh "rm -f /content/xfer.tgz /content/xfer.part_*" >/dev/null;;
  put)
    src=$1; parent=$2; b=$(basename "$src"); tmpd=$(mktemp -d); split -b 15M -d "$src" "$tmpd/$b.part_"
    for f in "$tmpd"/*.part_*; do
      ok=0; for t in 1 2 3 4; do
        timeout 180 colab upload -s "$name" "$f" "/content/$(basename "$f")" 2>&1 | tail -1 | grep -q Uploaded && { ok=1; break; }
      done; [ $ok = 1 ] || { echo "put: upload of $f failed"; rm -rf "$tmpd"; exit 3; }
    done; rm -rf "$tmpd"
    r=$(vsh "cd /content && cat $b.part_* > $b && rm -f $b.part_* && md5sum $b && mkdir -p $parent && tar xzf $b -C $parent && rm -f $b && echo PUT_OK")
    echo "$r" | tail -2; want=$(md5sum "$src" | awk '{print $1}')
    echo "$r" | grep -q "$want" && echo "$r" | grep -q PUT_OK || { echo "put: md5/untar failed"; exit 3; };;
  stop) timeout 120 colab stop -s "$name" 2>&1 | tail -1; timeout 60 colab usage;;
  *) echo "unknown command $cmd"; exit 2;;
esac
