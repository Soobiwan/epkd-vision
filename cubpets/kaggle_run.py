"""Runs one notebook's share of the CUB / Pets view-rules table (cubpets/plan.py) on the GPUs of a Kaggle session.

  python cubpets/kaggle_run.py --notebook 07            # what each kaggle/cubpets_NN.ipynb does
  python cubpets/kaggle_run.py --runs cub_sv2_prop_s0   # any list of runs
  python cubpets/kaggle_run.py --summary                # print the results found under --out

Steps:
 1. restore run folders from a previous version of the notebook attached as input (more epochs wins), so an
    interrupted run continues from its last finished epoch (latest.pt);
 2. build / verify the 96-px tensors (cubpets/get_data.py; content sha256 = the paper's tensors);
 3. fetch the ResNet32x4 teachers from the GitHub release (sha256 checked); a teacher that is not published yet is
    polled for, and the GPU runs another job of its queue meanwhile;
 4. one train.py process per GPU (queues in plan.NOTEBOOKS; an idle GPU takes the other GPU's last unstarted job);
    a crashed run is relaunched (resumes from latest.pt), at most 3 launches;
 5. time guard: after --max-hours minus 15 min no new launches, and each running job is stopped right after its next
    epoch checkpoint (or at --max-hours at the latest), so the session ends before Kaggle's 12-h limit with
    resumable state;
 6. writes results/<run>.json, summary.json, DONE (or INCOMPLETE), and a zip of everything.
"""
import argparse
import glob
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import get_data  # noqa: E402
import plan  # noqa: E402

T0 = time.time()


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')} +{(time.time() - T0) / 3600:.2f}h] {msg}", flush=True)


def commit():
    r = subprocess.run(["git", "-C", HERE, "rev-parse", "HEAD"], capture_output=True, text=True)
    return r.stdout.strip() or "unknown"


# ----------------------------------------------------------------------------- run folders
def progress(run_dir):
    """(done, last finished epoch with a checkpoint)."""
    if os.path.exists(os.path.join(run_dir, "done.json")):
        return True, 10 ** 6
    ep = 0
    lp = os.path.join(run_dir, "train.jsonl")
    if os.path.exists(lp) and os.path.exists(os.path.join(run_dir, "latest.pt")):
        for line in open(lp):
            try:
                ep = max(ep, json.loads(line)["epoch"])
            except Exception:  # noqa: BLE001
                pass
    return False, ep


def last_rec(run_dir):
    try:
        return json.loads(open(os.path.join(run_dir, "train.jsonl")).read().strip().splitlines()[-1])
    except Exception:  # noqa: BLE001
        return {}


def restore(names, runs_dir, search_root):
    """Copy run folders of an attached previous output (folders or cubpets_*.zip) if they are further along."""
    if not search_root or not os.path.isdir(search_root):
        return
    for name in names:
        dst = os.path.join(runs_dir, name)
        for prev in glob.glob(os.path.join(search_root, "**", "runs", name), recursive=True):
            if os.path.isdir(prev) and os.path.realpath(prev) != os.path.realpath(dst) \
                    and progress(prev)[1] > progress(dst)[1]:
                shutil.rmtree(dst, ignore_errors=True)
                shutil.copytree(prev, dst)
                log(f"restored {name} from {prev} (epoch {progress(dst)[1]}, done={progress(dst)[0]})")
    for zp in glob.glob(os.path.join(search_root, "**", "cubpets_*.zip"), recursive=True):
        try:
            zf = zipfile.ZipFile(zp)
        except zipfile.BadZipFile:
            continue
        for name in names:
            members = [m for m in zf.namelist() if f"runs/{name}/" in m and not m.endswith("/")]
            if not members:
                continue
            tmp = os.path.join(runs_dir, f".restore_{name}")
            shutil.rmtree(tmp, ignore_errors=True)
            for m in members:
                p = os.path.join(tmp, os.path.basename(m))
                os.makedirs(tmp, exist_ok=True)
                with zf.open(m) as src, open(p, "wb") as f:
                    shutil.copyfileobj(src, f)
            dst = os.path.join(runs_dir, name)
            if progress(tmp)[1] > progress(dst)[1]:
                shutil.rmtree(dst, ignore_errors=True)
                os.replace(tmp, dst)
                log(f"restored {name} from {zp} (epoch {progress(dst)[1]}, done={progress(dst)[0]})")
            else:
                shutil.rmtree(tmp, ignore_errors=True)


# ----------------------------------------------------------------------------- teachers
class Teachers:
    """Teacher checkpoints: local folder (--teacher-dir), else the GitHub release (polled until published)."""

    def __init__(self, tdir, local_dir=None, dummy=False, poll_s=120):
        self.tdir, self.local_dir, self.dummy, self.poll_s = tdir, local_dir, dummy, poll_s
        self.path, self.info, self.last_try = {}, {}, {}
        os.makedirs(tdir, exist_ok=True)

    def get(self, ds):
        if ds in self.path:
            return self.path[ds]
        if time.time() - self.last_try.get(ds, 0) < self.poll_s:
            return None
        self.last_try[ds] = time.time()
        f = plan.teacher_file(ds)
        dst = os.path.join(self.tdir, f + ".pth")
        if self.dummy:  # smoke test only: random-init teacher
            import torch
            import train
            torch.manual_seed(0)
            torch.save(train.build_model(plan.TEACHER, train.DATASETS[ds]["num_classes"]).state_dict(), dst)
            self.info[ds] = {"dummy": True}
            self.path[ds] = dst
            log(f"teacher {ds}: DUMMY random-init {plan.TEACHER} (smoke test)")
            return dst
        if self.local_dir and os.path.exists(os.path.join(self.local_dir, f + ".pth")):
            shutil.copyfile(os.path.join(self.local_dir, f + ".pth"), dst)
            jp = os.path.join(self.local_dir, f + ".json")
            self.info[ds] = json.load(open(jp)) if os.path.exists(jp) else {}
            if self.info[ds].get("sha256") and get_data.sha256_file(dst) != self.info[ds]["sha256"]:
                raise SystemExit(f"teacher {dst}: sha256 mismatch")
            self.path[ds] = dst
            log(f"teacher {ds}: {dst} (local) {self.info[ds]}")
            return dst
        try:
            with urllib.request.urlopen(f"{plan.RELEASE}/{f}.json", timeout=60) as r:
                info = json.loads(r.read())
        except (urllib.error.URLError, OSError, ValueError) as e:
            log(f"teacher {ds}: not published yet ({getattr(e, 'code', e)}); polling every {self.poll_s} s")
            return None
        try:
            get_data.download(f"{plan.RELEASE}/{f}.pth", dst, info["sha256"])
        except SystemExit as e:
            log(f"teacher {ds}: {e}; retrying later")
            return None
        self.info[ds], self.path[ds] = info, dst
        log(f"teacher {ds}: {dst} sha256 ok; {info}")
        return dst


# ----------------------------------------------------------------------------- results
def write_result(name, run_dir, res_dir, teachers, cmt):
    if not os.path.exists(os.path.join(run_dir, "done.json")):
        return None
    done = json.load(open(os.path.join(run_dir, "done.json")))
    cfg = json.load(open(os.path.join(run_dir, "config.json")))
    ds, pair, arm, seed = plan.parse(name)
    recs = {}
    for line in open(os.path.join(run_dir, "train.jsonl")):
        try:
            r = json.loads(line)
            recs[r["epoch"]] = r
        except Exception:  # noqa: BLE001
            pass
    res = dict(run=name, dataset=ds, pair=pair, arm=arm, seed=seed, teacher=cfg.get("teacher_ckpt"),
               student=cfg.get("student"), best=done["best"], best_epoch=done["best_epoch"], final=done["final"],
               teacher_acc=cfg.get("teacher_acc"), teacher_info=teachers.info.get(ds), epochs=cfg.get("epochs"),
               train_hours=sum(r.get("sec", 0) for r in recs.values()) / 3600, commit=cmt)
    os.makedirs(res_dir, exist_ok=True)
    json.dump(res, open(os.path.join(res_dir, name + ".json"), "w"), indent=1)
    return res


def summary(out):
    rows = []
    for p in sorted(glob.glob(os.path.join(out, "results", "*.json"))):
        rows.append(json.load(open(p)))
    done = {r["run"] for r in rows}
    print(f"{'run':24s} {'best':>7s} {'@ep':>4s} {'final':>7s} {'teacher':>8s} {'hours':>6s}")
    for r in rows:
        print(f"{r['run']:24s} {r['best']:7.2f} {r['best_epoch']:4d} {r['final']:7.2f} "
              f"{(r['teacher_acc'] or 0):8.2f} {r['train_hours']:6.2f}")
    for rd in sorted(glob.glob(os.path.join(out, "runs", "*"))):
        n = os.path.basename(rd)
        if n not in done and not n.startswith("."):
            r = last_rec(rd)
            print(f"{n:24s} unfinished: epoch {r.get('epoch', 0)}/240, best so far {r.get('best', 0):.2f}")
    return rows


# ----------------------------------------------------------------------------- main
def main():
    kaggle = os.path.isdir("/kaggle/working")
    ap = argparse.ArgumentParser()
    ap.add_argument("--notebook", help="notebook id in plan.NOTEBOOKS, e.g. 07")
    ap.add_argument("--runs", default="", help="comma-separated run names (instead of --notebook; one queue per GPU)")
    ap.add_argument("--out", default="/kaggle/working/cubpets" if kaggle else os.path.abspath("cubpets_out"))
    ap.add_argument("--work", default="/kaggle/temp/cubpets" if kaggle else os.path.abspath("cubpets_work"),
                    help="data, raw downloads and teachers (not part of the saved output)")
    ap.add_argument("--data-dir", default=None, help="folder with <ds>_uint8.pt (default WORK/data; hashes checked)")
    ap.add_argument("--restore-from", default="/kaggle/input" if kaggle else "",
                    help="search here for previous runs/<name> folders or cubpets_*.zip")
    ap.add_argument("--teacher-dir", default="", help="local folder with tea_<ds>_r32x4.pth (+ .json) instead of the release")
    ap.add_argument("--dummy-teacher", action="store_true", help="smoke test: random-init teacher")
    ap.add_argument("--teacher-wait-hours", type=float, default=8.0)
    ap.add_argument("--max-hours", type=float, default=11.5, help="session budget (Kaggle limit: 12 h)")
    ap.add_argument("--smoke", action="store_true",
                    help="tiny run: 5 epochs of 1 epoch per stage on 320 training images (all 5 stages)")
    ap.add_argument("--zip", type=int, default=1)
    ap.add_argument("--summary", action="store_true")
    a = ap.parse_args()
    for k in ("out", "work", "data_dir", "restore_from", "teacher_dir"):  # train.py runs with cwd = cubpets/
        if getattr(a, k):
            setattr(a, k, os.path.abspath(getattr(a, k)))
    if a.summary:
        summary(a.out)
        return

    if a.notebook:
        queues = [list(q) for q in plan.NOTEBOOKS[a.notebook]]
        tag = f"nb{a.notebook}"
    else:
        names = [n for n in a.runs.split(",") if n]
        queues = [names[0::2], names[1::2]]
        tag = "custom"
    names = [n for q in queues for n in q]
    for n in names:
        plan.parse(n)
    runs_dir, res_dir = os.path.join(a.out, "runs"), os.path.join(a.out, "results")
    os.makedirs(runs_dir, exist_ok=True)
    os.makedirs(a.work, exist_ok=True)
    cmt = commit()

    import torch
    n_gpu = torch.cuda.device_count()
    print(subprocess.run(["nvidia-smi", "--query-gpu=index,name,memory.total", "--format=csv"],
                         capture_output=True, text=True).stdout, flush=True)
    log(f"cubpets {tag}: epkd-vision {cmt}; torch {torch.__version__}; {n_gpu} GPU(s); runs {queues}")
    if n_gpu < 1:
        raise SystemExit("no GPU: set Accelerator to GPU T4 x2")
    if n_gpu < 2:
        log("WARNING: one GPU only (choose GPU T4 x2): the two queues run one after the other")
    gpus = list(range(min(2, n_gpu)))

    restore(names, runs_dir, a.restore_from)
    pending = [n for n in names if not progress(os.path.join(runs_dir, n))[0]]
    log(f"{len(names) - len(pending)} of {len(names)} runs already done; to run: {pending}")
    teachers = Teachers(os.path.join(a.work, "teachers"), a.teacher_dir or None, a.dummy_teacher)
    for n in names:  # results of runs finished in an earlier version
        if n not in pending:
            write_result(n, os.path.join(runs_dir, n), res_dir, teachers, cmt)

    data_dir = a.data_dir or os.path.join(a.work, "data")
    os.makedirs(data_dir, exist_ok=True)
    for ds in sorted({plan.parse(n)[0] for n in pending}, key=lambda d: [plan.parse(n)[0] for n in pending].index(d)):
        get_data.get(ds, data_dir, os.path.join(a.work, "raw"), a.restore_from or None)

    extra = []
    if a.smoke:
        import torch as _t
        sub = os.path.join(a.work, "smoke_subset.json")
        json.dump({"indices": list(range(320))}, open(sub, "w"))
        extra = ["--epochs", "5", "--stage-epochs", "1", "--subset-manifest", sub]
        _ = _t

    soft = T0 + (a.max_hours - 0.25) * 3600
    hard = T0 + a.max_hours * 3600
    running = {}  # gpu -> (name, Popen, fh, latest mtime at the soft deadline)
    failed, last_report, waiting_since = set(), 0.0, None

    def launch(g, name):
        ds = plan.parse(name)[0]
        rd = os.path.join(runs_dir, name)
        os.makedirs(rd, exist_ok=True)
        tries = int(open(os.path.join(rd, "tries")).read()) + 1 if os.path.exists(os.path.join(rd, "tries")) else 1
        open(os.path.join(rd, "tries"), "w").write(str(tries))
        args = plan.train_args(name, data_dir, teachers.path[ds]) + extra
        with open(os.path.join(rd, "cmd.txt"), "a") as f:
            f.write(f"# {time.strftime('%F %T')} kaggle {tag} gpu={g} commit={cmt} launch={tries}\n{' '.join(args)}\n")
        fh = open(os.path.join(rd, "stdout.log"), "a")
        p = subprocess.Popen([sys.executable, os.path.join(HERE, "train.py"), *args, "--run-dir", rd], cwd=HERE,
                             stdout=fh, stderr=subprocess.STDOUT,
                             env={**os.environ, "CUDA_VISIBLE_DEVICES": str(g), "PYTHONUNBUFFERED": "1"})
        running[g] = [name, p, fh, None]
        log(f"GPU {g}: started {name} (launch {tries}, resume from epoch {progress(rd)[1]}, "
            f"expected ~{plan.est_hours(name):.1f} h on a T4): {' '.join(args)}")

    def startable(name):
        rd = os.path.join(runs_dir, name)
        busy = {r[0] for r in running.values()}
        return name not in busy and name not in failed and not progress(rd)[0]

    while True:
        now = time.time()
        for g, (name, p, fh, mt) in list(running.items()):  # reap
            rd = os.path.join(runs_dir, name)
            if p.poll() is None:
                if now > soft:  # stop right after the next epoch checkpoint, or at the hard deadline
                    lp = os.path.join(rd, "latest.pt")
                    m = os.path.getmtime(lp) if os.path.exists(lp) else 0
                    if mt is None:
                        running[g][3] = m
                    elif m > mt or now > hard:
                        log(f"GPU {g}: time guard: stopping {name} at epoch {progress(rd)[1]} (resumable)")
                        p.send_signal(signal.SIGTERM)
                        try:
                            p.wait(60)
                        except subprocess.TimeoutExpired:
                            p.kill()
                continue
            fh.close()
            del running[g]
            r = last_rec(rd)
            if progress(rd)[0]:
                res = write_result(name, rd, res_dir, teachers, cmt)
                log(f"GPU {g}: {name} DONE: best {res['best']:.2f} @ {res['best_epoch']}, final {res['final']:.2f}")
            elif now > soft:
                log(f"GPU {g}: {name} stopped at epoch {r.get('epoch', 0)} by the time guard")
            else:
                tries = int(open(os.path.join(rd, "tries")).read())
                log(f"GPU {g}: {name} exited with code {p.returncode} at epoch {r.get('epoch', 0)} (launch {tries}); "
                    f"stdout tail:\n" + "".join(open(os.path.join(rd, "stdout.log")).readlines()[-15:]))
                if tries >= 3:
                    failed.add(name)
                    open(os.path.join(rd, "FAILED"), "w").write(f"{time.strftime('%F %T')} 3 launches\n")
        if now < soft:  # launch
            for g in gpus:
                if g in running:
                    continue
                own = queues[g] if g < len(queues) else []
                other = [n for k, q in enumerate(queues) if k != g for n in q]
                cands = [n for n in own if startable(n)] + [n for n in reversed(other) if startable(n)]
                for n in cands:
                    if teachers.get(plan.parse(n)[0]):
                        launch(g, n)
                        break
        left = [n for n in names if startable(n)]
        if not running and (not left or now >= soft):
            break
        if not running and left:  # everything left waits for a teacher
            waiting_since = waiting_since or now
            if now - waiting_since > a.teacher_wait_hours * 3600:
                log(f"gave up waiting for teachers after {a.teacher_wait_hours} h; not run: {left}")
                break
        else:
            waiting_since = None
        if now - last_report > 600:
            last_report = now
            st = {}
            for g, (name, p, fh, mt) in running.items():
                rd = os.path.join(runs_dir, name)
                r = last_rec(rd)
                st[name] = dict(gpu=g, epoch=r.get("epoch", 0), test=r.get("test_acc"), best=r.get("best"),
                                sec=r.get("sec"))
                log(f"   GPU {g} {name}: epoch {r.get('epoch', 0)}/{240 if not a.smoke else 5}, "
                    f"test {r.get('test_acc', 0):.2f}, best {r.get('best', 0):.2f}, {r.get('sec', 0):.0f} s/epoch")
            json.dump(dict(time=time.strftime('%F %T'), hours=(now - T0) / 3600, running=st, waiting=left,
                           done=[n for n in names if progress(os.path.join(runs_dir, n))[0]]),
                      open(os.path.join(a.out, "status.json"), "w"), indent=1)
        time.sleep(5 if now > soft else 20)

    done = [n for n in names if progress(os.path.join(runs_dir, n))[0]]
    rows = summary(a.out)
    json.dump(dict(notebook=tag, commit=cmt, done=done, not_done=[n for n in names if n not in done], results=rows),
              open(os.path.join(a.out, "summary.json"), "w"), indent=1)
    marker = "DONE" if len(done) == len(names) else "INCOMPLETE"
    for m in ("DONE", "INCOMPLETE"):
        if os.path.exists(os.path.join(a.out, m)):
            os.remove(os.path.join(a.out, m))
    open(os.path.join(a.out, marker), "w").write(f"{time.strftime('%F %T')} {len(done)}/{len(names)} done\n")
    log(f"{marker}: {len(done)}/{len(names)} runs done")
    if a.zip:
        zp = os.path.join(os.path.dirname(a.out), f"cubpets_{tag}.zip")
        with zipfile.ZipFile(zp + ".tmp", "w", zipfile.ZIP_DEFLATED) as zf:
            for root, _, files in os.walk(a.out):
                for f in files:
                    p = os.path.join(root, f)
                    zf.write(p, os.path.relpath(p, os.path.dirname(a.out)))
        os.replace(zp + ".tmp", zp)
        log(f"zipped {a.out} -> {zp} ({os.path.getsize(zp) / 2**20:.0f} MB)")


if __name__ == "__main__":
    main()
