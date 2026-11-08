"""Generates one Kaggle notebook per CIFAR-100 teacher -> student pair (EP-KD, 3 seeds, Kaggle GPU T4 x2).

Usage: python kaggle/make_notebooks.py   (writes kaggle/epkd_cifar100_<teacher>_to_<student>.ipynb)
"""
import json
import os

# (teacher, student, test accuracy of the RepDistiller teacher file, base learning rate)
# The ResNet56 file scores 72.41 (DKD's tables list 72.34).
# lr 0.01 for the MobileNetV2 / ShuffleNet students and 0.05 otherwise, as in CRD and DKD.
PAIRS = [
    ("resnet56", "resnet20", 72.41, 0.05),
    ("resnet110", "resnet32", 74.31, 0.05),
    ("resnet32x4", "resnet8x4", 79.42, 0.05),
    ("wrn_40_2", "wrn_16_2", 75.61, 0.05),
    ("wrn_40_2", "wrn_40_1", 75.61, 0.05),
    ("vgg13", "vgg8", 74.64, 0.05),
    ("resnet32x4", "ShuffleV1", 79.42, 0.01),
    ("wrn_40_2", "ShuffleV1", 75.61, 0.01),
    ("vgg13", "MobileNetV2", 74.64, 0.01),
    ("resnet32x4", "ShuffleV2", 79.42, 0.01),
]
PRETTY = {"resnet56": "ResNet56", "resnet20": "ResNet20", "resnet110": "ResNet110", "resnet32": "ResNet32",
          "resnet32x4": "ResNet32x4", "resnet8x4": "ResNet8x4", "wrn_40_2": "WRN-40-2", "wrn_16_2": "WRN-16-2",
          "wrn_40_1": "WRN-40-1", "vgg13": "VGG13", "vgg8": "VGG8", "ShuffleV1": "ShuffleNetV1",
          "ShuffleV2": "ShuffleNetV2", "MobileNetV2": "MobileNetV2"}

INTRO = """# EP-KD on CIFAR-100: {tp} → {sp} (3 seeds, Kaggle GPU T4 x2)

This notebook clones [Soobiwan/epkd-vision](https://github.com/Soobiwan/epkd-vision). It downloads CIFAR-100 and the
pretrained {tp} teacher, then trains the {sp} student with **EP-KD for vision**, using the paper recipe
(`scripts/cifar100.sh`), for seeds 0, 1 and 2.

| | |
|---|---|
| teacher | `{t}`. The standard CRD teacher from [RepDistiller](https://github.com/HobbitLong/RepDistiller), test accuracy {tacc:.2f}%. If that server fails, the notebook uses the mirror in the [mdistiller](https://github.com/megvii-research/mdistiller) release. |
| student | `{s}`, base learning rate {lr} (the CRD / DKD setting for this student) |
| EP-KD | `--views 1 --view-cache 1 --view-set sal1,sal2,sal3 --view-rotate 1 --passes 3,3,3,1,1 --anchor 1 --milestones 48,144,192,210` |
| KD (ours) | `--views 0 --milestones 150,180,210`, the same seeds. This is the "KD (ours)" row of the paper table; set `RUN_KD = False` to skip it. |
| reported | the **best test epoch** of each run, as mean ± sd over seeds, and the EP-KD − KD difference paired by seed |

**Kaggle settings (right panel):** Accelerator **GPU T4 x2**, Internet **on**. Then use **Save Version → Save & Run All
(Commit)**, so that the run continues after you close the browser.

**Sessions and resuming.** A Kaggle session ends after 12 h. Before that, the runner stops every job after
`TIME_BUDGET_H` hours and saves its resumable state under `/kaggle/working/runs/`, which Kaggle keeps as the version
output. If the summary at the end lists unfinished runs:

1. Open the notebook in the editor and choose **Add Input → Your Work → Notebooks**. Select this notebook, so that its
   latest output version is attached.
2. Run **Save & Run All** again. The restore cell finds the previous `runs/` folder under `/kaggle/input`, and each run
   continues from its last finished epoch.

The jobs run concurrently: `JOBS_PER_GPU` jobs on each GPU. The EP-KD jobs start first, and the KD jobs fill the free
slots."""

CONFIG = """# ---- configuration ----
TEACHER, STUDENT = "{t}", "{s}"
TEACHER_ACC = {tacc}      # test accuracy of the teacher checkpoint (sanity check only)
LR = {lr}                 # base learning rate (CRD / DKD setting for this student)
SEEDS = [0, 1, 2]
RUN_KD = True             # also train KD (ours) with the standard schedule, same seeds (paper table row + Delta)
JOBS_PER_GPU = 2          # concurrent training processes per GPU (data and augmentation live on the GPU)
TIME_BUDGET_H = 11.0      # stop and save before Kaggle's 12 h session limit
REPO = "https://github.com/Soobiwan/epkd-vision.git"
REPO_REF = "main"         # branch, tag or commit of epkd-vision
EXTRA_ARGS = []          # extra train.py flags for every run, e.g. ["--epochs", "2"] for a smoke test

PAIR = f"{{TEACHER}}_to_{{STUDENT}}"
CODE = "/kaggle/temp/epkd-vision"   # clone, data and teacher (not part of the saved output)
OUT = "/kaggle/working/runs"         # run folders (saved as the notebook output)
import time; T_START = time.time()"""

SETUP = """import os, subprocess, sys
os.makedirs("/kaggle/temp", exist_ok=True)
print(subprocess.run(["nvidia-smi", "--query-gpu=index,name,memory.total", "--format=csv"],
                     capture_output=True, text=True).stdout)
if not os.path.isdir(CODE):
    subprocess.run(["git", "clone", REPO, CODE], check=True)
subprocess.run(["git", "-C", CODE, "fetch", "--all", "--tags", "-q"], check=True)
subprocess.run(["git", "-C", CODE, "checkout", "-q", REPO_REF], check=True)
subprocess.run(["git", "-C", CODE, "pull", "-q", "--ff-only"], check=False)  # no-op on a detached commit
COMMIT = subprocess.run(["git", "-C", CODE, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
import torch
N_GPU = torch.cuda.device_count()
print(f"epkd-vision @ {COMMIT}; torch {torch.__version__}; {N_GPU} GPU(s)")
assert N_GPU >= 1, "no GPU: set Accelerator to GPU T4 x2"
if N_GPU < 2:
    print("WARNING: only one GPU visible; the jobs will share it (choose 'GPU T4 x2' for both GPUs)")"""

DATA = """# CIFAR-100 (python version): an attached Kaggle copy is used if present, otherwise the official download
import glob, shutil
data = os.path.join(CODE, "data")
os.makedirs(data, exist_ok=True)
if not os.path.isdir(os.path.join(data, "cifar-100-python")):
    found = [p for p in glob.glob("/kaggle/input/**/cifar-100-python", recursive=True)
             if os.path.exists(os.path.join(p, "train"))]
    if found:
        shutil.copytree(found[0], os.path.join(data, "cifar-100-python"))
        print("CIFAR-100 copied from", found[0])
    else:
        tgz = os.path.join(data, "cifar-100-python.tar.gz")
        subprocess.run(["curl", "-sSL", "--retry", "5", "-o", tgz,
                        "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"], check=True)
        subprocess.run(["tar", "-xzf", tgz, "-C", data], check=True)
        os.remove(tgz)
print(sorted(os.listdir(os.path.join(data, "cifar-100-python"))))"""

TEACHER_CELL = """# Pretrained teacher: RepDistiller (CRD) checkpoint, mdistiller mirror as fallback -> weights-only state dict
subprocess.run([sys.executable, "scripts/get_cifar_teachers.py", "--out", "teachers", TEACHER], cwd=CODE, check=True)
TEACHER_CKPT = os.path.join(CODE, "teachers", f"{TEACHER}_cifar100.pth")
print(TEACHER_CKPT, os.path.getsize(TEACHER_CKPT) // 1024, "KB")"""

RESTORE = """# Resume: copy run folders from a previous version of this notebook (attached via Add Input), if any
import json

def progress(run_dir):
    \"\"\"(done, last epoch logged) of a run folder.\"\"\"
    if os.path.exists(os.path.join(run_dir, "done.json")):
        return True, 10 ** 6
    ep = 0
    log = os.path.join(run_dir, "train.jsonl")
    if os.path.exists(log) and os.path.exists(os.path.join(run_dir, "latest.pt")):
        for line in open(log):
            try:
                ep = max(ep, json.loads(line)["epoch"])
            except Exception:
                pass
    return False, ep

os.makedirs(OUT, exist_ok=True)
for prev in glob.glob(f"/kaggle/input/**/runs/{PAIR}_*", recursive=True):
    if not os.path.isdir(prev):
        continue
    dst = os.path.join(OUT, os.path.basename(prev))
    if progress(prev)[1] > progress(dst)[1]:
        shutil.rmtree(dst, ignore_errors=True)
        shutil.copytree(prev, dst)
        print(f"restored {os.path.basename(prev)} (epoch {progress(dst)[1]}, done={progress(dst)[0]}) from {prev}")"""

RUN = """# Job runner: EP-KD (and KD) for every seed, JOBS_PER_GPU concurrent jobs per GPU, stopped at TIME_BUDGET_H
import signal
COMMON = ["--teacher", TEACHER, "--student", STUDENT, "--teacher-ckpt", TEACHER_CKPT,
          "--data-root", data, "--lr", str(LR)]
METHODS = {"epkd": ["--views", "1", "--view-cache", "1", "--view-set", "sal1,sal2,sal3", "--view-rotate", "1",
                    "--passes", "3,3,3,1,1", "--anchor", "1", "--milestones", "48,144,192,210"],
           "kd": ["--views", "0", "--milestones", "150,180,210"]}
jobs = [(m, s) for m in (["epkd", "kd"] if RUN_KD else ["epkd"]) for s in SEEDS]  # EP-KD first
pending = [j for j in jobs if not progress(os.path.join(OUT, f"{PAIR}_{j[0]}_s{j[1]}"))[0]]
print(f"{len(jobs) - len(pending)} done, {len(pending)} to run: {pending}")

def last_line(run_dir):
    try:
        return json.loads(open(os.path.join(run_dir, "train.jsonl")).read().strip().splitlines()[-1])
    except Exception:
        return {}

running = {}  # name -> (Popen, gpu, log file)
deadline = T_START + TIME_BUDGET_H * 3600
last_report = 0
while pending or running:
    for name, (p, g, fh) in list(running.items()):
        if p.poll() is not None:
            fh.close()
            del running[name]
            r = last_line(os.path.join(OUT, name))
            print(f"[{time.strftime('%H:%M')}] {name} exited ({p.returncode}); epoch {r.get('epoch')} best {r.get('best')}", flush=True)
    if time.time() > deadline:
        print(f"time budget of {TIME_BUDGET_H} h reached: stopping {list(running)}; they resume from latest.pt", flush=True)
        for p, g, fh in running.values():
            p.send_signal(signal.SIGTERM)
        for p, g, fh in running.values():
            try:
                p.wait(timeout=120)
            except subprocess.TimeoutExpired:
                p.kill()
            fh.close()
        running.clear()
        break
    load = [sum(g == k for _, g, _ in running.values()) for k in range(N_GPU)]
    while pending and min(load) < JOBS_PER_GPU:
        g = load.index(min(load))
        m, s = pending.pop(0)
        name = f"{PAIR}_{m}_s{s}"
        run_dir = os.path.join(OUT, name)
        os.makedirs(run_dir, exist_ok=True)
        fh = open(os.path.join(run_dir, "stdout.log"), "a")
        cmd = [sys.executable, "train.py", *COMMON, *METHODS[m], *EXTRA_ARGS, "--seed", str(s), "--run-dir", run_dir]
        p = subprocess.Popen(cmd, cwd=CODE, stdout=fh, stderr=subprocess.STDOUT,
                             env={**os.environ, "CUDA_VISIBLE_DEVICES": str(g)})
        running[name] = (p, g, fh)
        load[g] += 1
        print(f"[{time.strftime('%H:%M')}] started {name} on GPU {g}: {' '.join(cmd[1:])}", flush=True)
    if time.time() - last_report > 900:
        last_report = time.time()
        el = (time.time() - T_START) / 3600
        print(f"[{time.strftime('%H:%M')}] {el:.1f} h elapsed", flush=True)
        for name in running:
            r = last_line(os.path.join(OUT, name))
            print(f"   {name}: epoch {r.get('epoch', 0)}/240  test {r.get('test_acc')}  best {r.get('best')}  "
                  f"{r.get('sec', 0):.0f} s/epoch", flush=True)
    time.sleep(30)
print("runner finished")"""

SUMMARY = """# Summary: best test epoch per run, mean ± sd over seeds, EP-KD - KD paired by seed (95% t-CI)
import numpy as np
T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776}

def best_of(run_dir):
    best = None
    try:
        for line in open(os.path.join(run_dir, "train.jsonl")):
            r = json.loads(line)
            best = r["test_acc"] if best is None else max(best, r["test_acc"])
    except FileNotFoundError:
        pass
    return best

res, teacher_acc = {}, None
for m in METHODS:
    for s in SEEDS:
        rd = os.path.join(OUT, f"{PAIR}_{m}_s{s}")
        if os.path.isdir(rd):
            done, ep = progress(rd)
            res[(m, s)] = dict(best=best_of(rd), done=done, epoch=ep)
            try:
                teacher_acc = json.load(open(os.path.join(rd, "config.json")))["teacher_acc"]
            except Exception:
                pass
print(f"{PAIR}: teacher accuracy measured {teacher_acc} (expected {TEACHER_ACC})")
lines = [f"## {PAIR} (epkd-vision @ {COMMIT[:7]})", "", "| method | " + " | ".join(f"s{s}" for s in SEEDS) + " | mean ± sd |",
         "|---|" + "---|" * (len(SEEDS) + 1)]
summary = dict(pair=PAIR, teacher=TEACHER, student=STUDENT, lr=LR, commit=COMMIT, teacher_acc=teacher_acc, runs={})
for m in METHODS:
    cells, vals = [], []
    for s in SEEDS:
        r = res.get((m, s))
        if r is None or r["best"] is None:
            cells.append("-")
            continue
        summary["runs"][f"{m}_s{s}"] = r
        cells.append(f"{r['best']:.2f}" + ("" if r["done"] else f" (ep {r['epoch']}, running)"))
        if r["done"]:
            vals.append(r["best"])
    ms = f"{np.mean(vals):.2f} ± {np.std(vals, ddof=1):.2f} (n={len(vals)})" if len(vals) > 1 else "-"
    summary[m] = dict(mean=float(np.mean(vals)) if vals else None,
                      sd=float(np.std(vals, ddof=1)) if len(vals) > 1 else None, n=len(vals))
    lines.append(f"| {m} | " + " | ".join(cells) + f" | {ms} |")
paired = [res[("epkd", s)]["best"] - res[("kd", s)]["best"] for s in SEEDS
          if res.get(("epkd", s), {}).get("done") and res.get(("kd", s), {}).get("done")]
if len(paired) > 1:
    d, sd = float(np.mean(paired)), float(np.std(paired, ddof=1))
    h = T95[len(paired) - 1] * sd / np.sqrt(len(paired))
    summary["delta"] = dict(mean=d, ci95=[d - h, d + h], n=len(paired))
    lines += ["", f"EP-KD − KD (paired, n={len(paired)}): {d:+.2f} [{d - h:+.2f}, {d + h:+.2f}]"]
unfinished = [f"{m}_s{s}" for m in METHODS for s in SEEDS
              if (m == "epkd" or RUN_KD) and not res.get((m, s), {}).get("done")]
if unfinished:
    lines += ["", f"UNFINISHED: {unfinished}. Attach this version's output and run again (see the top cell)."]
text = "\\n".join(lines)
print(text)
open(f"/kaggle/working/summary_{PAIR}.md", "w").write(text + "\\n")
json.dump(summary, open(f"/kaggle/working/summary_{PAIR}.json", "w"), indent=1)"""


def cell(kind, src):
    c = {"cell_type": kind, "metadata": {}, "source": src.splitlines(keepends=True)}
    if kind == "code":
        c.update(execution_count=None, outputs=[])
    return c


def notebook(t, s, tacc, lr):
    fmt = dict(t=t, s=s, tp=PRETTY[t], sp=PRETTY[s], tacc=tacc, lr=lr)
    cells = [cell("markdown", INTRO.format(**fmt)), cell("code", CONFIG.format(**fmt)), cell("code", SETUP),
             cell("code", DATA), cell("code", TEACHER_CELL), cell("code", RESTORE), cell("code", RUN),
             cell("code", SUMMARY)]
    return {"cells": cells, "nbformat": 4, "nbformat_minor": 4,
            "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                         "language_info": {"name": "python"},
                         "kaggle": {"accelerator": "nvidiaTeslaT4", "isGpuEnabled": True,
                                    "isInternetEnabled": True, "language": "python",
                                    "sourceType": "notebook", "dataSources": []}}}


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    for t, s, tacc, lr in PAIRS:
        path = os.path.join(here, f"epkd_cifar100_{t}_to_{s}.ipynb")
        with open(path, "w") as f:
            json.dump(notebook(t, s, tacc, lr), f, indent=1)
        print(path)
