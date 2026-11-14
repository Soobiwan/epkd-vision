#!/usr/bin/env python3
"""VM-side detached runner for the full run: download -> precompute (+finalize +verify) -> val cache -> F1 -> F2.
Idempotent and resumable: every phase checks its completion marker on disk first.  Status in
<work>/logs/pipeline_status.json, log <work>/logs/pipeline.log.  Control: touch <work>/STOP (stops after the
current epoch's checkpoint; the trainer resumes from latest.pt on the next start).

    nohup python colab/pipeline.py --device tpu --work /content/work [--train-shards 294] > /content/work/logs/pipeline.log 2>&1 &
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
# teacher/student pairs: (F1 run name, F2 run name); configs are ref/configs/<run name>.json
PAIRS = {
    "F": ("F1_rn34_rn18_epkd_e001-080", "F2_rn34_rn18_epkd_e081-100_cosine"),   # ResNet34 -> ResNet18
    "E": ("E1_rn50_mnv1_epkd_e001-080", "E2_rn50_mnv1_epkd_e081-100_cosine"),   # ResNet50 -> MobileNetV1
}
N1, N2 = PAIRS[os.environ.get("EPKD_PAIR", "F")]
F1 = ROOT / f"ref/configs/{N1}.json"
F2 = ROOT / f"ref/configs/{N2}.json"
PY = sys.executable


def now() -> str:
    return dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


class Pipeline:
    def __init__(self, a: argparse.Namespace) -> None:
        self.a = a
        self.work = Path(a.work)
        self.logs = self.work / "logs"
        self.logs.mkdir(parents=True, exist_ok=True)
        self.status_path = self.logs / "pipeline_status.json"
        self.state: dict = {"started": now(), "events": []}
        self.env = dict(os.environ, PYTHONUNBUFFERED="1", EPKD_CACHE_DIR=str(self.work / "cache"), EPKD_IMAGENET_DIR=a.data)
        self.child: subprocess.Popen | None = None
        signal.signal(signal.SIGTERM, self._term)

    def _term(self, *_a):
        if self.child and self.child.poll() is None:
            self.child.terminate()
        sys.exit(1)

    def log(self, msg: str) -> None:
        print(f"[{now()}] {msg}", flush=True)
        self.state["events"] = (self.state.get("events", []) + [f"{now()} {msg}"])[-100:]
        self.status()

    def status(self, **kw) -> None:
        self.state.update({"time": now(), **kw})
        tmp = self.status_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.state, indent=1, default=str))
        tmp.replace(self.status_path)

    def stop_requested(self) -> bool:
        return (self.work / "STOP").exists()

    def run(self, args: list[str], logfile: str, cwd: Path = ROOT) -> int:
        self.log(f"run: {' '.join(args)} -> {logfile}")
        with open(self.logs / logfile, "a") as fh:
            fh.write(f"\n===== {now()} {' '.join(args)}\n")
            fh.flush()
            self.child = subprocess.Popen(args, stdout=fh, stderr=subprocess.STDOUT, cwd=str(cwd), env=self.env)
            rc = self.child.wait()
        self.child = None
        self.log(f"rc={rc}")
        return rc

    # ---------------------------------------------------------------- phases
    def download(self) -> None:
        st = Path(self.a.data) / "download_status.json"
        if st.exists() and json.loads(st.read_text()).get("complete"):
            self.log("download: complete")
            return
        self.status(phase="download")
        rc = self.run([PY, str(ROOT / "colab/download_hf.py"), "--out", self.a.data, "--train-shards", str(self.a.train_shards),
                       "--val-shards", str(self.a.val_shards), "--workers", str(self.a.download_workers)], "download.log")
        if rc != 0:
            raise SystemExit("download failed")

    def precompute(self) -> None:
        base = [PY, "-m", "epkd_colab.precompute_hybrid", "--configs", str(F1), str(F2), "--chunk", str(self.a.pre_chunk)]
        out = subprocess.run(base + ["--status"], capture_output=True, text=True, cwd=str(ROOT), env=self.env)
        try:
            s = json.loads(out.stdout.strip().splitlines()[-1])
        except Exception:  # noqa: BLE001
            self.log(f"precompute --status failed: {out.stdout[-500:]} {out.stderr[-800:]}")
            raise
        verify = self.logs / ("verify_F.json" if N1.startswith("F1_") else f"verify_{N1[0]}.json")
        if s.get("complete") and verify.exists() and json.loads(verify.read_text()).get("all_ok"):
            self.log(f"precompute: complete and verified ({s['roots']})")
            return
        self.status(phase="precompute", precompute=s)
        if s["done"] < s["total"]:
            self.run(base + ["--prepare"], "precompute.log")
            for attempt in range(5):
                rc = self.run(base + ["--device", self.a.device, "--head-k", str(self.a.head_k), "--union-batch", str(self.a.pre_batch),
                                      "--score-batch", "256", "--score-dtype", "bf16", "--cpu-procs", str(self.a.cpu_procs),
                                      "--decode-workers", str(self.a.pre_workers), "--log-every", "2000"], "precompute.log")
                out = subprocess.run(base + ["--status"], capture_output=True, text=True, cwd=str(ROOT), env=self.env)
                s = json.loads(out.stdout.strip().splitlines()[-1])
                self.status(precompute=s)
                if s["done"] >= s["total"]:
                    break
                if self.stop_requested():
                    raise SystemExit("STOP")
                self.log(f"precompute incomplete ({s['done']}/{s['total']}), retrying")
            if s["done"] < s["total"]:
                raise SystemExit("precompute failed")
        if not s.get("complete"):
            rc = self.run(base + ["--finalize"], "precompute.log")
            if rc != 0:
                raise SystemExit("finalize failed")
        # tolerant CPU-reference check: near-tie window swaps allowed (>= 95 % identical boxes); scores are bf16 and unused
        # by training (kd_gate off), so their tolerance is loose
        rc = self.run([PY, str(ROOT / "tools/verify_cache.py"), "--configs", str(F1), str(F2), "--n", str(self.a.verify_n), "--device",
                       "cpu", "--min-box-rate", "0.95", "--score-rtol", "2e-2", "--score-atol", "1e-3", "--score-gate", "off", "--out", str(verify)],
                      "verify.log")
        if rc != 0 or not json.loads(verify.read_text()).get("all_ok"):
            raise SystemExit("cache verification FAILED -- not training on it")
        self.log("precompute verified")

    def train(self, name: str, cfg: Path, last_epoch: int, init_from: Path | None) -> None:
        run_dir = self.work / "runs" / name
        done = run_dir / "DONE_EPOCH"
        if done.exists() and int(done.read_text().strip() or 0) >= last_epoch:
            self.log(f"{name}: done (epoch {done.read_text().strip()})")
            return
        self.status(phase=name)
        args = [PY, "-m", "epkd_colab.train", "--config", str(cfg), "--device", self.a.device, "--run-dir", str(run_dir),
                "--dtype", self.a.dtype, "--compile-scope", self.a.compile_scope, "--decode-procs", str(self.a.decode_procs),
                "--progress-every", "200", "--val-cache", str(self.work / "cache" / "val_u8.npy")] + self.a.train_args.split()
        if init_from is not None:
            args += ["--init-student-from", str(init_from)]
        failures = 0
        while True:
            if self.stop_requested():
                raise SystemExit("STOP")
            rc = self.run(args, f"{name}.log")
            if done.exists() and int(done.read_text().strip() or 0) >= last_epoch:
                return
            failures += 1
            if failures > self.a.max_failures:
                raise SystemExit(f"{name}: too many failures")
            self.log(f"{name}: exited rc={rc} before epoch {last_epoch}; restarting in 60 s (auto-resume)")
            time.sleep(60)

    def main(self) -> None:
        self.log(f"pipeline start device={self.a.device} work={self.work} pair={N1},{N2}")
        self.download()
        self.precompute()
        self.train(N1, F1, 80, None)
        self.train(N2, F2, 100, self.work / f"runs/{N1}/epoch_080.pt")
        self.status(phase="done")
        self.log("ALL DONE")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="tpu")
    ap.add_argument("--work", default="/content/work")
    ap.add_argument("--data", default="/media/e12056/ASi/imagenet/data")
    ap.add_argument("--train-shards", type=int, default=294)
    ap.add_argument("--val-shards", type=int, default=14)
    ap.add_argument("--download-workers", type=int, default=4)
    ap.add_argument("--pre-batch", type=int, default=32, help="union batch per head shape")
    ap.add_argument("--pre-workers", type=int, default=8, help="decode workers")
    ap.add_argument("--pre-chunk", type=int, default=32768)
    ap.add_argument("--head-k", type=int, default=20)
    ap.add_argument("--cpu-procs", type=int, default=max(2, (os.cpu_count() or 24) - 10))
    ap.add_argument("--verify-n", type=int, default=60)
    ap.add_argument("--dtype", default="bf16")
    ap.add_argument("--compile-scope", default="model", help="model: the only compiled scope whose updates are correct on torch_tpu")
    ap.add_argument("--decode-procs", type=int, default=6)
    ap.add_argument("--train-args", default="")
    ap.add_argument("--max-failures", type=int, default=20)
    a = ap.parse_args()
    Pipeline(a).main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
