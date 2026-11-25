"""Tabulates the CUB / Pets view-rules runs (ResNet32x4 teacher) from downloaded Kaggle outputs.

  python kaggle/collect_cubpets.py DIR_OR_ZIP [...]  [--csv out.csv] [--metric best|final]

Reads every cubpets/results/<run>.json (written by cubpets/kaggle_run.py) and every run folder with done.json +
config.json (e.g. runs/<run>/ of a local copy), from folders (searched recursively) and from cubpets_*.zip files.
If a run appears more than once, the copy with the most epochs logged wins (ties: the first found).
Only v3 runs are counted (frn arm, best-epoch teachers): results with "version": "v3", or run folders with
cubpets_run.json version v3. --any-version counts everything (e.g. the v1/v2 rfix / final-epoch-teacher runs).
Prints, per dataset and pair, arm x (mean ± sd over seeds, n, Δ vs prop = mean of the per-seed differences on the
seeds both arms have), and lists missing runs.
"""
import argparse
import glob
import json
import os
import statistics
import sys
import zipfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cubpets"))
import plan  # noqa: E402

ARM_LABEL = {"prop": "EP-KD (teacher proposals)", "frn": "random grid-free windows, fixed per stage",
             "low": "least-salient windows", "kd": "KD"}


def _from_run(name, done, cfg, n_ep):
    return dict(run=name, best=done["best"], final=done["final"], best_epoch=done["best_epoch"],
                teacher_acc=cfg.get("teacher_acc"), epochs=cfg.get("epochs"), n_ep=n_ep)


VERSION = "v3"


def scan(paths, any_version=False):
    found = {}

    def ok_mark(text):
        try:
            return any_version or json.loads(text).get("version") == VERSION
        except Exception:  # noqa: BLE001
            return any_version

    def add(r):
        try:
            plan.parse(r["run"])
        except (AssertionError, ValueError):
            return
        old = found.get(r["run"])
        if old is None or r.get("n_ep", 0) > old.get("n_ep", 0):
            found[r["run"]] = r

    for p in paths:
        zips = [p] if p.endswith(".zip") else glob.glob(os.path.join(p, "**", "*.zip"), recursive=True)
        if os.path.isdir(p):
            for rp in glob.glob(os.path.join(p, "**", "results", "*.json"), recursive=True):
                r = json.load(open(rp))
                r["n_ep"] = r.get("epochs") or 0
                if any_version or r.get("version") == VERSION:
                    add(r)
            for dp in glob.glob(os.path.join(p, "**", "done.json"), recursive=True):
                d = os.path.dirname(dp)
                mk = os.path.join(d, "cubpets_run.json")
                if os.path.exists(os.path.join(d, "config.json")) and ok_mark(open(mk).read() if os.path.exists(mk) else ""):
                    n = sum(1 for _ in open(os.path.join(d, "train.jsonl"))) if os.path.exists(
                        os.path.join(d, "train.jsonl")) else 0
                    add(_from_run(os.path.basename(d), json.load(open(dp)), json.load(open(os.path.join(d, "config.json"))), n))
        for zp in zips:
            try:
                zf = zipfile.ZipFile(zp)
            except zipfile.BadZipFile:
                continue
            names = set(zf.namelist())
            for m in names:
                if m.endswith("/done.json"):
                    d = m[:-len("done.json")]
                    if d + "config.json" in names and ok_mark(zf.read(d + "cubpets_run.json") if d + "cubpets_run.json" in names else ""):
                        n = len(zf.read(d + "train.jsonl").splitlines()) if d + "train.jsonl" in names else 0
                        add(_from_run(d.rstrip("/").split("/")[-1], json.loads(zf.read(m)),
                                      json.loads(zf.read(d + "config.json")), n))
    return found


def fmt(xs):
    if not xs:
        return "—"
    m = statistics.mean(xs)
    return f"{m:.2f} ± {statistics.stdev(xs):.2f}" if len(xs) > 1 else f"{m:.2f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--metric", default="best", choices=["best", "final"])
    ap.add_argument("--csv", default="")
    ap.add_argument("--any-version", action="store_true", help="also count v1/v2 runs (rfix arm, final-epoch teachers)")
    a = ap.parse_args()
    found = scan(a.paths, a.any_version)
    val = {}
    for name, r in found.items():
        ds, pair, arm, seed = plan.parse(name)
        val[(ds, pair, arm, seed)] = r[a.metric]
    rows = []
    for ds in plan.DATASETS:
        for pair in plan.PAIRS:
            t = [found[n]["teacher_acc"] for n in found if n.startswith(f"{ds}_{pair}_") and found[n].get("teacher_acc")]
            print(f"\n### {ds} | ResNet32x4 -> {plan.PAIRS[pair][0]} (teacher test acc {fmt(t[:1])}; {a.metric} test acc, %)")
            print("| arm | seeds 0/1/2 | mean ± sd | n | Δ vs prop |")
            print("|---|---|---|---|---|")
            for arm in plan.ARMS:
                xs = {s: val[(ds, pair, arm, s)] for s in plan.SEEDS if (ds, pair, arm, s) in val}
                ps = {s: val[(ds, pair, "prop", s)] for s in plan.SEEDS if (ds, pair, "prop", s) in val}
                common = sorted(set(xs) & set(ps))
                delta = "" if arm == "prop" or not common else \
                    f"{statistics.mean(xs[s] - ps[s] for s in common):+.2f} (n={len(common)})"
                seeds = " / ".join(f"{xs[s]:.2f}" if s in xs else "—" for s in plan.SEEDS)
                print(f"| {ARM_LABEL[arm]} | {seeds} | {fmt(list(xs.values()))} | {len(xs)} | {delta} |")
                rows.append(dict(dataset=ds, pair=pair, arm=arm, n=len(xs),
                                 mean=statistics.mean(xs.values()) if xs else None,
                                 sd=statistics.stdev(xs.values()) if len(xs) > 1 else None,
                                 delta_vs_prop=None if not delta else statistics.mean(xs[s] - ps[s] for s in common),
                                 **{f"s{s}": xs.get(s) for s in plan.SEEDS}))
    missing = [n for n in plan.all_runs() if n not in found]
    print(f"\n{len(found)} of {len(plan.all_runs())} runs found" + (f"; missing: {' '.join(missing)}" if missing else ""))
    if a.csv:
        import csv
        with open(a.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print("wrote", a.csv)


if __name__ == "__main__":
    main()
