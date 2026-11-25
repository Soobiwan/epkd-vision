"""Generates kaggle/cubpets_NN.ipynb (NN = 01..16): the CUB-200 / Oxford-Pets view-rules table for the ResNet32x4
teacher (students ResNet8x4 and ShuffleNetV2, arms prop / rfix / low / kd, seeds 0-2; 48 runs). Which runs each
notebook trains is in cubpets/plan.py (NOTEBOOKS); all logic is in cubpets/kaggle_run.py, so a notebook is three
cells: clone, run, summary. See kaggle/CUBPETS_README.md.

Usage: python kaggle/make_cubpets_notebooks.py [--ref TAG_OR_COMMIT]
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "cubpets"))
import plan  # noqa: E402

ARM = {"prop": "EP-KD, teacher proposals", "rfix": "EP-KD, random windows fixed per stage",
       "low": "EP-KD, least-salient windows", "kd": "KD"}
DS = {"cub": "CUB-200", "pets": "Oxford-Pets"}
PAIR = {"r8x4": "ResNet32x4 → ResNet8x4", "sv2": "ResNet32x4 → ShuffleNetV2"}

INTRO = """# EP-KD view rules on CUB-200 / Oxford-Pets (96 px), notebook {nb} of 16

Part of the CUB / Pets "view rules" table for the ResNet32x4 teacher (code: `cubpets/` of
[Soobiwan/epkd-vision](https://github.com/Soobiwan/epkd-vision), the code of the paper's CUB / Pets runs).
This notebook trains:

| GPU | runs, in order (expected hours on a T4) |
|---|---|
{rows}

Expected wall time: **~{hours:.1f} h** (uncertain by about ±40 %; a time guard stops cleanly at 11.5 h).

**Kaggle settings (right panel):** Accelerator **GPU T4 x2**, Internet **on**, no input needed. Then
**Save Version → Save & Run All (Commit)**; the run continues after you close the browser.

The script downloads CUB / Pets from Hugging Face and rebuilds the 96-px tensors (content sha256 checked against the
paper's), downloads the ResNet32x4 teachers from the GitHub release `cubpets-teachers` (sha256 checked; if a teacher
is not published yet, the script polls for it and the GPU runs another of its jobs meanwhile), and resumes runs from
a previous version of this notebook attached as input (**Add Input → Your Work → Notebooks → this notebook**, then
Save & Run All again). Results: `cubpets/results/*.json`, `cubpets/summary.json` and `cubpets_nb{nb}.zip` in the
output. Details: `kaggle/CUBPETS_README.md`."""

CLONE = """import os, subprocess
NOTEBOOK = "{nb}"
REPO = os.environ.get("CUBPETS_REPO", "https://github.com/Soobiwan/epkd-vision.git")
REF = os.environ.get("CUBPETS_REF", "{ref}")   # pinned tag of epkd-vision
CODE = "/kaggle/temp/epkd-vision" if os.path.isdir("/kaggle/working") else os.path.abspath("epkd-vision")
ARGS = os.environ.get("CUBPETS_ARGS", "")      # empty on Kaggle (local tests only)
os.makedirs(os.path.dirname(CODE), exist_ok=True)
if not os.path.isdir(CODE):
    subprocess.run(["git", "clone", "-q", REPO, CODE], check=True)
subprocess.run(["git", "-C", CODE, "fetch", "-q", "--tags", "--force", "origin"], check=True)
subprocess.run(["git", "-C", CODE, "checkout", "-q", "--force", REF], check=True)
print(subprocess.run(["git", "-C", CODE, "log", "-1", "--format=%H %cd %s"], capture_output=True, text=True).stdout)
print(subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout)"""

RUN = """# all runs of this notebook (data, teachers, both GPUs, resume, 11.5-h guard, zip); progress is printed every 10 min
!python {CODE}/cubpets/kaggle_run.py --notebook {NOTEBOOK} {ARGS}"""

SUMMARY = """!python {CODE}/cubpets/kaggle_run.py --summary {ARGS}"""


def cell(kind, src):
    c = {"cell_type": kind, "metadata": {}, "source": src.splitlines(keepends=True)}
    if kind == "code":
        c.update(execution_count=None, outputs=[])
    return c


def notebook(nb, ref):
    rows = []
    hours = 0.0
    for g, q in enumerate(plan.NOTEBOOKS[nb]):
        h = sum(plan.est_hours(n) for n in q)
        hours = max(hours, h)
        rows.append(f"| {g} | " + "; ".join(
            f"`{n}` ({DS[plan.parse(n)[0]]}, {PAIR[plan.parse(n)[1]]}, {ARM[plan.parse(n)[2]]}, "
            f"seed {plan.parse(n)[3]}, ~{plan.est_hours(n):.1f} h)" for n in q) + " |")
    cells = [cell("markdown", INTRO.format(nb=nb, rows="\n".join(rows), hours=hours)),
             cell("code", CLONE.format(nb=nb, ref=ref)), cell("code", RUN), cell("code", SUMMARY)]
    return {"cells": cells, "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
        "kaggle": {"accelerator": "nvidiaTeslaT4", "isInternetEnabled": True, "isGpuEnabled": True}},
        "nbformat": 4, "nbformat_minor": 4}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default="cubpets-v2")
    a = ap.parse_args()
    for nb in plan.NOTEBOOKS:
        p = os.path.join(HERE, f"cubpets_{nb}.ipynb")
        json.dump(notebook(nb, a.ref), open(p, "w"), indent=1)
        print(p)


if __name__ == "__main__":
    main()
