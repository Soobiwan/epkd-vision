"""Runs of the CUB / Pets view-rules table for the ResNet32x4 teacher, and their split over the Kaggle notebooks.

Run name: <dataset>_<pair>_<arm>_s<seed>, e.g. cub_sv2_prop_s0. Arms (only --view-set differs between the EP-KD arms):
  prop  EP-KD, the teacher's proposals (3 highest-CAM grid windows per stage)       --view-set sal1,sal2,sal3
  frn   EP-KD, 3 grid-free random windows (uniform over all valid pixel offsets),   --view-set frn1,frn2,frn3
        drawn once per image and stage and fixed for the stage (epkd2 runs {cub,pets}_frn_anc / frc_anc)
  low   EP-KD, the 3 least-salient grid windows                                    --view-set low1,low2,low3
  kd    plain KD, 240 epochs, LR x0.1 at 150/180/210                               --views 0
The command lines are those of the WRN-40-2 -> WRN-40-1 CUB / Pets runs (epkd2/cubucf/make_jobs.sh, runs
{cub,pets}_{prop_anc,frn_anc,low_anc,kd}_s*) with the teacher / student / learning rate of the pair.
"""

DATASETS = ("cub", "pets")
TEACHER = "resnet32x4"
# pair -> (student, base LR). LR: the CRD / DKD setting for the student (0.01 for ShuffleNets, 0.05 otherwise).
PAIRS = {"r8x4": ("resnet8x4", 0.05), "sv2": ("ShuffleV2", 0.01)}
SEEDS = (0, 1, 2)
ARMS = ("prop", "frn", "low", "kd")

# epkd2/cubucf/make_jobs.sh: COM / A (the paper's 48-96 px ladder, grid stride 24, view pad 6)
_EP = ["--view-sizes", "48,60,72,84,96", "--grid-stride", "24", "--view-pad", "6", "--view-cache", "1",
       "--view-rotate", "1", "--passes", "3,3,3,1,1", "--anchor", "1", "--milestones", "48,144,192,210", "--views", "1"]
ARM_ARGS = {
    "prop": _EP + ["--view-set", "sal1,sal2,sal3"],
    "frn": _EP + ["--view-set", "frn1,frn2,frn3"],
    "low": _EP + ["--view-set", "low1,low2,low3"],
    "kd": ["--views", "0"],
}

# Teacher checkpoints: GitHub release assets (best-test-epoch state dict + json with sha256 and test accuracy)
RELEASE = "https://github.com/Soobiwan/epkd-vision/releases/download/cubpets-teachers"


def teacher_file(ds):
    return f"tea_{ds}_r32x4"


def parse(name):
    ds, pair, arm, s = name.split("_")
    assert ds in DATASETS and pair in PAIRS and arm in ARMS and s[0] == "s", name
    return ds, pair, arm, int(s[1:])


def train_args(name, data_root, teacher_ckpt):
    """train.py arguments of a run (without --run-dir)."""
    ds, pair, arm, seed = parse(name)
    student, lr = PAIRS[pair]
    return (["--dataset", ds, "--data-root", data_root, "--teacher", TEACHER, "--teacher-ckpt", teacher_ckpt,
             "--student", student, "--lr", str(lr), "--bs", "64"] + ARM_ARGS[arm] + ["--seed", str(seed)])


# Expected hours on one Kaggle T4 (one job per GPU). Per-step times from a micro-benchmark on an RTX 2080 Ti (teacher
# forward + student forward/backward at bs 64, 96 px and at the view sizes; they match the CUB teacher's measured
# 24.8 s/epoch alone), times the step counts of the schedule, plus evaluation; a T4 taken as ~2.1x slower than a
# dedicated 2080 Ti for these AMP conv nets. Check: the WRN-40-2 -> WRN-40-1 CUB EP-KD runs took 3.8-4.1 h on a shared
# 2080 Ti. Uncertain by roughly +-30 %.
EST_H = {("cub", "sv2", "ep"): 9.2, ("cub", "r8x4", "ep"): 7.2, ("pets", "sv2", "ep"): 5.6, ("pets", "r8x4", "ep"): 4.4,
         ("cub", "sv2", "kd"): 3.1, ("cub", "r8x4", "kd"): 2.5, ("pets", "sv2", "kd"): 1.9, ("pets", "r8x4", "kd"): 1.5}


def est_hours(name):
    ds, pair, arm, _ = parse(name)
    return EST_H[(ds, pair, "kd" if arm == "kd" else "ep")]


# Notebook -> (GPU 0 queue, GPU 1 queue). Each GPU runs one job at a time and starts the first job of its queue whose
# teacher is available (so a Pets job fills the wait for the CUB teacher); a GPU with an empty queue takes the last
# not-yet-started job of the other GPU's queue. Longest runs (CUB EP-KD) one per GPU; Pets KD / CUB KD fill the
# GPUs of the shorter runs. Expected hours per GPU queue in the comments.
NOTEBOOKS = {
    "01": (["cub_sv2_prop_s0"], ["cub_sv2_frn_s0"]),                                             # 9.2 | 9.2
    "02": (["cub_sv2_low_s0"], ["cub_sv2_prop_s1"]),                                              # 9.2 | 9.2
    "03": (["cub_sv2_frn_s1"], ["cub_sv2_low_s1"]),                                              # 9.2 | 9.2
    "04": (["cub_sv2_prop_s2"], ["cub_sv2_frn_s2"]),                                             # 9.2 | 9.2
    "05": (["cub_sv2_low_s2"], ["cub_r8x4_prop_s0"]),                                             # 9.2 | 7.2
    "06": (["cub_r8x4_frn_s0", "pets_sv2_kd_s0"], ["cub_r8x4_low_s0", "pets_sv2_kd_s1"]),        # 9.1 | 9.1
    "07": (["cub_r8x4_prop_s1", "pets_sv2_kd_s2"], ["cub_r8x4_frn_s1", "pets_r8x4_kd_s0"]),      # 9.1 | 8.7
    "08": (["cub_r8x4_low_s1", "pets_r8x4_kd_s1"], ["cub_r8x4_prop_s2", "pets_r8x4_kd_s2"]),      # 8.7 | 8.7
    "09": (["cub_r8x4_frn_s2"], ["cub_r8x4_low_s2"]),                                            # 7.2 | 7.2
    "10": (["pets_sv2_prop_s0", "cub_sv2_kd_s0"], ["pets_sv2_frn_s0", "cub_sv2_kd_s1"]),         # 8.7 | 8.7
    "11": (["pets_sv2_low_s0", "cub_sv2_kd_s2"], ["pets_sv2_prop_s1", "cub_r8x4_kd_s0"]),         # 8.7 | 8.1
    "12": (["pets_sv2_frn_s1", "cub_r8x4_kd_s1"], ["pets_sv2_low_s1", "cub_r8x4_kd_s2"]),        # 8.1 | 8.1
    "13": (["pets_sv2_prop_s2"], ["pets_sv2_frn_s2"]),                                           # 5.6 | 5.6
    "14": (["pets_sv2_low_s2"], ["pets_r8x4_prop_s0"]),                                           # 5.6 | 4.4
    "15": (["pets_r8x4_frn_s0", "pets_r8x4_low_s0"], ["pets_r8x4_prop_s1", "pets_r8x4_frn_s1"]),  # 8.8 | 8.8
    "16": (["pets_r8x4_low_s1", "pets_r8x4_prop_s2"], ["pets_r8x4_frn_s2", "pets_r8x4_low_s2"]),  # 8.8 | 8.8
}


def all_runs():
    return [f"{ds}_{p}_{a}_s{s}" for ds in DATASETS for p in PAIRS for a in ARMS for s in SEEDS]


def _check():
    got = sorted(n for q0, q1 in NOTEBOOKS.values() for n in q0 + q1)
    assert got == sorted(all_runs()), set(all_runs()) ^ set(got)


_check()
