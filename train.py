"""EP-KD for vision classification: GPU-resident training script.

Evidence-Preserving Knowledge Distillation (EP-KD) trains the student on teacher-proposed restricted views of
each image (native-resolution crops ranked by the teacher's class-activation map), under a progressive evidence
budget that ends on the full image, while keeping the full-image distillation objective at every step.

`--views 0` gives plain KD; `--teacher none` trains a network with cross-entropy (used for teachers).
See README.md for the exact commands of the method, the baselines and the ablations.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import wrn  # noqa: E402

DATASETS = {
    "cifar100": dict(num_classes=100, size=32, pad=4,
                     mean=(0.5071, 0.4867, 0.4408), std=(0.2675, 0.2565, 0.2761)),
    "stl10": dict(num_classes=10, size=96, pad=12,
                  mean=(0.4467, 0.4398, 0.4066), std=(0.2603, 0.2566, 0.2713)),
    "tinyimagenet": dict(num_classes=200, size=64, pad=8,
                         mean=(0.4802, 0.4481, 0.3975), std=(0.2770, 0.2691, 0.2821)),
}


# ----------------------------------------------------------------------------- data
def load_dataset(name: str, root: str):
    """Returns uint8 NCHW tensors (train_x, train_y, test_x, test_y)."""
    if name == "cifar100":
        def _load(split):
            d = pickle.load(open(os.path.join(root, "cifar-100-python", split), "rb"), encoding="latin1")
            x = np.asarray(d["data"], dtype=np.uint8).reshape(-1, 3, 32, 32)
            return torch.from_numpy(x), torch.tensor(d["fine_labels"], dtype=torch.long)
        return (*_load("train"), *_load("test"))
    path = os.path.join(root, f"{name}_uint8.pt")
    d = torch.load(path)
    return d["train_x"], d["train_y"], d["test_x"], d["test_y"]


class GPUAugment:
    """Reflect-pad random crop + horizontal flip on normalised GPU tensors (= torchvision RandomCrop + Flip)."""

    def __init__(self, mean, std, size, pad, device):
        self.mean = torch.tensor(mean, device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(std, device=device).view(1, 3, 1, 1)
        self.size, self.pad = size, pad

    def normalize(self, xb_uint8):
        return (xb_uint8.float().div_(255.0) - self.mean) / self.std

    def __call__(self, xb_uint8):
        x = self.normalize(xb_uint8)
        b, s, p = x.size(0), self.size, self.pad
        xp = F.pad(x, (p, p, p, p), mode="reflect")
        off = torch.randint(0, 2 * p + 1, (b, 2), device=x.device)
        x = crop_batch(xp, off[:, 0], off[:, 1], s)
        flip = torch.rand(b, device=x.device) < 0.5
        return torch.where(flip.view(-1, 1, 1, 1), x.flip(-1), x)


def crop_batch(x, r, c, s):
    """Per-sample s x s crop with top-left (r, c)."""
    b, ch = x.size(0), x.size(1)
    ar = torch.arange(s, device=x.device)
    rows = (r.view(-1, 1) + ar).view(b, 1, s, 1)
    cols = (c.view(-1, 1) + ar).view(b, 1, 1, s)
    bi = torch.arange(b, device=x.device).view(b, 1, 1, 1)
    ci = torch.arange(ch, device=x.device).view(1, ch, 1, 1)
    return x[bi, ci, rows, cols]


def extract_view(x, r, c, s, view_type):
    """Restricted view of window (r, c, s): a native-resolution crop, or the window kept in place with the
    rest of the image set to the dataset mean (0 after normalisation)."""
    if view_type == "crop":
        return crop_batch(x, r, c, s)
    h = x.size(-1)
    ar = torch.arange(h, device=x.device)
    rin = (ar.view(1, -1) >= r.view(-1, 1)) & (ar.view(1, -1) < (r + s).view(-1, 1))
    cin = (ar.view(1, -1) >= c.view(-1, 1)) & (ar.view(1, -1) < (c + s).view(-1, 1))
    m = (rin.unsqueeze(2) & cin.unsqueeze(1)).unsqueeze(1).to(x.dtype)
    return x * m


# ----------------------------------------------------------------------------- models
def teacher_forward(t, x):
    """Lean WRN forward (eval mode): logits and the pre-pool feature map used for CAM."""
    out = t.block3(t.block2(t.block1(t.conv1(x))))
    f = t.relu(t.bn1(out))
    return t.fc(f.mean((2, 3))), f


def student_logits(s, x):
    return s(x)[0]


class NoBNUpdate:
    """Use batch statistics but leave BN running statistics untouched (views must not shift test-time BN)."""

    def __init__(self, model, enabled=True):
        self.bns = [m for m in model.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)] if enabled else []

    def __enter__(self):
        self.saved = [m.momentum for m in self.bns]
        for m in self.bns:
            m.momentum = 0.0

    def __exit__(self, *a):
        for m, mo in zip(self.bns, self.saved):
            m.momentum = mo


# ----------------------------------------------------------------------------- losses
def kd_per_sample(s_logits, t_logits, T):
    return F.kl_div(F.log_softmax(s_logits.float() / T, 1), F.softmax(t_logits.float() / T, 1),
                    reduction="none").sum(1) * (T * T)


def kl_teacher(t_full, t_view, tau):
    """D_T(v, x) = KL(p_T(.|x) || p_T(.|v)) at temperature tau, per sample."""
    p = F.softmax(t_full.float() / tau, 1)
    return (p * (F.log_softmax(t_full.float() / tau, 1) - F.log_softmax(t_view.float() / tau, 1))).sum(1)


# ----------------------------------------------------------------------------- evidence proposals
def cam_map(f, fc_w, y, out_size):
    """Gradient-free CAM of class y (exact Grad-CAM for a GAP-linear head), upsampled, normalised to max 1."""
    cam = torch.einsum("bchw,bc->bhw", f.float(), fc_w[y].float()).clamp_min(0).unsqueeze(1)
    cam = F.interpolate(cam, size=(out_size, out_size), mode="bilinear", align_corners=False)
    return cam / cam.flatten(1).amax(1).clamp_min(1e-8).view(-1, 1, 1, 1)


def sample_windows(cam, s, n, pos_temp, mode, generator=None):
    """Sample n window positions of size s per image. Returns rows, cols of shape [B, n]."""
    b, _, h, w = cam.shape
    gh, gw = h - s + 1, w - s + 1
    if mode == "random":
        idx = torch.randint(0, gh * gw, (b, n), device=cam.device)
    else:
        mass = F.avg_pool2d(cam, s, stride=1).flatten(1)  # [B, gh*gw]
        mass = mass / mass.amax(1, keepdim=True).clamp_min(1e-8)
        if pos_temp <= 0:
            if n == 1:
                idx = mass.argmax(1, keepdim=True)
            else:
                idx = mass.topk(n, 1).indices
        else:
            prob = torch.softmax((mass - 1.0) / pos_temp, 1)
            idx = torch.multinomial(prob, n, replacement=False)
    return idx // gw, idx % gw


def evidence_windows(cam, s, k, pos_temp, iou_thr):
    """k windows of size s on distinct evidence: greedy pick (argmax, or sampled at pos_temp), suppressing
    windows that overlap an earlier pick by IoU > iou_thr. Returns rows, cols of shape [B, k]."""
    gw = cam.size(-1) - s + 1
    mass = F.avg_pool2d(cam, s, stride=1).flatten(1)
    mass = mass / mass.amax(1, keepdim=True).clamp_min(1e-8)
    m = mass.clone()
    pos = torch.arange(mass.size(1), device=cam.device)
    pr, pc = (pos // gw).view(1, -1), (pos % gw).view(1, -1)
    rows, cols = [], []
    for _ in range(k):
        mu = torch.where(torch.isfinite(m).any(1, keepdim=True), m, mass)  # all suppressed: allow overlap
        if pos_temp <= 0:
            idx = mu.argmax(1)
        else:
            idx = torch.multinomial(torch.softmax((mu - 1.0) / pos_temp, 1), 1).squeeze(1)
        r, c = idx // gw, idx % gw
        rows.append(r)
        cols.append(c)
        inter = (s - (pr - r.view(-1, 1)).abs()).clamp_min(0) * (s - (pc - c.view(-1, 1)).abs()).clamp_min(0)
        m = torch.where(inter / (2 * s * s - inter) > iou_thr, torch.full_like(m, float("-inf")), m)
    return torch.stack(rows, 1), torch.stack(cols, 1)


# ----------------------------------------------------------------------------- schedules
def lr_at(epoch, args):
    if args.sched == "cosine":
        return args.lr * 0.5 * (1 + math.cos(math.pi * epoch / args.epochs))
    lr = args.lr
    for m in args.milestones:
        if epoch >= m:
            lr *= 0.1
    return lr


def view_size_at(epoch, args):
    """Global per-stage view size (the budget b_t realised as crop size). Returns full size when views are off."""
    if args.view_until > 0 and epoch >= args.view_until:
        return 10 ** 6
    if args.size_sched == "random":  # interleaved control: any restricted size at every step
        sizes = args.view_sizes[:-1] if len(args.view_sizes) > 1 else args.view_sizes
        return sizes[int(torch.randint(len(sizes), (1,)))]
    k = min(epoch // args.stage_epochs, len(args.view_sizes) - 1)
    return args.view_sizes[k]


def stage_index(epoch, args):
    return min(epoch // args.stage_epochs, len(args.view_sizes) - 1)


def passes_at(epoch, args):
    """Passes over the training set in this epoch (each pass draws new augmentations and new views)."""
    return args.passes[min(stage_index(epoch, args), len(args.passes) - 1)]


def views_active(epoch, args, full):
    if args.views <= 0 or (args.view_until > 0 and epoch >= args.view_until):
        return False
    if args.size_mode == "verified":
        return gamma_at(epoch, args) < 1.0
    if args.size_sched == "random":
        return True
    return args.view_sizes[stage_index(epoch, args)] < full


def gamma_at(epoch, args):
    """Budget interpolation gamma_t in [0, 1] (staged: constant within a stage)."""
    n = len(args.view_sizes) - 1  # last entry is the full image
    k = min(epoch // args.stage_epochs, n)
    return k / n


# ----------------------------------------------------------------------------- views
def make_views(x, y, t_full, cams, epoch, args, full, pass_idx=0):
    """Build the restricted views for one batch.

    Returns a list of groups (size, img_idx, v, t_view, D) plus stats. Every group has a single crop size so the
    student and teacher forwards are batched per size.
    """
    groups = []
    b = x.size(0)
    cam = cams["gt"]
    if args.size_mode == "global" and args.view_set != ["sal"]:
        # teacher-reasoning views: distinct evidence for the label (gt1, gt2, ...) and for the runner-up class (conf)
        s = view_size_at(epoch, args)
        if s >= full:
            return groups
        members = [args.view_set[pass_idx % len(args.view_set)]] if args.view_rotate else args.view_set
        n_gt = max([int(m[2:]) for m in args.view_set if m.startswith("gt")] + [0])
        if n_gt:
            gr, gc = evidence_windows(cam, s, n_gt, args.pos_temp, args.iou_thr)
        conf_win = {m: evidence_windows(cams[m], s, 1, args.pos_temp, args.iou_thr)
                    for m in members if m.startswith("conf")}
        ar = torch.arange(b, device=x.device)
        for m in members:
            if m.startswith("conf"):
                r, c = conf_win[m][0][:, 0], conf_win[m][1][:, 0]
            else:
                r, c = gr[:, int(m[2:]) - 1], gc[:, int(m[2:]) - 1]
            v = extract_view(x, r, c, s, args.view_type)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=args.amp):
                tv = teacher_forward(args.teacher, v)[0].float()
            groups.append((s, ar, v, tv, kl_teacher(t_full, tv, args.ver_tau)))
        return groups
    if args.size_mode == "global":
        s = view_size_at(epoch, args)
        if s >= full:
            return groups
        n_cand = max(args.cands, args.views)
        rr, cc = sample_windows(cam, s, n_cand, args.pos_temp, args.pos)
        vs, ds, tl = [], [], []
        for k in range(n_cand):
            v = extract_view(x, rr[:, k], cc[:, k], s, args.view_type)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=args.amp):
                tv, _ = teacher_forward(args.teacher, v)
            vs.append(v)
            tl.append(tv.float())
            ds.append(kl_teacher(t_full, tv, args.ver_tau))
        D = torch.stack(ds, 1)  # [B, n_cand]
        if args.select == "first" or n_cand == args.views:
            order = torch.arange(n_cand, device=x.device).expand(b, n_cand)
        elif args.select == "maxfid":
            order = D.argsort(1)
        elif args.select == "minfid":
            order = D.argsort(1, descending=True)
        elif args.select == "eps":
            # keep sampling order, eligible (D <= eps) first, then the rest by fidelity
            key = torch.where(D <= args.eps, torch.arange(n_cand, device=x.device).float().expand(b, -1),
                              n_cand + D)
            order = key.argsort(1)
        else:
            raise ValueError(args.select)
        V = torch.stack(vs, 1)
        TL = torch.stack(tl, 1)
        ar = torch.arange(b, device=x.device)
        for j in range(args.views):
            k = order[:, j]
            groups.append((s, ar, V[ar, k], TL[ar, k], D[ar, k]))
        return groups

    # size_mode == "verified": EP-KD Eqs. (2) and (4) with per-image budgets
    sizes = args.cand_sizes
    D_all, V_all, T_all = [], [], []
    for s in sizes:
        rr, cc = sample_windows(cam, s, 1, args.pos_temp, args.pos)
        V_all.append(extract_view(x, rr[:, 0], cc[:, 0], s, args.view_type))
    if args.view_type == "mask":  # all candidates share the full resolution: one teacher pass
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=args.amp):
            tv_all, _ = teacher_forward(args.teacher, torch.cat(V_all))
        T_all = list(tv_all.float().split(b))
    else:
        for v in V_all:
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=args.amp):
                T_all.append(teacher_forward(args.teacher, v)[0].float())
    D_all = [kl_teacher(t_full, tv, args.ver_tau) for tv in T_all]
    D = torch.stack(D_all, 1)  # [B, S]
    cost = torch.tensor([(s / full) ** 2 for s in sizes], device=x.device)
    ok = D <= args.eps
    # minimal verified cost C(v*_eps(x)); the full image (cost 1) is always verified
    c_star = torch.where(ok, cost.expand_as(D), torch.ones_like(D)).amin(1)
    g = gamma_at(epoch, args)
    if g >= 1.0:
        return groups
    budget = (1 - g) * c_star + g
    # Eq. (4): verified view whose cost is closest to the budget; ties -> smaller discrepancy
    gap = (cost.view(1, -1) - budget.view(-1, 1)).abs()
    gap = torch.where(ok, gap, torch.full_like(gap, float("inf")))
    full_gap = (1.0 - budget)
    if args.unverified_fallback == "largest":
        # images whose teacher needs the full image still get the largest restricted candidate
        gap_fb = torch.where(ok.any(1, keepdim=True), gap,
                             torch.where(torch.arange(len(sizes), device=x.device) == len(sizes) - 1, 0.0, float("inf")))
        gap = gap_fb
    key = gap + 1e-6 * D
    best = key.argmin(1)
    best_gap = gap.gather(1, best.view(-1, 1)).squeeze(1)
    use_view = best_gap <= full_gap + 1e-9 if args.unverified_fallback != "largest" else torch.ones_like(best_gap, dtype=torch.bool)
    if args.view_type == "mask":  # one group with per-image window sizes
        idx = torch.nonzero(use_view).squeeze(1)
        if idx.numel() >= 2:
            sel = best[idx]
            groups.append((torch.tensor(sizes, device=x.device)[sel], idx, torch.stack(V_all, 1)[idx, sel],
                           torch.stack(T_all, 1)[idx, sel], D[idx, sel]))
        return groups
    for j, s in enumerate(sizes):
        idx = torch.nonzero(use_view & (best == j)).squeeze(1)
        if idx.numel() < 2:
            continue
        groups.append((s, idx, V_all[j][idx], T_all[j][idx], D[idx, j]))
    return groups


# ----------------------------------------------------------------------------- cached views (old TGSC pipeline)
@torch.no_grad()
def build_view_cache(teacher, trx, try_, aug, s, args):
    """Per-image view windows of size s on the clean (un-augmented) image, fixed for the whole stage.
    Members: salK = K-th window by mean CAM mass on a stride-`grid_stride` grid (old proposal ordering),
    gtK / confK = distinct label / k-th non-label evidence (evidence_windows, deterministic).
    Returns int64 [N, len(view_set), 2] (row, col)."""
    out = []
    members = args.view_set
    n_sal = max([int(m[3:]) for m in members if m.startswith("sal")] + [0])
    n_gt = max([int(m[2:]) for m in members if m.startswith("gt")] + [0])
    confs = [m for m in members if m.startswith("conf")]
    for i in range(0, trx.size(0), 1000):
        x = aug.normalize(trx[i:i + 1000])
        y = try_[i:i + 1000]
        with torch.autocast("cuda", dtype=torch.float16, enabled=args.amp):
            t, f = teacher_forward(teacher, x)
        full = x.size(-1)
        cam = cam_map(f, teacher.fc.weight, y, full)
        pos = {}
        if n_sal:
            g = args.grid_stride
            mass = F.avg_pool2d(cam, s, stride=g)
            gw = mass.size(-1)
            k = min(n_sal, mass[0].numel())
            top = mass.flatten(1).topk(k, 1).indices
            for j in range(n_sal):
                tj = top[:, min(j, k - 1)]
                pos["sal%d" % (j + 1)] = ((tj // gw) * g, (tj % gw) * g)
        if n_gt:
            r, c = evidence_windows(cam, s, n_gt, 0.0, args.iou_thr)
            for j in range(n_gt):
                pos["gt%d" % (j + 1)] = (r[:, j], c[:, j])
        if confs:
            ranked = t.float().scatter(1, y.view(-1, 1), float("-inf")).topk(max(int(m[4:] or 1) for m in confs), 1).indices
            for m in confs:
                r, c = evidence_windows(cam_map(f, teacher.fc.weight, ranked[:, int(m[4:] or 1) - 1], full), s, 1, 0.0,
                                        args.iou_thr)
                pos[m] = (r[:, 0], c[:, 0])
        out.append(torch.stack([torch.stack(pos[m], 1) for m in members], 1))
    return torch.cat(out).long()


def make_cached_views(xc, idx, t_full, cache, s, args, pass_idx, rot_off=None):
    """Crop the cached windows from the clean batch xc, then RandomCrop(s, pad=view_pad, reflect) + flip on each crop.
    With rot_off (per-image offsets), each image still gets each member once per epoch but batches mix members."""
    k = len(args.view_set)
    members = [pass_idx % k] if args.view_rotate else range(k)
    b = xc.size(0)
    ar = torch.arange(b, device=xc.device)
    groups = []
    for j in members:
        jj = (j + rot_off[idx]) % k if (rot_off is not None and args.view_rotate) else torch.full_like(idx, j)
        v = crop_batch(xc, cache[idx, jj, 0], cache[idx, jj, 1], s)
        p = args.view_pad
        if p > 0:
            vp = F.pad(v, (p, p, p, p), mode="reflect")
            off = torch.randint(0, 2 * p + 1, (b, 2), device=xc.device)
            v = crop_batch(vp, off[:, 0], off[:, 1], s)
        flip = torch.rand(b, device=xc.device) < 0.5
        v = torch.where(flip.view(-1, 1, 1, 1), v.flip(-1), v)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=args.amp):
            tv = teacher_forward(args.teacher, v)[0].float()
        groups.append((s, ar, v, tv, kl_teacher(t_full, tv, args.ver_tau)))
    return groups


# ----------------------------------------------------------------------------- train / eval
@torch.no_grad()
def evaluate(model, xs, ys, aug, bs=1000):
    model.eval()
    correct = 0
    for i in range(0, xs.size(0), bs):
        x = aug.normalize(xs[i:i + bs])
        correct += (student_logits(model, x).argmax(1) == ys[i:i + bs]).sum().item()
    model.train()
    return 100.0 * correct / xs.size(0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="cifar100")
    ap.add_argument("--data-root", default="./data", help="folder with cifar-100-python/ or <dataset>_uint8.pt")
    ap.add_argument("--teacher", default="wrn_40_2")
    ap.add_argument("--student", default="wrn_40_1")
    ap.add_argument("--teacher-ckpt", default=None, help="teacher state dict (or a dict with key 'model')")
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=240)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--wd", type=float, default=5e-4)
    ap.add_argument("--sched", default="step", choices=["step", "cosine"])
    ap.add_argument("--milestones", default="150,180,210")
    ap.add_argument("--T", type=float, default=4.0)
    ap.add_argument("--ce-w", type=float, default=0.1)
    ap.add_argument("--kd-w", type=float, default=0.9)
    ap.add_argument("--amp", type=int, default=1)
    # EP-KD
    ap.add_argument("--views", type=int, default=0, help="restricted views per image per step (0 = plain KD)")
    ap.add_argument("--anchor", type=float, default=1.0, help="weight of the full-image objective")
    ap.add_argument("--view-w", type=float, default=1.0)
    ap.add_argument("--view-ce", type=float, default=0.1, help="CE weight on views")
    ap.add_argument("--view-ce-mode", default="all", choices=["all", "valid"])
    ap.add_argument("--view-sizes", default="16,20,24,28,32")
    ap.add_argument("--stage-epochs", type=int, default=48)
    ap.add_argument("--passes", default="1", help="passes over the data per epoch, per stage (e.g. 3,3,3,1,1)")
    ap.add_argument("--size-sched", default="staged", choices=["staged", "random"])
    ap.add_argument("--view-set", default="sal", help="sal (sampled saliency) or a list of gt1,gt2,...,conf")
    ap.add_argument("--view-rotate", type=int, default=0, help="1 = one member of --view-set per pass, in turn")
    ap.add_argument("--iou-thr", type=float, default=0.3)
    ap.add_argument("--view-cache", type=int, default=0, help="1 = views fixed per stage from the clean image (old TGSC)")
    ap.add_argument("--grid-stride", type=int, default=8, help="window grid stride for salK members")
    ap.add_argument("--view-pad", type=int, default=2, help="RandomCrop padding applied to cached views")
    ap.add_argument("--rotate-mix", type=int, default=0, help="1 = per-image random rotation offset (batches mix members)")
    ap.add_argument("--stage-view-sets", default="", help="';'-separated view sets per stage (overrides --view-set)")
    ap.add_argument("--xview-w", type=float, default=0.0, help="weight of KD(T(x) || S(v)) on verified views")
    ap.add_argument("--xview-eps", type=float, default=0.5, help="verification tolerance for the cross-view term")
    ap.add_argument("--view-until", type=int, default=0, help="no views from this epoch on (0 = follow the stages)")
    ap.add_argument("--size-mode", default="global", choices=["global", "verified"])
    ap.add_argument("--view-type", default="crop", choices=["crop", "mask"])
    ap.add_argument("--cand-sizes", default="12,16,20,24,28")
    ap.add_argument("--unverified-fallback", default="none", choices=["none", "largest"])
    ap.add_argument("--pos", default="saliency", choices=["saliency", "random"])
    ap.add_argument("--pos-temp", type=float, default=0.1)
    ap.add_argument("--cands", type=int, default=1)
    ap.add_argument("--select", default="first", choices=["first", "maxfid", "minfid", "eps"])
    ap.add_argument("--eps", type=float, default=0.5)
    ap.add_argument("--ver-tau", type=float, default=1.0)
    ap.add_argument("--gate", default="off", choices=["off", "on"])
    ap.add_argument("--sigma", type=float, default=1.0)
    ap.add_argument("--tau-g", type=float, default=0.75)
    ap.add_argument("--bn-views", type=int, default=0, help="1 = views update BN running stats")
    args = ap.parse_args()
    args.milestones = [int(m) for m in args.milestones.split(",") if m]
    args.view_sizes = [int(s) for s in args.view_sizes.split(",")]
    args.cand_sizes = [int(s) for s in args.cand_sizes.split(",")]
    args.amp = bool(args.amp)
    args.passes = [int(p) for p in args.passes.split(",")]
    args.view_set = args.view_set.split(",")
    args.stage_view_sets = [v.split(",") for v in args.stage_view_sets.split(";") if v]

    os.makedirs(args.run_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.benchmark = args.size_mode == "global" or args.view_type == "mask"
    dev = torch.device("cuda")
    meta = DATASETS[args.dataset]
    full = meta["size"]

    trx, try_, tex, tey = load_dataset(args.dataset, args.data_root)
    trx, try_, tex, tey = trx.to(dev), try_.to(dev), tex.to(dev), tey.to(dev)
    aug = GPUAugment(meta["mean"], meta["std"], full, meta["pad"], dev)

    if args.teacher == "none":  # plain cross-entropy training (teachers / CE baselines)
        teacher, t_acc = None, 0.0
        args.views = 0
    else:
        if args.teacher_ckpt is None:
            raise SystemExit("--teacher-ckpt is required unless --teacher none")
        teacher = getattr(wrn, args.teacher)(num_classes=meta["num_classes"]).to(dev)
        ck = torch.load(args.teacher_ckpt, map_location="cpu", weights_only=False)
        teacher.load_state_dict(ck["model"] if "model" in ck else ck)
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad_(False)
        t_acc = evaluate(teacher, tex, tey, aug)
        teacher.eval()
    args.teacher = teacher

    student = getattr(wrn, args.student)(num_classes=meta["num_classes"]).to(dev)
    opt = torch.optim.SGD(student.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.wd)
    scaler = torch.amp.GradScaler("cuda", enabled=bool(args.amp))

    start, best, best_ep = 0, 0.0, -1
    latest = os.path.join(args.run_dir, "latest.pt")
    if os.path.exists(latest):
        st = torch.load(latest, map_location=dev, weights_only=False)
        student.load_state_dict(st["student"])
        opt.load_state_dict(st["opt"])
        scaler.load_state_dict(st["scaler"])
        start, best, best_ep = st["epoch"], st["best"], st["best_ep"]
        torch.set_rng_state(st["rng"].cpu())
        torch.cuda.set_rng_state(st["cuda_rng"].cpu())

    cfg = {k: v for k, v in vars(args).items() if k != "teacher"}
    cfg["teacher_acc"] = t_acc
    with open(os.path.join(args.run_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=1)
    log = open(os.path.join(args.run_dir, "train.jsonl"), "a")
    print(f"teacher acc {t_acc:.2f}; start epoch {start}", flush=True)

    n = trx.size(0)
    steps_per_epoch = n // args.bs  # drop_last: every step is a full batch
    view_cache = None
    for epoch in range(start, args.epochs):
        lr = lr_at(epoch, args)
        for g in opt.param_groups:
            g["lr"] = lr
        t0 = time.time()
        active = views_active(epoch, args, full)
        use_full = (not active) or args.anchor > 0  # without an anchor, restricted stages train on views only
        n_pass = passes_at(epoch, args)
        if args.stage_view_sets:  # per-stage view family, e.g. reasoning views while they are distinct
            args.view_set = args.stage_view_sets[min(stage_index(epoch, args), len(args.stage_view_sets) - 1)]
        if active and args.view_cache:
            s_now = view_size_at(epoch, args)
            if view_cache is None or view_cache[2] != (s_now, tuple(args.view_set)):
                view_cache = (s_now, build_view_cache(teacher, trx, try_, aug, s_now, args), (s_now, tuple(args.view_set)))
                print(f"view cache: size {s_now}, members {args.view_set}", flush=True)
        stats = dict(loss=0.0, full=0.0, view=0.0, D=0.0, q=0.0, nview=0.0, size=0.0, correct=0, xok=0.0)
        rot_off = torch.randint(0, len(args.view_set), (n,), device=dev) if args.rotate_mix else None
        for it in range(steps_per_epoch * n_pass):
            if it % steps_per_epoch == 0:
                perm = torch.randperm(n, device=dev)
            j = it % steps_per_epoch
            idx = perm[j * args.bs:(j + 1) * args.bs]
            x = aug(trx[idx])
            y = try_[idx]
            if teacher is not None:
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=bool(args.amp)):
                    t_full, feat = teacher_forward(teacher, x)
                t_full = t_full.float()
            loss = 0.0
            if use_full:
                with torch.autocast("cuda", dtype=torch.float16, enabled=bool(args.amp)):
                    s_full = student_logits(student, x)
                if teacher is None:
                    l_full = F.cross_entropy(s_full.float(), y)
                else:
                    l_full = args.ce_w * F.cross_entropy(s_full.float(), y) \
                        + args.kd_w * kd_per_sample(s_full, t_full, args.T).mean()
                loss = (args.anchor if active else 1.0) * l_full
                stats["full"] += l_full.item()
                stats["correct"] += (s_full.argmax(1) == y).sum().item()
            if active:
                cams = {"gt": cam_map(feat, teacher.fc.weight, y, full)}
                confs = [m for m in args.view_set if m.startswith("conf")]
                if confs:  # evidence behind the teacher's k-th ranked non-label class (conf = conf1 = runner-up)
                    ranked = t_full.scatter(1, y.view(-1, 1), float("-inf")).topk(max(int(m[4:] or 1) for m in confs), 1).indices
                    for m in confs:
                        cams[m] = cam_map(feat, teacher.fc.weight, ranked[:, int(m[4:] or 1) - 1], full)
                if args.view_cache:
                    groups = make_cached_views(aug.normalize(trx[idx]), idx, t_full, view_cache[1], view_cache[0], args,
                                               it // steps_per_epoch, rot_off)
                else:
                    groups = make_views(x, y, t_full, cams, epoch, args, full, pass_idx=it // steps_per_epoch)
                l_view = 0.0
                for s, gi, v, tv, D in groups:
                    # views leave BN running stats alone when the full image is also trained on this step
                    with NoBNUpdate(student, enabled=use_full and not args.bn_views), \
                            torch.autocast("cuda", dtype=torch.float16, enabled=bool(args.amp)):
                        sv = student_logits(student, v)
                    q = torch.exp(-D / args.sigma)
                    w = (q / args.tau_g).clamp(max=1.0) if args.gate == "on" else torch.ones_like(q)
                    ce_v = F.cross_entropy(sv.float(), y[gi], reduction="none")
                    if args.view_ce_mode == "valid":  # y_v: the image label is a valid target only if the view keeps it
                        ce_v = ce_v * (tv.argmax(1) == y[gi]).float()
                    per = args.view_ce * ce_v + args.kd_w * w * kd_per_sample(sv, tv, args.T)
                    if args.xview_w > 0:  # local-to-global: verified views (D <= eps) also match T(x), Lemma 1
                        ok = (D <= args.xview_eps).float()
                        per = per + args.xview_w * args.kd_w * ok * kd_per_sample(sv, t_full[gi], args.T)
                        stats["xok"] += ok.sum().item()
                    l_view = l_view + per.sum() / x.size(0)
                    stats["D"] += D.sum().item()
                    stats["q"] += q.sum().item()
                    stats["nview"] += gi.numel()
                    stats["size"] += float(s * gi.numel()) if isinstance(s, int) else float(s.float().sum())
                if groups:
                    loss = loss + args.view_w * l_view / (len(groups) if args.size_mode == "global" else 1)
                    stats["view"] += float(l_view)
            if not torch.is_tensor(loss):  # no full objective and no view in this batch
                continue
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            stats["loss"] += loss.item()
        acc = evaluate(student, tex, tey, aug)
        if acc > best:
            best, best_ep = acc, epoch + 1
            torch.save(student.state_dict(), os.path.join(args.run_dir, "best_test.pth"))
        nv = max(1.0, stats["nview"])
        ns = steps_per_epoch * n_pass
        rec = dict(epoch=epoch + 1, lr=lr, test_acc=acc, best=best, best_epoch=best_ep, passes=n_pass,
                   loss=stats["loss"] / ns, l_full=stats["full"] / ns,
                   l_view=stats["view"] / ns,
                   train_acc=100.0 * stats["correct"] / (ns * args.bs),
                   view_D=stats["D"] / nv, view_q=stats["q"] / nv, xview_frac=stats["xok"] / nv,
                   view_frac=stats["nview"] / (ns * args.bs), view_size=stats["size"] / nv,
                   sec=time.time() - t0)
        log.write(json.dumps(rec) + "\n")
        log.flush()
        print(json.dumps(rec), flush=True)
        torch.save(dict(student=student.state_dict(), opt=opt.state_dict(), scaler=scaler.state_dict(),
                        epoch=epoch + 1, best=best, best_ep=best_ep, rng=torch.get_rng_state(),
                        cuda_rng=torch.cuda.get_rng_state()), latest + ".tmp")
        os.replace(latest + ".tmp", latest)
    torch.save(student.state_dict(), os.path.join(args.run_dir, "final.pth"))
    with open(os.path.join(args.run_dir, "done.json"), "w") as f:
        json.dump(dict(best=best, best_epoch=best_ep, final=acc), f)
    print(f"done best {best:.2f}@{best_ep} final {acc:.2f}", flush=True)


if __name__ == "__main__":
    main()
