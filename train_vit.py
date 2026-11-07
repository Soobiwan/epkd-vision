"""EP-KD for Vision Transformers: ViT-S/16 teacher (OFA-KD CIFAR-100 checkpoint) -> DeiT-T/16
trained from scratch on CIFAR-100 at 224 px with the OFA-KD recipe.

Recipe (OFA-KD configs/cifar/vit_mlp.yaml, 8 GPUs x 128 = global batch 1024; reproduced here by 8 x 128 gradient
accumulation): AdamW lr 5e-4, wd 0.05, cosine to 1e-5 over 300 epochs (per-epoch steps, timm), 20 warmup epochs from
5e-7, clip-grad 5, RandAug rand-m9-mstd0.5-inc1, RRC(0.08-1), hflip, random erasing 0.25 (pixel), mixup 0.8 /
cutmix 1.0 (switch 0.5), label smoothing 0.1. Loss as OFA's KD: soft-target CE (weight 1) + KL at tau=1 (weight 1).

EP-KD (same method as the CNN recipe; only the view family and the relevance change):
  * view v = a square window of s x s patch tokens on the 14 x 14 grid, fed with the tokens' ORIGINAL position
    embeddings and the CLS token (no resizing, no mask tokens; the other tokens are absent),
  * relevance r_T = the teacher's last-layer CLS->patch attention (mean over heads) on the full augmented image,
    taken from the anchor forward; the 3 most relevant windows (greedy, IoU suppression) are the proposals,
  * own-view targets p_T(.|v), full-image anchor at every step, one proposal per image per pass (pass p uses the
    (p mod 3)-th proposal), 5 stages of 60 epochs with sides 7,9,10,12 then the full image, passes 3,3,3,1,1.
  `--views 0` gives KD (passes 1) or KD_sm (passes 3,3,3,1,1, same steps as EP-KD, no views).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from PIL import Image

import timm
from timm.data import Mixup, create_transform
from timm.loss import SoftTargetCrossEntropy

import timm.data.auto_augment as _taa  # noqa: E402


def _posterize_compat(img, bits_to_keep, **__):
    """timm posterize with the pre-Pillow-11 behaviour that OFA-KD ran with: 0 bits kept gives a black image
    (Pillow >= 11 raises). PosterizeIncreasing reaches 0 bits at magnitude 10 (rand-m9-mstd0.5-inc1)."""
    if bits_to_keep >= 8:
        return img
    if bits_to_keep <= 0:
        return img.point(lambda _: 0)
    return _taa.ImageOps.posterize(img, bits_to_keep)


for _k in ("Posterize", "PosterizeIncreasing", "PosterizeOriginal"):
    if _k in _taa.NAME_TO_OP:
        _taa.NAME_TO_OP[_k] = _posterize_compat

CIFAR_MEAN = (0.5071, 0.4865, 0.4409)  # OFA-KD train.py
CIFAR_STD = (0.2673, 0.2564, 0.2762)
GRID = 14


# ----------------------------------------------------------------------------- data
def test_cache(root, path):
    """CIFAR-100 test set with the timm eval transform (bicubic Resize 256 + CenterCrop 224, PIL), as uint8."""
    if os.path.exists(path):
        return torch.load(path, weights_only=True)
    ds = torchvision.datasets.CIFAR100(root, train=False)
    tf = torchvision.transforms.Compose([torchvision.transforms.Resize(256, interpolation=Image.BICUBIC),
                                         torchvision.transforms.CenterCrop(224)])
    xs = np.stack([np.asarray(tf(img), dtype=np.uint8) for img, _ in ds]).transpose(0, 3, 1, 2)
    d = dict(x=torch.from_numpy(np.ascontiguousarray(xs)), y=torch.tensor(ds.targets, dtype=torch.long))
    torch.save(d, path + ".tmp")
    os.replace(path + ".tmp", path)
    return d


@torch.no_grad()
def evaluate(model, test, dev, bs=500):
    model.eval()
    mean = torch.tensor(CIFAR_MEAN, device=dev).view(1, 3, 1, 1)
    std = torch.tensor(CIFAR_STD, device=dev).view(1, 3, 1, 1)
    correct = 0
    for i in range(0, test["x"].size(0), bs):
        x = (test["x"][i:i + bs].to(dev, non_blocking=True).float().div_(255) - mean) / std
        with torch.autocast("cuda", dtype=torch.float16):
            correct += (model(x).argmax(1).cpu() == test["y"][i:i + bs]).sum().item()
    model.train()
    return 100.0 * correct / test["x"].size(0)


# ----------------------------------------------------------------------------- token views
def vit_forward(m, x, keep=None, cls_attn=False):
    """timm VisionTransformer forward on the tokens `keep` ([B, K] patch indices, original positions) or on all
    tokens. With cls_attn, also returns the last block's CLS->patch attention [B, N_kept], mean over heads."""
    t = m._pos_embed(m.patch_embed(x))  # [B, 1 + 196, C], CLS first, position embeddings added
    if keep is not None:
        idx = torch.cat([keep.new_zeros(keep.size(0), 1), keep + 1], 1)
        t = t.gather(1, idx.unsqueeze(-1).expand(-1, -1, t.size(-1)))
    t = m.norm_pre(m.patch_drop(t))
    for blk in m.blocks[:-1]:
        t = blk(t)
    a = None
    if cls_attn:
        last = m.blocks[-1]
        at = last.attn
        b, n, c = t.shape
        qkv = at.qkv(last.norm1(t)).reshape(b, n, 3, at.num_heads, c // at.num_heads).permute(2, 0, 3, 1, 4)
        q, k = at.q_norm(qkv[0]), at.k_norm(qkv[1])
        s = (q[:, :, :1].float() * at.scale) @ k.float().transpose(-2, -1)  # [B, H, 1, N]
        a = s.softmax(-1).mean(1)[:, 0, 1:]  # [B, N - 1]
    t = m.blocks[-1](t)
    return m.forward_head(m.norm(t)), a


def window_tokens(rows, cols, s):
    """Patch indices [B, s*s] of the s x s windows with top-left (rows, cols) on the 14 x 14 grid."""
    ar = torch.arange(s, device=rows.device)
    r = (rows.view(-1, 1, 1) + ar.view(1, -1, 1))
    c = (cols.view(-1, 1, 1) + ar.view(1, 1, -1))
    return (r * GRID + c).flatten(1)


def propose_windows(attn, s, k, iou_thr):
    """k windows of side s by teacher CLS-attention mass, greedy with IoU suppression; once every window is
    suppressed, the next most relevant window not yet picked. Returns rows, cols [B, k]."""
    a = attn.float().view(-1, 1, GRID, GRID)
    g = GRID - s + 1
    mass = F.avg_pool2d(a, s, stride=1).flatten(1)  # [B, g*g]
    pos = torch.arange(g * g, device=a.device)
    pr, pc = (pos // g).view(1, -1), (pos % g).view(1, -1)
    m, used = mass.clone(), torch.zeros_like(mass, dtype=torch.bool)
    rows, cols = [], []
    for _ in range(k):
        avail = torch.isfinite(m).any(1, keepdim=True)
        mu = torch.where(avail, m, mass.masked_fill(used, float("-inf")))
        idx = mu.argmax(1)
        r, c = idx // g, idx % g
        rows.append(r)
        cols.append(c)
        used[torch.arange(idx.numel(), device=a.device), idx] = True
        inter = (s - (pr - r.view(-1, 1)).abs()).clamp_min(0) * (s - (pc - c.view(-1, 1)).abs()).clamp_min(0)
        m = m.masked_fill(inter / (2 * s * s - inter) > iou_thr, float("-inf")).masked_fill(used, float("-inf"))
    return torch.stack(rows, 1), torch.stack(cols, 1)


def kd_loss(s_logits, t_logits, tau):
    """OFA-KD kd_loss: KL(p_T || p_S) at temperature tau, batchmean, times tau^2."""
    return F.kl_div(F.log_softmax(s_logits.float() / tau, 1), F.softmax(t_logits.float() / tau, 1),
                    reduction="batchmean") * tau * tau


def kl_teacher(t_full, t_view, tau):
    p = F.softmax(t_full.float() / tau, 1)
    return (p * (F.log_softmax(t_full.float() / tau, 1) - F.log_softmax(t_view.float() / tau, 1))).sum(1)


# ----------------------------------------------------------------------------- schedules
def lr_at(epoch, a):
    """timm CosineLRScheduler stepped per epoch (t_initial = epochs, warmup not prefixed)."""
    if epoch < a.warmup_epochs:
        return a.warmup_lr + epoch * (a.lr - a.warmup_lr) / a.warmup_epochs
    return a.min_lr + 0.5 * (a.lr - a.min_lr) * (1 + math.cos(math.pi * epoch / a.epochs))


def stage_index(epoch, a):
    return min(epoch // a.stage_epochs, len(a.view_sides) - 1)


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--data-root", default="./data", help="folder with cifar-100-python/")
    ap.add_argument("--test-cache", default="./data/cifar100_test_224_bicubic_uint8.pt",
                    help="built on first use: the test set under the timm eval transform, uint8")
    ap.add_argument("--teacher", default="vit_small_patch16_224")
    ap.add_argument("--teacher-ckpt",
                    default="teachers/vit_small_patch16_224_cifar100_weights.pt",
                    help="weights-only state dict (scripts/get_vit_teacher.sh)")
    ap.add_argument("--student", default="deit_tiny_patch16_224")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--bs", type=int, default=128, help="micro-batch")
    ap.add_argument("--accum", type=int, default=8, help="micro-batches per optimizer step (OFA: 8 GPUs x 128)")
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--min-lr", type=float, default=1e-5)
    ap.add_argument("--warmup-lr", type=float, default=5e-7)
    ap.add_argument("--warmup-epochs", type=int, default=20)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--clip-grad", type=float, default=5.0)
    ap.add_argument("--drop-path", type=float, default=0.0)
    ap.add_argument("--mixup", type=float, default=0.8)
    ap.add_argument("--cutmix", type=float, default=1.0)
    ap.add_argument("--smoothing", type=float, default=0.1)
    ap.add_argument("--aa", default="rand-m9-mstd0.5-inc1")
    ap.add_argument("--reprob", type=float, default=0.25)
    ap.add_argument("--gt-w", type=float, default=1.0)
    ap.add_argument("--kd-w", type=float, default=1.0)
    ap.add_argument("--tau", type=float, default=1.0)
    ap.add_argument("--workers", type=int, default=6)
    # EP-KD
    ap.add_argument("--views", type=int, default=0, help="1 = EP-KD token-window views; 0 = KD / KD_sm")
    ap.add_argument("--view-w", type=float, default=1.0)
    ap.add_argument("--view-sides", default="7,9,10,12,14", help="window side per stage on the 14x14 grid")
    ap.add_argument("--stage-epochs", type=int, default=60)
    ap.add_argument("--passes", default="1", help="passes over the data per epoch, per stage (EP-KD: 3,3,3,1,1)")
    ap.add_argument("--n-prop", type=int, default=3)
    ap.add_argument("--iou-thr", type=float, default=0.3)
    ap.add_argument("--anchor", type=float, default=1.0)
    # smoke test
    ap.add_argument("--subset", type=int, default=0, help=">0: first N training images (smoke test)")
    ap.add_argument("--max-micro", type=int, default=0, help=">0: micro-batches per pass (smoke test)")
    a = ap.parse_args()
    a.view_sides = [int(s) for s in a.view_sides.split(",")]
    a.passes = [int(p) for p in a.passes.split(",")]

    os.makedirs(a.run_dir, exist_ok=True)
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    dev = torch.device("cuda")

    test = test_cache(a.data_root, a.test_cache)
    tf = create_transform(224, is_training=True, scale=(0.08, 1.0), ratio=(3 / 4, 4 / 3), hflip=0.5,
                          color_jitter=0.4, auto_augment=a.aa, interpolation="random", mean=CIFAR_MEAN,
                          std=CIFAR_STD, re_prob=a.reprob, re_mode="pixel", re_count=1)
    ds = torchvision.datasets.CIFAR100(a.data_root, train=True, transform=tf)
    if a.subset:
        ds = torch.utils.data.Subset(ds, list(range(a.subset)))
    g = torch.Generator().manual_seed(a.seed)
    loader = torch.utils.data.DataLoader(ds, batch_size=a.bs, shuffle=True, drop_last=True, num_workers=a.workers,
                                         pin_memory=True, persistent_workers=True, generator=g, prefetch_factor=4)
    mixup = Mixup(mixup_alpha=a.mixup, cutmix_alpha=a.cutmix, prob=1.0, switch_prob=0.5, mode="batch",
                  label_smoothing=a.smoothing, num_classes=100)
    soft_ce = SoftTargetCrossEntropy()

    teacher = timm.create_model(a.teacher, num_classes=100)
    teacher.load_state_dict(torch.load(a.teacher_ckpt, map_location="cpu", weights_only=True))
    teacher.to(dev).eval().requires_grad_(False)
    t_acc = evaluate(teacher, test, dev)
    teacher.eval()

    student = timm.create_model(a.student, num_classes=100, drop_path_rate=a.drop_path).to(dev)
    opt = torch.optim.AdamW(student.parameters(), lr=a.lr, weight_decay=a.wd)
    scaler = torch.amp.GradScaler("cuda")

    start, best, best_ep = 0, 0.0, -1
    latest = os.path.join(a.run_dir, "latest.pt")
    if os.path.exists(latest):
        st = torch.load(latest, map_location=dev, weights_only=False)  # our own file
        student.load_state_dict(st["student"])
        opt.load_state_dict(st["opt"])
        scaler.load_state_dict(st["scaler"])
        start, best, best_ep = st["epoch"], st["best"], st["best_ep"]
        torch.set_rng_state(st["rng"].cpu())
        torch.cuda.set_rng_state(st["cuda_rng"].cpu())
        g.manual_seed(a.seed * 100003 + start)

    cfg = dict(vars(a), teacher_acc=t_acc, timm=timm.__version__, torch=torch.__version__)
    with open(os.path.join(a.run_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=1)
    log = open(os.path.join(a.run_dir, "train.jsonl"), "a")
    print(f"teacher acc {t_acc:.2f}; start epoch {start}", flush=True)

    for epoch in range(start, a.epochs):
        lr = lr_at(epoch, a)
        for pg in opt.param_groups:
            pg["lr"] = lr
        si = stage_index(epoch, a)
        side = a.view_sides[si]
        active = a.views > 0 and side < GRID
        n_pass = a.passes[min(si, len(a.passes) - 1)]
        st_ = dict(loss=0.0, full=0.0, view=0.0, D=0.0, nview=0, n=0, correct=0, steps=0, wait=0.0)
        t0 = time.time()
        micro = 0
        opt.zero_grad(set_to_none=True)
        for p in range(n_pass):
            tw = time.time()
            for i, (x, y) in enumerate(loader):
                if a.max_micro and i >= a.max_micro:
                    break
                x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
                st_["wait"] += time.time() - tw
                x, ys = mixup(x, y)
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
                    t_full, attn = vit_forward(teacher, x, cls_attn=active)
                with torch.autocast("cuda", dtype=torch.float16):
                    s_full = student(x)
                l_full = a.gt_w * soft_ce(s_full.float(), ys) + a.kd_w * kd_loss(s_full, t_full, a.tau)
                loss = (a.anchor if active else 1.0) * l_full
                if active:
                    rows, cols = propose_windows(attn, side, a.n_prop, a.iou_thr)
                    j = p % a.n_prop
                    keep = window_tokens(rows[:, j], cols[:, j], side)
                    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
                        tv = vit_forward(teacher, x, keep)[0]
                    with torch.autocast("cuda", dtype=torch.float16):
                        sv = vit_forward(student, x, keep)[0]
                    l_view = a.gt_w * soft_ce(sv.float(), ys) + a.kd_w * kd_loss(sv, tv, a.tau)
                    loss = loss + a.view_w * l_view
                    st_["view"] += l_view.item()
                    st_["D"] += kl_teacher(t_full, tv, a.tau).sum().item()
                    st_["nview"] += x.size(0)
                scaler.scale(loss / a.accum).backward()
                micro += 1
                if micro % a.accum == 0:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(student.parameters(), a.clip_grad)
                    scaler.step(opt)
                    scaler.update()
                    opt.zero_grad(set_to_none=True)
                    st_["steps"] += 1
                st_["loss"] += loss.item()
                st_["full"] += l_full.item()
                st_["n"] += x.size(0)
                st_["correct"] += (s_full.argmax(1) == y).sum().item()
                tw = time.time()
            # a partial accumulation at the end of a pass is dropped (timm: drop_last over the global batch)
            opt.zero_grad(set_to_none=True)
            micro = 0
        acc = evaluate(student, test, dev)
        if acc > best:
            best, best_ep = acc, epoch + 1
            torch.save(student.state_dict(), os.path.join(a.run_dir, "best_test.pth"))
        nm = max(1, st_["n"] // a.bs)
        sec = time.time() - t0
        rec = dict(epoch=epoch + 1, lr=lr, test_acc=acc, best=best, best_epoch=best_ep, passes=n_pass, side=side,
                   views=int(active), steps=st_["steps"], loss=st_["loss"] / nm, l_full=st_["full"] / nm,
                   l_view=st_["view"] / nm, view_D=st_["D"] / max(1, st_["nview"]),
                   train_acc_mixed=100.0 * st_["correct"] / max(1, st_["n"]), sec=sec,
                   img_s=st_["n"] / sec, data_wait=st_["wait"], mem_gb=torch.cuda.max_memory_allocated() / 2 ** 30)
        log.write(json.dumps(rec) + "\n")
        log.flush()
        print(json.dumps(rec), flush=True)
        torch.save(dict(student=student.state_dict(), opt=opt.state_dict(), scaler=scaler.state_dict(),
                        epoch=epoch + 1, best=best, best_ep=best_ep, rng=torch.get_rng_state(),
                        cuda_rng=torch.cuda.get_rng_state()), latest + ".tmp")
        os.replace(latest + ".tmp", latest)
        g.manual_seed(a.seed * 100003 + epoch + 1)
    torch.save(student.state_dict(), os.path.join(a.run_dir, "final.pth"))
    with open(os.path.join(a.run_dir, "done.json"), "w") as f:
        json.dump(dict(best=best, best_epoch=best_ep, final=acc, teacher_acc=t_acc), f)
    print(f"done best {best:.2f}@{best_ep} final {acc:.2f}", flush=True)


if __name__ == "__main__":
    main()
