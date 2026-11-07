"""Off-centre CIFAR-100 (CIFAR-100-Canvas): each 32x32 CIFAR-100 image is pasted at a uniformly random position into a
64x64 canvas, so the object covers 25% of the image and is off-centre. The background is a smooth colour field
(another training image average-pooled to 4x4 and bilinearly upsampled), which has natural colour statistics but no
recognisable object. Positions and backgrounds are fixed per image (seeded) for train and test.

usage: python scripts/prepare_canvas.py CIFAR_ROOT OUT_DIR   (CIFAR_ROOT holds cifar-100-python/)
writes OUT_DIR/cifar100canvas_uint8.pt with train_x/train_y/test_x/test_y (uint8 NCHW) and train_pos/test_pos.
"""
import os
import pickle
import sys

import numpy as np
import torch
import torch.nn.functional as F


def _load(root, split):
    d = pickle.load(open(os.path.join(root, "cifar-100-python", split), "rb"), encoding="latin1")
    return torch.from_numpy(np.asarray(d["data"], dtype=np.uint8).reshape(-1, 3, 32, 32)), \
        torch.tensor(d["fine_labels"], dtype=torch.long)


def canvas(x, pool, g):
    n = x.size(0)
    bg_src = pool[torch.randint(0, pool.size(0), (n,), generator=g)].float()
    bg = F.interpolate(F.adaptive_avg_pool2d(bg_src, 4), size=(64, 64), mode="bilinear", align_corners=False)
    pos = torch.randint(0, 33, (n, 2), generator=g)
    out = bg.round().clamp(0, 255).to(torch.uint8)
    for i in range(n):
        r, c = int(pos[i, 0]), int(pos[i, 1])
        out[i, :, r:r + 32, c:c + 32] = x[i]
    return out, pos


def main():
    root, out_dir = sys.argv[1], sys.argv[2]
    trx, try_ = _load(root, "train")
    tex, tey = _load(root, "test")
    g = torch.Generator().manual_seed(20261105)
    ctr, ptr = canvas(trx, trx, g)
    cte, pte = canvas(tex, trx, g)  # test backgrounds also come from training images (no test-label leakage)
    f = ctr.float() / 255.0
    mean = f.mean((0, 2, 3)).tolist()
    std = f.permute(1, 0, 2, 3).reshape(3, -1).std(1).tolist()
    print("mean", [round(m, 4) for m in mean], "std", [round(s, 4) for s in std])
    torch.save(dict(train_x=ctr, train_y=try_, test_x=cte, test_y=tey, train_pos=ptr, test_pos=pte,
                    mean=mean, std=std), os.path.join(out_dir, "cifar100canvas_uint8.pt"))


if __name__ == "__main__":
    main()
