"""Compares every operation of mvkd.randaugment with the PIL original of CRLD / VRM (mdistiller/dataset/cifar100.py).
usage: python tests/test_randaugment.py [CIFAR_ROOT]   (needs a GPU or runs on CPU)"""
import os
import pickle
import sys

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageEnhance, ImageOps

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import mvkd  # noqa: E402

# ---- PIL reference: copied from github.com/arcaninez/crld mdistiller/dataset/cifar100.py
PIL_OPS = {
    "AutoContrast": lambda img, v: ImageOps.autocontrast(img),
    "Brightness": lambda img, v: ImageEnhance.Brightness(img).enhance(v),
    "Color": lambda img, v: ImageEnhance.Color(img).enhance(v),
    "Contrast": lambda img, v: ImageEnhance.Contrast(img).enhance(v),
    "Equalize": lambda img, v: ImageOps.equalize(img),
    "Identity": lambda img, v: img,
    "Posterize": lambda img, v: ImageOps.posterize(img, max(1, int(v))),
    "Rotate": lambda img, v: img.rotate(v),
    "Sharpness": lambda img, v: ImageEnhance.Sharpness(img).enhance(v),
    "ShearX": lambda img, v: img.transform(img.size, Image.AFFINE, (1, v, 0, 0, 1, 0)),
    "ShearY": lambda img, v: img.transform(img.size, Image.AFFINE, (1, 0, 0, v, 1, 0)),
    "Solarize": lambda img, v: ImageOps.solarize(img, v),
    "TranslateX": lambda img, v: img.transform(img.size, Image.AFFINE, (1, 0, v * img.size[0], 0, 1, 0)),
    "TranslateY": lambda img, v: img.transform(img.size, Image.AFFINE, (1, 0, 0, 0, 1, v * img.size[1])),
}


def pil_cutout(img, v, x0, y0):
    v = v * img.size[0]
    if v <= 0:
        return img
    w, h = img.size
    x0 = int(max(0, x0 - v / 2.)); y0 = int(max(0, y0 - v / 2.))
    x1 = min(w, x0 + v); y1 = min(h, y0 + v)
    img = img.copy()
    ImageDraw.Draw(img).rectangle((x0, y0, x1, y1), (125, 123, 114))
    return img


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else "./data"
    d = pickle.load(open(os.path.join(root, "cifar-100-python", "train"), "rb"), encoding="latin1")
    x = np.asarray(d["data"][:256], dtype=np.uint8).reshape(-1, 3, 32, 32)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(0)
    worst = 0.0
    for sz in (32, 16):  # full images and EP-KD-sized views
        xs = x[:, :, :sz, :sz]
        t = torch.from_numpy(xs.copy()).float().to(dev)
        pils = [Image.fromarray(np.ascontiguousarray(a.transpose(1, 2, 0))) for a in xs]
        for name, lo, hi in mvkd.AUGMENT_LIST:
            v = lo + (hi - lo) * rng.random(len(pils))
            vt = torch.tensor(v, device=dev, dtype=torch.float64)
            got = mvkd.OPS[name](t, vt if name in ("Rotate", "ShearX", "ShearY", "TranslateX", "TranslateY")
                                 else vt.float()).cpu().numpy()
            ref = np.stack([np.asarray(PIL_OPS[name](p, float(vv))).transpose(2, 0, 1) for p, vv in zip(pils, v)])
            diff = np.abs(got.astype(np.float64) - ref)
            exact = (diff.reshape(len(pils), -1).max(1) == 0).mean()
            worst = max(worst, 1 - exact)
            print(f"{sz:2d}px {name:12s} images identical {100 * exact:6.2f}%   mean|diff| {diff.mean():.4f}   max {diff.max():.0f}")
        v = rng.random(len(pils)) * 0.5
        x0, y0 = rng.random(len(pils)) * sz, rng.random(len(pils)) * sz
        got = mvkd.cutout(t, torch.tensor(v, device=dev).float(), torch.tensor(x0, device=dev).float(),
                          torch.tensor(y0, device=dev).float()).cpu().numpy()
        ref = np.stack([np.asarray(pil_cutout(p, a, b, c)).transpose(2, 0, 1) for p, a, b, c in zip(pils, v, x0, y0)])
        diff = np.abs(got - ref)
        print(f"{sz:2d}px {'Cutout':12s} images identical {100 * (diff.reshape(len(pils), -1).max(1) == 0).mean():6.2f}%   mean|diff| {diff.mean():.4f}")
    print("worst share of non-identical images: %.2f%%" % (100 * worst))


if __name__ == "__main__":
    main()


def test_pipeline(dev="cuda" if torch.cuda.is_available() else "cpu"):
    """The vectorised randaugment equals a per-sample loop over the same random draws."""
    x = torch.randint(0, 256, (64, 3, 32, 32), device=dev).float()
    g1 = torch.Generator(device=dev).manual_seed(5)
    got = mvkd.randaugment(x, gen=g1)
    g2 = torch.Generator(device=dev).manual_seed(5)
    out = x
    geo = ("Rotate", "ShearX", "ShearY", "TranslateX", "TranslateY")
    for _ in range(2):
        choice = torch.randint(0, len(mvkd.AUGMENT_LIST), (64,), device=dev, generator=g2)
        u = torch.rand(64, device=dev, generator=g2, dtype=torch.float64)
        rows = []
        for i in range(64):
            name, lo, hi = mvkd.AUGMENT_LIST[int(choice[i])]
            val = (lo + float(hi - lo) * u[i:i + 1])
            rows.append(mvkd.OPS[name](out[i:i + 1], val if name in geo else val.to(out.dtype)))
        out = torch.cat(rows)
    v = torch.rand(64, device=dev, generator=g2) * 0.5
    x0 = torch.rand(64, device=dev, generator=g2) * 32
    y0 = torch.rand(64, device=dev, generator=g2) * 32
    ref = mvkd.cutout(out, v, x0, y0)
    print("pipeline: vectorised == per-sample loop:", bool((got == ref).all()))


if __name__ == "__main__":
    test_pipeline()
