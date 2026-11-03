"""Convert STL-10 (labelled split) and Tiny-ImageNet-200 parquet files (HF mirrors) to uint8 NCHW tensors.

usage: python prepare_data.py RAW_DIR OUT_DIR stl10 tiny
  stl10: RAW_DIR/stl10/{train,test}.parquet         (tanganke/stl10)
  tiny:  RAW_DIR/tiny/{train,valid}.parquet         (zh-plus/tiny-imagenet; valid = official val split)
"""
import io
import os
import sys

import numpy as np
import pyarrow.parquet as pq
import torch
from PIL import Image


def _load(path, size):
    t = pq.read_table(path).to_pydict()
    img_col = "image" if "image" in t else [k for k in t if k != "label"][0]
    xs = []
    for im in t[img_col]:
        a = np.asarray(Image.open(io.BytesIO(im["bytes"])).convert("RGB"))
        assert a.shape == (size, size, 3), a.shape
        xs.append(a)
    x = torch.from_numpy(np.stack(xs).transpose(0, 3, 1, 2).copy())
    return x, torch.tensor(t["label"], dtype=torch.long)


def main():
    raw, out, names = sys.argv[1], sys.argv[2], sys.argv[3:]
    for name in names:
        if name == "stl10":
            trx, try_ = _load(os.path.join(raw, "stl10", "train.parquet"), 96)
            tex, tey = _load(os.path.join(raw, "stl10", "test.parquet"), 96)
            fn = "stl10_uint8.pt"
        else:
            trx, try_ = _load(os.path.join(raw, "tiny", "train.parquet"), 64)
            tex, tey = _load(os.path.join(raw, "tiny", "valid.parquet"), 64)
            fn = "tinyimagenet_uint8.pt"
        d = dict(train_x=trx, train_y=try_, test_x=tex, test_y=tey)
        torch.save(d, os.path.join(out, fn))
        print(name, {k: tuple(v.shape) for k, v in d.items()},
              "classes", int(try_.max()) + 1, "per-class train", torch.bincount(try_).min().item(), torch.bincount(try_).max().item(),
              "per-class test", torch.bincount(tey).min().item(), torch.bincount(tey).max().item())


if __name__ == "__main__":
    main()
