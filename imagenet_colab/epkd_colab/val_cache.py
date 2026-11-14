"""Cached validation inputs: the reference eval transform (PIL decode -> resize short 256 -> center crop 224)
computed once with the unmodified reference function ``_image_bytes_to_eval_tensor`` up to the uint8 crop,
stored as a (N, 3, 224, 224) uint8 memmap (+ labels).  Normalisation ((x/255 - mean)/std, fp32) happens on the
device in the same arithmetic as ``TF.to_tensor`` + ``TF.normalize``; the student is evaluated in fp32 as in
``DDPKDTrainer._evaluate``.  Bit-identical inputs to the reference eval loader."""
from __future__ import annotations

from multiprocessing import Pool
import os
from pathlib import Path
from typing import Iterator

import numpy as np
import pyarrow.parquet as pq
import torch
from PIL import Image
from io import BytesIO

from tgsc_kd.imagenet_parquet import ParquetSplitInfo, _load_image_bytes, _resize_short_side
from torchvision.transforms import functional as TF


def _crop_u8(args: tuple[bytes, int, int]) -> np.ndarray:
    b, resize_short, crop = args
    image = _load_image_bytes(b)
    image = _resize_short_side(image, resize_short)
    image = TF.center_crop(image, [crop, crop])
    return TF.pil_to_tensor(image).numpy()  # (3, crop, crop) uint8 == to_tensor(image) * 255 exactly


class ValCache:
    def __init__(self, path: str | Path, split: ParquetSplitInfo, data_cfg, procs: int = 8) -> None:
        self.path = Path(path)
        self.labels_path = self.path.with_suffix(".labels.npy")
        self.crop = int(data_cfg.eval_crop_size)
        self.n = int(split.total_rows)
        if not (self.path.exists() and self.labels_path.exists()):
            self._build(split, int(data_cfg.eval_resize_short), procs)
        self.x = np.load(self.path, mmap_mode="r")
        self.y = np.load(self.labels_path)
        assert self.x.shape[0] == self.y.shape[0] == self.n, (self.x.shape, self.y.shape, self.n)

    def _build(self, split: ParquetSplitInfo, resize_short: int, procs: int) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(self.path.name + ".tmp.npy")
        x = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.uint8, shape=(self.n, 3, self.crop, self.crop))
        y = np.full(self.n, -1, dtype=np.int64)
        i = 0
        with Pool(procs) as pool:
            for shard_idx, p in enumerate(split.paths):
                pf = pq.ParquetFile(p)
                for batch in pf.iter_batches(columns=["image.bytes", "label"], batch_size=256):
                    if i >= self.n:  # split limited by max_val_samples: stop at its row count
                        break
                    blobs = batch.column(0).field("bytes").to_pylist()[: self.n - i]
                    labels = batch.column(1).to_pylist()[: self.n - i]
                    args = [(b, resize_short, self.crop) for b in blobs]
                    for k, arr in enumerate(pool.map(_crop_u8, args, chunksize=8)):
                        x[i + k] = arr
                        y[i + k] = int(labels[k])
                    i += len(blobs)
                print(f"[val-cache] {i}/{self.n}", flush=True)
                if i >= self.n:
                    break
        x.flush()
        del x
        np.save(self.labels_path, y)
        os.replace(tmp, self.path)

    def batches(self, batch_size: int, device: torch.device) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        """(uint8 (b,3,224,224) on device, labels on device); skips label < 0 as the reference loader does."""
        keep = np.flatnonzero(self.y >= 0)
        for s in range(0, len(keep), batch_size):
            idx = keep[s: s + batch_size]
            x = torch.from_numpy(np.ascontiguousarray(self.x[idx[0]: idx[-1] + 1])) if np.all(np.diff(idx) == 1) else torch.from_numpy(self.x[idx])
            y = torch.from_numpy(self.y[idx])
            yield x.to(device, non_blocking=True), y.to(device, non_blocking=True)
