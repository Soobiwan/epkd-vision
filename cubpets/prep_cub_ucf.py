"""CUB-200-2011 and UCF101 (single frames) as uint8 NCHW tensors for epkd2 trainers.

usage:
  python -I prep_cub_ucf.py cub  RAW_CUB_DIR  OUT_DIR [SIZE]     RAW_CUB_DIR has images.zip (or images/), images.txt,
                                                                image_class_labels.txt, train_test_split.txt,
                                                                optional bounding_boxes.txt
  python -I prep_cub_ucf.py ucf  RAW_UCF_DIR  OUT_DIR [SIZE] [TRAIN_FRAMES]
                                                                RAW_UCF_DIR has UCF-101/ (videos) and ucfTrainTestlist/
  python -I prep_cub_ucf.py cubbox RAW_CUB_DIR OUT_DIR [SIZE]  add bounding boxes to an existing cub_uint8.pt

CUB: the WHOLE image is resized (squashed) to SIZE x SIZE, so the bird is never cut off and all background is kept;
the official train/test split (5994 / 5794). Boxes (x0, y0, x1, y1) are stored in SIZE-pixel coordinates.
UCF101: official split 1 (9537 / 3783 videos). Train: TRAIN_FRAMES frames per video at evenly spaced positions
(k + 1) / (TRAIN_FRAMES + 1); test: the middle frame of each video. Frames are resized (squashed) 320x240 -> SIZE x SIZE.
"""
import io
import os
import subprocess
import sys
import zipfile
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch
from PIL import Image


def _stats(x):
    xf = x.float().div(255)
    return [round(v, 4) for v in xf.mean((0, 2, 3)).tolist()], [round(v, 4) for v in xf.std((0, 2, 3)).tolist()]


def _read_table(path):
    return [ln.split() for ln in open(path) if ln.strip()]


# ----------------------------------------------------------------------------- CUB
def prep_cub(raw, out, size):
    names = {int(i): p for i, p in _read_table(os.path.join(raw, "images.txt"))}
    labels = {int(i): int(c) - 1 for i, c in _read_table(os.path.join(raw, "image_class_labels.txt"))}
    split = {int(i): int(t) for i, t in _read_table(os.path.join(raw, "train_test_split.txt"))}
    zf = zipfile.ZipFile(os.path.join(raw, "images.zip")) if os.path.exists(os.path.join(raw, "images.zip")) else None
    members = {}
    if zf is not None:
        for m in zf.namelist():
            members[m.split("images/", 1)[-1] if "images/" in m else m] = m
    xs = {0: [], 1: []}
    ys = {0: [], 1: []}
    ids = {0: [], 1: []}
    wh = {0: [], 1: []}
    for i in sorted(names):
        p = names[i]
        if zf is not None:
            im = Image.open(io.BytesIO(zf.read(members[p])))
        else:
            im = Image.open(os.path.join(raw, "images", p))
        im = im.convert("RGB")
        w, h = im.size
        a = np.asarray(im.resize((size, size), Image.BICUBIC))
        t = 1 - split[i]  # train_test_split: 1 = train -> bucket 0
        xs[t].append(a)
        ys[t].append(labels[i])
        ids[t].append(i)
        wh[t].append((w, h))
    d = {}
    for t, nm in ((0, "train"), (1, "test")):
        d[f"{nm}_x"] = torch.from_numpy(np.stack(xs[t]).transpose(0, 3, 1, 2).copy())
        d[f"{nm}_y"] = torch.tensor(ys[t], dtype=torch.long)
        d[f"{nm}_id"] = torch.tensor(ids[t], dtype=torch.long)
        d[f"{nm}_wh"] = torch.tensor(wh[t], dtype=torch.long)
    _add_boxes(d, raw, size)
    path = os.path.join(out, "cub_uint8.pt" if size == 96 else f"cub{size}_uint8.pt")
    torch.save(d, path)
    _report("cub", d, path)


def _add_boxes(d, raw, size):
    bb = os.path.join(raw, "bounding_boxes.txt")
    if not os.path.exists(bb):
        print("no bounding_boxes.txt: boxes not stored (run `cubbox` later)")
        return False
    box = {int(r[0]): [float(v) for v in r[1:5]] for r in _read_table(bb)}  # x, y, w, h in original pixels
    for nm in ("train", "test"):
        out = []
        for i, (w, h) in zip(d[f"{nm}_id"].tolist(), d[f"{nm}_wh"].tolist()):
            x, y, bw, bh = box[i]
            out.append([x / w * size, y / h * size, (x + bw) / w * size, (y + bh) / h * size])
        d[f"{nm}_box"] = torch.tensor(out, dtype=torch.float32)
    frac = ((d["train_box"][:, 2] - d["train_box"][:, 0]) * (d["train_box"][:, 3] - d["train_box"][:, 1])) / size ** 2
    print(f"bird box area / image area (train): mean {frac.mean():.3f}, median {frac.median():.3f}, "
          f"q10 {frac.quantile(0.1):.3f}, q90 {frac.quantile(0.9):.3f}")
    return True


def add_cub_boxes(raw, out, size):
    path = os.path.join(out, "cub_uint8.pt")
    d = torch.load(path)
    if _add_boxes(d, raw, size):
        torch.save(d, path)
        print("boxes added to", path)


# ----------------------------------------------------------------------------- UCF101
FFMPEG = None


def _frames(args):
    """Decode the frames of one video at the given relative positions, resized to size x size (RGB uint8)."""
    path, positions, size = args
    raw = subprocess.run([FFMPEG, "-v", "error", "-i", path, "-vf", f"scale={size}:{size}:flags=bicubic",
                          "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], capture_output=True, check=True).stdout
    n = len(raw) // (size * size * 3)
    v = np.frombuffer(raw[:n * size * size * 3], dtype=np.uint8).reshape(n, size, size, 3)
    idx = [min(n - 1, int(round(p * (n - 1)))) for p in positions]
    return v[idx]


def prep_ucf(raw, out, size, train_frames):
    global FFMPEG
    import imageio_ffmpeg
    FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
    lst = os.path.join(raw, "ucfTrainTestlist")
    cls = {n: int(i) - 1 for i, n in _read_table(os.path.join(lst, "classInd.txt"))}
    train = [r[0] for r in _read_table(os.path.join(lst, "trainlist01.txt"))]
    test = [r[0] for r in _read_table(os.path.join(lst, "testlist01.txt"))]
    d = {}
    for nm, vids, pos in (("train", train, [(k + 1) / (train_frames + 1) for k in range(train_frames)]),
                          ("test", test, [0.5])):
        jobs = [(os.path.join(raw, "UCF-101", v), pos, size) for v in vids]
        with ProcessPoolExecutor(int(os.environ.get("PREP_WORKERS", "12"))) as ex:
            fr = list(ex.map(_frames, jobs, chunksize=16))
        x = np.concatenate(fr)
        y = np.repeat([cls[v.split("/")[0]] for v in vids], len(pos))
        g = np.repeat(np.arange(len(vids)), len(pos))
        d[f"{nm}_x"] = torch.from_numpy(x.transpose(0, 3, 1, 2).copy())
        d[f"{nm}_y"] = torch.tensor(y, dtype=torch.long)
        d[f"{nm}_video"] = torch.tensor(g, dtype=torch.long)
        print(nm, len(vids), "videos", x.shape[0], "frames", flush=True)
    path = os.path.join(out, "ucf101_uint8.pt")
    torch.save(d, path)
    _report("ucf101", d, path)


def _shard_frames(args):
    """Stream one frames shard (tar.gz of <cls>/<video>/frame.NNN.image.jpeg, a video's frames contiguous) and keep the
    frames at the requested relative positions of the videos in `want` (video -> positions)."""
    import tarfile
    path, want, size = args
    out = {}
    cur, buf = None, {}

    def flush():
        if cur in want and buf:
            keys = sorted(buf)
            n = len(keys)
            imgs = []
            for p in want[cur]:
                k = keys[min(n - 1, int(round(p * (n - 1))))]
                im = Image.open(io.BytesIO(buf[k])).convert("RGB").resize((size, size), Image.BICUBIC)
                imgs.append(np.asarray(im))
            out[cur] = np.stack(imgs)

    with tarfile.open(path, "r|gz") as tf:
        for m in tf:
            if not m.isfile() or not m.name.endswith(".jpeg"):
                continue
            parts = m.name.split("/")
            vid = parts[-2]
            if vid != cur:
                flush()
                cur, buf = vid, {}
            if vid in want:
                buf[int(parts[-1].split(".")[1])] = tf.extractfile(m).read()
        flush()
    return out


def prep_ucf_frames(raw, out, size, train_frames):
    """UCF101 from the pre-extracted frames mirror (mavi88/ucf101_frames, shards/*.tar.gz), official split 1."""
    import glob
    lst = os.path.join(raw, "ucfTrainTestlist")
    cls = {n: int(i) - 1 for i, n in _read_table(os.path.join(lst, "classInd.txt"))}
    vcls = {}
    for f in ("trainlist01.txt", "testlist01.txt"):
        for r in _read_table(os.path.join(lst, f)):
            vcls[r[0].split("/")[1][:-4]] = cls[r[0].split("/")[0]]  # folder name (video names differ in case)
    train = [r[0].split("/")[1][:-4] for r in _read_table(os.path.join(lst, "trainlist01.txt"))]
    test = [r[0].split("/")[1][:-4] for r in _read_table(os.path.join(lst, "testlist01.txt"))]
    pos_tr = [(k + 1) / (train_frames + 1) for k in range(train_frames)]
    want = {v: pos_tr for v in train}
    want.update({v: [0.5] for v in test})
    shards = sorted(glob.glob(os.path.join(raw, "shards", "*.tar.gz")))
    got = {}
    with ProcessPoolExecutor(min(len(shards), int(os.environ.get("PREP_WORKERS", "13")))) as ex:
        for o in ex.map(_shard_frames, [(s, want, size) for s in shards]):
            got.update(o)
    miss = [v for v in train + test if v not in got]
    print(f"{len(got)} videos decoded, {len(miss)} missing", miss[:10], flush=True)
    d = {}
    for nm, vids in (("train", train), ("test", test)):
        vids = [v for v in vids if v in got]
        x = np.concatenate([got[v] for v in vids])
        y = np.concatenate([[vcls[v]] * len(got[v]) for v in vids])
        g = np.concatenate([[i] * len(got[v]) for i, v in enumerate(vids)])
        d[f"{nm}_x"] = torch.from_numpy(x.transpose(0, 3, 1, 2).copy())
        d[f"{nm}_y"] = torch.tensor(y, dtype=torch.long)
        d[f"{nm}_video"] = torch.tensor(g, dtype=torch.long)
        print(nm, len(vids), "videos", x.shape[0], "frames", flush=True)
    path = os.path.join(out, "ucf101_uint8.pt")
    torch.save(d, path)
    _report("ucf101", d, path)


def _pets_annotations(raw):
    """Head boxes (official, trainval only) and trimaps from the annotations archive (a truncated archive is fine:
    whatever was read before the truncation is kept). Returns {stem: (x0, y0, x1, y1, W, H)}, {stem: png bytes}."""
    import re
    import tarfile
    boxes, trimaps = {}, {}
    for arc in ("pets_ann_full.tar.gz", "pets_ann.tar.gz"):
        path = os.path.join(raw, arc)
        if not os.path.exists(path):
            continue
        try:
            with tarfile.open(path, "r|gz") as tf:
                for m in tf:
                    if not m.isfile() or "/._" in m.name:
                        continue
                    stem = os.path.basename(m.name).rsplit(".", 1)[0]
                    if m.name.endswith(".xml") and stem not in boxes:
                        t = tf.extractfile(m).read().decode(errors="ignore")
                        wh = re.search(r"<width>(\d+)</width>\s*<height>(\d+)</height>", t)
                        bb = re.search(r"<xmin>(\d+)</xmin>\s*<ymin>(\d+)</ymin>\s*<xmax>(\d+)</xmax>\s*<ymax>(\d+)</ymax>", t)
                        if wh and bb:
                            boxes[stem] = tuple(float(v) for v in bb.groups()) + tuple(float(v) for v in wh.groups())
                    elif "trimaps/" in m.name and m.name.endswith(".png") and stem not in trimaps:
                        trimaps[stem] = tf.extractfile(m).read()
        except (tarfile.ReadError, EOFError, OSError, zipfile.BadZipFile) as e:
            print(f"{arc}: truncated ({type(e).__name__}); kept {len(boxes)} boxes, {len(trimaps)} trimaps so far")
    return boxes, trimaps


def prep_pets(raw, out, size):
    """Oxford-IIIT Pets, official trainval/test split (HF timm/oxford-iiit-pet parquet), whole image squashed to SIZE.
    train_box: official head box in SIZE-px coordinates (NaN where absent); train_fg: trimap pet-or-border mask at
    SIZE (uint8 0/1; 255 everywhere if the trimap is absent)."""
    import pyarrow.parquet as pq
    boxes, trimaps = _pets_annotations(raw)
    d = {}
    for nm in ("train", "test"):
        t = pq.read_table(os.path.join(raw, f"{nm}.parquet")).to_pydict()
        xs, bx, fg = [], [], []
        for im, iid in zip(t["image"], t["image_id"]):
            pim = Image.open(io.BytesIO(im["bytes"])).convert("RGB")
            w, h = pim.size
            xs.append(np.asarray(pim.resize((size, size), Image.BICUBIC)))
            b = boxes.get(iid)
            if b is not None:
                x0, y0, x1, y1, W, H = b
                bx.append([x0 / W * size, y0 / H * size, x1 / W * size, y1 / H * size])
            else:
                bx.append([float("nan")] * 4)
            if iid in trimaps:  # 1 = pet, 2 = background, 3 = border
                tm = np.asarray(Image.open(io.BytesIO(trimaps[iid])).resize((size, size), Image.NEAREST))
                fg.append((tm != 2).astype(np.uint8))
            else:
                fg.append(np.full((size, size), 255, np.uint8))
        d[f"{nm}_x"] = torch.from_numpy(np.stack(xs).transpose(0, 3, 1, 2).copy())
        d[f"{nm}_y"] = torch.tensor(t["label"], dtype=torch.long)
        d[f"{nm}_box"] = torch.tensor(bx, dtype=torch.float32)
        d[f"{nm}_fg"] = torch.from_numpy(np.stack(fg))
        d[f"{nm}_id"] = list(t["image_id"])
        ok = ~torch.isnan(d[f"{nm}_box"][:, 0])
        print(f"{nm}: {len(xs)} images, head boxes {int(ok.sum())}, trimaps {int((d[f'{nm}_fg'][:, 0, 0] != 255).sum())}")
    b = d["train_box"][~torch.isnan(d["train_box"][:, 0])]
    frac = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]) / size ** 2
    print(f"head box area / image area (train): mean {frac.mean():.3f}, median {frac.median():.3f}")
    path = os.path.join(out, "pets_uint8.pt" if size == 96 else f"pets{size}_uint8.pt")
    torch.save(d, path)
    _report("pets", {k: v for k, v in d.items() if torch.is_tensor(v)}, path)


def _report(name, d, path):
    print(name, {k: tuple(v.shape) for k, v in d.items()}, "->", path)
    print("classes", int(d["train_y"].max()) + 1, "per-class train", torch.bincount(d["train_y"]).min().item(),
          torch.bincount(d["train_y"]).max().item())
    m, s = _stats(d["train_x"])
    print(f"mean={tuple(m)}, std={tuple(s)}")


if __name__ == "__main__":
    what, raw, out = sys.argv[1], sys.argv[2], sys.argv[3]
    size = int(sys.argv[4]) if len(sys.argv) > 4 else 96
    if what == "cub":
        prep_cub(raw, out, size)
    elif what == "cubbox":
        add_cub_boxes(raw, out, size)
    elif what == "pets":
        prep_pets(raw, out, size)
    elif what == "ucfframes":
        prep_ucf_frames(raw, out, size, int(sys.argv[5]) if len(sys.argv) > 5 else 3)
    else:
        prep_ucf(raw, out, size, int(sys.argv[5]) if len(sys.argv) > 5 else 3)
