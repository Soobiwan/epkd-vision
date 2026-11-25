"""Builds data/cub_uint8.pt and data/pets_uint8.pt (96 px) exactly as the paper's CUB / Pets runs used them.

The raw files are the ones the local tensors were built from (Hugging Face mirrors, pinned revisions, sha256 checked):
  CUB-200-2011  alkzar90/CC6204-Hackaton-Cub-Dataset  data/images.zip + images.txt + image_class_labels.txt +
                train_test_split.txt (images.txt identical to the official list; official split 5994 / 5794)
  Oxford Pets   timm/oxford-iiit-pet                  data/{train,test}-00000-of-00001.parquet (official trainval / test)
They are converted by prep_cub_ucf.py (byte copy of the local prep script). The training code reads only train_x,
train_y, test_x and test_y; their content sha256 must equal DATA_SHA256 below (the hashes of the local
epkd2/data/{cub,pets}_uint8.pt). Boxes / trimaps (analysis only) are not built here.

JPEG decoding and the bicubic resize come from Pillow, and the pixels depend on its version: Pillow 13.0.0 reproduces
the paper tensors (11.3.0 and 12.0.0 do not). The prep therefore runs with Pillow 13.0.0, installed into a private
folder (pip --target) when the installed version differs; the installed Pillow is the fallback. A prepared
<ds>_uint8.pt found under --fallback-dir (e.g. an attached Kaggle dataset) is used if its hashes match.

usage: python cubpets/get_data.py --out DATA_DIR [--work TMP_DIR] [--fallback-dir /kaggle/input] cub pets
"""
import argparse
import glob
import hashlib
import os
import shutil
import subprocess
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
PILLOW_PIN = "13.0.0"  # gives the paper tensors; 11.3.0 and 12.0.0 give other pixels (checked)

HF = "https://huggingface.co/datasets"
SOURCES = {
    "cub": (f"{HF}/alkzar90/CC6204-Hackaton-Cub-Dataset/resolve/f40d8c4463aed3f9eb67c4b7ff4baaab1aa18e15/data", [
        ("images.zip", "fd4bcf950d96f4dc3d4be023843337aa83f0b260ea74d6d0d432745476052f24"),
        ("images.txt", "e1cfc5ed1f4bd1b838040784b6f75c44f7a25d9b31707a3adab49ec694618c1f"),
        ("image_class_labels.txt", "e9df8c96d897994a245746fe2c98914b9b31f5342bddc6e8b06bdfbf45e49b02"),
        ("train_test_split.txt", "8bbbfcc09b4d3c9c6e3e40c9874bd863eb6658f4f456e472df1b33d3403d2600"),
    ]),
    "pets": (f"{HF}/timm/oxford-iiit-pet/resolve/089695c834a7deb60505b7cc506672db1c31a6aa/data", [
        ("train-00000-of-00001.parquet", "b9ff432b33c77f4b50014684f25686a1cd523090686571743a6b7a257af27422", "train.parquet"),
        ("test-00000-of-00001.parquet", "0997d4744ee6740e96f17f56b40b97c70076a4e4dad565bc63d0ffd03906f754", "test.parquet"),
    ]),
}
# sha256 of the raw tensor bytes (tensor.numpy().tobytes()) of the local epkd2/data/<ds>_uint8.pt
DATA_SHA256 = {
    "cub": {
        "train_x": "aed820e0ac8a9610f64eb6809ac0e720454bc07237285b73ce4abc765ced3df5",
        "train_y": "504ee97fe1a024934b1d7727ebf94c4621f84571f4a8105c3ede63c8af21bd76",
        "test_x": "9bfd31b3fbffec5c444b5b58db2d20eda0bab39e74f40d360fe71591d943aa0d",
        "test_y": "6dec8923d905ccb4759adf932e31ef0567822a98827dbba621ef166d35477c82",
    },
    "pets": {
        "train_x": "2e7f8d4bfdc23a9b3e6f065f835da57028e1179fafce8472f32e71648df61623",
        "train_y": "716985b054a221e7a127e15a303bdd12958c0cca01262b74de5e2789d2726617",
        "test_x": "5e695c3ea96ceb7b8982f9ec4c49379959366c42bcfe2ec286d674cfea909420",
        "test_y": "2be1587ba6a84fd844810a57d66f522315a086c93ac5e846b18994d92f6b5f16",
    },
}


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def tensor_hashes(path):
    import torch
    d = torch.load(path)
    return {k: hashlib.sha256(d[k].contiguous().numpy().tobytes()).hexdigest() for k in DATA_SHA256_KEYS}


DATA_SHA256_KEYS = ("train_x", "train_y", "test_x", "test_y")


def check(ds, path):
    got = tensor_hashes(path)
    bad = [k for k in DATA_SHA256_KEYS if got[k] != DATA_SHA256[ds][k]]
    if bad:
        print(f"[get_data] {ds}: {path} differs from the paper tensors in {bad}", flush=True)
    return not bad


def download(url, dst, sha, tries=5):
    if os.path.exists(dst) and sha256_file(dst) == sha:
        return
    for k in range(tries):
        try:
            t0 = time.time()
            tmp = dst + ".part"
            with urllib.request.urlopen(url, timeout=120) as r, open(tmp, "wb") as f:
                shutil.copyfileobj(r, f, 1 << 22)
            got = sha256_file(tmp)
            if got != sha:
                raise IOError(f"sha256 {got} != {sha}")
            os.replace(tmp, dst)
            print(f"[get_data] {os.path.basename(dst)}: {os.path.getsize(dst) / 2**20:.0f} MB in {time.time() - t0:.0f} s",
                  flush=True)
            return
        except Exception as e:  # noqa: BLE001
            print(f"[get_data] download {url} failed ({e}); retry {k + 1}/{tries}", flush=True)
            time.sleep(10 * (k + 1))
    raise SystemExit(f"could not download {url}")


def pillow_pinned(work):
    """Folder with Pillow PILLOW_PIN (pip --target), for PYTHONPATH."""
    tgt = os.path.join(work, f"pillow{PILLOW_PIN}")
    if not os.path.isdir(os.path.join(tgt, "PIL")):
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "--target", tgt,
                        f"pillow=={PILLOW_PIN}"], check=True)
    return tgt


def prep(ds, raw, out, pythonpath=None):
    env = dict(os.environ)
    if pythonpath:
        env["PYTHONPATH"] = pythonpath + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    code = ("import sys, PIL; sys.argv = ['prep_cub_ucf.py', %r, %r, %r]; print('[get_data] Pillow', PIL.__version__, "
            "flush=True); import runpy; runpy.run_path(%r, run_name='__main__')"
            % (ds, raw, out, os.path.join(HERE, "prep_cub_ucf.py")))
    subprocess.run([sys.executable, "-c", code], check=True, env=env)


def get(ds, out, work, fallback_dir=None):
    path = os.path.join(out, f"{ds}_uint8.pt")
    if os.path.exists(path) and check(ds, path):
        print(f"[get_data] {ds}: {path} ok (hashes match)", flush=True)
        return path
    if fallback_dir and os.path.isdir(fallback_dir):  # an attached copy of the prepared tensors
        for cand in sorted(glob.glob(os.path.join(fallback_dir, "**", f"{ds}_uint8.pt"), recursive=True)):
            if check(ds, cand):
                shutil.copyfile(cand, path)
                print(f"[get_data] {ds}: copied from {cand} (hashes match)", flush=True)
                return path
    base, files = SOURCES[ds]
    raw = os.path.join(work, f"raw_{ds}")
    os.makedirs(raw, exist_ok=True)
    for f in files:
        download(f"{base}/{f[0]}", os.path.join(raw, f[2] if len(f) > 2 else f[0]), f[1])
    import PIL
    tries = [None] if PIL.__version__ == PILLOW_PIN else ["pin", None]
    for how in tries:
        t0 = time.time()
        try:
            prep(ds, raw, out, pythonpath=pillow_pinned(work) if how == "pin" else None)
        except (subprocess.CalledProcessError, OSError) as e:
            print(f"[get_data] {ds}: prep with {'Pillow ' + PILLOW_PIN if how else 'the installed Pillow'} failed ({e})",
                  flush=True)
            continue
        print(f"[get_data] {ds}: prep {time.time() - t0:.0f} s", flush=True)
        if check(ds, path):
            print(f"[get_data] {ds}: {path} ok (hashes match)", flush=True)
            return path
    if os.path.exists(path):
        os.replace(path, path + ".MISMATCH")
    raise SystemExit(f"{ds}: the rebuilt tensors differ from the paper's; attach the prepared {ds}_uint8.pt "
                     f"(Kaggle dataset, see kaggle/CUBPETS_README.md) and run again")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("datasets", nargs="+", choices=sorted(SOURCES))
    ap.add_argument("--out", required=True)
    ap.add_argument("--work", default=None, help="download / scratch folder (default: OUT/raw)")
    ap.add_argument("--fallback-dir", default=None)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    work = a.work or os.path.join(a.out, "raw")
    os.makedirs(work, exist_ok=True)
    for ds in a.datasets:
        get(ds, a.out, work, a.fallback_dir)


if __name__ == "__main__":
    main()
