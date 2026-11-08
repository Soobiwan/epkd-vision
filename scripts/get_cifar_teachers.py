"""Downloads the standard CIFAR-100 teachers of the CRD / DKD benchmark and saves weights-only state dicts.

Source: RepDistiller (Tian et al., CRD, ICLR 2020; github.com/HobbitLong/RepDistiller), which publishes
<name>_vanilla/ckpt_epoch_240.pth. Fallback: the same teachers in the mdistiller release (Zhao et al., DKD,
CVPR 2022; github.com/megvii-research/mdistiller, cifar_teachers.tar).

Usage: python scripts/get_cifar_teachers.py [--out teachers] resnet56 resnet110 resnet32x4 vgg13 wrn_40_2
Writes <out>/<name>_cifar100.pth (a plain state dict, loadable with weights_only=True).
"""
import argparse
import hashlib
import io
import os
import sys
import tarfile
import tempfile
import time
import urllib.request

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from train import load_checkpoint  # noqa: E402

REPDISTILLER = "https://shape2prog.csail.mit.edu/repo/{name}_vanilla/ckpt_epoch_240.pth"
MDISTILLER = "https://github.com/megvii-research/mdistiller/releases/download/checkpoints/cifar_teachers.tar"
# sha256 of the RepDistiller files and their CIFAR-100 test accuracy (both checked 2026-11-08)
TEACHERS = {
    "resnet56": ("4143895c82105da0a73236230db11aee5be39b660fe21e1323037c774f994155", 72.41),
    "resnet110": ("75dc04faf468c078f1dfa010b51069fa3a8a4b91145379574e05c8e63e7441a9", 74.31),
    "resnet32x4": ("22aa12e2b632b19ddde155ccf2388671842ddd304824b3095e0a7fff364beb4f", 79.42),
    "vgg13": ("ff25893af9e28fe12cfcc4fd0c699596754eb89bfafd521dfb1301cd281b47f8", 74.64),
    "wrn_40_2": ("41100a8e8478b326645eb9aab09c18fc8ed34c1ab4004d658cb652125a81ca18", 75.61),
    "ResNet50": ("9f546083e2e4e5aaa733b328f34c354acb93b168952684f7742f556ba1253c3d", 79.34),
}


def fetch(url, timeout=600, tries=4):
    """Whole-file download; a truncated transfer raises (IncompleteRead) and is retried."""
    for k in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "epkd-vision"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                data = r.read()
                n = r.headers.get("Content-Length")
            if n is not None and len(data) != int(n):
                raise IOError(f"got {len(data)} of {n} bytes")
            return data
        except Exception as e:
            if k == tries - 1:
                raise
            print(f"  download attempt {k + 1} failed ({e}); retrying")
            time.sleep(10 * (k + 1))


def to_state_dict(raw):
    with tempfile.NamedTemporaryFile(suffix=".pth") as f:
        f.write(raw)
        f.flush()
        ck = load_checkpoint(f.name)  # weights_only=True
    return ck["model"] if isinstance(ck, dict) and "model" in ck else ck


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="+", choices=sorted(TEACHERS))
    ap.add_argument("--out", default="teachers")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    tar_bytes = None
    for name in args.names:
        dst = os.path.join(args.out, f"{name}_cifar100.pth")
        if os.path.exists(dst):
            print(f"{name}: {dst} exists")
            continue
        sha, acc = TEACHERS[name]
        raw = None
        try:
            raw = fetch(REPDISTILLER.format(name=name))
            got = hashlib.sha256(raw).hexdigest()
            if sha and got != sha:
                print(f"{name}: RepDistiller file has sha256 {got}, expected {sha}; trying mdistiller")
                raw = None
            else:
                print(f"{name}: RepDistiller download OK ({len(raw) / 1e6:.1f} MB)")
        except Exception as e:  # server down / blocked
            print(f"{name}: RepDistiller download failed ({e}); trying mdistiller")
        if raw is None:
            if tar_bytes is None:
                tar_bytes = fetch(MDISTILLER, timeout=1800)
            with tarfile.open(fileobj=io.BytesIO(tar_bytes)) as tf:
                member = next(m for m in tf.getmembers()
                              if m.name.endswith(f"{name}_vanilla/ckpt_epoch_240.pth"))
                raw = tf.extractfile(member).read()
            print(f"{name}: mdistiller download OK")
        torch.save(to_state_dict(raw), dst)
        print(f"{name}: saved {dst} (expected test accuracy {acc:.2f})")


if __name__ == "__main__":
    main()
