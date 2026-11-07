"""Extracts the model weights ('state_dict') of a timm/OFA-KD training checkpoint into a plain state dict.

The checkpoint is read with torch.load(weights_only=True); only argparse.Namespace (the stored training args) is
allowed in addition to tensors, so no code from the file is executed.
Usage: python scripts/convert_ofa_checkpoint.py IN.pth OUT.pt
"""
import argparse
import sys

import torch

torch.serialization.add_safe_globals([argparse.Namespace])
ck = torch.load(sys.argv[1], map_location="cpu", weights_only=True)
sd = ck["state_dict"] if "state_dict" in ck else ck
torch.save(sd, sys.argv[2])
print(f"{sys.argv[2]}: {len(sd)} tensors; checkpoint metric {ck.get('metric', 'n/a')}")
