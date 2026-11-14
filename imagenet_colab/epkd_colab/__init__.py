"""Single-GPU (Colab A100) EP-KD ImageNet pipeline built on the unmodified reference code in ../ref.

Import side effect: puts ../ref/src and ../ref on sys.path so that ``tgsc_kd`` (reference package) and the
reference ``model.python`` definitions are importable.
"""
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
REF = ROOT / "ref"
for p in (REF / "src", REF, REF / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
