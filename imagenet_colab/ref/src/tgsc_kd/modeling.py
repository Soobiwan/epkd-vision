from __future__ import annotations

import re
from pathlib import Path
import sys
from typing import Callable

import torch
import torch.nn as nn
from torch.hub import load_state_dict_from_url

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.python.ShuffleNetv1 import ShuffleV1
from model.python.ShuffleNetv2 import ShuffleV2
from model.python.mobilenetv1_imagenet import MobileNetV1 as ImageNetMobileNetV1
from model.python.mobilenetv2 import mobile_half
from model.python.resnet import (
    resnet8,
    resnet8x4,
    resnet14,
    resnet20,
    resnet32,
    resnet32x4,
    resnet44,
    resnet56,
    resnet110,
)
from model.python.resnet_imagenet import MODEL_URLS as IMAGENET_MODEL_URLS
from model.python.resnet_imagenet import ResNet18 as ImageNetResNet18
from model.python.resnet_imagenet import ResNet34 as ImageNetResNet34
from model.python.resnet_imagenet import ResNet50 as ImageNetResNet50
from model.python.resnetv2 import ResNet50 as CifarResNet50
from model.python.vgg import vgg8_bn, vgg11_bn, vgg13_bn, vgg16_bn, vgg19_bn
from model.python.wrn import wrn_16_1, wrn_16_2, wrn_40_1, wrn_40_2


MODEL_ROOT = ROOT / "model"


def _normalize_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", name.lower())


MODEL_BUILDERS: dict[str, tuple[str, Callable[..., nn.Module]]] = {
    _normalize_name("resnet8"): ("resnet8", resnet8),
    _normalize_name("resnet14"): ("resnet14", resnet14),
    _normalize_name("resnet20"): ("resnet20", resnet20),
    _normalize_name("resnet32"): ("resnet32", resnet32),
    _normalize_name("resnet44"): ("resnet44", resnet44),
    _normalize_name("resnet56"): ("resnet56", resnet56),
    _normalize_name("resnet110"): ("resnet110", resnet110),
    _normalize_name("resnet8x4"): ("resnet8x4", resnet8x4),
    _normalize_name("resnet32x4"): ("resnet32x4", resnet32x4),
    _normalize_name("ResNet18"): ("ResNet18", ImageNetResNet18),
    _normalize_name("ResNet34"): ("ResNet34", ImageNetResNet34),
    _normalize_name("ResNet50"): ("ResNet50", ImageNetResNet50),
    _normalize_name("RN50"): ("RN50", ImageNetResNet50),
    _normalize_name("wrn_16_1"): ("wrn_16_1", wrn_16_1),
    _normalize_name("wrn_16_2"): ("wrn_16_2", wrn_16_2),
    _normalize_name("wrn_40_1"): ("wrn_40_1", wrn_40_1),
    _normalize_name("wrn_40_2"): ("wrn_40_2", wrn_40_2),
    _normalize_name("vgg8"): ("vgg8", vgg8_bn),
    _normalize_name("vgg11"): ("vgg11", vgg11_bn),
    _normalize_name("vgg13"): ("vgg13", vgg13_bn),
    _normalize_name("vgg16"): ("vgg16", vgg16_bn),
    _normalize_name("vgg19"): ("vgg19", vgg19_bn),
    _normalize_name("MobileNetV2"): ("MobileNetV2", mobile_half),
    _normalize_name("MobileNetV1"): ("MobileNetV1", ImageNetMobileNetV1),
    _normalize_name("MNv1"): ("MNv1", ImageNetMobileNetV1),
    _normalize_name("ShuffleV1"): ("ShuffleV1", ShuffleV1),
    _normalize_name("ShuffleV2"): ("ShuffleV2", ShuffleV2),
}

DEFAULT_TEACHER_CHECKPOINTS: dict[str, Path] = {
    _normalize_name("resnet56"): MODEL_ROOT / "cifar_teachers" / "resnet56_vanilla" / "ckpt_epoch_240.pth",
    _normalize_name("resnet110"): MODEL_ROOT / "cifar_teachers" / "resnet110_vanilla" / "ckpt_epoch_240.pth",
    _normalize_name("resnet32x4"): MODEL_ROOT / "cifar_teachers" / "resnet32x4_vanilla" / "ckpt_epoch_240.pth",
    _normalize_name("ResNet50"): MODEL_ROOT / "imagenet_teachers" / "ResNet50_vanilla" / "resnet50-19c8e357.pth",
    _normalize_name("RN50"): MODEL_ROOT / "imagenet_teachers" / "ResNet50_vanilla" / "resnet50-19c8e357.pth",
    _normalize_name("ResNet34"): MODEL_ROOT / "imagenet_teachers" / "ResNet34_vanilla" / "resnet34-333f7ec4.pth",
    _normalize_name("wrn_40_2"): MODEL_ROOT / "cifar_teachers" / "wrn_40_2_vanilla" / "ckpt_epoch_240.pth",
    _normalize_name("vgg13"): MODEL_ROOT / "cifar_teachers" / "vgg13_vanilla" / "ckpt_epoch_240.pth",
}

AUTO_DOWNLOAD_TEACHERS: dict[str, str] = {
    _normalize_name("ResNet34"): IMAGENET_MODEL_URLS["resnet34"],
    _normalize_name("ResNet50"): IMAGENET_MODEL_URLS["resnet50"],
    _normalize_name("RN50"): IMAGENET_MODEL_URLS["resnet50"],
}


def available_models() -> list[str]:
    return sorted(display_name for display_name, _builder in MODEL_BUILDERS.values())


def build_model(name: str, *, num_classes: int) -> nn.Module:
    key = _normalize_name(name)
    if key not in MODEL_BUILDERS:
        raise KeyError(f"Unknown model '{name}'. Use --list-models to inspect supported names.")
    _display_name, builder = MODEL_BUILDERS[key]
    if key == _normalize_name("ResNet50") and num_classes != 1000:
        return CifarResNet50(num_classes=num_classes)
    return builder(num_classes=num_classes)


def _maybe_download_teacher_checkpoint(model_name: str, path: Path) -> None:
    key = _normalize_name(model_name)
    if key not in AUTO_DOWNLOAD_TEACHERS or path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    state_dict = load_state_dict_from_url(AUTO_DOWNLOAD_TEACHERS[key], progress=True)
    torch.save(state_dict, path)


def resolve_teacher_checkpoint(model_name: str, checkpoint_path: str | None = None) -> Path:
    if checkpoint_path:
        return (ROOT / checkpoint_path).resolve() if not Path(checkpoint_path).is_absolute() else Path(checkpoint_path)
    key = _normalize_name(model_name)
    if key not in DEFAULT_TEACHER_CHECKPOINTS:
        raise KeyError(f"No default teacher checkpoint is registered for '{model_name}'.")
    path = DEFAULT_TEACHER_CHECKPOINTS[key]
    _maybe_download_teacher_checkpoint(model_name, path)
    return path


def _extract_state_dict(payload: object) -> dict[str, torch.Tensor]:
    if isinstance(payload, dict):
        for key in ("model", "state_dict", "student"):
            value = payload.get(key)
            if isinstance(value, dict):
                return value
        if payload and all(isinstance(value, torch.Tensor) for value in payload.values()):
            return payload  # type: ignore[return-value]
    raise TypeError("Checkpoint does not contain a usable state dict.")


def _strip_module_prefix(key: str) -> str:
    while key.startswith("module."):
        key = key[len("module."):]
    return key


def _looks_like_mdistiller_mobilenetv1(state_dict: dict[str, torch.Tensor]) -> bool:
    keys = {_strip_module_prefix(key) for key in state_dict}
    return "model.0.0.weight" in keys and "fc.weight" in keys


def _adapt_mdistiller_mobilenetv1_keys(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    adapted: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        new_key = _strip_module_prefix(key)
        if new_key.startswith("model."):
            parts = new_key.split(".")
            if len(parts) >= 3 and parts[1].isdigit():
                layer_idx = int(parts[1])
                sub_idx = parts[2]
                suffix = ".".join(parts[3:])
                suffix = f".{suffix}" if suffix else ""
                if layer_idx == 0:
                    new_key = f"stem.{sub_idx}{suffix}"
                elif 1 <= layer_idx <= 13:
                    feature_idx = layer_idx - 1
                    if sub_idx in {"0", "1"}:
                        new_key = f"features.{feature_idx}.depthwise.{sub_idx}{suffix}"
                    elif sub_idx in {"3", "4"}:
                        pointwise_idx = int(sub_idx) - 3
                        new_key = f"features.{feature_idx}.pointwise.{pointwise_idx}{suffix}"
        elif new_key.startswith("fc."):
            new_key = "classifier." + new_key[len("fc."):]
        adapted[new_key] = value
    return adapted


def _normalize_state_dict_keys(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    if _looks_like_mdistiller_mobilenetv1(state_dict):
        return _adapt_mdistiller_mobilenetv1_keys(state_dict)

    normalized: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        new_key = _strip_module_prefix(key)
        for prefix in ("module.", "model."):
            if new_key.startswith(prefix):
                new_key = new_key[len(prefix):]
        normalized[new_key] = value
    return normalized


def load_checkpoint(model: nn.Module, checkpoint_path: str | Path, *, map_location: str | torch.device = "cpu") -> Path:
    path = Path(checkpoint_path)
    if not path.is_absolute():
        path = ROOT / path
    path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    state_dict = _normalize_state_dict_keys(_extract_state_dict(checkpoint))
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"Checkpoint load mismatch for {path}: missing={len(missing)}, unexpected={len(unexpected)}"
        )
    return path
