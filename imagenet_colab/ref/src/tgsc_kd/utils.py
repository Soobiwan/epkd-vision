from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


def unwrap_logits(output: Any) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (tuple, list)) and output:
        return unwrap_logits(output[0])
    if isinstance(output, dict):
        for key in ("logits", "output", "out"):
            if key in output:
                return unwrap_logits(output[key])
    raise TypeError(f"Unsupported model output type: {type(output)!r}")


def forward_logits(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    return unwrap_logits(model(x))


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        torch.multiprocessing.set_sharing_strategy("file_system")
    except (RuntimeError, ValueError):
        pass
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = forward_logits(model, x)
        pred = logits.argmax(1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    model.train()
    return 100.0 * correct / max(1, total)


@torch.no_grad()
def accuracy(output: torch.Tensor, target: torch.Tensor, topk: Sequence[int] = (1,)) -> list[torch.Tensor]:
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))
    scores: list[torch.Tensor] = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        scores.append(correct_k.mul_(100.0 / batch_size))
    return scores


@torch.no_grad()
def evaluate_topk(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, float, float]:
    model.eval()
    top1_sum = 0.0
    top5_sum = 0.0
    loss_sum = 0.0
    total = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = forward_logits(model, x)
        loss = F.cross_entropy(logits, y)
        acc1, acc5 = accuracy(logits, y, topk=(1, min(5, logits.size(1))))
        batch_size = y.size(0)
        total += batch_size
        loss_sum += float(loss.item()) * batch_size
        top1_sum += float(acc1.item()) * batch_size
        top5_sum += float(acc5.item()) * batch_size
    model.train()
    if total == 0:
        return 0.0, 0.0, 0.0
    return top1_sum / total, top5_sum / total, loss_sum / total


def build_stage_plan(stages: Sequence[tuple[int, int]]) -> list[int]:
    per_epoch_res: list[int] = []
    for resolution, epochs in stages:
        per_epoch_res.extend([int(resolution)] * int(epochs))
    return per_epoch_res


def resolve_checkpoint_dir(save_path: str, checkpoint_dir: str | None = None) -> Path:
    if checkpoint_dir:
        return Path(checkpoint_dir)
    save = Path(save_path)
    stem = save.stem if save.suffix else save.name
    return save.parent / stem


def _last_linear(layer: nn.Module) -> Optional[nn.Linear]:
    if isinstance(layer, nn.Linear):
        return layer
    if isinstance(layer, nn.Sequential):
        for module in reversed(layer):
            if isinstance(module, nn.Linear):
                return module
    return None


def infer_num_classes(model: nn.Module, device: Optional[torch.device] = None) -> int:
    base = model.module if isinstance(model, nn.DataParallel) else model
    for attr in ("fc", "classifier", "head", "linear"):
        layer = getattr(base, attr, None)
        linear = _last_linear(layer) if layer is not None else None
        if linear is not None:
            return linear.out_features
    with torch.no_grad():
        dev = device or next(base.parameters()).device
        dummy = torch.zeros(1, 3, 32, 32, device=dev)
        logits = forward_logits(base, dummy)
    return int(logits.shape[1])


def pick_target_layer(model: nn.Module) -> nn.Module:
    base = model.module if isinstance(model, (nn.DataParallel, nn.parallel.DistributedDataParallel)) else model

    if hasattr(base, "layer4") and isinstance(base.layer4, nn.Sequential):
        last_block = base.layer4[-1]
        if hasattr(last_block, "conv3"):
            return last_block.conv3
        if hasattr(last_block, "conv2"):
            return last_block.conv2
        return last_block

    if hasattr(base, "block3"):
        if isinstance(base.block3, nn.Sequential):
            last_block = base.block3[-1]
            if hasattr(last_block, "conv2"):
                return last_block.conv2
            return last_block
        if hasattr(base.block3, "layer"):
            last_block = base.block3.layer[-1]
            if hasattr(last_block, "conv2"):
                return last_block.conv2
            return last_block

    if hasattr(base, "layer3") and isinstance(base.layer3, nn.Sequential):
        last_block = base.layer3[-1]
        if hasattr(last_block, "conv2"):
            return last_block.conv2
        if hasattr(last_block, "conv3"):
            return last_block.conv3
        return last_block

    if hasattr(base, "features") and isinstance(base.features, nn.Sequential):
        for layer in reversed(base.features):
            if isinstance(layer, nn.Conv2d):
                return layer
        return base.features[-1]

    if hasattr(base, "stage4"):
        return base.stage4[-1] if isinstance(base.stage4, nn.Sequential) else base.stage4

    if hasattr(base, "conv5"):
        return base.conv5

    last_conv = None
    for name, module in base.named_modules():
        if isinstance(module, nn.Conv2d) and "classifier" not in name.lower() and "fc" not in name.lower():
            last_conv = module

    if last_conv is not None:
        return last_conv

    raise ValueError(f"Could not infer a Grad-CAM target layer for {type(base).__name__}.")
