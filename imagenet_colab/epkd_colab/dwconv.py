"""Depthwise 3x3 convolution as 9 shifted multiply-adds (opt-in TPU speed experiment, 11-13).

On the TPU a depthwise conv (groups == channels) cannot use the matrix unit and its backward (weight gradient) is
slow; written as shifted elementwise products it is plain vector work for XLA in both directions.  Same maths as
``nn.Conv2d(C, C, 3, stride, padding=1, groups=C, bias=False)`` up to floating-point summation order.

``replace_depthwise(model)`` swaps such convs in place; the replacement module REUSES the original ``weight``
Parameter (same name, same tensor), so the optimizer state, checkpoints and state_dict keys are unchanged.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ShiftDepthwise3x3(nn.Module):
    def __init__(self, conv: nn.Conv2d, acc_fp32: bool = True) -> None:
        super().__init__()
        assert conv.kernel_size == (3, 3) and conv.padding == (1, 1) and conv.dilation == (1, 1) and conv.bias is None
        assert conv.groups == conv.in_channels == conv.out_channels
        self.weight = conv.weight  # (C, 1, 3, 3), shared Parameter
        self.stride = int(conv.stride[0])
        self.acc_fp32 = bool(acc_fp32)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = self.stride
        n, c, h, w = x.shape
        ho, wo = (h - 1) // s + 1, (w - 1) // s + 1
        xp = F.pad(x, (1, 1, 1, 1))
        wt = self.weight.to(x.dtype).view(c, 9)
        acc = None
        for i in range(3):
            for j in range(3):
                tap = xp[:, :, i: i + s * (ho - 1) + 1: s, j: j + s * (wo - 1) + 1: s]
                term = tap * wt[:, 3 * i + j].view(1, c, 1, 1)
                if self.acc_fp32:
                    term = term.float()
                acc = term if acc is None else acc + term
        return acc.to(x.dtype)


def replace_depthwise(model: nn.Module, acc_fp32: bool = True) -> int:
    n = 0
    for name, child in list(model.named_children()):
        if (isinstance(child, nn.Conv2d) and child.groups > 1 and child.groups == child.in_channels == child.out_channels
                and child.kernel_size == (3, 3) and child.padding == (1, 1) and child.bias is None):
            setattr(model, name, ShiftDepthwise3x3(child, acc_fp32))
            n += 1
        else:
            n += replace_depthwise(child, acc_fp32)
    return n
