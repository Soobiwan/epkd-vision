from __future__ import annotations

import math

import torch
import torch.nn as nn


class _ConvBNReLU(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, groups: int = 1):
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                groups=groups,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class _DepthwiseSeparableBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int):
        super().__init__()
        self.depthwise = _ConvBNReLU(in_channels, in_channels, stride=stride, groups=in_channels)
        self.pointwise = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        return self.pointwise(x)


class MobileNetV1ImageNet(nn.Module):
    def __init__(self, num_classes: int = 1000):
        super().__init__()
        self.stem = _ConvBNReLU(3, 32, stride=2)

        cfg = [
            (64, 1),
            (128, 2),
            (128, 1),
            (256, 2),
            (256, 1),
            (512, 2),
            (512, 1),
            (512, 1),
            (512, 1),
            (512, 1),
            (512, 1),
            (1024, 2),
            (1024, 1),
        ]

        blocks: list[nn.Module] = []
        in_channels = 32
        for out_channels, stride in cfg:
            blocks.append(_DepthwiseSeparableBlock(in_channels, out_channels, stride))
            in_channels = out_channels
        self.features = nn.Sequential(*blocks)
        # Adaptive pooling keeps the classifier head compatible with TGSC's varying crop sizes.
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(1024, num_classes)

        self._initialize_weights()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, list[torch.Tensor] | torch.Tensor]]:
        x = self.stem(x)
        feat0 = x

        x = self.features[0](x)
        x = self.features[1](x)
        x = self.features[2](x)
        feat1 = x

        x = self.features[3](x)
        x = self.features[4](x)
        feat2 = x

        x = self.features[5](x)
        for idx in range(6, 11):
            x = self.features[idx](x)
        feat3 = x

        x = self.features[11](x)
        x = self.features[12](x)
        feat4 = x

        x = self.avgpool(x)
        pooled = torch.flatten(x, 1)
        logits = self.classifier(pooled)

        feats = {
            "pooled_feat": pooled,
            "feats": [feat0, feat1, feat2, feat3, feat4],
        }
        return logits, feats

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                n = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
                module.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1.0)
                module.bias.data.zero_()
            elif isinstance(module, nn.Linear):
                module.weight.data.normal_(0, 0.01)
                module.bias.data.zero_()


def MobileNetV1(*, num_classes: int = 1000, **_: object) -> MobileNetV1ImageNet:
    return MobileNetV1ImageNet(num_classes=num_classes)
