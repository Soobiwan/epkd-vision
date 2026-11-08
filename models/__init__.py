"""CIFAR model zoo of the CRD / DKD benchmark (RepDistiller, mdistiller), with resolution-agnostic pooling so that
16-px restricted views run through every network. WRNs come from ../wrn.py (unchanged).

Every model returns (logits, feats) with feats["feats"][-1] the pre-pool feature map, so that
logits = classifier(feats["feats"][-1].mean((2, 3))) and the class-activation map is well defined.
"""
import wrn

from .resnet import resnet20, resnet32, resnet56, resnet110, resnet8x4, resnet32x4
from .vgg import vgg8_bn, vgg13_bn
from .mobilenetv2 import mobile_half
from .ShuffleNetv1 import ShuffleV1
from .ShuffleNetv2 import ShuffleV2
from .resnetv2 import ResNet50  # RepDistiller / mdistiller resnetv2.py (already adaptive-pooled)

# names follow the CRD / mdistiller teacher folders (<name>_vanilla/ckpt_epoch_240.pth)
MODELS = {
    "resnet20": resnet20, "resnet32": resnet32, "resnet56": resnet56, "resnet110": resnet110,
    "resnet8x4": resnet8x4, "resnet32x4": resnet32x4,
    "wrn_16_1": wrn.wrn_16_1, "wrn_16_2": wrn.wrn_16_2, "wrn_40_1": wrn.wrn_40_1, "wrn_40_2": wrn.wrn_40_2,
    "vgg8": vgg8_bn, "vgg13": vgg13_bn,
    "MobileNetV2": lambda num_classes: mobile_half(num_classes),
    "ShuffleV1": ShuffleV1, "ShuffleV2": ShuffleV2,
    "ResNet50": ResNet50,
}


def build_model(name, num_classes):
    if name not in MODELS:
        raise SystemExit(f"unknown model {name}; choose from {sorted(MODELS)}")
    return MODELS[name](num_classes=num_classes)


def classifier(model):
    """Final linear layer (its weight gives the CAM)."""
    for attr in ("fc", "linear", "classifier"):
        m = getattr(model, attr, None)
        if m is not None:
            return m[-1] if hasattr(m, "__getitem__") else m
    raise AttributeError(f"{type(model).__name__} has no fc / linear / classifier")
