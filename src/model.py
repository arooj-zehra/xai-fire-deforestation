"""Model builders. Only two architectures are allowed (scope freeze)."""
import torch.nn as nn
from torchvision import models as tvm

MODELS = ("mobilenet_v3_small", "resnet18")


def build_model(name, num_classes, pretrained=True):
    if name == "mobilenet_v3_small":
        m = tvm.mobilenet_v3_small(weights=tvm.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None)
        m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, num_classes)
    elif name == "resnet18":
        m = tvm.resnet18(weights=tvm.ResNet18_Weights.DEFAULT if pretrained else None)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
    else:
        raise ValueError(f"model must be one of {MODELS}")
    return m


def head(m):
    return m.classifier if hasattr(m, "classifier") else m.fc


def set_trainable(m, backbone):
    """backbone=False -> only the classifier head trains; True -> everything trains."""
    for p in m.parameters():
        p.requires_grad = backbone
    for p in head(m).parameters():
        p.requires_grad = True


def gradcam_layer(m):
    """Last conv block, used for Grad-CAM in Phase 5."""
    return m.features[-1] if hasattr(m, "features") else m.layer4[-1]
