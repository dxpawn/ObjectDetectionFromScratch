"""Pretrained ResNet feature extractor (the only pretrained component).

Returns the C3/C4/C5 feature maps (strides 8/16/32). BatchNorm running stats are
frozen (FCOS/RetinaNet convention) for stability with small detection batches.
"""

from __future__ import annotations

import torch.nn as nn
from torchvision.models import (
    ResNet18_Weights,
    ResNet34_Weights,
    ResNet50_Weights,
    ResNet101_Weights,
    resnet18,
    resnet34,
    resnet50,
    resnet101,
)

_BUILDERS = {
    "resnet18": (resnet18, ResNet18_Weights.DEFAULT, (128, 256, 512)),
    "resnet34": (resnet34, ResNet34_Weights.DEFAULT, (128, 256, 512)),
    "resnet50": (resnet50, ResNet50_Weights.DEFAULT, (512, 1024, 2048)),
    # resnet101 shares resnet50's C3/C4/C5 channel counts -> FPN/head unchanged.
    "resnet101": (resnet101, ResNet101_Weights.DEFAULT, (512, 1024, 2048)),
}


class ResNetBackbone(nn.Module):
    def __init__(
        self,
        name: str = "resnet34",
        pretrained: bool = True,
        freeze_bn: bool = True,
        freeze_stem: bool = True,
    ) -> None:
        super().__init__()
        if name not in _BUILDERS:
            raise ValueError(f"Unsupported backbone: {name}")
        builder, weights, out_channels = _BUILDERS[name]
        net = builder(weights=weights if pretrained else None)

        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)
        self.layer1 = net.layer1  # stride 4  (C2)
        self.layer2 = net.layer2  # stride 8  (C3)
        self.layer3 = net.layer3  # stride 16 (C4)
        self.layer4 = net.layer4  # stride 32 (C5)
        self.out_channels = out_channels

        self._freeze_bn = freeze_bn
        if freeze_stem:
            for module in (self.stem, self.layer1):
                for param in module.parameters():
                    param.requires_grad_(False)
        if freeze_bn:
            self._set_bn_eval()

    def _set_bn_eval(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()
                for param in module.parameters():
                    param.requires_grad_(False)

    def train(self, mode: bool = True):  # keep BN frozen even in train mode
        super().train(mode)
        if self._freeze_bn:
            self._set_bn_eval()
        return self

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        c3 = self.layer2(x)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        return c3, c4, c5
