"""FCOS detector: backbone + FPN + shared head, plus location-grid helper.

Everything works in PIXEL coordinates of the (letterboxed) network input:
  - location centers are pixel centers of each feature-map cell,
  - the head's regression outputs are pixel (l,t,r,b) distances.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .backbone import ResNetBackbone
from .fpn import FPN
from .head import FCOSHead

STRIDES = (8, 16, 32, 64, 128)


class FCOSDetector(nn.Module):
    def __init__(
        self,
        num_classes: int = 5,
        backbone: str = "resnet34",
        pretrained: bool = True,
        fpn_channels: int = 256,
        freeze_bn: bool = True,
        freeze_stem: bool = True,
    ) -> None:
        super().__init__()
        self.strides = STRIDES
        self.num_classes = num_classes
        self.backbone = ResNetBackbone(
            backbone, pretrained=pretrained, freeze_bn=freeze_bn, freeze_stem=freeze_stem
        )
        self.fpn = FPN(self.backbone.out_channels, fpn_channels)
        self.head = FCOSHead(fpn_channels, num_classes, num_levels=len(STRIDES))

    def forward(self, x):
        c3, c4, c5 = self.backbone(x)
        features = self.fpn(c3, c4, c5)
        cls_outs, reg_outs, ctr_outs = self.head(features)
        return cls_outs, reg_outs, ctr_outs

    @staticmethod
    def locations_for_features(features, strides, device=None):
        """Pixel-center coordinates for every cell of every level.

        Returns a list (per level) of (H*W, 2) tensors of (x, y) centers.
        """
        locations = []
        for feat, stride in zip(features, strides):
            h, w = feat.shape[-2:]
            shift_x = (torch.arange(0, w, device=device or feat.device) + 0.5) * stride
            shift_y = (torch.arange(0, h, device=device or feat.device) + 0.5) * stride
            ys, xs = torch.meshgrid(shift_y, shift_x, indexing="ij")
            locs = torch.stack([xs.reshape(-1), ys.reshape(-1)], dim=1)
            locations.append(locs)
        return locations
