"""Shared FCOS detection head.

Two 4-conv towers (classification, regression) with GroupNorm, shared across all
pyramid levels. Outputs per level:
  - cls_logits: (B, num_classes, H, W)  -> sigmoid, focal loss
  - reg:        (B, 4, H, W)            -> positive (l,t,r,b) distances IN PIXELS
  - centerness: (B, 1, H, W)            -> logit, BCE

Regression uses ``exp(scale_i * raw)`` with a per-level learnable scalar so one
shared head can serve levels whose object scales differ by orders of magnitude.
Working directly in pixels (rather than stride units) keeps decode/target code
simple; the learnable per-level scale absorbs the magnitude difference.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class Scale(nn.Module):
    def __init__(self, init: float = 1.0) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(float(init)))

    def forward(self, x):
        return x * self.scale


class FCOSHead(nn.Module):
    def __init__(
        self,
        in_channels: int = 256,
        num_classes: int = 5,
        num_convs: int = 4,
        num_levels: int = 5,
        prior_prob: float = 0.01,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes

        cls_tower, reg_tower = [], []
        for _ in range(num_convs):
            cls_tower += [
                nn.Conv2d(in_channels, in_channels, 3, padding=1),
                nn.GroupNorm(32, in_channels),
                nn.ReLU(inplace=True),
            ]
            reg_tower += [
                nn.Conv2d(in_channels, in_channels, 3, padding=1),
                nn.GroupNorm(32, in_channels),
                nn.ReLU(inplace=True),
            ]
        self.cls_tower = nn.Sequential(*cls_tower)
        self.reg_tower = nn.Sequential(*reg_tower)

        self.cls_logits = nn.Conv2d(in_channels, num_classes, 3, padding=1)
        self.bbox_reg = nn.Conv2d(in_channels, 4, 3, padding=1)
        self.centerness = nn.Conv2d(in_channels, 1, 3, padding=1)
        self.scales = nn.ModuleList(Scale(1.0) for _ in range(num_levels))

        # Init tower + prediction convs.
        for module in [self.cls_tower, self.reg_tower, self.cls_logits,
                       self.bbox_reg, self.centerness]:
            for layer in module.modules():
                if isinstance(layer, nn.Conv2d):
                    nn.init.normal_(layer.weight, std=0.01)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)
        # Focal-loss prior bias on the classification output.
        bias = -math.log((1 - prior_prob) / prior_prob)
        nn.init.constant_(self.cls_logits.bias, bias)

    def forward(self, features):
        cls_outs, reg_outs, ctr_outs = [], [], []
        for level, feat in enumerate(features):
            cls_feat = self.cls_tower(feat)
            reg_feat = self.reg_tower(feat)

            cls_outs.append(self.cls_logits(cls_feat))
            ctr_outs.append(self.centerness(reg_feat))
            # Positive pixel distances; clamp scaled logit for AMP-safe exp.
            raw = self.scales[level](self.bbox_reg(reg_feat))
            reg_outs.append(torch.exp(raw.clamp(max=12.0)))
        return cls_outs, reg_outs, ctr_outs
