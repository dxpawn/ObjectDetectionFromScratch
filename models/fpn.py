"""Feature Pyramid Network: C3,C4,C5 -> P3,P4,P5,P6,P7 (strides 8..128).

Follows the RetinaNet/FCOS construction: 1x1 laterals, top-down upsample-and-add,
3x3 output convs, plus P6 (stride-2 conv on C5) and P7 (stride-2 conv on ReLU(P6))
for large objects.
"""

from __future__ import annotations

import torch.nn as nn
import torch.nn.functional as F


class FPN(nn.Module):
    def __init__(self, in_channels=(128, 256, 512), out_channels: int = 256) -> None:
        super().__init__()
        c3, c4, c5 = in_channels

        self.lat3 = nn.Conv2d(c3, out_channels, 1)
        self.lat4 = nn.Conv2d(c4, out_channels, 1)
        self.lat5 = nn.Conv2d(c5, out_channels, 1)

        self.out3 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.out4 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.out5 = nn.Conv2d(out_channels, out_channels, 3, padding=1)

        # P6 from C5, P7 from P6 (FCOS uses C5 for P6).
        self.p6 = nn.Conv2d(c5, out_channels, 3, stride=2, padding=1)
        self.p7 = nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=1)

        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_uniform_(module.weight, a=1)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, c3, c4, c5):
        p5 = self.lat5(c5)
        p4 = self.lat4(c4) + F.interpolate(p5, size=c4.shape[-2:], mode="nearest")
        p3 = self.lat3(c3) + F.interpolate(p4, size=c3.shape[-2:], mode="nearest")

        p3 = self.out3(p3)
        p4 = self.out4(p4)
        p5 = self.out5(p5)
        p6 = self.p6(c5)
        p7 = self.p7(F.relu(p6))
        return [p3, p4, p5, p6, p7]
