from __future__ import annotations

import torch
from torch import Tensor, nn

from .common import ConvBNAct, validate_nchw


class LSKAttention(nn.Module):
    """Large Selective Kernel attention adapted to an NCHW residual block.

    Mechanism: parallel/local-to-long-range depthwise branches are spatially selected
    by pooled descriptors. This is a clean YOLO neck adapter, not a copy of LSKNet.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.channels = channels
        self.dw5 = nn.Conv2d(channels, channels, 5, padding=2, groups=channels)
        self.dw7_d3 = nn.Conv2d(
            channels, channels, 7, padding=9, dilation=3, groups=channels
        )
        hidden = max(channels // 2, 8)
        self.proj1 = nn.Conv2d(channels, hidden, 1)
        self.proj2 = nn.Conv2d(channels, hidden, 1)
        self.selector = nn.Conv2d(2, 2, 7, padding=3)
        self.expand = nn.Conv2d(hidden, channels, 1)
        self.out = ConvBNAct(channels, channels, 1, activation=False)

    def forward(self, x: Tensor) -> Tensor:
        validate_nchw(x, self.channels)
        a1 = self.dw5(x)
        a2 = self.dw7_d3(a1)
        b1, b2 = self.proj1(a1), self.proj2(a2)
        fused = torch.cat([b1, b2], dim=1)
        avg = fused.mean(dim=1, keepdim=True)
        mx = fused.amax(dim=1, keepdim=True)
        weights = torch.sigmoid(self.selector(torch.cat([avg, mx], dim=1)))
        selected = b1 * weights[:, 0:1] + b2 * weights[:, 1:2]
        gate = torch.sigmoid(self.expand(selected))
        return x + self.out(x * gate)
