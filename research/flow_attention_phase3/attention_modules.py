from __future__ import annotations

import torch
from torch import nn


class EMAAttention(nn.Module):
    """Efficient Multi-Scale Attention (EMA), adapted as a channel-preserving YOLO block.

    The implementation follows the ICASSP 2023 EMA formulation: channel grouping,
    directional pooling, local 3x3 context and cross-spatial weighting. The module
    preserves NCHW shape and is intended for a single P3 detection-side branch.
    """

    def __init__(self, channels: int, factor: int = 8) -> None:
        super().__init__()
        if channels <= 0:
            raise ValueError("channels must be positive")
        if factor <= 0 or channels % factor != 0:
            raise ValueError("factor must be positive and divide channels")

        self.groups = factor
        group_channels = channels // factor
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.conv1x1 = nn.Conv2d(group_channels, group_channels, kernel_size=1)
        self.conv3x3 = nn.Conv2d(
            group_channels, group_channels, kernel_size=3, padding=1
        )
        self.norm = nn.GroupNorm(group_channels, group_channels)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        g = self.groups
        group_x = x.reshape(b * g, c // g, h, w)

        x_h = group_x.mean(dim=3, keepdim=True)
        x_w = group_x.mean(dim=2, keepdim=True).transpose(2, 3)
        hw = self.conv1x1(torch.cat((x_h, x_w), dim=2))
        a_h, a_w = torch.split(hw, (h, w), dim=2)
        a_w = a_w.transpose(2, 3)

        x1 = self.norm(group_x * a_h.sigmoid() * a_w.sigmoid())
        x2 = self.conv3x3(group_x)

        q1 = self.softmax(self.pool(x1).reshape(b * g, 1, -1))
        k1 = x2.reshape(b * g, c // g, h * w)
        q2 = self.softmax(self.pool(x2).reshape(b * g, 1, -1))
        k2 = x1.reshape(b * g, c // g, h * w)

        weights = (torch.bmm(q1, k1) + torch.bmm(q2, k2)).reshape(
            b * g, 1, h, w
        )
        return (group_x * weights.sigmoid()).reshape(b, c, h, w)


class CAAAttention(nn.Module):
    """Context Anchor Attention (CAA), adapted from PKINet (CVPR 2024).

    Average pooling forms a local context anchor. Depthwise horizontal and vertical
    strip convolutions then aggregate long-range context before a sigmoid gate.
    Input and output shapes are identical.
    """

    def __init__(self, channels: int, kernel_size: int = 11) -> None:
        super().__init__()
        if channels <= 0:
            raise ValueError("channels must be positive")
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer")

        padding = kernel_size // 2
        self.avg_pool = nn.AvgPool2d(kernel_size=7, stride=1, padding=3)
        self.pre = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(),
        )
        self.h_conv = nn.Conv2d(
            channels,
            channels,
            kernel_size=(1, kernel_size),
            padding=(0, padding),
            groups=channels,
            bias=False,
        )
        self.v_conv = nn.Conv2d(
            channels,
            channels,
            kernel_size=(kernel_size, 1),
            padding=(padding, 0),
            groups=channels,
            bias=False,
        )
        self.post = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attention = self.avg_pool(x)
        attention = self.pre(attention)
        attention = self.h_conv(attention)
        attention = self.v_conv(attention)
        attention = self.post(attention).sigmoid()
        return x * attention


class _ZPool(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat(
            (x.max(dim=1, keepdim=True).values, x.mean(dim=1, keepdim=True)),
            dim=1,
        )


class _TripletGate(nn.Module):
    def __init__(self, kernel_size: int = 7) -> None:
        super().__init__()
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer")
        self.compress = _ZPool()
        self.spatial = nn.Sequential(
            nn.Conv2d(
                2,
                1,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                bias=False,
            ),
            nn.BatchNorm2d(1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = self.spatial(self.compress(x)).sigmoid()
        return x * scale


class TripletAttention(nn.Module):
    """Triplet Attention (WACV 2021) with explicit cross-dimension interactions.

    ``channels`` is accepted for a uniform Ultralytics parse_model contract; the
    attention itself is parameterized independently of channel count.
    """

    def __init__(
        self,
        channels: int,
        no_spatial: bool = False,
        kernel_size: int = 7,
    ) -> None:
        super().__init__()
        if channels <= 0:
            raise ValueError("channels must be positive")
        self.no_spatial = no_spatial
        self.cw = _TripletGate(kernel_size)
        self.hc = _TripletGate(kernel_size)
        self.hw = None if no_spatial else _TripletGate(kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_perm1 = x.permute(0, 2, 1, 3).contiguous()
        x_out1 = self.cw(x_perm1).permute(0, 2, 1, 3).contiguous()

        x_perm2 = x.permute(0, 3, 2, 1).contiguous()
        x_out2 = self.hc(x_perm2).permute(0, 3, 2, 1).contiguous()

        if self.no_spatial:
            return 0.5 * (x_out1 + x_out2)

        x_out3 = self.hw(x)
        return (x_out1 + x_out2 + x_out3) / 3.0


__all__ = ["EMAAttention", "CAAAttention", "TripletAttention"]
