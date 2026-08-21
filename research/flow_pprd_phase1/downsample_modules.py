from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
import torch.nn.functional as F


def _autopad(k: int) -> int:
    return k // 2


class ConvBNAct(nn.Module):
    """Small standalone Conv-BN-SiLU block for research integration."""

    def __init__(self, c1: int, c2: int, k: int = 1, s: int = 1, g: int = 1) -> None:
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, _autopad(k), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        return self.act(self.bn(self.conv(x)))


def _fuse_conv_bn(conv: nn.Conv2d, bn: nn.BatchNorm2d) -> tuple[Tensor, Tensor]:
    """Fuse Conv2d + BatchNorm2d weights for inference reparameterization."""

    if conv.bias is not None:
        bias = conv.bias
    else:
        bias = torch.zeros(conv.weight.shape[0], device=conv.weight.device, dtype=conv.weight.dtype)
    std = torch.sqrt(bn.running_var + bn.eps)
    scale = bn.weight / std
    weight = conv.weight * scale.reshape(-1, 1, 1, 1)
    bias = bn.bias + (bias - bn.running_mean) * scale
    return weight, bias


class RepContextDown(nn.Module):
    """RepDown-inspired cheap contextual stride-2 downsampling.

    Training graph: pointwise projection -> 7x7 DW stride-2 + 3x3 DW stride-2 -> sum -> SiLU.
    Deployment graph: the two depthwise branches are fused into one 7x7 DW stride-2 convolution.

    This is a paper-level implementation of the reparameterizable pattern, not a claim of
    byte-identical reproduction of an external YOLO-ULM code release.
    """

    def __init__(self, c1: int, c2: int) -> None:
        super().__init__()
        self.pre = ConvBNAct(c1, c2, k=1, s=1)
        self.dw7 = nn.Conv2d(c2, c2, 7, 2, 3, groups=c2, bias=False)
        self.bn7 = nn.BatchNorm2d(c2)
        self.dw3 = nn.Conv2d(c2, c2, 3, 2, 1, groups=c2, bias=False)
        self.bn3 = nn.BatchNorm2d(c2)
        self.act = nn.SiLU(inplace=True)
        self.reparam: nn.Conv2d | None = None

    def forward(self, x: Tensor) -> Tensor:
        x = self.pre(x)
        if self.reparam is not None:
            return self.act(self.reparam(x))
        return self.act(self.bn7(self.dw7(x)) + self.bn3(self.dw3(x)))

    @torch.no_grad()
    def switch_to_deploy(self) -> "RepContextDown":
        """Fuse the two depthwise training branches into one 7x7 depthwise conv."""

        if self.reparam is not None:
            return self
        self.eval()
        w7, b7 = _fuse_conv_bn(self.dw7, self.bn7)
        w3, b3 = _fuse_conv_bn(self.dw3, self.bn3)
        w3 = F.pad(w3, [2, 2, 2, 2])
        rep = nn.Conv2d(
            self.dw7.in_channels,
            self.dw7.out_channels,
            7,
            2,
            3,
            groups=self.dw7.groups,
            bias=True,
        ).to(device=w7.device, dtype=w7.dtype)
        rep.weight.copy_(w7 + w3)
        rep.bias.copy_(b7 + b3)
        self.reparam = rep
        del self.dw7, self.bn7, self.dw3, self.bn3
        return self


class SPDDown(nn.Module):
    """Space-to-depth downsampling followed by a pointwise projection."""

    def __init__(self, c1: int, c2: int) -> None:
        super().__init__()
        self.unshuffle = nn.PixelUnshuffle(2)
        self.project = ConvBNAct(4 * c1, c2, k=1, s=1)

    def forward(self, x: Tensor) -> Tensor:
        if x.shape[-2] % 2 or x.shape[-1] % 2:
            raise ValueError("SPDDown requires even spatial dimensions")
        return self.project(self.unshuffle(x))


@dataclass(frozen=True)
class PartialSplit:
    detail_in: int
    context_in: int
    detail_out: int
    context_out: int


def _partial_split(c1: int, c2: int, detail_ratio: float) -> PartialSplit:
    if c1 < 2 or c2 < 2:
        raise ValueError("PartialPolyphaseRepDown requires at least 2 input/output channels")
    if not 0.0 < detail_ratio < 1.0:
        raise ValueError("detail_ratio must be in (0, 1)")
    detail_in = min(c1 - 1, max(1, int(round(c1 * detail_ratio))))
    detail_out = min(c2 - 1, max(1, int(round(c2 * detail_ratio))))
    return PartialSplit(detail_in, c1 - detail_in, detail_out, c2 - detail_out)


class PartialPolyphaseRepDown(nn.Module):
    """PPRD: partial polyphase preservation + reparameterizable context downsampling.

    A fixed fraction of channels pays the cost of full 2x2 phase preservation via PixelUnshuffle.
    Remaining channels use a cheap reparameterizable context branch. Outputs are concatenated with
    no attention/gating so the first-stage ablation isolates the proposed mechanism.
    """

    def __init__(self, c1: int, c2: int, detail_ratio: float = 0.25) -> None:
        super().__init__()
        self.detail_ratio = float(detail_ratio)
        self.split = _partial_split(c1, c2, self.detail_ratio)
        self.detail = SPDDown(self.split.detail_in, self.split.detail_out)
        self.context = RepContextDown(self.split.context_in, self.split.context_out)

    def forward(self, x: Tensor) -> Tensor:
        xd, xc = torch.split(x, [self.split.detail_in, self.split.context_in], dim=1)
        return torch.cat((self.detail(xd), self.context(xc)), dim=1)

    def switch_to_deploy(self) -> "PartialPolyphaseRepDown":
        self.context.switch_to_deploy()
        return self


__all__ = ["RepContextDown", "SPDDown", "PartialPolyphaseRepDown"]
