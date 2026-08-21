"""DySample research implementation for FloW-Img Phase 2.

Adapted from the official MIT-licensed ICCV 2023 implementation:
https://github.com/tiny-smart/dysample
Paper: "Learning to Upsample by Learning to Sample".

The sampling formulation is preserved. This research copy adds explicit argument validation,
type hints, and ``meshgrid(..., indexing='ij')`` to avoid PyTorch indexing ambiguity.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class DySample(nn.Module):
    """Ultra-lightweight content-aware upsampling via learned sampling offsets.

    Args:
        in_channels: Number of input/output feature channels.
        scale: Integer spatial upsampling factor. Phase 2 uses ``2``.
        style: ``"lp"`` (linear -> pixel shuffle) or ``"pl"`` (pixel shuffle -> linear).
        groups: Number of channel groups sampled with independent offsets.
        dyscope: Enable the optional learned offset-scope branch from the official implementation.

    Notes:
        - Channel count is preserved.
        - Phase-2 default is ``scale=2, style="lp", groups=4, dyscope=False``.
        - The module relies only on standard PyTorch operators; no custom CUDA extension is required.
    """

    def __init__(
        self,
        in_channels: int,
        scale: int = 2,
        style: str = "lp",
        groups: int = 4,
        dyscope: bool = False,
    ) -> None:
        super().__init__()
        if scale < 2:
            raise ValueError("scale must be >= 2")
        if style not in {"lp", "pl"}:
            raise ValueError("style must be 'lp' or 'pl'")
        if in_channels < groups or in_channels % groups != 0:
            raise ValueError("in_channels must be divisible by groups and >= groups")
        if style == "pl" and (in_channels < scale**2 or in_channels % scale**2 != 0):
            raise ValueError("style='pl' requires in_channels divisible by scale**2")

        self.in_channels = int(in_channels)
        self.scale = int(scale)
        self.style = style
        self.groups = int(groups)
        self.dyscope = bool(dyscope)

        offset_in_channels = in_channels // (scale**2) if style == "pl" else in_channels
        offset_out_channels = 2 * groups if style == "pl" else 2 * groups * (scale**2)

        self.offset = nn.Conv2d(offset_in_channels, offset_out_channels, kernel_size=1)
        nn.init.normal_(self.offset.weight, mean=0.0, std=0.001)
        nn.init.constant_(self.offset.bias, 0.0)

        if dyscope:
            self.scope = nn.Conv2d(offset_in_channels, offset_out_channels, kernel_size=1, bias=False)
            nn.init.constant_(self.scope.weight, 0.0)

        self.register_buffer("init_pos", self._make_init_pos(), persistent=True)

    def _make_init_pos(self) -> Tensor:
        h = torch.arange(
            (-self.scale + 1) / 2,
            (self.scale - 1) / 2 + 1,
            dtype=torch.float32,
        ) / self.scale
        grid_y, grid_x = torch.meshgrid(h, h, indexing="ij")
        return (
            torch.stack((grid_y, grid_x))
            .transpose(1, 2)
            .repeat(1, self.groups, 1)
            .reshape(1, -1, 1, 1)
        )

    def _sample(self, x: Tensor, offset: Tensor) -> Tensor:
        b, _, h, w = offset.shape
        offset = offset.view(b, 2, -1, h, w)

        coords_h = torch.arange(h, dtype=x.dtype, device=x.device) + 0.5
        coords_w = torch.arange(w, dtype=x.dtype, device=x.device) + 0.5
        grid_w, grid_h = torch.meshgrid(coords_w, coords_h, indexing="ij")
        coords = torch.stack((grid_w, grid_h)).transpose(1, 2).unsqueeze(0).unsqueeze(2)

        normalizer = x.new_tensor([w, h]).view(1, 2, 1, 1, 1)
        coords = 2.0 * (coords + offset) / normalizer - 1.0
        coords = F.pixel_shuffle(coords.view(b, -1, h, w), self.scale)
        coords = (
            coords.view(b, 2, -1, self.scale * h, self.scale * w)
            .permute(0, 2, 3, 4, 1)
            .contiguous()
            .flatten(0, 1)
        )

        sampled = F.grid_sample(
            x.reshape(b * self.groups, -1, h, w),
            coords,
            mode="bilinear",
            align_corners=False,
            padding_mode="border",
        )
        return sampled.view(b, -1, self.scale * h, self.scale * w)

    def _forward_lp(self, x: Tensor) -> Tensor:
        if self.dyscope:
            offset = self.offset(x) * self.scope(x).sigmoid() * 0.5 + self.init_pos
        else:
            offset = self.offset(x) * 0.25 + self.init_pos
        return self._sample(x, offset)

    def _forward_pl(self, x: Tensor) -> Tensor:
        x_shuffled = F.pixel_shuffle(x, self.scale)
        if self.dyscope:
            offset = (
                F.pixel_unshuffle(
                    self.offset(x_shuffled) * self.scope(x_shuffled).sigmoid(),
                    self.scale,
                )
                * 0.5
                + self.init_pos
            )
        else:
            offset = F.pixel_unshuffle(self.offset(x_shuffled), self.scale) * 0.25 + self.init_pos
        return self._sample(x, offset)

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_pl(x) if self.style == "pl" else self._forward_lp(x)


__all__ = ["DySample"]
