from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn


def _odd_kernel_tuple(values: Sequence[int], *, minimum: int = 1) -> tuple[int, ...]:
    kernels = tuple(int(v) for v in values)
    if not kernels:
        raise ValueError("at least one kernel is required")
    if any(k < minimum or k % 2 == 0 for k in kernels):
        raise ValueError(f"kernels must be odd integers >= {minimum}")
    if len(set(kernels)) != len(kernels):
        raise ValueError("kernels must be unique")
    return kernels


class SegNeXtMSCA(nn.Module):
    """Canonical SegNeXt-style Multi-Scale Convolutional Attention control.

    This is the M1 control. A local depthwise convolution is followed by parallel
    separable strip-convolution branches. Their responses are summed, mixed by a
    pointwise convolution, and multiplied with the untouched input feature.
    """

    def __init__(
        self,
        channels: int,
        local_kernel: int = 5,
        strip_kernels: Sequence[int] = (7, 11, 21),
    ) -> None:
        super().__init__()
        if channels <= 0:
            raise ValueError("channels must be positive")
        if local_kernel <= 0 or local_kernel % 2 == 0:
            raise ValueError("local_kernel must be a positive odd integer")
        kernels = _odd_kernel_tuple(strip_kernels)

        self.local = nn.Conv2d(
            channels,
            channels,
            kernel_size=local_kernel,
            padding=local_kernel // 2,
            groups=channels,
        )
        self.strip_branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        channels,
                        channels,
                        kernel_size=(1, k),
                        padding=(0, k // 2),
                        groups=channels,
                    ),
                    nn.Conv2d(
                        channels,
                        channels,
                        kernel_size=(k, 1),
                        padding=(k // 2, 0),
                        groups=channels,
                    ),
                )
                for k in kernels
            ]
        )
        self.mix = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        local = self.local(x)
        attention = local
        for branch in self.strip_branches:
            attention = attention + branch(local)
        attention = self.mix(attention)
        return x * attention


class ContextContrastMSCA(nn.Module):
    """Water-context contrast adaptation of MSCA for tiny floating-object features.

    Unlike standard MSCA, large spatial context is *not* treated as an extra target
    feature. Fixed directional average filters estimate structured background at
    multiple scales. The block then derives:

    - signed residual: local feature minus the multi-scale background estimate;
    - contrast: polarity-invariant local/background disagreement;
    - anisotropy: disagreement between horizontal and vertical contrast, useful as
      a cue for elongated wave/reflection structures.

    A small pointwise gate reads [local, contrast, anisotropy]. The output is an
    identity-preserving signed residual adapter:

        y = x + gamma * sigmoid(gate(.)) * residual

    ``gamma`` is per-channel and zero-initialized by default, so the module is an
    exact identity at initialization. Scale weights are global learned scalars,
    not image-conditioned kernel selection.
    """

    def __init__(
        self,
        channels: int,
        context_kernels: Sequence[int] = (3, 5, 7),
        reduction: int = 8,
        gamma_init: float = 0.0,
    ) -> None:
        super().__init__()
        if channels <= 0:
            raise ValueError("channels must be positive")
        if reduction <= 0:
            raise ValueError("reduction must be positive")
        kernels = _odd_kernel_tuple(context_kernels, minimum=3)

        self.channels = channels
        self.context_kernels = kernels

        self.local = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1,
            groups=channels,
            bias=False,
        )
        # Start from an interpretable local identity instead of a random high-pass
        # filter. The kernel stays trainable after initialization.
        nn.init.zeros_(self.local.weight)
        with torch.no_grad():
            self.local.weight[:, 0, 1, 1] = 1.0

        self.horizontal_context = nn.ModuleList(
            [
                nn.AvgPool2d(
                    kernel_size=(1, k),
                    stride=1,
                    padding=(0, k // 2),
                    count_include_pad=False,
                )
                for k in kernels
            ]
        )
        self.vertical_context = nn.ModuleList(
            [
                nn.AvgPool2d(
                    kernel_size=(k, 1),
                    stride=1,
                    padding=(k // 2, 0),
                    count_include_pad=False,
                )
                for k in kernels
            ]
        )

        # Equal scale contribution at initialization. These are global scalars,
        # deliberately avoiding per-image dynamic receptive-field selection.
        self.scale_logits = nn.Parameter(torch.zeros(len(kernels)))

        hidden = max(8, channels // reduction)
        self.gate = nn.Sequential(
            nn.Conv2d(channels * 3, hidden, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=True),
        )
        # Neutral 0.5 gate when the residual path first wakes up.
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.zeros_(self.gate[-1].bias)

        self.gamma = nn.Parameter(
            torch.full((1, channels, 1, 1), float(gamma_init))
        )

    def compute_descriptors(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return local, residual, contrast, anisotropy and normalized scale weights."""
        local = self.local(x)
        weights = self.scale_logits.softmax(dim=0)

        background = torch.zeros_like(local)
        contrast = torch.zeros_like(local)
        anisotropy = torch.zeros_like(local)

        for weight, h_pool, v_pool in zip(
            weights, self.horizontal_context, self.vertical_context
        ):
            horizontal = h_pool(local)
            vertical = v_pool(local)
            dh = (local - horizontal).abs()
            dv = (local - vertical).abs()

            background = background + weight * 0.5 * (horizontal + vertical)
            contrast = contrast + weight * 0.5 * (dh + dv)
            anisotropy = anisotropy + weight * (dh - dv).abs()

        residual = local - background
        return local, residual, contrast, anisotropy, weights

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        local, residual, contrast, anisotropy, _ = self.compute_descriptors(x)
        gate_input = torch.cat((local, contrast, anisotropy), dim=1)
        gate = self.gate(gate_input).sigmoid()
        return x + self.gamma * gate * residual


__all__ = ["SegNeXtMSCA", "ContextContrastMSCA"]
