from __future__ import annotations

import math
from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import nn


def _validate_odd_kernel(value: int, *, minimum: int = 3) -> int:
    value = int(value)
    if value < minimum or value % 2 == 0:
        raise ValueError(f"kernel must be an odd integer >= {minimum}, got {value}")
    return value


def _identity_depthwise(channels: int, kernel_size: int = 3) -> nn.Conv2d:
    kernel_size = _validate_odd_kernel(kernel_size)
    conv = nn.Conv2d(
        channels,
        channels,
        kernel_size=kernel_size,
        padding=kernel_size // 2,
        groups=channels,
        bias=False,
    )
    nn.init.zeros_(conv.weight)
    with torch.no_grad():
        conv.weight[:, 0, kernel_size // 2, kernel_size // 2] = 1.0
    return conv


def _fuse_conv_bn(conv: nn.Conv2d, bn: nn.BatchNorm2d) -> tuple[torch.Tensor, torch.Tensor]:
    if conv.bias is None:
        conv_bias = torch.zeros(conv.weight.shape[0], device=conv.weight.device, dtype=conv.weight.dtype)
    else:
        conv_bias = conv.bias
    inv_std = torch.rsqrt(bn.running_var + bn.eps)
    scale = bn.weight * inv_std
    kernel = conv.weight * scale.reshape(-1, 1, 1, 1)
    bias = bn.bias + (conv_bias - bn.running_mean) * scale
    return kernel, bias


def _expand_dilated_kernel(kernel: torch.Tensor, dilation: int) -> torch.Tensor:
    if dilation == 1:
        return kernel
    k = kernel.shape[-1]
    effective = dilation * (k - 1) + 1
    out = kernel.new_zeros(kernel.shape[0], kernel.shape[1], effective, effective)
    out[..., ::dilation, ::dilation] = kernel
    return out


def _center_pad_kernel(kernel: torch.Tensor, target_size: int) -> torch.Tensor:
    current = kernel.shape[-1]
    if current > target_size:
        raise ValueError(f"kernel size {current} exceeds target size {target_size}")
    if current == target_size:
        return kernel
    total = target_size - current
    left = total // 2
    right = total - left
    return F.pad(kernel, (left, right, left, right))


class DilatedReparamDW(nn.Module):
    """Depthwise large kernel with train-time dilated branches and deploy-time fusion.

    This independently re-implements the Dilated Reparam idea used by UniRepLKNet.
    It is intentionally limited to depthwise stride-1 convolution because that is
    the only form required by the FloW Phase-5 control.
    """

    _PRESETS: dict[int, tuple[tuple[int, ...], tuple[int, ...]]] = {
        17: ((5, 9, 3, 3, 3), (1, 2, 4, 5, 7)),
        15: ((5, 7, 3, 3, 3), (1, 2, 3, 5, 7)),
        13: ((5, 7, 3, 3, 3), (1, 2, 3, 4, 5)),
        11: ((5, 5, 3, 3, 3), (1, 2, 3, 4, 5)),
    }

    def __init__(
        self,
        channels: int,
        kernel_size: int = 17,
        branch_kernels: Sequence[int] | None = None,
        branch_dilations: Sequence[int] | None = None,
        deploy: bool = False,
    ) -> None:
        super().__init__()
        if channels <= 0:
            raise ValueError("channels must be positive")
        kernel_size = _validate_odd_kernel(kernel_size, minimum=11)
        self.channels = int(channels)
        self.kernel_size = kernel_size
        self.deploy = bool(deploy)

        if deploy:
            self.reparam = nn.Conv2d(
                channels,
                channels,
                kernel_size,
                padding=kernel_size // 2,
                groups=channels,
                bias=True,
            )
            return

        if branch_kernels is None or branch_dilations is None:
            if kernel_size not in self._PRESETS:
                raise ValueError(
                    f"no frozen branch preset for kernel_size={kernel_size}; "
                    f"provide branch_kernels and branch_dilations explicitly"
                )
            branch_kernels, branch_dilations = self._PRESETS[kernel_size]

        branch_kernels = tuple(int(k) for k in branch_kernels)
        branch_dilations = tuple(int(d) for d in branch_dilations)
        if len(branch_kernels) != len(branch_dilations) or not branch_kernels:
            raise ValueError("branch kernels/dilations must be non-empty and have the same length")

        self.lk = nn.Conv2d(
            channels,
            channels,
            kernel_size,
            padding=kernel_size // 2,
            groups=channels,
            bias=False,
        )
        self.lk_bn = nn.BatchNorm2d(channels)

        self.branch_kernels = branch_kernels
        self.branch_dilations = branch_dilations
        self.branches = nn.ModuleList()
        for k, d in zip(branch_kernels, branch_dilations):
            _validate_odd_kernel(k)
            effective = d * (k - 1) + 1
            if effective > kernel_size:
                raise ValueError(
                    f"branch effective kernel {effective} exceeds large kernel {kernel_size}"
                )
            self.branches.append(
                nn.Sequential(
                    nn.Conv2d(
                        channels,
                        channels,
                        k,
                        padding=d * (k // 2),
                        dilation=d,
                        groups=channels,
                        bias=False,
                    ),
                    nn.BatchNorm2d(channels),
                )
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.deploy:
            return self.reparam(x)
        y = self.lk_bn(self.lk(x))
        for branch in self.branches:
            y = y + branch(x)
        return y

    @torch.no_grad()
    def get_equivalent_kernel_bias(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.deploy:
            return self.reparam.weight.detach().clone(), self.reparam.bias.detach().clone()

        kernel, bias = _fuse_conv_bn(self.lk, self.lk_bn)
        for (k, d), branch in zip(zip(self.branch_kernels, self.branch_dilations), self.branches):
            branch_kernel, branch_bias = _fuse_conv_bn(branch[0], branch[1])
            branch_kernel = _expand_dilated_kernel(branch_kernel, d)
            branch_kernel = _center_pad_kernel(branch_kernel, self.kernel_size)
            kernel = kernel + branch_kernel
            bias = bias + branch_bias
        return kernel, bias

    @torch.no_grad()
    def switch_to_deploy(self) -> "DilatedReparamDW":
        if self.deploy:
            return self
        kernel, bias = self.get_equivalent_kernel_bias()
        reparam = nn.Conv2d(
            self.channels,
            self.channels,
            self.kernel_size,
            padding=self.kernel_size // 2,
            groups=self.channels,
            bias=True,
        ).to(device=kernel.device, dtype=kernel.dtype)
        reparam.weight.copy_(kernel)
        reparam.bias.copy_(bias)
        self.reparam = reparam
        del self.lk
        del self.lk_bn
        del self.branches
        self.deploy = True
        return self


class UniRepLKControl(nn.Module):
    """Phase-5 M1: generic large-receptive-field control for P3 classification only."""

    def __init__(
        self,
        channels: int,
        kernel_size: int = 17,
        gamma_init: float = 0.0,
        deploy: bool = False,
    ) -> None:
        super().__init__()
        if channels <= 0:
            raise ValueError("channels must be positive")
        self.context = DilatedReparamDW(channels, kernel_size=kernel_size, deploy=deploy)
        self.project = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.gamma = nn.Parameter(torch.full((1, channels, 1, 1), float(gamma_init)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        response = self.project(F.silu(self.context(x)))
        return x + self.gamma * response

    @torch.no_grad()
    def switch_to_deploy(self) -> "UniRepLKControl":
        self.context.switch_to_deploy()
        return self


class StripLKC(nn.Module):
    """Phase-5 M2: directional water-context contrast using large strip kernels.

    Horizontal/vertical depthwise strip kernels are initialized as directional
    averages but remain trainable. The injected feature is derived from directional
    disagreement rather than an attention mask.
    """

    def __init__(
        self,
        channels: int,
        strip_kernel: int = 17,
        gamma_init: float = 0.0,
    ) -> None:
        super().__init__()
        if channels <= 0:
            raise ValueError("channels must be positive")
        strip_kernel = _validate_odd_kernel(strip_kernel)
        self.channels = int(channels)
        self.strip_kernel = strip_kernel

        self.local = _identity_depthwise(channels, 3)
        self.horizontal = nn.Conv2d(
            channels,
            channels,
            kernel_size=(1, strip_kernel),
            padding=(0, strip_kernel // 2),
            groups=channels,
            bias=False,
        )
        self.vertical = nn.Conv2d(
            channels,
            channels,
            kernel_size=(strip_kernel, 1),
            padding=(strip_kernel // 2, 0),
            groups=channels,
            bias=False,
        )
        nn.init.constant_(self.horizontal.weight, 1.0 / strip_kernel)
        nn.init.constant_(self.vertical.weight, 1.0 / strip_kernel)

        self.project = nn.Conv2d(channels * 3, channels, kernel_size=1, bias=False)
        self.gamma = nn.Parameter(torch.full((1, channels, 1, 1), float(gamma_init)))

    def compute_descriptors(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        local = self.local(x)
        horizontal = self.horizontal(local)
        vertical = self.vertical(local)
        d_h = local - horizontal
        d_v = local - vertical
        return local, horizontal, vertical, d_h, d_v

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        local, _, _, d_h, d_v = self.compute_descriptors(x)
        response = self.project(torch.cat((local, d_h, d_v), dim=1))
        return x + self.gamma * response


class CenterExcludedPeripheralConv(nn.Module):
    """PeLK-inspired center-excluded peripheral depthwise context operator.

    Spatial locations at similar eccentricity share one depthwise coefficient.
    The central square is structurally masked out, so this branch cannot directly
    consume the tiny-object core. Sharing grows logarithmically toward the edge.
    The learned shared kernel can be materialized into one static depthwise Conv2d
    for export/deployment.
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int = 17,
        center_kernel: int = 5,
        deploy: bool = False,
    ) -> None:
        super().__init__()
        if channels <= 0:
            raise ValueError("channels must be positive")
        kernel_size = _validate_odd_kernel(kernel_size)
        center_kernel = _validate_odd_kernel(center_kernel)
        if center_kernel >= kernel_size:
            raise ValueError("center_kernel must be smaller than kernel_size")
        self.channels = int(channels)
        self.kernel_size = kernel_size
        self.center_kernel = center_kernel
        self.deploy = bool(deploy)

        basis = self._build_basis(kernel_size, center_kernel)
        self.num_bins = int(basis.shape[0])
        self.register_buffer("basis", basis, persistent=True)

        if deploy:
            self.reparam = nn.Conv2d(
                channels,
                channels,
                kernel_size,
                padding=kernel_size // 2,
                groups=channels,
                bias=False,
            )
        else:
            self.shared_weight = nn.Parameter(torch.full((channels, self.num_bins), 1.0 / self.num_bins))

    @staticmethod
    def _build_basis(kernel_size: int, center_kernel: int) -> torch.Tensor:
        radius = kernel_size // 2
        center_radius = center_kernel // 2
        bins: dict[int, list[tuple[int, int]]] = {}
        for y in range(kernel_size):
            for x in range(kernel_size):
                eccentricity = max(abs(y - radius), abs(x - radius))
                if eccentricity <= center_radius:
                    continue
                distance_from_center_band = eccentricity - center_radius
                bin_id = int(math.floor(math.log2(distance_from_center_band)))
                bins.setdefault(bin_id, []).append((y, x))

        basis = torch.zeros(len(bins), kernel_size, kernel_size)
        for out_idx, bin_id in enumerate(sorted(bins)):
            points = bins[bin_id]
            value = 1.0 / len(points)
            for y, x in points:
                basis[out_idx, y, x] = value
        return basis

    def effective_kernel(self) -> torch.Tensor:
        if self.deploy:
            return self.reparam.weight[:, 0]
        return torch.einsum("cb,bhw->chw", self.shared_weight, self.basis)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.deploy:
            return self.reparam(x)
        kernel = self.effective_kernel().unsqueeze(1)
        return F.conv2d(
            x,
            kernel,
            padding=self.kernel_size // 2,
            groups=self.channels,
        )

    @torch.no_grad()
    def switch_to_deploy(self) -> "CenterExcludedPeripheralConv":
        if self.deploy:
            return self
        kernel = self.effective_kernel().unsqueeze(1)
        reparam = nn.Conv2d(
            self.channels,
            self.channels,
            self.kernel_size,
            padding=self.kernel_size // 2,
            groups=self.channels,
            bias=False,
        ).to(device=kernel.device, dtype=kernel.dtype)
        reparam.weight.copy_(kernel)
        self.reparam = reparam
        del self.shared_weight
        self.deploy = True
        return self


class CEPConvLKC(nn.Module):
    """Phase-5 M3: center-excluded large peripheral context for tiny objects."""

    def __init__(
        self,
        channels: int,
        kernel_size: int = 17,
        center_kernel: int = 5,
        gamma_init: float = 0.0,
        deploy: bool = False,
    ) -> None:
        super().__init__()
        self.local = _identity_depthwise(channels, 3)
        self.peripheral = CenterExcludedPeripheralConv(
            channels,
            kernel_size=kernel_size,
            center_kernel=center_kernel,
            deploy=deploy,
        )
        self.project = nn.Conv2d(channels * 3, channels, kernel_size=1, bias=False)
        self.gamma = nn.Parameter(torch.full((1, channels, 1, 1), float(gamma_init)))

    def compute_descriptors(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        local = self.local(x)
        peripheral = self.peripheral(local)
        contrast = local - peripheral
        return local, peripheral, contrast

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        local, peripheral, contrast = self.compute_descriptors(x)
        response = self.project(torch.cat((local, peripheral, contrast), dim=1))
        return x + self.gamma * response

    @torch.no_grad()
    def switch_to_deploy(self) -> "CEPConvLKC":
        self.peripheral.switch_to_deploy()
        return self


__all__ = [
    "DilatedReparamDW",
    "UniRepLKControl",
    "StripLKC",
    "CenterExcludedPeripheralConv",
    "CEPConvLKC",
]
