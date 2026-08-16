"""Reference-only Round-3 modules for the YOLO11 water-surface attention track.

IMPORTANT
---------
This file is a research reference, NOT part of the detection-failure-probe package.
Codex should port the selected classes into the actual YOLO11/Ultralytics engineering
repository and adapt imports/configuration to the existing Round-2 DBRA implementation.

The code intentionally does not implement DBRA again. A fixed, already-validated DBRA
module must be passed into the composite wrappers so Round-3 cannot silently change the
parent DBRA mechanism/hyperparameters.

Sources/mechanisms:
- GRN: ConvNeXt V2, CVPR 2023, https://github.com/facebookresearch/ConvNeXt-V2
- Slide Attention: Slide-Transformer, CVPR 2023,
  https://github.com/LeapLabTHU/Slide-Transformer
- Focal Modulation: FocalNet, NeurIPS 2022,
  https://github.com/microsoft/FocalNet

Project adaptations:
- NCHW tensors throughout.
- Dynamic H/W for Slide instead of a fixed input_resolution.
- Near-DBRA residual scales for new companion branches.
- No modification of YOLO box/DFL/regression path.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class GRN2d(nn.Module):
    """NCHW equivalent of ConvNeXt-V2 Global Response Normalization.

    Official GRN is channel-last. For x=[B,C,H,W], spatial response is computed over
    H/W and normalized across C. gamma/beta are zero-initialized, so this module starts
    as exact identity.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = float(eps)
        self.gamma = nn.Parameter(torch.zeros(1, dim, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gx = torch.norm(x, p=2, dim=(2, 3), keepdim=True)
        nx = gx / (gx.mean(dim=1, keepdim=True) + self.eps)
        return x + self.gamma * (x * nx) + self.beta


class SlideAttention2d(nn.Module):
    """Dynamic-resolution NCHW port of upstream SlideAttention semantics.

    This intentionally mirrors the official Slide-Swin implementation's layout:
    qkv is projected to 3*C channels, then reshaped directly to
    [B*num_heads, 3*head_dim, H, W] before q/k/v slicing. That layout may look unusual,
    but changing it to a conventional q/k/v-then-head split creates a different module.

    Other upstream details retained:
    - local k x k K/V neighborhood;
    - frozen designed depthwise-shift WEIGHTS;
    - fixed-shift bias remains trainable, matching upstream Conv2d behavior;
    - parallel learnable depthwise shift/deformation path;
    - truncated-normal relative local bias;
    - softmax over k^2 local neighbors.

    Project-only changes are dynamic H/W, NCHW I/O, and Conv2d(1x1) replacing the
    equivalent per-token Linear projections.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        kernel_size: int = 3,
        qkv_bias: bool = True,
        proj_bias: bool = True,
    ):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}")
        if kernel_size < 1 or kernel_size % 2 != 1:
            raise ValueError("kernel_size must be a positive odd integer")

        self.dim = int(dim)
        self.num_heads = int(num_heads)
        self.head_dim = self.dim // self.num_heads
        self.kernel_size = int(kernel_size)
        self.num_neighbors = self.kernel_size * self.kernel_size
        self.scale = self.head_dim**-0.5

        self.qkv = nn.Conv2d(self.dim, 3 * self.dim, 1, bias=qkv_bias)
        self.proj = nn.Conv2d(self.dim, self.dim, 1, bias=proj_bias)

        self.fixed_shift = nn.Conv2d(
            self.head_dim,
            self.num_neighbors * self.head_dim,
            kernel_size=self.kernel_size,
            padding=self.kernel_size // 2,
            groups=self.head_dim,
            bias=True,
        )
        self.learned_shift = nn.Conv2d(
            self.head_dim,
            self.num_neighbors * self.head_dim,
            kernel_size=self.kernel_size,
            padding=self.kernel_size // 2,
            groups=self.head_dim,
            bias=True,
        )

        self.relative_bias = nn.Parameter(
            torch.zeros(1, self.num_heads, 1, self.num_neighbors, 1, 1)
        )
        self._init_fixed_shift()
        nn.init.trunc_normal_(self.relative_bias, std=0.02)

    def _init_fixed_shift(self) -> None:
        k = self.kernel_size
        base = torch.zeros(self.num_neighbors, 1, k, k)
        for idx in range(self.num_neighbors):
            base[idx, 0, idx // k, idx % k] = 1.0

        weight = base.repeat(self.head_dim, 1, 1, 1)
        with torch.no_grad():
            self.fixed_shift.weight.copy_(weight)
        # Match upstream: only the designed kernel weights are frozen.
        self.fixed_shift.weight.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = x.shape
        if channels != self.dim:
            raise ValueError(f"expected C={self.dim}, got C={channels}")

        # Source-faithful layout: qkv -> [B*heads, 3*head_dim, H, W] directly.
        packed = self.qkv(x).reshape(
            batch * self.num_heads,
            3 * self.head_dim,
            height,
            width,
        )
        q = packed[:, : self.head_dim]
        k = packed[:, self.head_dim : 2 * self.head_dim]
        v = packed[:, 2 * self.head_dim :]

        q = (q * self.scale).reshape(
            batch,
            self.num_heads,
            self.head_dim,
            1,
            height,
            width,
        )

        k = (self.fixed_shift(k) + self.learned_shift(k)).reshape(
            batch,
            self.num_heads,
            self.head_dim,
            self.num_neighbors,
            height,
            width,
        )
        v = (self.fixed_shift(v) + self.learned_shift(v)).reshape(
            batch,
            self.num_heads,
            self.head_dim,
            self.num_neighbors,
            height,
            width,
        )

        k = k + self.relative_bias
        logits = (q * k).sum(dim=2, keepdim=True)
        attn = torch.softmax(logits, dim=3)

        y = (attn * v).sum(dim=3).reshape(batch, channels, height, width)
        return self.proj(y)


class FocalModulation2dLite(nn.Module):
    """Constrained NCHW adaptation of FocalNet focal modulation.

    Round-3 default deliberately uses focal_level=1 and focal_window=3. This avoids
    turning the experiment into another large-context sweep after earlier negative
    LSK/CAA evidence.
    """

    def __init__(
        self,
        dim: int,
        focal_window: int = 3,
        focal_level: int = 1,
        focal_factor: int = 2,
        normalize_modulator: bool = True,
    ):
        super().__init__()
        if focal_level < 1:
            raise ValueError("focal_level must be >= 1")

        self.dim = int(dim)
        self.focal_level = int(focal_level)
        self.normalize_modulator = bool(normalize_modulator)

        self.pre = nn.Conv2d(
            self.dim,
            2 * self.dim + self.focal_level + 1,
            kernel_size=1,
            bias=True,
        )

        focal_layers = []
        for level in range(self.focal_level):
            kernel = focal_window + focal_factor * level
            if kernel % 2 != 1:
                raise ValueError("all focal kernels must be odd")
            focal_layers.append(
                nn.Sequential(
                    nn.Conv2d(
                        self.dim,
                        self.dim,
                        kernel_size=kernel,
                        padding=kernel // 2,
                        groups=self.dim,
                        bias=False,
                    ),
                    nn.GELU(),
                )
            )
        self.focal_layers = nn.ModuleList(focal_layers)

        self.modulator_proj = nn.Conv2d(self.dim, self.dim, 1, bias=True)
        self.out_proj = nn.Conv2d(self.dim, self.dim, 1, bias=True)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        packed = self.pre(x)
        q, ctx, gates = torch.split(
            packed,
            [self.dim, self.dim, self.focal_level + 1],
            dim=1,
        )

        aggregated = torch.zeros_like(ctx)
        current = ctx
        for level, layer in enumerate(self.focal_layers):
            current = layer(current)
            aggregated = aggregated + current * gates[:, level : level + 1]

        global_ctx = self.act(current.mean(dim=(2, 3), keepdim=True))
        aggregated = aggregated + global_ctx * gates[:, self.focal_level :]

        if self.normalize_modulator:
            aggregated = aggregated / float(self.focal_level + 1)

        modulator = self.modulator_proj(aggregated)
        return self.out_proj(q * modulator)


class DBRAGRNPost(nn.Module):
    """Primary R3-G1: fixed DBRA followed by zero-init GRN."""

    def __init__(self, dim: int, dbra: nn.Module, eps: float = 1e-6):
        super().__init__()
        self.dbra = dbra
        self.grn = GRN2d(dim=dim, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.grn(self.dbra(x))


class SlideDBRAParallel(nn.Module):
    """Primary R3-S1: preserve local evidence in parallel with fixed DBRA."""

    def __init__(
        self,
        dbra: nn.Module,
        slide: nn.Module,
        local_scale_init: float = 1e-3,
    ):
        super().__init__()
        self.dbra = dbra
        self.slide = slide
        self.local_scale = nn.Parameter(torch.tensor(float(local_scale_init)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        routed = self.dbra(x)
        local = self.slide(x)
        return routed + self.local_scale * local


class SlideThenDBRA(nn.Module):
    """Secondary R3-S2: weak local preconditioner before fixed DBRA."""

    def __init__(
        self,
        dbra: nn.Module,
        slide: nn.Module,
        local_scale_init: float = 1e-3,
    ):
        super().__init__()
        self.dbra = dbra
        self.slide = slide
        self.local_scale = nn.Parameter(torch.tensor(float(local_scale_init)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        local_refined = x + self.local_scale * self.slide(x)
        return self.dbra(local_refined)


class FocalDBRAParallel(nn.Module):
    """Primary R3-F1: weak FocalMod correction parallel to fixed DBRA."""

    def __init__(
        self,
        dbra: nn.Module,
        focal: nn.Module,
        focal_scale_init: float = 1e-3,
    ):
        super().__init__()
        self.dbra = dbra
        self.focal = focal
        self.focal_scale = nn.Parameter(torch.tensor(float(focal_scale_init)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        routed = self.dbra(x)
        modulation = self.focal(x)
        return routed + self.focal_scale * modulation


class DBRAThenFocal(nn.Module):
    """Secondary R3-F2: weak FocalMod residual after fixed DBRA."""

    def __init__(
        self,
        dbra: nn.Module,
        focal: nn.Module,
        focal_scale_init: float = 1e-3,
    ):
        super().__init__()
        self.dbra = dbra
        self.focal = focal
        self.focal_scale = nn.Parameter(torch.tensor(float(focal_scale_init)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        routed = self.dbra(x)
        return routed + self.focal_scale * self.focal(routed)


def choose_head_count(dim: int, max_heads: int = 4) -> int:
    """Pick the largest simple divisor <= max_heads for Slide reference configs."""
    for heads in range(min(max_heads, dim), 0, -1):
        if dim % heads == 0:
            return heads
    return 1


__all__ = [
    "GRN2d",
    "SlideAttention2d",
    "FocalModulation2dLite",
    "DBRAGRNPost",
    "SlideDBRAParallel",
    "SlideThenDBRA",
    "FocalDBRAParallel",
    "DBRAThenFocal",
    "choose_head_count",
]
