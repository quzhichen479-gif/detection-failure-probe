from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn


class ConvBNAct(nn.Sequential):
    """Small local helper used only by the research modules in this directory."""

    def __init__(
        self,
        c1: int,
        c2: int,
        k: int = 1,
        s: int = 1,
        p: Optional[int] = None,
        g: int = 1,
        d: int = 1,
        act: bool = True,
    ) -> None:
        if p is None:
            p = ((k - 1) * d) // 2
        layers: list[nn.Module] = [
            nn.Conv2d(c1, c2, k, s, p, dilation=d, groups=g, bias=False),
            nn.BatchNorm2d(c2),
        ]
        if act:
            layers.append(nn.SiLU(inplace=True))
        super().__init__(*layers)


def dfl_entropy_map(
    reg_logits: torch.Tensor,
    reg_max: int = 16,
    eps: float = 1e-9,
) -> torch.Tensor:
    """Return normalized DFL entropy, shape ``[B, 1, H, W]``.

    ``reg_logits`` is the raw YOLO box branch output before DFL projection and must
    have ``4 * reg_max`` channels. Entropy is averaged over left/top/right/bottom.
    The output is normalized to [0, 1] by log(reg_max).
    """

    if reg_logits.ndim != 4:
        raise ValueError("reg_logits must be BCHW")
    b, c, h, w = reg_logits.shape
    expected = 4 * reg_max
    if c != expected:
        raise ValueError(f"expected {expected} box channels, got {c}")

    dtype = reg_logits.dtype
    probs = reg_logits.float().view(b, 4, reg_max, h, w).softmax(dim=2)
    entropy = -(probs * probs.clamp_min(eps).log()).sum(dim=2)
    entropy = entropy / math.log(float(reg_max))
    return entropy.mean(dim=1, keepdim=True).to(dtype=dtype)


@dataclass(frozen=True)
class URP2Stats:
    mean_uncertainty: float
    mean_gate: float
    active_fraction_050: float


class URP2Refiner(nn.Module):
    """Uncertainty-Routed P2 residual refiner for the YOLO11 P3 prediction level.

    Research contract
    -----------------
    * P2 supplies shallow high-resolution detail; P3 supplies semantic context.
    * The router is computed from the *existing* P3 DFL distributions.
    * V1 uses a dense implementation with a detached soft gate. This tests the
      scientific mechanism without claiming sparse runtime acceleration.
    * The module emits residual corrections for the stock P3 raw regression and
      classification logits. P4/P5 remain untouched.

    The intended Ultralytics integration is inside a small ``Detect`` subclass:
    compute stock P3 raw box/class logits once, pass them here together with P2/P3,
    then continue the normal concatenate/decode/loss path with the refined logits.
    """

    def __init__(
        self,
        p2_channels: int,
        p3_channels: int,
        nc: int = 1,
        reg_max: int = 16,
        hidden: int = 64,
        uncertainty_threshold: float = 0.55,
        temperature: float = 0.15,
        detach_router: bool = True,
        reg_residual_scale: float = 0.25,
        cls_residual_scale: float = 1.0,
    ) -> None:
        super().__init__()
        if min(p2_channels, p3_channels, nc, reg_max, hidden) <= 0:
            raise ValueError("channel counts, nc, reg_max and hidden must be positive")
        if temperature <= 0:
            raise ValueError("temperature must be positive")

        self.nc = nc
        self.reg_max = reg_max
        self.uncertainty_threshold = float(uncertainty_threshold)
        self.temperature = float(temperature)
        self.detach_router = bool(detach_router)
        self.reg_residual_scale = float(reg_residual_scale)
        self.cls_residual_scale = float(cls_residual_scale)

        # P2/4 -> P3/8. Stride-2 is deliberate: UR-P2 uses P2 evidence but does not
        # add a fourth dense detection head.
        self.p2_down = ConvBNAct(p2_channels, hidden, k=3, s=2)
        self.p3_proj = ConvBNAct(p3_channels, hidden, k=1, s=1)

        # Explicit detail/context discrepancy rather than generic attention.
        self.mix = nn.Sequential(
            ConvBNAct(hidden * 3 + 1, hidden, k=3),
            ConvBNAct(hidden, hidden, k=3, g=hidden),
            ConvBNAct(hidden, hidden, k=1),
        )
        self.delta_reg = nn.Conv2d(hidden, 4 * reg_max, 1)
        self.delta_cls = nn.Conv2d(hidden, nc, 1)

        # Start from the exact stock detector behaviour.
        nn.init.zeros_(self.delta_reg.weight)
        nn.init.zeros_(self.delta_reg.bias)
        nn.init.zeros_(self.delta_cls.weight)
        nn.init.zeros_(self.delta_cls.bias)

    def route(self, reg_logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        uncertainty = dfl_entropy_map(reg_logits, self.reg_max)
        source = uncertainty.detach() if self.detach_router else uncertainty
        gate = torch.sigmoid(
            (source - self.uncertainty_threshold) / self.temperature
        )
        return uncertainty, gate

    def forward(
        self,
        p2: torch.Tensor,
        p3: torch.Tensor,
        p3_reg_logits: torch.Tensor,
        p3_cls_logits: torch.Tensor,
        return_stats: bool = False,
    ):
        if p3_cls_logits.ndim != 4 or p3_cls_logits.shape[1] != self.nc:
            raise ValueError(f"p3_cls_logits must have {self.nc} channels")
        if p3.shape[-2:] != p3_reg_logits.shape[-2:]:
            raise ValueError("P3 feature and P3 regression logits must share HxW")

        detail = self.p2_down(p2)
        if detail.shape[-2:] != p3.shape[-2:]:
            detail = F.interpolate(detail, size=p3.shape[-2:], mode="bilinear", align_corners=False)
        semantic = self.p3_proj(p3)
        uncertainty, gate = self.route(p3_reg_logits)

        discrepancy = (detail - semantic).abs()
        mixed = self.mix(torch.cat((detail, semantic, discrepancy, gate), dim=1))
        mixed = mixed * gate

        # Bounded regression residual reduces the chance of destabilizing DFL early.
        reg_delta = torch.tanh(self.delta_reg(mixed)) * self.reg_residual_scale
        cls_delta = self.delta_cls(mixed) * self.cls_residual_scale
        refined_reg = p3_reg_logits + reg_delta
        refined_cls = p3_cls_logits + cls_delta

        if not return_stats:
            return refined_reg, refined_cls

        stats = URP2Stats(
            mean_uncertainty=float(uncertainty.detach().mean().cpu()),
            mean_gate=float(gate.detach().mean().cpu()),
            active_fraction_050=float((gate.detach() >= 0.5).float().mean().cpu()),
        )
        return refined_reg, refined_cls, stats


class SDASHead(nn.Module):
    """Shallow Detail Auxiliary Supervision head.

    Attach to the stock backbone P2/4 feature during training. It predicts a
    one-channel GT-center heatmap. The head and its loss are removed for inference.

    V1 intentionally supervises all GT centers; it does not depend on the earlier
    unreliable FloW size statistics. A later size gate must be justified by a new
    seed79-aligned audit before being enabled.
    """

    def __init__(self, channels: int, hidden: Optional[int] = None) -> None:
        super().__init__()
        if channels <= 0:
            raise ValueError("channels must be positive")
        hidden = hidden or max(16, channels // 2)
        self.head = nn.Sequential(
            ConvBNAct(channels, hidden, k=3),
            nn.Conv2d(hidden, 1, 1),
        )
        nn.init.constant_(self.head[-1].bias, -2.19)  # ~0.10 initial probability

    def forward(self, p2: torch.Tensor) -> torch.Tensor:
        return self.head(p2)


def _gaussian2d(radius: int, sigma: float, device, dtype) -> torch.Tensor:
    diameter = 2 * radius + 1
    x = torch.arange(diameter, device=device, dtype=dtype) - radius
    y = x[:, None]
    return torch.exp(-(x * x + y * y) / (2 * sigma * sigma))


def _draw_gaussian(
    heatmap: torch.Tensor,
    center_x: int,
    center_y: int,
    radius: int,
) -> None:
    h, w = heatmap.shape
    radius = max(0, int(radius))
    if not (0 <= center_x < w and 0 <= center_y < h):
        return
    if radius == 0:
        heatmap[center_y, center_x] = 1.0
        return

    gaussian = _gaussian2d(
        radius,
        sigma=max(1.0, (2 * radius + 1) / 6.0),
        device=heatmap.device,
        dtype=heatmap.dtype,
    )
    left = min(center_x, radius)
    right = min(w - center_x - 1, radius)
    top = min(center_y, radius)
    bottom = min(h - center_y - 1, radius)

    masked_hm = heatmap[
        center_y - top : center_y + bottom + 1,
        center_x - left : center_x + right + 1,
    ]
    masked_g = gaussian[
        radius - top : radius + bottom + 1,
        radius - left : radius + right + 1,
    ]
    torch.maximum(masked_hm, masked_g, out=masked_hm)


def build_sdas_target(
    batch_size: int,
    feature_hw: tuple[int, int],
    batch_idx: torch.Tensor,
    bboxes_xywhn: torch.Tensor,
    min_radius: int = 0,
    max_radius: int = 4,
) -> torch.Tensor:
    """Build SDAS center heatmaps from normalized xywh boxes.

    Args:
        batch_idx: ``[N]`` image indices.
        bboxes_xywhn: ``[N, 4]`` normalized ``cx, cy, w, h`` in [0, 1].
    """

    h, w = feature_hw
    if batch_size <= 0 or h <= 0 or w <= 0:
        raise ValueError("batch_size and feature dimensions must be positive")
    if bboxes_xywhn.ndim != 2 or bboxes_xywhn.shape[-1] != 4:
        raise ValueError("bboxes_xywhn must be Nx4")
    if batch_idx.numel() != bboxes_xywhn.shape[0]:
        raise ValueError("batch_idx and bboxes must have the same length")

    device = bboxes_xywhn.device
    dtype = bboxes_xywhn.dtype
    target = torch.zeros((batch_size, 1, h, w), device=device, dtype=dtype)

    for idx, box in zip(batch_idx.long().tolist(), bboxes_xywhn):
        if idx < 0 or idx >= batch_size:
            continue
        cx, cy, bw, bh = box
        x = int(torch.clamp(cx * w, 0, w - 1).item())
        y = int(torch.clamp(cy * h, 0, h - 1).item())
        box_w = max(float(bw * w), 1.0)
        box_h = max(float(bh * h), 1.0)
        radius = int(round(0.25 * min(box_w, box_h)))
        radius = max(min_radius, min(max_radius, radius))
        _draw_gaussian(target[idx, 0], x, y, radius)
    return target


def sdas_focal_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    alpha: float = 2.0,
    beta: float = 4.0,
) -> torch.Tensor:
    """CenterNet-style focal loss for SDAS heatmap supervision."""

    if logits.shape != target.shape:
        raise ValueError("logits and target must have the same shape")
    pred = logits.sigmoid().clamp_(1e-6, 1 - 1e-6)
    pos = target.eq(1).to(logits.dtype)
    neg = target.lt(1).to(logits.dtype)
    neg_weight = (1 - target).pow(beta)

    pos_loss = -(1 - pred).pow(alpha) * pred.log() * pos
    neg_loss = -pred.pow(alpha) * (1 - pred).log() * neg_weight * neg
    num_pos = pos.sum()
    if num_pos.item() < 1:
        return neg_loss.sum()
    return (pos_loss.sum() + neg_loss.sum()) / num_pos


class WCRF(nn.Module):
    """Water-Context Residual Fusion, a channel-preserving P3 side-branch block.

    WCRF does not use parallel 1/3/5-kernel attention. It explicitly estimates
    local evidence L, broader context C, their residual D=C-L, and learns how much
    of that contextual residual should be injected. ``gamma`` starts at zero so a
    newly inserted block begins as an exact identity residual branch.
    """

    def __init__(
        self,
        channels: int,
        hidden: Optional[int] = None,
        context_kernel: int = 7,
        context_dilation: int = 2,
    ) -> None:
        super().__init__()
        if channels <= 0:
            raise ValueError("channels must be positive")
        if context_kernel <= 0 or context_kernel % 2 == 0:
            raise ValueError("context_kernel must be a positive odd integer")
        if context_dilation <= 0:
            raise ValueError("context_dilation must be positive")
        hidden = hidden or max(16, channels // 2)

        self.local = nn.Sequential(
            ConvBNAct(channels, channels, k=3, g=channels),
            ConvBNAct(channels, hidden, k=1),
        )
        self.context = nn.Sequential(
            nn.AvgPool2d(context_kernel, stride=1, padding=context_kernel // 2),
            ConvBNAct(
                channels,
                channels,
                k=3,
                g=channels,
                d=context_dilation,
            ),
            ConvBNAct(channels, hidden, k=1),
        )
        self.gate = nn.Sequential(
            nn.Conv2d(hidden * 3, hidden, 1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, hidden, 1),
            nn.Sigmoid(),
        )
        self.out = ConvBNAct(hidden, channels, k=1, act=False)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        local = self.local(x)
        context = self.context(x)
        delta = context - local
        gate = self.gate(torch.cat((local, context, delta.abs()), dim=1))
        fused = local + gate * delta
        return x + self.gamma * self.out(fused)


__all__ = [
    "URP2Refiner",
    "URP2Stats",
    "dfl_entropy_map",
    "SDASHead",
    "build_sdas_target",
    "sdas_focal_loss",
    "WCRF",
]
