"""YOLO11-DDFL v1 reference implementation.

This track adapts the Densified Distribution Focal Loss (D-DFL) idea to
Ultralytics YOLO11 / 8.4.113 without importing the YOLOv9/DSDL training stack.

Design goals:
- keep native TAL, CIoU, BCE, feature pyramid, decoder geometry and NMS;
- keep the original YOLO11 regression-tower hidden width (base reg_max=16);
- change only the final box-logit support from 16 uniform bins to 20 bins;
- preserve the original maximum representable distance (15 feature units);
- densify the support only near zero, where tiny-object l/t/r/b targets cluster;
- reinitialize only the final box-output conv when moving 64 -> 80 channels;
- provide a matched-capacity uniform-20 control.

This is NOT a faithful reproduction of full DSDL. In particular, it does not use
D-TAL or Signed DFL. Native YOLO11 TAL selects anchors inside GT boxes, so native
positive l/t/r/b targets are non-negative; negative support is intentionally omitted.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
import torch.nn.functional as F
from torch import Tensor, nn


STANDARD16 = tuple(float(x) for x in range(16))
UNIFORM20 = tuple(float(x) for x in torch.linspace(0.0, 15.0, 20).tolist())
DDFL20 = (
    0.0,
    0.25,
    0.50,
    0.75,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    5.0,
    6.0,
    7.0,
    8.0,
    9.0,
    10.0,
    11.0,
    12.0,
    13.0,
    14.0,
    15.0,
)


@dataclass(frozen=True)
class YOLO11DDFLConfig:
    """Frozen first-round configurations."""

    mode: str = "ddfl20"
    base_reg_max: int = 16
    target_max_margin: float = 0.01
    eps: float = 1e-7

    @property
    def reg_values(self) -> tuple[float, ...]:
        return support_for_mode(self.mode)

    @property
    def num_bins(self) -> int:
        return len(self.reg_values)

    @property
    def output_channels(self) -> int:
        return 4 * self.num_bins

    @property
    def base_output_channels(self) -> int:
        return 4 * self.base_reg_max


def _validate_support(values: Iterable[float]) -> tuple[float, ...]:
    values = tuple(float(v) for v in values)
    if len(values) < 2:
        raise ValueError("DFL support needs at least two values")
    if values[0] < 0:
        raise ValueError("YOLO11-DDFL v1 intentionally uses non-negative support")
    if any(b <= a for a, b in zip(values[:-1], values[1:])):
        raise ValueError(f"support must be strictly increasing, got {values}")
    return values


def support_for_mode(mode: str) -> tuple[float, ...]:
    """Return frozen regression support for an ablation mode."""
    table = {
        "standard16": STANDARD16,
        "uniform20": UNIFORM20,
        "ddfl20": DDFL20,
    }
    if mode not in table:
        raise ValueError(f"unknown DFL mode {mode!r}; choose from {sorted(table)}")
    return _validate_support(table[mode])


class NonUniformDFLIntegral(nn.Module):
    """Decode 4 categorical distance distributions by expectation over arbitrary support.

    Input shape follows Ultralytics Detect inference: [B, 4*K, A].
    Output shape is [B, 4, A] with continuous l/t/r/b distances.
    """

    def __init__(self, reg_values: Iterable[float]) -> None:
        super().__init__()
        values = _validate_support(reg_values)
        self.num_bins = len(values)
        self.register_buffer("reg_values", torch.tensor(values, dtype=torch.float32), persistent=True)

    def forward(self, x: Tensor) -> Tensor:
        b, c, a = x.shape
        expected_c = 4 * self.num_bins
        if c != expected_c:
            raise ValueError(f"expected {expected_c} regression channels, got {c}")
        probs = x.view(b, 4, self.num_bins, a).softmax(dim=2)
        support = self.reg_values.to(device=x.device, dtype=probs.dtype).view(1, 1, self.num_bins, 1)
        return (probs * support).sum(dim=2)


class NonUniformDFLoss(nn.Module):
    """Two-neighbor interpolation DFL for arbitrary strictly increasing support.

    This generalizes Ultralytics' standard integer-bin DFL. For a target y between
    support values r_l <= y <= r_r, the loss is the linear interpolation of the
    two categorical negative log-likelihoods.
    """

    def __init__(
        self,
        reg_values: Iterable[float],
        target_max_margin: float = 0.01,
        eps: float = 1e-7,
    ) -> None:
        super().__init__()
        values = _validate_support(reg_values)
        if target_max_margin < 0:
            raise ValueError("target_max_margin must be >= 0")
        self.num_bins = len(values)
        self.target_max_margin = float(target_max_margin)
        self.eps = float(eps)
        self.register_buffer("reg_values", torch.tensor(values, dtype=torch.float32), persistent=True)

    def clamp_target(self, target: Tensor) -> Tensor:
        values = self.reg_values.to(device=target.device, dtype=target.dtype)
        lo = values[0]
        hi = values[-1] - self.target_max_margin
        if hi <= lo:
            hi = values[-1]
        return target.clamp(min=lo, max=hi)

    def forward(self, pred_dist: Tensor, target_ltrb: Tensor) -> Tensor:
        """Return per-positive DFL averaged over l/t/r/b, shape [N,1].

        pred_dist may be [N,4,K] or [N*4,K]. target_ltrb is [N,4].
        """
        if target_ltrb.ndim != 2 or target_ltrb.shape[-1] != 4:
            raise ValueError(f"target_ltrb must be [N,4], got {tuple(target_ltrb.shape)}")
        n = target_ltrb.shape[0]
        if pred_dist.ndim == 2:
            if pred_dist.shape != (n * 4, self.num_bins):
                raise ValueError(
                    f"2D pred_dist must be [N*4,{self.num_bins}], got {tuple(pred_dist.shape)} for N={n}"
                )
            pred = pred_dist.view(n, 4, self.num_bins)
        elif pred_dist.ndim == 3:
            if pred_dist.shape != (n, 4, self.num_bins):
                raise ValueError(
                    f"3D pred_dist must be [N,4,{self.num_bins}], got {tuple(pred_dist.shape)}"
                )
            pred = pred_dist
        else:
            raise ValueError(f"pred_dist must be 2D or 3D, got ndim={pred_dist.ndim}")

        # Use FP32 for search/interpolation/log-softmax stability under AMP.
        target = self.clamp_target(target_ltrb.float())
        support = self.reg_values.to(device=target.device, dtype=target.dtype)

        # right_idx is the first support value >= target. Exact support points therefore
        # receive full mass on that exact bin. The minimum boundary is handled by same-bin logic.
        right_idx = torch.searchsorted(support, target.contiguous(), right=False).clamp(0, self.num_bins - 1)
        left_idx = (right_idx - 1).clamp(0, self.num_bins - 1)

        left_value = support[left_idx]
        right_value = support[right_idx]
        same = right_idx.eq(left_idx)
        denom = (right_value - left_value).clamp_min(self.eps)
        w_right = torch.where(same, torch.zeros_like(target), (target - left_value) / denom).clamp(0.0, 1.0)
        w_left = 1.0 - w_right

        logp = F.log_softmax(pred.float(), dim=-1)
        lp_left = logp.gather(-1, left_idx.unsqueeze(-1)).squeeze(-1)
        lp_right = logp.gather(-1, right_idx.unsqueeze(-1)).squeeze(-1)
        loss_side = -(w_left * lp_left + w_right * lp_right)
        return loss_side.mean(dim=-1, keepdim=True)


def native_standard_dfl_reference(pred_dist: Tensor, target_ltrb: Tensor, reg_max: int = 16) -> Tensor:
    """Reference copy of the Ultralytics 8.4.113 integer-bin DFL math for parity tests."""
    target = target_ltrb.clamp(0, reg_max - 1 - 0.01)
    tl = target.long()
    tr = tl + 1
    wl = tr - target
    wr = 1.0 - wl
    pred = pred_dist.view(-1, reg_max)
    logp = F.log_softmax(pred.float(), dim=1)
    loss = -(
        logp.gather(1, tl.reshape(-1, 1)).reshape(tl.shape) * wl
        + logp.gather(1, tr.reshape(-1, 1)).reshape(tl.shape) * wr
    )
    return loss.mean(-1, keepdim=True)


def head_contract(mode: str) -> dict[str, object]:
    """Describe the minimal YOLO11 Detect-head shape change for the selected mode."""
    values = support_for_mode(mode)
    return {
        "mode": mode,
        "base_reg_max": 16,
        "num_bins": len(values),
        "base_box_output_channels": 64,
        "new_box_output_channels": 4 * len(values),
        "keep_regression_tower_hidden_width": True,
        "support_min": values[0],
        "support_max": values[-1],
    }
