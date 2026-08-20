"""SG-WICG v1 reference implementation.

This module is intentionally framework-light. It is designed to be copied into an
Ultralytics 8.4.113 worktree and called only for foreground box-regression samples.
It does NOT modify TAL, DFL, classification loss, Detect heads, decoding, or NMS.

Frozen v1 formulation:
    Inner-CIoU(r=1.25)
      -> + GCD
      -> + GT-short-side scale gate (tau=12 px, T=2 px)
      -> + WIoUv3-style non-monotonic focusing with TAL-weighted mean preservation.

Important baseline rule:
    A0/CIoU must continue to use Ultralytics' native bbox_iou(..., CIoU=True)
    implementation. Do not route A0 through this file.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.distributed as dist
from torch import Tensor, nn


@dataclass(frozen=True)
class SGWICGConfig:
    """Frozen v1 defaults. Do not tune in the first six-cell ablation."""

    mode: str = "sg_wicg"
    inner_ratio: float = 1.25
    gcd_fixed_weight: float = 0.50
    sg_tau_px: float = 12.0
    sg_temp_px: float = 2.0
    wise_alpha: float = 1.70
    wise_delta: float = 2.70
    wise_ema_rate: float = 0.01
    wise_mean_init: float = 1.0
    eps: float = 1e-6


VALID_MODES = {
    "inner_ciou",
    "gcd",
    "inner_gcd_fixed",
    "sg_icg",
    "sg_wicg",
}


def _xyxy_geometry(boxes: Tensor, eps: float) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Return center x/y and positive width/height for aligned xyxy boxes."""
    x1, y1, x2, y2 = boxes.unbind(-1)
    w = (x2 - x1).clamp_min(eps)
    h = (y2 - y1).clamp_min(eps)
    cx = (x1 + x2) * 0.5
    cy = (y1 + y2) * 0.5
    return cx, cy, w, h


def aligned_iou(pred: Tensor, target: Tensor, eps: float = 1e-7) -> Tensor:
    """Aligned IoU for N predicted/target xyxy pairs; output shape [N]."""
    inter_lt = torch.maximum(pred[:, :2], target[:, :2])
    inter_rb = torch.minimum(pred[:, 2:], target[:, 2:])
    inter_wh = (inter_rb - inter_lt).clamp_min(0)
    inter = inter_wh[:, 0] * inter_wh[:, 1]

    p_wh = (pred[:, 2:] - pred[:, :2]).clamp_min(0)
    t_wh = (target[:, 2:] - target[:, :2]).clamp_min(0)
    p_area = p_wh[:, 0] * p_wh[:, 1]
    t_area = t_wh[:, 0] * t_wh[:, 1]
    union = (p_area + t_area - inter).clamp_min(eps)
    return (inter / union).clamp(0.0, 1.0)


def _scale_about_center(boxes: Tensor, ratio: float) -> Tensor:
    """Scale box width/height around fixed centers."""
    center = (boxes[:, :2] + boxes[:, 2:]) * 0.5
    half = (boxes[:, 2:] - boxes[:, :2]) * (0.5 * ratio)
    return torch.cat((center - half, center + half), dim=-1)


def inner_ciou_loss(pred: Tensor, target: Tensor, ratio: float = 1.25, eps: float = 1e-7) -> Tensor:
    """Per-positive Inner-CIoU loss.

    L = 1 - IoU(inner boxes) + rho^2/c^2 + alpha*v.
    The CIoU geometry penalty is computed on the original boxes.
    """
    if ratio <= 0:
        raise ValueError(f"inner ratio must be > 0, got {ratio}")

    pred_i = _scale_about_center(pred, ratio)
    target_i = _scale_about_center(target, ratio)
    inner_iou = aligned_iou(pred_i, target_i, eps=eps)
    iou = aligned_iou(pred, target, eps=eps)

    pcx, pcy, pw, ph = _xyxy_geometry(pred, eps)
    tcx, tcy, tw, th = _xyxy_geometry(target, eps)
    rho2 = (pcx - tcx).square() + (pcy - tcy).square()

    enc_lt = torch.minimum(pred[:, :2], target[:, :2])
    enc_rb = torch.maximum(pred[:, 2:], target[:, 2:])
    enc_wh = (enc_rb - enc_lt).clamp_min(eps)
    c2 = enc_wh[:, 0].square() + enc_wh[:, 1].square() + eps

    v = (4.0 / torch.pi**2) * (torch.atan(tw / th) - torch.atan(pw / ph)).square()
    with torch.no_grad():
        alpha = v / (1.0 - iou + v + eps)

    return 1.0 - inner_iou + rho2 / c2 + alpha * v


def gcd_loss(pred: Tensor, target: Tensor, eps: float = 1e-6) -> Tensor:
    """Per-positive Gaussian Combined Distance loss using the published symmetric form.

    Computation is forced to FP32 for stable divisions/sqrt/exp under AMP.
    Output is converted back to the original prediction dtype.
    """
    out_dtype = pred.dtype
    p = pred.float()
    t = target.float()

    pcx, pcy, pw, ph = _xyxy_geometry(p, eps)
    tcx, tcy, tw, th = _xyxy_geometry(t, eps)
    dx = pcx - tcx
    dy = pcy - tcy
    dw = pw - tw
    dh = ph - th

    center_p = (dx / (pw + eps)).square() + (dy / (ph + eps)).square()
    shape_t = 0.25 * ((dw / (tw + eps)).square() + (dh / (th + eps)).square())
    center_t = (dx / (tw + eps)).square() + (dy / (th + eps)).square()
    shape_p = 0.25 * ((dw / (pw + eps)).square() + (dh / (ph + eps)).square())

    gcd2 = 0.5 * (center_p + shape_t + center_t + shape_p)
    similarity = torch.exp(-torch.sqrt(gcd2.clamp_min(0.0) + eps))
    return (1.0 - similarity).to(out_dtype)


def scale_gate_from_target(
    target_xyxy_grid: Tensor,
    stride_pos: Tensor,
    tau_px: float = 12.0,
    temp_px: float = 2.0,
) -> Tuple[Tensor, Tensor]:
    """Return lambda and target short-side pixels for foreground samples.

    target boxes are in YOLO grid units. stride_pos is one scalar stride per
    foreground sample. Lambda approaches 1 for tiny objects (GCD-dominant) and
    0 for larger objects (Inner-CIoU-dominant).
    """
    if temp_px <= 0:
        raise ValueError(f"sg_temp_px must be > 0, got {temp_px}")
    wh_grid = (target_xyxy_grid[:, 2:] - target_xyxy_grid[:, :2]).clamp_min(0)
    stride_pos = stride_pos.reshape(-1).to(device=wh_grid.device, dtype=wh_grid.dtype)
    wh_px = wh_grid * stride_pos[:, None]
    short_px = wh_px.amin(dim=-1)
    lam = torch.sigmoid((tau_px - short_px) / temp_px)
    return lam.detach(), short_px.detach()


def _global_mean_detached(values: Tensor) -> Tensor:
    """Mean over positives, synchronized across DDP ranks when initialized."""
    values = values.detach()
    total = values.sum()
    count = torch.tensor(float(values.numel()), device=values.device, dtype=values.dtype)
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(total, op=dist.ReduceOp.SUM)
        dist.all_reduce(count, op=dist.ReduceOp.SUM)
    return total / count.clamp_min(1.0)


def _global_weighted_mean_detached(values: Tensor, weights: Tensor, eps: float) -> Tensor:
    """Weighted mean synchronized across DDP ranks."""
    v = values.detach().reshape(-1)
    w = weights.detach().reshape(-1).to(v.dtype)
    numerator = (v * w).sum()
    denominator = w.sum()
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(numerator, op=dist.ReduceOp.SUM)
        dist.all_reduce(denominator, op=dist.ReduceOp.SUM)
    return numerator / denominator.clamp_min(eps)


class WiseFocus(nn.Module):
    """WIoUv3-style non-monotonic focusing, separated from geometry.

    Quality q=1-IoU is detached. The final coefficient is additionally normalized
    to have TAL-weighted global mean 1, so this component redistributes regression
    gradients instead of silently changing the total box-loss gain.
    """

    def __init__(
        self,
        alpha: float = 1.70,
        delta: float = 2.70,
        ema_rate: float = 0.01,
        mean_init: float = 1.0,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if alpha <= 0 or delta <= 0:
            raise ValueError("wise_alpha and wise_delta must be > 0")
        if not 0.0 <= ema_rate <= 1.0:
            raise ValueError("wise_ema_rate must be in [0, 1]")
        self.alpha = float(alpha)
        self.delta = float(delta)
        self.ema_rate = float(ema_rate)
        self.eps = float(eps)
        self.register_buffer("iou_mean", torch.tensor(float(mean_init), dtype=torch.float32))

    @torch.no_grad()
    def update_mean(self, q: Tensor) -> None:
        if q.numel() == 0:
            return
        q_mean = _global_mean_detached(q.float())
        self.iou_mean.mul_(1.0 - self.ema_rate).add_(self.ema_rate * q_mean)

    def forward(self, plain_iou: Tensor, tal_weight: Tensor, update_state: bool = True) -> Tuple[Tensor, Dict[str, Tensor]]:
        q = (1.0 - plain_iou.detach()).clamp_min(0.0)
        if update_state and self.training:
            self.update_mean(q)

        mean = self.iou_mean.to(device=q.device, dtype=q.dtype).clamp_min(self.eps)
        beta = q / mean
        divisor = self.delta * torch.pow(torch.as_tensor(self.alpha, device=q.device, dtype=q.dtype), beta - self.delta)
        raw = beta / divisor.clamp_min(self.eps)

        raw_mean = _global_weighted_mean_detached(raw, tal_weight, self.eps)
        gain = (raw / raw_mean.clamp_min(self.eps)).detach()
        diagnostics = {
            "wise_iou_mean": mean.detach(),
            "wise_beta_mean": beta.detach().mean() if beta.numel() else beta.new_tensor(0.0),
            "wise_gain_mean": _global_weighted_mean_detached(gain, tal_weight, self.eps).detach(),
            "wise_gain_max": gain.detach().max() if gain.numel() else gain.new_tensor(0.0),
        }
        return gain, diagnostics


class SGWICGBoxLoss(nn.Module):
    """Foreground-only box regression loss for A1-A5.

    Args to forward:
        pred_xyxy_grid: [N,4] positive predicted boxes in YOLO grid units.
        target_xyxy_grid: [N,4] aligned targets in YOLO grid units.
        tal_weight: [N] or [N,1], sum of target_scores over classes for each positive.
        stride_pos: [N] stride corresponding to each positive anchor.
        target_scores_sum: Ultralytics' existing normalization denominator.

    Returns:
        scalar loss normalized exactly like Ultralytics box loss, plus detached diagnostics.
    """

    def __init__(self, cfg: SGWICGConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or SGWICGConfig()
        if self.cfg.mode not in VALID_MODES:
            raise ValueError(f"unsupported SG-WICG mode: {self.cfg.mode}; valid={sorted(VALID_MODES)}")
        self.wise = WiseFocus(
            alpha=self.cfg.wise_alpha,
            delta=self.cfg.wise_delta,
            ema_rate=self.cfg.wise_ema_rate,
            mean_init=self.cfg.wise_mean_init,
            eps=self.cfg.eps,
        )

    def forward(
        self,
        pred_xyxy_grid: Tensor,
        target_xyxy_grid: Tensor,
        tal_weight: Tensor,
        stride_pos: Tensor,
        target_scores_sum: Tensor,
        *,
        update_wise: bool = True,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        if pred_xyxy_grid.numel() == 0:
            zero = pred_xyxy_grid.sum() * 0.0
            return zero, {}

        w = tal_weight.reshape(-1).to(dtype=pred_xyxy_grid.dtype)
        denom = torch.as_tensor(target_scores_sum, device=pred_xyxy_grid.device, dtype=pred_xyxy_grid.dtype).clamp_min(self.cfg.eps)

        l_inner = inner_ciou_loss(pred_xyxy_grid, target_xyxy_grid, self.cfg.inner_ratio, self.cfg.eps)
        l_gcd = gcd_loss(pred_xyxy_grid, target_xyxy_grid, self.cfg.eps)
        plain_iou = aligned_iou(pred_xyxy_grid, target_xyxy_grid, self.cfg.eps)

        lam = pred_xyxy_grid.new_zeros((pred_xyxy_grid.shape[0],))
        short_px = pred_xyxy_grid.new_zeros((pred_xyxy_grid.shape[0],))
        diagnostics: Dict[str, Tensor] = {
            "inner_loss_mean": l_inner.detach().mean(),
            "gcd_loss_mean": l_gcd.detach().mean(),
            "plain_iou_mean": plain_iou.detach().mean(),
        }

        mode = self.cfg.mode
        if mode == "inner_ciou":
            per_pos = l_inner
        elif mode == "gcd":
            per_pos = l_gcd
        elif mode == "inner_gcd_fixed":
            fixed = float(self.cfg.gcd_fixed_weight)
            if not 0.0 <= fixed <= 1.0:
                raise ValueError(f"gcd_fixed_weight must be in [0,1], got {fixed}")
            per_pos = (1.0 - fixed) * l_inner + fixed * l_gcd
        else:
            lam, short_px = scale_gate_from_target(
                target_xyxy_grid,
                stride_pos,
                tau_px=self.cfg.sg_tau_px,
                temp_px=self.cfg.sg_temp_px,
            )
            per_pos = (1.0 - lam) * l_inner + lam * l_gcd
            diagnostics.update(
                {
                    "sg_lambda_mean": lam.mean().detach(),
                    "sg_lambda_p10": torch.quantile(lam.float(), 0.10).detach(),
                    "sg_lambda_p50": torch.quantile(lam.float(), 0.50).detach(),
                    "sg_lambda_p90": torch.quantile(lam.float(), 0.90).detach(),
                    "target_short_px_mean": short_px.mean().detach(),
                }
            )

            if mode == "sg_wicg":
                gain, wise_diag = self.wise(plain_iou, w, update_state=update_wise)
                per_pos = gain.to(per_pos.dtype) * per_pos
                diagnostics.update(wise_diag)

        loss = (per_pos * w).sum() / denom
        diagnostics["box_reg_loss"] = loss.detach()
        return loss, diagnostics


def foreground_strides(stride_tensor: Tensor, fg_mask: Tensor) -> Tensor:
    """Extract one stride value per foreground sample from Ultralytics tensors.

    Ultralytics 8.4.113 stride_tensor is [num_anchors,1], while fg_mask is
    [batch,num_anchors]. This helper expands without materializing a repeated copy.
    """
    if stride_tensor.ndim != 2 or stride_tensor.shape[-1] != 1:
        raise ValueError(f"expected stride_tensor [A,1], got {tuple(stride_tensor.shape)}")
    expanded = stride_tensor.unsqueeze(0).expand(fg_mask.shape[0], -1, -1)
    return expanded[fg_mask].reshape(-1)
