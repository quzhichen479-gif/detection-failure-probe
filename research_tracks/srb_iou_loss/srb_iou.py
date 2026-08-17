"""Smooth Resolution-Bounded IoU (SRB-IoU) reference implementation.

This file is intentionally standalone.  It does not patch Ultralytics, change
assignment, or alter DFL.  The caller must pass decoded bounding boxes in a
single coordinate system and express ``delta`` in the same units.
"""

from __future__ import annotations

import math
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

Reduction = Literal["none", "mean", "sum"]


def stable_log_cosh(x: Tensor) -> Tensor:
    """Compute log(cosh(x)) without overflowing for large |x|."""
    ax = x.abs()
    return ax + F.softplus(-2.0 * ax) - math.log(2.0)


def _validate_boxes(boxes: Tensor, name: str) -> None:
    if boxes.ndim == 0 or boxes.shape[-1] != 4:
        raise ValueError(f"{name} must have shape (..., 4), got {tuple(boxes.shape)}")
    if not torch.isfinite(boxes).all():
        raise ValueError(f"{name} contains NaN or Inf")


def _reduce(loss: Tensor, reduction: Reduction) -> Tensor:
    if reduction == "none":
        return loss
    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    raise ValueError(f"Unsupported reduction: {reduction}")


def rb_overlap_mismatch(
    pred_xyxy: Tensor,
    target_xyxy: Tensor,
    *,
    delta: float | Tensor = 1.0,
    eps: float = 1e-7,
    validate: bool = True,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Return the resolution-bounded symmetric-difference ratio.

    The core mismatch is

        m = (U - I) / (U + A_delta)

    with

        A_delta = 2 * delta * (w_g + h_g) + 4 * delta**2.

    ``delta`` must be expressed in the same coordinate system as the boxes.
    For example, when boxes are represented in feature-grid coordinates at
    stride ``s``, a one-input-pixel floor is ``delta = 1 / s``.
    """
    if validate:
        _validate_boxes(pred_xyxy, "pred_xyxy")
        _validate_boxes(target_xyxy, "target_xyxy")

    if pred_xyxy.shape != target_xyxy.shape:
        raise ValueError(
            "pred_xyxy and target_xyxy must have identical shapes, got "
            f"{tuple(pred_xyxy.shape)} and {tuple(target_xyxy.shape)}"
        )

    original_dtype = pred_xyxy.dtype
    work_dtype = (
        torch.float32
        if original_dtype in (torch.float16, torch.bfloat16)
        else original_dtype
    )

    pred = pred_xyxy.to(work_dtype)
    target = target_xyxy.to(device=pred.device, dtype=work_dtype)

    px1, py1, px2, py2 = pred.unbind(dim=-1)
    gx1, gy1, gx2, gy2 = target.unbind(dim=-1)

    # YOLO decoders should already guarantee valid decoded boxes.  clamp_min
    # protects numerical execution but must not be used to hide a broken
    # decoder; callers can enable validation around their integration point.
    pw = (px2 - px1).clamp_min(eps)
    ph = (py2 - py1).clamp_min(eps)
    gw = (gx2 - gx1).clamp_min(eps)
    gh = (gy2 - gy1).clamp_min(eps)

    pred_area = pw * ph
    gt_area = gw * gh

    ix1 = torch.maximum(px1, gx1)
    iy1 = torch.maximum(py1, gy1)
    ix2 = torch.minimum(px2, gx2)
    iy2 = torch.minimum(py2, gy2)

    iw = (ix2 - ix1).clamp_min(0.0)
    ih = (iy2 - iy1).clamp_min(0.0)
    inter = iw * ih
    union = pred_area + gt_area - inter

    delta_t = torch.as_tensor(delta, device=pred.device, dtype=work_dtype)
    if torch.any(delta_t <= 0):
        raise ValueError("delta must be strictly positive")

    a_delta = 2.0 * delta_t * (gw + gh) + 4.0 * delta_t.square()
    mismatch = (union - inter) / (union + a_delta + eps)

    # beta is not a fitted size threshold.  It is derived from the fraction of
    # the delta-dilated target support that belongs to the resolution boundary.
    beta = (a_delta / (gt_area + a_delta + eps)).clamp_min(eps)

    return mismatch, {
        "inter": inter,
        "union": union,
        "pred_area": pred_area,
        "gt_area": gt_area,
        "a_delta": a_delta,
        "beta": beta,
        "gt_w": gw,
        "gt_h": gh,
    }


def srb_iou_loss(
    pred_xyxy: Tensor,
    target_xyxy: Tensor,
    *,
    delta: float | Tensor = 1.0,
    lambda_edge: float = 1.0,
    eps: float = 1e-7,
    reduction: Reduction = "none",
    validate: bool = True,
    return_components: bool = False,
) -> Tensor | tuple[Tensor, dict[str, Tensor]]:
    """Compute Smooth Resolution-Bounded IoU loss.

    SRB-IoU is

        L = beta * log(cosh(m / beta)) + lambda_edge * L_edge

    where

        m = (U - I) / (U + A_delta)
        beta = A_delta / (A_g + A_delta)

    and ``L_edge`` is the mean robust error of the four box edges normalized by
    the delta-expanded GT width/height.

    The overlap term changes the local conditioning of IoU near pixel-limited
    boxes.  The edge term supplies bounded geometric guidance when boxes do not
    overlap.  No NWD, EIoU, Inner-IoU, assignment, or classification term is
    included here.

    Mixed precision:
        Geometry is promoted to float32 for fp16/bfloat16 inputs.  The returned
        loss intentionally stays in the work dtype for numerical stability.
    """
    if lambda_edge < 0:
        raise ValueError("lambda_edge must be non-negative")

    mismatch, geom = rb_overlap_mismatch(
        pred_xyxy,
        target_xyxy,
        delta=delta,
        eps=eps,
        validate=validate,
    )

    work_dtype = mismatch.dtype
    pred = pred_xyxy.to(work_dtype)
    target = target_xyxy.to(device=pred.device, dtype=work_dtype)
    px1, py1, px2, py2 = pred.unbind(dim=-1)
    gx1, gy1, gx2, gy2 = target.unbind(dim=-1)

    beta = geom["beta"]
    overlap_arg = mismatch / beta
    overlap_loss = beta * stable_log_cosh(overlap_arg)

    sx = geom["gt_w"] + 2.0 * torch.as_tensor(
        delta, device=pred.device, dtype=work_dtype
    )
    sy = geom["gt_h"] + 2.0 * torch.as_tensor(
        delta, device=pred.device, dtype=work_dtype
    )

    edge_residuals = torch.stack(
        (
            (px1 - gx1) / sx,
            (px2 - gx2) / sx,
            (py1 - gy1) / sy,
            (py2 - gy2) / sy,
        ),
        dim=-1,
    )
    edge_loss = stable_log_cosh(edge_residuals).mean(dim=-1)

    per_box = overlap_loss + float(lambda_edge) * edge_loss
    loss = _reduce(per_box, reduction)

    if not return_components:
        return loss

    components = {
        "total_per_box": per_box.detach(),
        "overlap": overlap_loss.detach(),
        "edge": edge_loss.detach(),
        "mismatch": mismatch.detach(),
        "beta": beta.detach(),
        "a_delta": geom["a_delta"].detach(),
        "inter": geom["inter"].detach(),
        "union": geom["union"].detach(),
    }
    return loss, components


def rb_iou_v0_loss(
    pred_xyxy: Tensor,
    target_xyxy: Tensor,
    *,
    delta: float | Tensor = 1.0,
    lambda_edge: float = 1.0,
    eps: float = 1e-7,
    reduction: Reduction = "none",
) -> Tensor:
    """Early RB-v0 retained only for loss-surface ablations.

    Do not use this as the final training proposal.  It keeps a cusp at exact
    matching because the overlap part is the unsmoothed mismatch ``m``.
    """
    mismatch, geom = rb_overlap_mismatch(
        pred_xyxy,
        target_xyxy,
        delta=delta,
        eps=eps,
    )
    pred = pred_xyxy.to(mismatch.dtype)
    target = target_xyxy.to(device=pred.device, dtype=mismatch.dtype)
    px1, py1, px2, py2 = pred.unbind(dim=-1)
    gx1, gy1, gx2, gy2 = target.unbind(dim=-1)
    delta_t = torch.as_tensor(delta, device=pred.device, dtype=mismatch.dtype)
    sx = geom["gt_w"] + 2.0 * delta_t
    sy = geom["gt_h"] + 2.0 * delta_t
    z = torch.stack(
        (
            (px1 - gx1) / sx,
            (px2 - gx2) / sx,
            (py1 - gy1) / sy,
            (py2 - gy2) / sy,
        ),
        dim=-1,
    )
    edge = stable_log_cosh(z).mean(dim=-1)
    return _reduce(mismatch + float(lambda_edge) * edge, reduction)


class SRBIoULoss(nn.Module):
    """``nn.Module`` wrapper for :func:`srb_iou_loss`."""

    def __init__(
        self,
        delta: float = 1.0,
        lambda_edge: float = 1.0,
        eps: float = 1e-7,
        reduction: Reduction = "none",
    ) -> None:
        super().__init__()
        if delta <= 0:
            raise ValueError("delta must be strictly positive")
        if lambda_edge < 0:
            raise ValueError("lambda_edge must be non-negative")
        self.delta = float(delta)
        self.lambda_edge = float(lambda_edge)
        self.eps = float(eps)
        self.reduction = reduction

    def forward(self, pred_xyxy: Tensor, target_xyxy: Tensor) -> Tensor:
        return srb_iou_loss(
            pred_xyxy,
            target_xyxy,
            delta=self.delta,
            lambda_edge=self.lambda_edge,
            eps=self.eps,
            reduction=self.reduction,
        )
