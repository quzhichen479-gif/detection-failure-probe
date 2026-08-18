"""Reference implementations for the EIoU / Inner-EIoU / SQA-Inner-EIoU ablation family.

The four intended experiments are:

L1: EIoU
L2: Inner-EIoU with fixed ratio=1.2
L3: Inner-EIoU with fixed ratio=0.8
L4: SQA-Inner-EIoU with detached smooth quality controller

All functions operate on xyxy boxes and return per-box losses by default.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class LossComponents:
    """Detached diagnostic components for one loss evaluation."""

    iou: Tensor
    inner_iou: Tensor
    center: Tensor
    width: Tensor
    height: Tensor
    ratio: Tensor


def _validate_boxes(pred: Tensor, target: Tensor) -> None:
    if pred.shape != target.shape or pred.shape[-1] != 4:
        raise ValueError("pred and target must share shape (..., 4) in xyxy format")


def _box_geometry(boxes: Tensor, eps: float) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    x1, y1, x2, y2 = boxes.unbind(dim=-1)
    w = (x2 - x1).clamp_min(eps)
    h = (y2 - y1).clamp_min(eps)
    cx = (x1 + x2) * 0.5
    cy = (y1 + y2) * 0.5
    return x1, y1, x2, y2, w, h, cx, cy


def bbox_iou_xyxy(pred: Tensor, target: Tensor, eps: float = 1e-7) -> Tensor:
    """Standard IoU for aligned xyxy boxes."""
    _validate_boxes(pred, target)
    px1, py1, px2, py2, pw, ph, _, _ = _box_geometry(pred, eps)
    gx1, gy1, gx2, gy2, gw, gh, _, _ = _box_geometry(target, eps)

    ix1 = torch.maximum(px1, gx1)
    iy1 = torch.maximum(py1, gy1)
    ix2 = torch.minimum(px2, gx2)
    iy2 = torch.minimum(py2, gy2)
    iw = (ix2 - ix1).clamp_min(0.0)
    ih = (iy2 - iy1).clamp_min(0.0)
    inter = iw * ih
    union = pw * ph + gw * gh - inter
    return inter / union.clamp_min(eps)


def scale_boxes_about_center(boxes: Tensor, ratio: float | Tensor, eps: float = 1e-7) -> Tensor:
    """Scale box width/height around its own center without moving the center."""
    _validate_boxes(boxes, boxes)
    _, _, _, _, w, h, cx, cy = _box_geometry(boxes, eps)
    ratio_t = torch.as_tensor(ratio, device=boxes.device, dtype=boxes.dtype)
    if torch.any(ratio_t <= 0):
        raise ValueError("ratio must be > 0")
    while ratio_t.ndim < w.ndim:
        ratio_t = ratio_t.unsqueeze(-1)
    rw = w * ratio_t
    rh = h * ratio_t
    return torch.stack((cx - rw * 0.5, cy - rh * 0.5, cx + rw * 0.5, cy + rh * 0.5), dim=-1)


def _eiou_geometry_terms(pred: Tensor, target: Tensor, eps: float) -> tuple[Tensor, Tensor, Tensor]:
    """EIoU center, width, and height penalties using the enclosing box normalization."""
    px1, py1, px2, py2, pw, ph, pcx, pcy = _box_geometry(pred, eps)
    gx1, gy1, gx2, gy2, gw, gh, gcx, gcy = _box_geometry(target, eps)

    cw = (torch.maximum(px2, gx2) - torch.minimum(px1, gx1)).clamp_min(eps)
    ch = (torch.maximum(py2, gy2) - torch.minimum(py1, gy1)).clamp_min(eps)

    center = ((pcx - gcx).square() + (pcy - gcy).square()) / (cw.square() + ch.square() + eps)
    width = (pw - gw).square() / (cw.square() + eps)
    height = (ph - gh).square() / (ch.square() + eps)
    return center, width, height


def _reduce(loss: Tensor, reduction: str) -> Tensor:
    if reduction == "none":
        return loss
    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    raise ValueError(f"unsupported reduction: {reduction}")


def eiou_loss(
    pred: Tensor,
    target: Tensor,
    *,
    eps: float = 1e-7,
    reduction: str = "none",
    return_components: bool = False,
) -> Tensor | tuple[Tensor, LossComponents]:
    """L1: EIoU = 1 - IoU + center + width + height."""
    _validate_boxes(pred, target)
    iou = bbox_iou_xyxy(pred, target, eps)
    center, width, height = _eiou_geometry_terms(pred, target, eps)
    loss = 1.0 - iou + center + width + height
    reduced = _reduce(loss, reduction)
    if not return_components:
        return reduced
    components = LossComponents(
        iou=iou.detach(),
        inner_iou=iou.detach(),
        center=center.detach(),
        width=width.detach(),
        height=height.detach(),
        ratio=torch.ones_like(iou).detach(),
    )
    return reduced, components


def inner_eiou_loss(
    pred: Tensor,
    target: Tensor,
    *,
    ratio: float | Tensor,
    eps: float = 1e-7,
    reduction: str = "none",
    return_components: bool = False,
) -> Tensor | tuple[Tensor, LossComponents]:
    """L2/L3: Inner-EIoU with a fixed or externally supplied auxiliary-box ratio."""
    _validate_boxes(pred, target)
    iou = bbox_iou_xyxy(pred, target, eps)
    pred_inner = scale_boxes_about_center(pred, ratio, eps)
    target_inner = scale_boxes_about_center(target, ratio, eps)
    inner_iou = bbox_iou_xyxy(pred_inner, target_inner, eps)
    center, width, height = _eiou_geometry_terms(pred, target, eps)
    loss = 1.0 - inner_iou + center + width + height
    reduced = _reduce(loss, reduction)
    if not return_components:
        return reduced
    ratio_t = torch.as_tensor(ratio, device=iou.device, dtype=iou.dtype)
    ratio_t = torch.ones_like(iou) * ratio_t
    components = LossComponents(
        iou=iou.detach(),
        inner_iou=inner_iou.detach(),
        center=center.detach(),
        width=width.detach(),
        height=height.detach(),
        ratio=ratio_t.detach(),
    )
    return reduced, components


def smooth_quality_ratio(iou: Tensor, eta: float = 0.20) -> Tensor:
    """Detached smooth quality controller used by SQA-Inner-EIoU.

    q = stop_gradient(IoU)
    g(q) = 3q^2 - 2q^3
    r(q) = 1 + eta * (1 - 2g(q))

    For eta=0.20, r smoothly transitions from 1.20 -> 1.00 -> 0.80.
    The controller has zero slope at q=0 and q=1.
    """
    if not 0.0 < eta < 1.0:
        raise ValueError("eta must satisfy 0 < eta < 1")
    q = iou.detach().clamp(0.0, 1.0)
    g = 3.0 * q.square() - 2.0 * q.pow(3)
    return 1.0 + float(eta) * (1.0 - 2.0 * g)


def sqa_inner_eiou_loss(
    pred: Tensor,
    target: Tensor,
    *,
    eta: float = 0.20,
    eps: float = 1e-7,
    reduction: str = "none",
    return_components: bool = False,
) -> Tensor | tuple[Tensor, LossComponents]:
    """L4: Smooth Quality-Adaptive Inner-EIoU.

    The IoU-derived ratio is detached: quality controls which local geometry is
    used for this optimization step, but the controller itself does not create
    a second gradient path through IoU.
    """
    _validate_boxes(pred, target)
    iou = bbox_iou_xyxy(pred, target, eps)
    ratio = smooth_quality_ratio(iou, eta=eta)
    pred_inner = scale_boxes_about_center(pred, ratio, eps)
    target_inner = scale_boxes_about_center(target, ratio, eps)
    inner_iou = bbox_iou_xyxy(pred_inner, target_inner, eps)
    center, width, height = _eiou_geometry_terms(pred, target, eps)
    loss = 1.0 - inner_iou + center + width + height
    reduced = _reduce(loss, reduction)
    if not return_components:
        return reduced
    components = LossComponents(
        iou=iou.detach(),
        inner_iou=inner_iou.detach(),
        center=center.detach(),
        width=width.detach(),
        height=height.detach(),
        ratio=ratio.detach(),
    )
    return reduced, components


def loss_by_id(
    loss_id: str,
    pred: Tensor,
    target: Tensor,
    *,
    eps: float = 1e-7,
    reduction: str = "none",
) -> Tensor:
    """Convenience dispatcher matching the preregistered L1-L4 experiment IDs."""
    key = loss_id.upper()
    if key == "L1":
        return eiou_loss(pred, target, eps=eps, reduction=reduction)
    if key == "L2":
        return inner_eiou_loss(pred, target, ratio=1.2, eps=eps, reduction=reduction)
    if key == "L3":
        return inner_eiou_loss(pred, target, ratio=0.8, eps=eps, reduction=reduction)
    if key == "L4":
        return sqa_inner_eiou_loss(pred, target, eta=0.20, eps=eps, reduction=reduction)
    raise ValueError(f"unknown loss_id={loss_id!r}; expected L1, L2, L3, or L4")
