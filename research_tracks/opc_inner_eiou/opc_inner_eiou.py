"""Reference implementation of fixed Inner-EIoU(0.8) and OPC-Inner-EIoU.

This research track is intentionally self-contained. It does not depend on the
older SQA branch and does not modify the repository package.

L3 reference:
    Inner-EIoU with fixed auxiliary-box ratio r0 = 0.8.

L5 candidate:
    Overlap-Preserving Contractive Inner-EIoU (OPC-Inner-EIoU).

The OPC controller keeps r=0.8 whenever the contracted auxiliary boxes can
safely preserve overlap. Only in the geometric risk band 0.8 < u < 1 does it
relax contraction toward r=1. It never enlarges boxes beyond the original size.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class LossComponents:
    """Detached diagnostic values for one aligned-box loss evaluation."""

    iou: Tensor
    inner_iou: Tensor
    center: Tensor
    width: Tensor
    height: Tensor
    ratio: Tensor
    u: Tensor
    u_x: Tensor
    u_y: Tensor


def _validate_boxes(pred: Tensor, target: Tensor) -> None:
    if pred.shape != target.shape or pred.shape[-1] != 4:
        raise ValueError("pred and target must share shape (..., 4) in xyxy format")
    if not pred.is_floating_point() or not target.is_floating_point():
        raise TypeError("pred and target must be floating-point tensors")


def _work_tensors(pred: Tensor, target: Tensor) -> tuple[Tensor, Tensor]:
    """Promote low-precision geometry to fp32 while preserving autograd."""
    _validate_boxes(pred, target)
    if pred.dtype in (torch.float16, torch.bfloat16):
        pred = pred.float()
    if target.dtype in (torch.float16, torch.bfloat16):
        target = target.float()
    return pred, target


def _box_geometry(
    boxes: Tensor,
    eps: float,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    x1, y1, x2, y2 = boxes.unbind(dim=-1)
    w = (x2 - x1).clamp_min(eps)
    h = (y2 - y1).clamp_min(eps)
    cx = (x1 + x2) * 0.5
    cy = (y1 + y2) * 0.5
    return x1, y1, x2, y2, w, h, cx, cy


def bbox_iou_xyxy(pred: Tensor, target: Tensor, eps: float = 1e-7) -> Tensor:
    """Aligned IoU for xyxy boxes."""
    pred, target = _work_tensors(pred, target)
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


def scale_boxes_about_center(
    boxes: Tensor,
    ratio: float | Tensor,
    eps: float = 1e-7,
) -> Tensor:
    """Scale width and height around each box center; centers stay unchanged."""
    if not boxes.is_floating_point() or boxes.shape[-1] != 4:
        raise ValueError("boxes must be floating-point xyxy tensors with shape (..., 4)")
    _, _, _, _, w, h, cx, cy = _box_geometry(boxes, eps)
    ratio_t = torch.as_tensor(ratio, device=boxes.device, dtype=boxes.dtype)
    if torch.any(ratio_t <= 0):
        raise ValueError("ratio must be > 0")
    while ratio_t.ndim < w.ndim:
        ratio_t = ratio_t.unsqueeze(-1)
    rw = w * ratio_t
    rh = h * ratio_t
    return torch.stack(
        (cx - rw * 0.5, cy - rh * 0.5, cx + rw * 0.5, cy + rh * 0.5),
        dim=-1,
    )


def _eiou_geometry_terms(
    pred: Tensor,
    target: Tensor,
    eps: float,
) -> tuple[Tensor, Tensor, Tensor]:
    """EIoU center, width, and height penalties with enclosing-box normalization."""
    px1, py1, px2, py2, pw, ph, pcx, pcy = _box_geometry(pred, eps)
    gx1, gy1, gx2, gy2, gw, gh, gcx, gcy = _box_geometry(target, eps)

    cw = (torch.maximum(px2, gx2) - torch.minimum(px1, gx1)).clamp_min(eps)
    ch = (torch.maximum(py2, gy2) - torch.minimum(py1, gy1)).clamp_min(eps)
    center = ((pcx - gcx).square() + (pcy - gcy).square()) / (
        cw.square() + ch.square() + eps
    )
    width = (pw - gw).square() / (cw.square() + eps)
    height = (ph - gh).square() / (ch.square() + eps)
    return center, width, height


def overlap_state_u(
    pred: Tensor,
    target: Tensor,
    *,
    eps: float = 1e-7,
    detach: bool = True,
) -> tuple[Tensor, Tensor, Tensor]:
    """Return the normalized center-separation state u=max(u_x,u_y).

    u_x = 2|cx-cx_g| / (w+w_g)
    u_y = 2|cy-cy_g| / (h+h_g)

    For valid positive-area axis-aligned boxes, u < 1 is equivalent to positive
    overlap in both x and y directions. After scaling both boxes by ratio r,
    the corresponding auxiliary boxes overlap whenever u < r.
    """
    pred, target = _work_tensors(pred, target)
    _, _, _, _, pw, ph, pcx, pcy = _box_geometry(pred, eps)
    _, _, _, _, gw, gh, gcx, gcy = _box_geometry(target, eps)

    u_x = 2.0 * (pcx - gcx).abs() / (pw + gw + eps)
    u_y = 2.0 * (pcy - gcy).abs() / (ph + gh + eps)
    u = torch.maximum(u_x, u_y)
    if detach:
        return u.detach(), u_x.detach(), u_y.detach()
    return u, u_x, u_y


def opc_ratio_from_u(u: Tensor, r0: float = 0.8) -> Tensor:
    """Map detached overlap state u to an overlap-preserving contraction ratio.

    Piecewise controller:
        r = r0,                                  u <= r0
        z = (u-r0)/(1-r0)
        r = r0 + (1-r0)*(2z-z^2),               r0 < u < 1
        r = 1,                                   u >= 1

    In the transition band, r-u=(1-r0)z(1-z)>0, so auxiliary overlap is
    preserved whenever the original boxes still overlap. The controller is
    contractive only: r0 <= r <= 1.
    """
    if not 0.0 < r0 < 1.0:
        raise ValueError("r0 must satisfy 0 < r0 < 1")
    if u.requires_grad:
        raise ValueError("u must be detached before entering the OPC controller")

    r0_t = torch.as_tensor(r0, device=u.device, dtype=u.dtype)
    one = torch.ones_like(u)
    z = ((u - r0_t) / (1.0 - r0)).clamp(0.0, 1.0)
    transition = r0_t + (1.0 - r0) * (2.0 * z - z.square())
    ratio = torch.where(u <= r0_t, torch.full_like(u, r0), transition)
    ratio = torch.where(u >= 1.0, one, ratio)
    return ratio.clamp(min=r0, max=1.0)


def opc_ratio(
    pred: Tensor,
    target: Tensor,
    *,
    r0: float = 0.8,
    eps: float = 1e-7,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Return detached OPC ratio and the corresponding u/u_x/u_y diagnostics."""
    u, u_x, u_y = overlap_state_u(pred, target, eps=eps, detach=True)
    ratio = opc_ratio_from_u(u, r0=r0)
    return ratio, u, u_x, u_y


def _reduce(loss: Tensor, reduction: str) -> Tensor:
    if reduction == "none":
        return loss
    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    raise ValueError(f"unsupported reduction: {reduction}")


def inner_eiou_loss(
    pred: Tensor,
    target: Tensor,
    *,
    ratio: float = 0.8,
    eps: float = 1e-7,
    reduction: str = "none",
    return_components: bool = False,
) -> Tensor | tuple[Tensor, LossComponents]:
    """Fixed-ratio Inner-EIoU reference; ratio=0.8 reproduces L3."""
    pred, target = _work_tensors(pred, target)
    iou = bbox_iou_xyxy(pred, target, eps)
    pred_inner = scale_boxes_about_center(pred, ratio, eps)
    target_inner = scale_boxes_about_center(target, ratio, eps)
    inner_iou = bbox_iou_xyxy(pred_inner, target_inner, eps)
    center, width, height = _eiou_geometry_terms(pred, target, eps)
    loss = 1.0 - inner_iou + center + width + height
    reduced = _reduce(loss, reduction)
    if not return_components:
        return reduced

    u, u_x, u_y = overlap_state_u(pred, target, eps=eps, detach=True)
    ratio_t = torch.full_like(iou, float(ratio))
    components = LossComponents(
        iou=iou.detach(),
        inner_iou=inner_iou.detach(),
        center=center.detach(),
        width=width.detach(),
        height=height.detach(),
        ratio=ratio_t,
        u=u,
        u_x=u_x,
        u_y=u_y,
    )
    return reduced, components


def opc_inner_eiou_loss(
    pred: Tensor,
    target: Tensor,
    *,
    r0: float = 0.8,
    eps: float = 1e-7,
    reduction: str = "none",
    return_components: bool = False,
) -> Tensor | tuple[Tensor, LossComponents]:
    """L5: Overlap-Preserving Contractive Inner-EIoU.

    The ratio controller is detached from autograd. It selects the auxiliary
    geometry used for the current optimization step, but does not create an
    extra gradient path pred -> u -> r -> IoU_inner.
    """
    pred, target = _work_tensors(pred, target)
    iou = bbox_iou_xyxy(pred, target, eps)
    ratio, u, u_x, u_y = opc_ratio(pred, target, r0=r0, eps=eps)
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
        u=u,
        u_x=u_x,
        u_y=u_y,
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
    """Dispatch only the current best control (L3) and new candidate (L5)."""
    key = loss_id.upper()
    if key == "L3":
        return inner_eiou_loss(pred, target, ratio=0.8, eps=eps, reduction=reduction)
    if key == "L5":
        return opc_inner_eiou_loss(pred, target, r0=0.8, eps=eps, reduction=reduction)
    raise ValueError(f"unknown loss_id={loss_id!r}; expected L3 or L5")
