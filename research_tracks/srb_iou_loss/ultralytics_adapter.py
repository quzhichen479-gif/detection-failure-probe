"""Small adapter helpers for integrating SRB-IoU into Ultralytics YOLO loss.

This is not a monkey patch.  Copy the relevant helper into the local frozen
Ultralytics 8.4.113 engineering tree only after verifying the exact BboxLoss
signature and coordinate system there.
"""

from __future__ import annotations

import torch
from srb_iou import srb_iou_loss
from torch import Tensor


def positive_stride_values(stride_tensor: Tensor, fg_mask: Tensor) -> Tensor:
    """Return one stride value for every positive candidate.

    Supported common layouts:

    - stride_tensor: [A, 1], fg_mask: [B, A]
    - stride_tensor: [A],    fg_mask: [B, A]
    - stride_tensor: [B, A], fg_mask: [B, A]
    - stride_tensor: [B, A, 1], fg_mask: [B, A]
    """
    if fg_mask.ndim != 2:
        raise ValueError(f"fg_mask must be [B, A], got {tuple(fg_mask.shape)}")
    batch, anchors = fg_mask.shape

    stride = stride_tensor
    if stride.ndim == 2 and stride.shape == (anchors, 1):
        stride = stride.squeeze(-1).unsqueeze(0).expand(batch, -1)
    elif stride.ndim == 1 and stride.shape[0] == anchors:
        stride = stride.unsqueeze(0).expand(batch, -1)
    elif stride.ndim == 3 and stride.shape == (batch, anchors, 1):
        stride = stride.squeeze(-1)
    elif stride.ndim == 2 and stride.shape == (batch, anchors):
        pass
    else:
        raise ValueError(
            "Unsupported stride_tensor layout: "
            f"stride={tuple(stride_tensor.shape)}, fg={tuple(fg_mask.shape)}"
        )

    positive = stride[fg_mask]
    if positive.numel() == 0:
        return positive
    if not torch.isfinite(positive).all() or torch.any(positive <= 0):
        raise ValueError("Positive stride values must be finite and > 0")
    return positive


def weighted_srb_for_ultralytics(
    pred_bboxes: Tensor,
    target_bboxes: Tensor,
    target_scores: Tensor,
    target_scores_sum: Tensor,
    fg_mask: Tensor,
    stride_tensor: Tensor,
    *,
    delta_pixel: float = 1.0,
    lambda_edge: float = 1.0,
    eps: float = 1e-7,
) -> Tensor:
    """Compute TAL-weighted SRB-IoU while preserving Ultralytics weighting.

    Expected integration assumption:
        ``pred_bboxes`` and ``target_bboxes`` are decoded in feature-grid units,
        so one input pixel becomes ``delta_grid = delta_pixel / stride`` for
        each positive candidate.

    If the local frozen implementation already supplies boxes in input-pixel
    units, do not use this conversion; call ``srb_iou_loss(..., delta=1.0)``
    directly instead.
    """
    if pred_bboxes.shape != target_bboxes.shape or pred_bboxes.shape[-1] != 4:
        raise ValueError("pred_bboxes and target_bboxes must share shape [B, A, 4]")
    if fg_mask.shape != pred_bboxes.shape[:2]:
        raise ValueError("fg_mask must match the [B, A] candidate dimensions")
    if target_scores.shape[:2] != fg_mask.shape:
        raise ValueError("target_scores must match the [B, A] candidate dimensions")

    if not torch.any(fg_mask):
        return pred_bboxes.sum() * 0.0

    pred_pos = pred_bboxes[fg_mask]
    target_pos = target_bboxes[fg_mask]
    stride_pos = positive_stride_values(stride_tensor, fg_mask).to(pred_pos.dtype)
    delta_pos = float(delta_pixel) / stride_pos

    per_box = srb_iou_loss(
        pred_pos,
        target_pos,
        delta=delta_pos,
        lambda_edge=lambda_edge,
        eps=eps,
        reduction="none",
    )

    weight = target_scores.sum(dim=-1)[fg_mask].to(per_box.dtype)
    normalizer = target_scores_sum.to(per_box.dtype).clamp_min(eps)
    return (per_box * weight).sum() / normalizer
