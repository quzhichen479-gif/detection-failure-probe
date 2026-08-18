"""Ultralytics 8.4.113 adapter helpers for L3/L5 bbox regression.

This helper preserves TaskAlignedAssigner weighting and normalization and only
replaces the per-positive IoU-style box regression term. DFL remains untouched.
"""

from __future__ import annotations

from opc_inner_eiou import loss_by_id
from torch import Tensor


def weighted_loss_for_ultralytics(
    loss_id: str,
    pred_bboxes: Tensor,
    target_bboxes: Tensor,
    target_scores: Tensor,
    target_scores_sum: Tensor,
    fg_mask: Tensor,
    *,
    eps: float = 1e-7,
) -> Tensor:
    """Return TAL-weighted L3/L5 bbox regression for positive candidates.

    Expected common shapes:
        pred_bboxes:   [B, A, 4]
        target_bboxes: [B, A, 4]
        target_scores: [B, A, C]
        fg_mask:       [B, A]

    OPC uses dimensionless ratios and normalized center separation, so no
    stride-specific pixel conversion is required. Prediction and target boxes
    only need to share the same xyxy coordinate system.
    """
    if pred_bboxes.shape != target_bboxes.shape or pred_bboxes.shape[-1] != 4:
        raise ValueError("pred_bboxes and target_bboxes must share shape [B, A, 4]")
    if fg_mask.shape != pred_bboxes.shape[:2]:
        raise ValueError("fg_mask must match the [B, A] candidate dimensions")
    if target_scores.shape[:2] != fg_mask.shape:
        raise ValueError("target_scores must match the [B, A] candidate dimensions")

    if not fg_mask.any():
        return pred_bboxes.sum() * 0.0

    pred_pos = pred_bboxes[fg_mask]
    target_pos = target_bboxes[fg_mask]
    per_box = loss_by_id(
        loss_id,
        pred_pos,
        target_pos,
        eps=eps,
        reduction="none",
    )
    weight = target_scores.sum(dim=-1)[fg_mask].to(per_box.dtype)
    normalizer = target_scores_sum.to(per_box.dtype).clamp_min(eps)
    return (per_box * weight).sum() / normalizer
