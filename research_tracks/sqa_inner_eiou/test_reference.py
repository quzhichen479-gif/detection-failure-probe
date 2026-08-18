"""Executable reference tests for the EIoU / Inner-EIoU / SQA family.

Run inside a PyTorch environment:

    python research_tracks/sqa_inner_eiou/test_reference.py

These tests are kept outside the package-level pytest suite because this
repository does not declare PyTorch as a hard dependency.
"""

from __future__ import annotations

import torch
from losses import (
    bbox_iou_xyxy,
    eiou_loss,
    inner_eiou_loss,
    loss_by_id,
    smooth_quality_ratio,
    sqa_inner_eiou_loss,
)


def _box(
    cx: float,
    cy: float,
    w: float,
    h: float,
    *,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    return torch.tensor(
        [[cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0]],
        dtype=dtype,
    )


def test_exact_match_all_zero() -> None:
    target = _box(0.0, 0.0, 8.0, 4.0)
    for loss_id in ("L1", "L2", "L3", "L4"):
        pred = target.clone().requires_grad_(True)
        loss = loss_by_id(loss_id, pred, target, reduction="mean")
        loss.backward()
        assert abs(float(loss.detach())) < 1e-12
        assert torch.isfinite(pred.grad).all()


def test_sqa_controller_endpoints_and_midpoint() -> None:
    q = torch.tensor([0.0, 0.5, 1.0], dtype=torch.float64, requires_grad=True)
    ratio = smooth_quality_ratio(q, eta=0.20)
    expected = torch.tensor([1.2, 1.0, 0.8], dtype=torch.float64)
    assert torch.allclose(ratio, expected, atol=1e-12, rtol=0.0)
    assert not ratio.requires_grad


def test_sqa_controller_is_monotonic() -> None:
    q = torch.linspace(0.0, 1.0, 101, dtype=torch.float64)
    ratio = smooth_quality_ratio(q, eta=0.20)
    assert torch.all(ratio[1:] <= ratio[:-1])
    assert float(ratio.min()) >= 0.8 - 1e-12
    assert float(ratio.max()) <= 1.2 + 1e-12


def test_l1_matches_explicit_eiou_call() -> None:
    target = _box(0.0, 0.0, 8.0, 4.0)
    pred = _box(0.7, -0.3, 7.2, 4.8)
    direct = eiou_loss(pred, target, reduction="none")
    dispatched = loss_by_id("L1", pred, target, reduction="none")
    assert torch.allclose(direct, dispatched, atol=1e-12, rtol=0.0)


def test_l2_and_l3_dispatch_fixed_ratios() -> None:
    target = _box(0.0, 0.0, 8.0, 4.0)
    pred = _box(0.9, 0.2, 8.0, 4.0)
    l2 = loss_by_id("L2", pred, target, reduction="none")
    ref2 = inner_eiou_loss(pred, target, ratio=1.2, reduction="none")
    l3 = loss_by_id("L3", pred, target, reduction="none")
    ref3 = inner_eiou_loss(pred, target, ratio=0.8, reduction="none")
    assert torch.allclose(l2, ref2, atol=1e-12, rtol=0.0)
    assert torch.allclose(l3, ref3, atol=1e-12, rtol=0.0)


def test_broad_ratio_reduces_overlap_penalty_for_low_quality_overlap() -> None:
    target = _box(0.0, 0.0, 8.0, 8.0)
    pred = _box(5.0, 0.0, 8.0, 8.0)
    base_iou = bbox_iou_xyxy(pred, target)
    _, broad = inner_eiou_loss(pred, target, ratio=1.2, reduction="mean", return_components=True)
    assert float(base_iou) < 0.5
    assert float(broad.inner_iou) > float(base_iou)


def test_fine_ratio_increases_high_quality_overlap_sensitivity() -> None:
    target = _box(0.0, 0.0, 8.0, 8.0)
    pred = _box(0.25, 0.0, 8.0, 8.0)
    base_iou = bbox_iou_xyxy(pred, target)
    _, fine = inner_eiou_loss(pred, target, ratio=0.8, reduction="mean", return_components=True)
    assert float(base_iou) > 0.8
    assert float(fine.inner_iou) < float(base_iou)


def test_sqa_uses_broad_ratio_for_low_quality_and_fine_for_high_quality() -> None:
    target = _box(0.0, 0.0, 8.0, 8.0)
    low = _box(5.0, 0.0, 8.0, 8.0)
    high = _box(0.25, 0.0, 8.0, 8.0)
    _, low_c = sqa_inner_eiou_loss(low, target, return_components=True, reduction="mean")
    _, high_c = sqa_inner_eiou_loss(high, target, return_components=True, reduction="mean")
    assert float(low_c.ratio) > 1.0
    assert float(high_c.ratio) < 1.0


def test_backward_finite_for_tiny_and_elongated_boxes() -> None:
    cases = [
        (_box(0.2, 0.1, 4.0, 3.0), _box(0.0, 0.0, 4.0, 3.0)),
        (_box(0.5, -0.2, 20.0, 4.0), _box(0.0, 0.0, 20.0, 4.0)),
    ]
    for pred0, target in cases:
        for loss_id in ("L1", "L2", "L3", "L4"):
            pred = pred0.clone().requires_grad_(True)
            loss = loss_by_id(loss_id, pred, target, reduction="mean")
            loss.backward()
            assert torch.isfinite(loss)
            assert torch.isfinite(pred.grad).all()


def test_gradcheck_l4_in_smooth_region() -> None:
    target = torch.tensor([[0.0, 0.0, 8.0, 8.0]], dtype=torch.float64)
    pred = torch.tensor([[0.7, 0.4, 8.4, 7.7]], dtype=torch.float64, requires_grad=True)

    def fn(x: torch.Tensor) -> torch.Tensor:
        return sqa_inner_eiou_loss(x, target, eta=0.20, reduction="sum")

    assert torch.autograd.gradcheck(fn, (pred,), eps=1e-6, atol=3e-5, rtol=3e-4)


def main() -> None:
    tests = [
        test_exact_match_all_zero,
        test_sqa_controller_endpoints_and_midpoint,
        test_sqa_controller_is_monotonic,
        test_l1_matches_explicit_eiou_call,
        test_l2_and_l3_dispatch_fixed_ratios,
        test_broad_ratio_reduces_overlap_penalty_for_low_quality_overlap,
        test_fine_ratio_increases_high_quality_overlap_sensitivity,
        test_sqa_uses_broad_ratio_for_low_quality_and_fine_for_high_quality,
        test_backward_finite_for_tiny_and_elongated_boxes,
        test_gradcheck_l4_in_smooth_region,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"All {len(tests)} reference tests passed.")


if __name__ == "__main__":
    main()
