"""Executable reference tests for OPC-Inner-EIoU.

Run in a PyTorch environment:

    python research_tracks/opc_inner_eiou/test_reference.py

The repository does not declare PyTorch as a hard dependency, so these tests are
kept outside the package-level pytest suite.
"""

from __future__ import annotations

import torch

from opc_inner_eiou import (
    bbox_iou_xyxy,
    inner_eiou_loss,
    loss_by_id,
    opc_inner_eiou_loss,
    opc_ratio,
    opc_ratio_from_u,
    scale_boxes_about_center,
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


def test_exact_match_zero_for_l3_and_l5() -> None:
    target = _box(0.0, 0.0, 8.0, 4.0)
    for loss_id in ("L3", "L5"):
        pred = target.clone().requires_grad_(True)
        loss = loss_by_id(loss_id, pred, target, reduction="mean")
        loss.backward()
        assert abs(float(loss.detach())) < 1e-12
        assert torch.isfinite(pred.grad).all()


def test_opc_matches_l3_in_safe_contraction_region() -> None:
    target = _box(0.0, 0.0, 8.0, 8.0)
    pred = _box(2.0, 0.0, 8.0, 8.0)
    ratio, u, _, _ = opc_ratio(pred, target, r0=0.8)
    assert float(u) < 0.8
    assert abs(float(ratio) - 0.8) < 1e-12
    l3 = inner_eiou_loss(pred, target, ratio=0.8, reduction="none")
    l5 = opc_inner_eiou_loss(pred, target, r0=0.8, reduction="none")
    assert torch.allclose(l3, l5, atol=1e-12, rtol=0.0)


def test_ratio_controller_known_points() -> None:
    u = torch.tensor([0.0, 0.8, 0.85, 0.9, 0.99, 1.0, 1.2], dtype=torch.float64)
    ratio = opc_ratio_from_u(u, r0=0.8)
    expected = torch.tensor(
        [0.8, 0.8, 0.8875, 0.95, 0.9995, 1.0, 1.0],
        dtype=torch.float64,
    )
    assert torch.allclose(ratio, expected, atol=1e-12, rtol=0.0)


def test_controller_is_contract_only() -> None:
    u = torch.linspace(0.0, 2.0, 1001, dtype=torch.float64)
    ratio = opc_ratio_from_u(u, r0=0.8)
    assert float(ratio.min()) >= 0.8 - 1e-12
    assert float(ratio.max()) <= 1.0 + 1e-12


def test_transition_ratio_strictly_exceeds_u() -> None:
    u = torch.linspace(0.801, 0.999, 199, dtype=torch.float64)
    ratio = opc_ratio_from_u(u, r0=0.8)
    assert torch.all(ratio > u)


def test_l3_can_collapse_inner_overlap_but_opc_preserves_it() -> None:
    # Equal 8x8 boxes with dx=7.2 give u=0.9. Original boxes overlap, but
    # fixed r=0.8 auxiliary boxes do not because u > 0.8.
    target = _box(0.0, 0.0, 8.0, 8.0)
    pred = _box(7.2, 0.0, 8.0, 8.0)
    original_iou = bbox_iou_xyxy(pred, target)
    fixed_inner_iou = bbox_iou_xyxy(
        scale_boxes_about_center(pred, 0.8),
        scale_boxes_about_center(target, 0.8),
    )
    ratio, u, _, _ = opc_ratio(pred, target, r0=0.8)
    opc_inner_iou = bbox_iou_xyxy(
        scale_boxes_about_center(pred, ratio),
        scale_boxes_about_center(target, ratio),
    )
    assert 0.8 < float(u) < 1.0
    assert float(original_iou) > 0.0
    assert abs(float(fixed_inner_iou)) < 1e-12
    assert float(opc_inner_iou) > 0.0


def test_centered_scale_mismatch_keeps_same_iou_under_fixed_contraction() -> None:
    target = _box(0.0, 0.0, 8.0, 4.0)
    pred = _box(0.0, 0.0, 10.0, 5.0)
    base = bbox_iou_xyxy(pred, target)
    inner = bbox_iou_xyxy(
        scale_boxes_about_center(pred, 0.8),
        scale_boxes_about_center(target, 0.8),
    )
    assert torch.allclose(base, inner, atol=1e-12, rtol=0.0)


def test_opc_controller_is_detached() -> None:
    target = _box(0.0, 0.0, 8.0, 8.0)
    pred = _box(7.2, 0.0, 8.0, 8.0).requires_grad_(True)
    ratio, u, u_x, u_y = opc_ratio(pred, target, r0=0.8)
    assert not ratio.requires_grad
    assert not u.requires_grad
    assert not u_x.requires_grad
    assert not u_y.requires_grad


def test_backward_finite_for_tiny_and_elongated_boxes() -> None:
    cases = [
        (_box(0.2, 0.1, 4.0, 3.0), _box(0.0, 0.0, 4.0, 3.0)),
        (_box(0.5, -0.2, 20.0, 4.0), _box(0.0, 0.0, 20.0, 4.0)),
        (_box(3.7, 0.0, 4.0, 4.0), _box(0.0, 0.0, 4.0, 4.0)),
    ]
    for pred0, target in cases:
        for loss_id in ("L3", "L5"):
            pred = pred0.clone().requires_grad_(True)
            loss = loss_by_id(loss_id, pred, target, reduction="mean")
            loss.backward()
            assert torch.isfinite(loss)
            assert torch.isfinite(pred.grad).all()


def test_mixed_precision_is_finite() -> None:
    for dtype in (torch.float16, torch.bfloat16):
        target = _box(0.0, 0.0, 8.0, 4.0, dtype=dtype)
        pred = _box(1.0, 0.5, 8.0, 4.0, dtype=dtype).requires_grad_(True)
        loss = opc_inner_eiou_loss(pred, target, reduction="mean")
        assert loss.dtype == torch.float32
        loss.backward()
        assert torch.isfinite(loss)
        assert torch.isfinite(pred.grad).all()


def test_gradcheck_in_safe_constant_ratio_region() -> None:
    # Stay in u < 0.8 so the detached controller is locally constant r=0.8;
    # numerical and autograd derivatives therefore describe the same function.
    target = torch.tensor([[0.0, 0.0, 8.0, 8.0]], dtype=torch.float64)
    pred = torch.tensor([[1.1, 0.7, 8.4, 7.6]], dtype=torch.float64, requires_grad=True)

    def fn(x: torch.Tensor) -> torch.Tensor:
        return opc_inner_eiou_loss(x, target, r0=0.8, reduction="sum")

    assert torch.autograd.gradcheck(fn, (pred,), eps=1e-6, atol=3e-5, rtol=3e-4)


def main() -> None:
    tests = [
        test_exact_match_zero_for_l3_and_l5,
        test_opc_matches_l3_in_safe_contraction_region,
        test_ratio_controller_known_points,
        test_controller_is_contract_only,
        test_transition_ratio_strictly_exceeds_u,
        test_l3_can_collapse_inner_overlap_but_opc_preserves_it,
        test_centered_scale_mismatch_keeps_same_iou_under_fixed_contraction,
        test_opc_controller_is_detached,
        test_backward_finite_for_tiny_and_elongated_boxes,
        test_mixed_precision_is_finite,
        test_gradcheck_in_safe_constant_ratio_region,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"All {len(tests)} OPC reference tests passed.")


if __name__ == "__main__":
    main()
