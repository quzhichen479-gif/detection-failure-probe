"""Executable reference tests for SRB-IoU.

Run from this directory or provide it on PYTHONPATH:

    python research_tracks/srb_iou_loss/test_srb_iou_reference.py

These tests are intentionally outside the package-level pytest suite because the
main repository does not declare PyTorch as a hard dependency.
"""

from __future__ import annotations

import math

import torch
from srb_iou import srb_iou_loss, stable_log_cosh


def _box(cx: float, cy: float, w: float, h: float, dtype: torch.dtype) -> torch.Tensor:
    return torch.tensor(
        [[cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0]],
        dtype=dtype,
    )


def _scalar_loss(pred: torch.Tensor, target: torch.Tensor, delta: float = 1.0) -> float:
    return float(srb_iou_loss(pred, target, delta=delta, reduction="mean").detach())


def test_exact_match_zero_and_zero_gradient() -> None:
    target = _box(0.0, 0.0, 8.0, 8.0, torch.float64)
    pred = target.clone().requires_grad_(True)
    loss = srb_iou_loss(pred, target, delta=1.0, reduction="mean")
    loss.backward()
    assert abs(float(loss)) < 1e-12
    assert torch.isfinite(pred.grad).all()
    assert float(pred.grad.abs().max()) < 1e-10


def test_translation_monotonic() -> None:
    target = _box(0.0, 0.0, 8.0, 8.0, torch.float64)
    offsets = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 12.0]
    values = [
        _scalar_loss(_box(d, 0.0, 8.0, 8.0, torch.float64), target)
        for d in offsets
    ]
    assert all(b > a for a, b in zip(values, values[1:], strict=True))


def test_non_overlap_has_correct_translation_gradient() -> None:
    target = _box(0.0, 0.0, 8.0, 8.0, torch.float64)
    pred = _box(20.0, 0.0, 8.0, 8.0, torch.float64).requires_grad_(True)
    loss = srb_iou_loss(pred, target, delta=1.0, reduction="mean")
    loss.backward()
    assert torch.isfinite(pred.grad).all()
    # Shifting the complete prediction further right changes x1 and x2 equally.
    # A positive directional derivative means gradient descent moves it left.
    dloss_dshift_x = pred.grad[0, 0] + pred.grad[0, 2]
    assert float(dloss_dshift_x) > 0.0


def test_centered_scale_has_unique_sampled_minimum_at_target() -> None:
    target = _box(0.0, 0.0, 8.0, 8.0, torch.float64)
    sizes = [3.0, 4.0, 6.0, 7.0, 8.0, 9.0, 10.0, 12.0, 16.0]
    values = [
        _scalar_loss(_box(0.0, 0.0, s, s, torch.float64), target)
        for s in sizes
    ]
    best = min(range(len(values)), key=values.__getitem__)
    assert sizes[best] == 8.0
    assert values[best] == 0.0


def test_tiny_box_gradient_is_finite() -> None:
    target = _box(0.0, 0.0, 0.5, 0.5, torch.float64)
    pred = _box(0.05, 0.0, 0.5, 0.5, torch.float64).requires_grad_(True)
    loss = srb_iou_loss(pred, target, delta=1.0, reduction="mean")
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(pred.grad).all()
    assert float(pred.grad.abs().max()) < 1.0


def test_extreme_aspect_ratio_is_finite() -> None:
    target = _box(0.0, 0.0, 32.0, 4.0, torch.float64)
    pred = _box(0.7, -0.2, 31.0, 4.5, torch.float64).requires_grad_(True)
    loss = srb_iou_loss(pred, target, delta=1.0, reduction="mean")
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(pred.grad).all()


def test_mixed_precision_inputs_return_finite_fp32_loss() -> None:
    for dtype in (torch.float16, torch.bfloat16):
        target = _box(0.0, 0.0, 8.0, 4.0, dtype)
        pred = _box(1.0, 0.5, 8.0, 4.0, dtype).requires_grad_(True)
        loss = srb_iou_loss(pred, target, delta=1.0, reduction="mean")
        assert loss.dtype == torch.float32
        loss.backward()
        assert torch.isfinite(loss)
        assert torch.isfinite(pred.grad).all()


def test_stable_log_cosh_large_residual() -> None:
    x = torch.tensor([-1e4, -100.0, 0.0, 100.0, 1e4], dtype=torch.float32)
    y = stable_log_cosh(x)
    assert torch.isfinite(y).all()
    assert abs(float(y[2])) < 1e-7
    assert math.isclose(float(y[-1]), 1e4 - math.log(2.0), rel_tol=1e-6)


def test_gradcheck_in_smooth_overlap_region() -> None:
    target = torch.tensor([[0.0, 0.0, 8.0, 8.0]], dtype=torch.float64)
    pred = torch.tensor([[1.1, 0.7, 8.4, 7.6]], dtype=torch.float64, requires_grad=True)

    def fn(x: torch.Tensor) -> torch.Tensor:
        return srb_iou_loss(
            x,
            target,
            delta=1.0,
            reduction="sum",
            validate=False,
        )

    assert torch.autograd.gradcheck(fn, (pred,), eps=1e-6, atol=2e-5, rtol=2e-4)


def test_invalid_delta_rejected() -> None:
    target = _box(0.0, 0.0, 8.0, 8.0, torch.float32)
    try:
        srb_iou_loss(target, target, delta=0.0)
    except ValueError as exc:
        assert "delta" in str(exc)
    else:
        raise AssertionError("delta=0 must be rejected")


def main() -> None:
    tests = [
        test_exact_match_zero_and_zero_gradient,
        test_translation_monotonic,
        test_non_overlap_has_correct_translation_gradient,
        test_centered_scale_has_unique_sampled_minimum_at_target,
        test_tiny_box_gradient_is_finite,
        test_extreme_aspect_ratio_is_finite,
        test_mixed_precision_inputs_return_finite_fp32_loss,
        test_stable_log_cosh_large_residual,
        test_gradcheck_in_smooth_overlap_region,
        test_invalid_delta_rejected,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"All {len(tests)} SRB-IoU reference tests passed.")


if __name__ == "__main__":
    main()
