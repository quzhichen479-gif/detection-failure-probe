"""Reference invariants for SG-WICG v1.

Run before touching Ultralytics integration:
    pytest -q research_tracks/sg_wicg/test_reference.py
"""

import torch

from research_tracks.sg_wicg.sg_wicg import (
    SGWICGBoxLoss,
    SGWICGConfig,
    WiseFocus,
    aligned_iou,
    gcd_loss,
    inner_ciou_loss,
    scale_gate_from_target,
)


def _boxes(dtype=torch.float32):
    target = torch.tensor(
        [[10.0, 10.0, 20.0, 20.0], [2.0, 3.0, 8.0, 15.0], [30.0, 10.0, 50.0, 18.0]],
        dtype=dtype,
    )
    pred = torch.tensor(
        [[11.0, 10.0, 21.0, 20.0], [3.0, 4.0, 9.0, 16.0], [29.0, 11.0, 49.0, 19.0]],
        dtype=dtype,
        requires_grad=True,
    )
    return pred, target


def test_identical_boxes_are_zero():
    b = torch.tensor([[1.0, 2.0, 8.0, 11.0], [3.0, 4.0, 6.0, 7.0]], dtype=torch.float32)
    assert torch.allclose(inner_ciou_loss(b, b), torch.zeros(2), atol=1e-6)
    assert torch.allclose(gcd_loss(b, b), torch.zeros(2), atol=1e-6)


def test_gcd_scale_invariance():
    pred, target = _boxes()
    a = gcd_loss(pred, target)
    for scale in (2.0, 4.0):
        b = gcd_loss(pred * scale, target * scale)
        assert torch.allclose(a, b, atol=2e-6, rtol=2e-6)


def test_non_overlap_gcd_is_finite_and_has_gradient():
    pred = torch.tensor([[0.0, 0.0, 2.0, 2.0]], requires_grad=True)
    target = torch.tensor([[20.0, 10.0, 24.0, 15.0]])
    loss = gcd_loss(pred, target).sum()
    assert torch.isfinite(loss)
    loss.backward()
    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all()
    assert pred.grad.abs().sum() > 0


def test_scale_gate_is_monotonic_and_centered():
    target = torch.tensor([[0.0, 0.0, 8.0, 8.0], [0.0, 0.0, 12.0, 12.0], [0.0, 0.0, 16.0, 16.0]])
    lam, short = scale_gate_from_target(target, torch.ones(3), tau_px=12.0, temp_px=2.0)
    assert short.tolist() == [8.0, 12.0, 16.0]
    assert lam[0] > lam[1] > lam[2]
    assert torch.allclose(lam[1], torch.tensor(0.5), atol=1e-7)


def test_wise_tal_weighted_mean_is_one():
    plain_iou = torch.tensor([0.1, 0.4, 0.7, 0.9])
    tal_weight = torch.tensor([0.2, 0.5, 0.9, 0.4])
    wise = WiseFocus()
    wise.eval()
    gain, _ = wise(plain_iou, tal_weight, update_state=False)
    weighted_mean = (gain * tal_weight).sum() / tal_weight.sum()
    assert torch.allclose(weighted_mean, torch.tensor(1.0), atol=1e-6)


def test_all_ablation_modes_are_finite_and_backward():
    pred0, target = _boxes()
    tal_weight = torch.tensor([0.3, 0.8, 0.5])
    stride = torch.tensor([8.0, 8.0, 16.0])
    denom = tal_weight.sum()
    for mode in ("inner_ciou", "gcd", "inner_gcd_fixed", "sg_icg", "sg_wicg"):
        pred = pred0.detach().clone().requires_grad_(True)
        loss_fn = SGWICGBoxLoss(SGWICGConfig(mode=mode))
        loss, diag = loss_fn(pred, target, tal_weight, stride, denom)
        assert torch.isfinite(loss), (mode, diag)
        loss.backward()
        assert pred.grad is not None
        assert torch.isfinite(pred.grad).all(), mode


def test_plain_iou_range():
    pred, target = _boxes()
    iou = aligned_iou(pred, target)
    assert ((0.0 <= iou) & (iou <= 1.0)).all()
