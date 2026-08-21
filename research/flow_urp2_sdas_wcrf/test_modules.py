from __future__ import annotations

import torch

from modules import (
    SDASHead,
    URP2Refiner,
    WCRF,
    build_sdas_target,
    dfl_entropy_map,
    sdas_focal_loss,
)


def test_dfl_entropy_shape_and_range():
    x = torch.randn(2, 64, 20, 20)
    u = dfl_entropy_map(x, reg_max=16)
    assert u.shape == (2, 1, 20, 20)
    assert torch.isfinite(u).all()
    assert float(u.min()) >= 0.0
    assert float(u.max()) <= 1.00001


def test_urp2_identity_at_initialization_and_backward():
    torch.manual_seed(0)
    module = URP2Refiner(64, 128, nc=1, reg_max=16, hidden=32)
    p2 = torch.randn(2, 64, 40, 40, requires_grad=True)
    p3 = torch.randn(2, 128, 20, 20, requires_grad=True)
    reg = torch.randn(2, 64, 20, 20, requires_grad=True)
    cls = torch.randn(2, 1, 20, 20, requires_grad=True)
    out_reg, out_cls = module(p2, p3, reg, cls)
    assert out_reg.shape == reg.shape
    assert out_cls.shape == cls.shape
    # Delta heads are zero initialized, so insertion starts from stock logits.
    assert torch.allclose(out_reg, reg, atol=1e-7, rtol=0)
    assert torch.allclose(out_cls, cls, atol=1e-7, rtol=0)
    (out_reg.mean() + out_cls.mean()).backward()
    assert reg.grad is not None and cls.grad is not None


def test_sdas_target_and_loss():
    head = SDASHead(64, hidden=32)
    p2 = torch.randn(2, 64, 40, 40, requires_grad=True)
    logits = head(p2)
    batch_idx = torch.tensor([0, 1])
    boxes = torch.tensor([[0.50, 0.50, 0.10, 0.08], [0.25, 0.75, 0.04, 0.05]])
    target = build_sdas_target(2, (40, 40), batch_idx, boxes)
    assert target.shape == logits.shape
    assert target.max() == 1
    loss = sdas_focal_loss(logits, target)
    assert torch.isfinite(loss)
    loss.backward()
    assert p2.grad is not None


def test_sdas_empty_gt_is_finite():
    logits = torch.randn(2, 1, 40, 40, requires_grad=True)
    target = build_sdas_target(
        2,
        (40, 40),
        torch.empty(0, dtype=torch.long),
        torch.empty(0, 4),
    )
    loss = sdas_focal_loss(logits, target)
    assert torch.isfinite(loss)
    loss.backward()


def test_wcrf_shape_identity_start_and_backward():
    torch.manual_seed(0)
    module = WCRF(128, hidden=64)
    x = torch.randn(2, 128, 20, 20, requires_grad=True)
    y = module(x)
    assert y.shape == x.shape
    # gamma starts at zero, giving an exact identity insertion at step 0.
    assert torch.allclose(y, x, atol=1e-7, rtol=0)
    y.mean().backward()
    assert x.grad is not None


def test_amp_like_half_precision_forward_when_supported():
    if not torch.cuda.is_available():
        return
    device = torch.device("cuda")
    module = WCRF(64, hidden=32).to(device).half()
    x = torch.randn(1, 64, 20, 20, device=device, dtype=torch.float16)
    y = module(x)
    assert y.dtype == torch.float16
    assert torch.isfinite(y).all()
