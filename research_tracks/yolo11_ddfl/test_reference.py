"""Reference invariants for YOLO11-DDFL v1.

Run from this directory or add it to PYTHONPATH:
    pytest -q research_tracks/yolo11_ddfl/test_reference.py
"""

import torch

from yolo11_ddfl import (
    DDFL20,
    STANDARD16,
    UNIFORM20,
    NonUniformDFLIntegral,
    NonUniformDFLoss,
    head_contract,
    native_standard_dfl_reference,
)


def test_support_contracts():
    assert len(STANDARD16) == 16
    assert len(UNIFORM20) == 20
    assert len(DDFL20) == 20
    assert STANDARD16[0] == DDFL20[0] == 0.0
    assert STANDARD16[-1] == DDFL20[-1] == 15.0
    assert sum(v <= 2.0 for v in DDFL20) > sum(v <= 2.0 for v in UNIFORM20)


def test_head_contract_keeps_base_tower_width_and_changes_only_output_budget():
    c = head_contract("ddfl20")
    assert c["base_reg_max"] == 16
    assert c["num_bins"] == 20
    assert c["base_box_output_channels"] == 64
    assert c["new_box_output_channels"] == 80
    assert c["keep_regression_tower_hidden_width"] is True
    assert c["support_max"] == 15.0


def test_standard16_loss_matches_ultralytics_math():
    torch.manual_seed(0)
    n = 19
    logits = torch.randn(n, 4, 16, dtype=torch.float32, requires_grad=True)
    target = torch.rand(n, 4) * 14.8
    ours = NonUniformDFLoss(STANDARD16)(logits, target)
    native = native_standard_dfl_reference(logits, target, reg_max=16)
    assert torch.allclose(ours, native, atol=1e-6, rtol=1e-6)


def test_standard16_integral_matches_arange_expectation():
    torch.manual_seed(1)
    b, a = 2, 31
    x = torch.randn(b, 64, a)
    ours = NonUniformDFLIntegral(STANDARD16)(x)
    p = x.view(b, 4, 16, a).softmax(2)
    ref = (p * torch.arange(16, dtype=p.dtype).view(1, 1, 16, 1)).sum(2)
    assert torch.allclose(ours, ref, atol=1e-6, rtol=1e-6)


def test_exact_support_target_reduces_to_single_bin_nll():
    support = DDFL20
    loss_fn = NonUniformDFLoss(support)
    k = support.index(0.5)
    logits = torch.full((1, 4, len(support)), -3.0)
    logits[:, :, k] = 4.0
    target = torch.full((1, 4), 0.5)
    loss = loss_fn(logits, target)
    logp = torch.log_softmax(logits.float(), -1)
    ref = -logp[:, :, k].mean(-1, keepdim=True)
    assert torch.allclose(loss, ref, atol=1e-6, rtol=1e-6)


def test_between_dense_bins_uses_linear_interpolation():
    support = DDFL20
    loss_fn = NonUniformDFLoss(support)
    left = support.index(0.25)
    right = support.index(0.50)
    logits = torch.randn(1, 4, len(support))
    target = torch.full((1, 4), 0.375)  # exact midpoint
    loss = loss_fn(logits, target)
    logp = torch.log_softmax(logits.float(), -1)
    ref = -0.5 * (logp[:, :, left] + logp[:, :, right])
    ref = ref.mean(-1, keepdim=True)
    assert torch.allclose(loss, ref, atol=1e-6, rtol=1e-6)


def test_ddfl20_integral_shape_and_gradient_are_finite():
    torch.manual_seed(2)
    x = torch.randn(3, 80, 101, requires_grad=True)
    y = NonUniformDFLIntegral(DDFL20)(x)
    assert y.shape == (3, 4, 101)
    y.mean().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def test_ddfl20_loss_gradient_is_finite():
    torch.manual_seed(3)
    n = 23
    logits = torch.randn(n, 4, 20, requires_grad=True)
    target = torch.rand(n, 4) * 14.9
    loss = NonUniformDFLoss(DDFL20)(logits, target).mean()
    assert torch.isfinite(loss)
    loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_target_clamp_preserves_yolo11_range():
    loss_fn = NonUniformDFLoss(DDFL20, target_max_margin=0.01)
    target = torch.tensor([[-2.0, 0.0, 15.0, 100.0]])
    clamped = loss_fn.clamp_target(target)
    assert clamped.min().item() >= 0.0
    assert clamped.max().item() <= 14.990001


def test_uniform20_and_ddfl20_have_matched_capacity():
    assert len(UNIFORM20) == len(DDFL20) == 20
    assert head_contract("uniform20")["new_box_output_channels"] == head_contract("ddfl20")["new_box_output_channels"] == 80
