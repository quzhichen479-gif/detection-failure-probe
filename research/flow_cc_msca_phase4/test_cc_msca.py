from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from cc_msca import ContextContrastMSCA, SegNeXtMSCA


@pytest.mark.parametrize(
    ("module", "expected_params"),
    [
        (SegNeXtMSCA(64), 11200),
        (ContextContrastMSCA(64), 2771),
    ],
)
def test_shape_finite_backward_and_parameter_budget(module, expected_params):
    x = torch.randn(2, 64, 33, 35, requires_grad=True)
    y = module(x)

    assert y.shape == x.shape
    assert torch.isfinite(y).all()
    assert sum(p.numel() for p in module.parameters()) == expected_params

    y.mean().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def test_cc_msca_is_exact_identity_at_default_initialization():
    module = ContextContrastMSCA(32)
    x = torch.randn(2, 32, 19, 23)
    y = module(x)
    assert torch.equal(y, x)


def test_cc_msca_constant_feature_has_zero_context_contrast_initially():
    module = ContextContrastMSCA(16)
    x = torch.ones(1, 16, 17, 21)
    local, residual, contrast, anisotropy, weights = module.compute_descriptors(x)

    assert torch.allclose(local, x)
    assert torch.count_nonzero(residual) == 0
    assert torch.count_nonzero(contrast) == 0
    assert torch.count_nonzero(anisotropy) == 0
    assert torch.allclose(weights, torch.full_like(weights, 1.0 / 3.0))


def test_cc_msca_internal_path_receives_gradients_after_residual_wakes_up():
    module = ContextContrastMSCA(16)
    with torch.no_grad():
        module.gamma.fill_(1e-3)

    x = torch.randn(2, 16, 19, 23, requires_grad=True)
    module(x).square().mean().backward()

    assert module.local.weight.grad is not None
    assert module.local.weight.grad.abs().sum() > 0
    assert module.gate[-1].weight.grad is not None
    assert module.gate[-1].weight.grad.abs().sum() > 0
    assert module.scale_logits.grad is not None
    assert module.scale_logits.grad.abs().sum() > 0


@pytest.mark.parametrize("kernels", [(3, 4, 7), (3, 3, 7), (1, 3, 5)])
def test_cc_msca_rejects_invalid_context_kernels(kernels):
    with pytest.raises(ValueError):
        ContextContrastMSCA(32, context_kernels=kernels)


def test_msca_rejects_even_local_kernel():
    with pytest.raises(ValueError):
        SegNeXtMSCA(32, local_kernel=4)
