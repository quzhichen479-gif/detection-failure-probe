from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from attention_modules import CAAAttention, EMAAttention, TripletAttention


@pytest.mark.parametrize(
    ("module", "expected_params"),
    [
        (EMAAttention(64, factor=8), 672),
        (CAAAttention(64, kernel_size=11), 9856),
        (TripletAttention(64, kernel_size=7), 300),
    ],
)
def test_shape_backward_and_parameter_budget(module, expected_params):
    x = torch.randn(2, 64, 80, 80, requires_grad=True)
    y = module(x)

    assert y.shape == x.shape
    assert torch.isfinite(y).all()
    assert sum(p.numel() for p in module.parameters()) == expected_params

    y.mean().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def test_ema_requires_divisible_groups():
    with pytest.raises(ValueError):
        EMAAttention(60, factor=8)


def test_caa_requires_odd_kernel():
    with pytest.raises(ValueError):
        CAAAttention(64, kernel_size=10)


def test_triplet_requires_positive_channels():
    with pytest.raises(ValueError):
        TripletAttention(0)
