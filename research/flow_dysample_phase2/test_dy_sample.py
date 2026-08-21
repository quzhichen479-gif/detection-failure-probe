from __future__ import annotations

import torch
import torch.nn.functional as F

from dy_sample import DySample


def test_lp_shape_and_finite_backward() -> None:
    x = torch.randn(2, 64, 7, 9, requires_grad=True)
    module = DySample(64, scale=2, style="lp", groups=4, dyscope=False)
    y = module(x)
    assert y.shape == (2, 64, 14, 18)
    assert torch.isfinite(y).all()
    y.square().mean().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def test_pl_shape() -> None:
    x = torch.randn(1, 64, 5, 8)
    module = DySample(64, scale=2, style="pl", groups=4, dyscope=False)
    assert module(x).shape == (1, 64, 10, 16)


def test_dyscope_shape() -> None:
    x = torch.randn(1, 64, 5, 8)
    module = DySample(64, scale=2, style="lp", groups=4, dyscope=True)
    assert module(x).shape == (1, 64, 10, 16)


def test_zero_offset_matches_regular_bilinear_grid() -> None:
    x = torch.randn(1, 64, 5, 8)
    module = DySample(64, scale=2, style="lp", groups=4, dyscope=False).eval()
    torch.nn.init.zeros_(module.offset.weight)
    torch.nn.init.zeros_(module.offset.bias)

    y = module(x)
    reference = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
    torch.testing.assert_close(y, reference, atol=1e-6, rtol=1e-6)


def test_invalid_group_configuration_is_rejected() -> None:
    try:
        DySample(62, groups=4)
    except ValueError:
        return
    raise AssertionError("DySample must reject channels that are not divisible by groups")
