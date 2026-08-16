"""Reference tests for the Round-4 FreqFusion YOLO adapter.

These tests validate the project-authored wrapper contract without copying or requiring the
third-party FreqFusion implementation. The real YOLO engineering repository must add
additional tests against the pinned upstream operator.
"""

from __future__ import annotations

import torch
import torch.nn as nn

import freqfusion_yolo_adapter as adapter_mod


class DummyFreqFusion(nn.Module):
    """Minimal differentiable stand-in matching the upstream call/return contract."""

    def __init__(self, hr_channels: int, lr_channels: int, **kwargs):
        super().__init__()
        self.hr_proj = nn.Conv2d(hr_channels, hr_channels, 1)
        self.lr_proj = nn.Conv2d(lr_channels, lr_channels, 1)

    def forward(self, hr_feat, lr_feat, use_checkpoint=False):
        del use_checkpoint
        hr_refined = self.hr_proj(hr_feat)
        lr_up = torch.nn.functional.interpolate(lr_feat, scale_factor=2, mode="nearest")
        lr_up = self.lr_proj(lr_up)
        mask = hr_refined[:, :1]
        return mask, hr_refined, lr_up


def _install_dummy():
    adapter_mod.FreqFusion = DummyFreqFusion


def test_parse_helper_preserves_hr_lr_order_and_channel_sum():
    channels = [3, 16, 32, 64, 128, 256]
    parsed, out_c = adapter_mod.infer_freqfusion_parse_args(
        channel_table=channels,
        from_indices=[4, 3],
        yaml_args=[],
    )
    assert parsed[:2] == [128, 64]
    assert out_c == 192


def test_freqfusion_concat_shape():
    _install_dummy()
    module = adapter_mod.FreqFusionConcat(hr_channels=32, lr_channels=48)
    hr = torch.randn(2, 32, 40, 40)
    lr = torch.randn(2, 48, 20, 20)
    out = module([hr, lr])
    assert out.shape == (2, 80, 40, 40)


def test_freqfusion_concat_rejects_reversed_or_wrong_spatial_ratio():
    _install_dummy()
    module = adapter_mod.FreqFusionConcat(hr_channels=32, lr_channels=48)
    hr = torch.randn(1, 32, 40, 40)
    lr = torch.randn(1, 48, 21, 20)
    try:
        module([hr, lr])
    except ValueError as exc:
        assert "2:1" in str(exc)
    else:
        raise AssertionError("invalid spatial ratio was not rejected")


def test_gradients_reach_both_wrapper_branches():
    _install_dummy()
    module = adapter_mod.FreqFusionConcat(hr_channels=16, lr_channels=24)
    hr = torch.randn(2, 16, 32, 32, requires_grad=True)
    lr = torch.randn(2, 24, 16, 16, requires_grad=True)

    loss = module([hr, lr]).square().mean()
    loss.backward()

    assert hr.grad is not None and torch.isfinite(hr.grad).all()
    assert lr.grad is not None and torch.isfinite(lr.grad).all()
    for p in module.parameters():
        assert p.grad is not None
        assert torch.isfinite(p.grad).all()


def test_output_channel_metadata():
    _install_dummy()
    module = adapter_mod.FreqFusionConcat(hr_channels=64, lr_channels=64)
    assert module.out_channels == 128
