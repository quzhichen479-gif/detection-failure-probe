from __future__ import annotations

import pytest
import torch

from large_kernel_context import (
    CEPConvLKC,
    CenterExcludedPeripheralConv,
    DilatedReparamDW,
    StripLKC,
    UniRepLKControl,
)


@pytest.mark.parametrize("module_cls", [UniRepLKControl, StripLKC, CEPConvLKC])
def test_phase5_blocks_preserve_shape_and_start_as_identity(module_cls):
    torch.manual_seed(0)
    x = torch.randn(2, 16, 32, 32)
    module = module_cls(16)
    y = module(x)
    assert y.shape == x.shape
    torch.testing.assert_close(y, x, rtol=0.0, atol=0.0)


def test_strip_lkc_descriptors_have_expected_shape():
    x = torch.randn(1, 8, 24, 24)
    module = StripLKC(8, strip_kernel=17)
    descriptors = module.compute_descriptors(x)
    assert len(descriptors) == 5
    assert all(t.shape == x.shape for t in descriptors)


def test_center_excluded_kernel_is_exactly_zero_in_core():
    module = CenterExcludedPeripheralConv(4, kernel_size=17, center_kernel=5)
    kernel = module.effective_kernel()
    center = kernel.shape[-1] // 2
    core = kernel[:, center - 2 : center + 3, center - 2 : center + 3]
    torch.testing.assert_close(core, torch.zeros_like(core), rtol=0.0, atol=0.0)
    assert module.num_bins >= 2


def test_center_excluded_peripheral_conv_deploy_equivalence():
    torch.manual_seed(1)
    x = torch.randn(2, 6, 20, 20)
    module = CenterExcludedPeripheralConv(6, kernel_size=17, center_kernel=5).eval()
    y_train = module(x)
    module.switch_to_deploy()
    y_deploy = module(x)
    torch.testing.assert_close(y_train, y_deploy, rtol=1e-5, atol=1e-6)


def test_dilated_reparam_deploy_equivalence():
    torch.manual_seed(2)
    x = torch.randn(2, 8, 24, 24)
    module = DilatedReparamDW(8, kernel_size=17).eval()
    y_train = module(x)
    module.switch_to_deploy()
    y_deploy = module(x)
    torch.testing.assert_close(y_train, y_deploy, rtol=1e-4, atol=2e-5)


def test_unireplk_control_deploy_equivalence_with_active_residual():
    torch.manual_seed(3)
    x = torch.randn(1, 8, 24, 24)
    module = UniRepLKControl(8, kernel_size=17, gamma_init=0.2).eval()
    before = module(x)
    module.switch_to_deploy()
    after = module(x)
    torch.testing.assert_close(before, after, rtol=1e-4, atol=2e-5)


@pytest.mark.parametrize("module_cls", [UniRepLKControl, StripLKC, CEPConvLKC])
def test_phase5_blocks_backprop_when_residual_is_active(module_cls):
    torch.manual_seed(4)
    x = torch.randn(2, 8, 16, 16, requires_grad=True)
    module = module_cls(8, gamma_init=0.1)
    loss = module(x).square().mean()
    loss.backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    grads = [p.grad for p in module.parameters() if p.requires_grad and p.grad is not None]
    assert grads
    assert all(torch.isfinite(g).all() for g in grads)


def test_invalid_center_kernel_rejected():
    with pytest.raises(ValueError):
        CenterExcludedPeripheralConv(8, kernel_size=17, center_kernel=17)


def test_yolo11_wrapper_changes_only_p3_classification_path():
    from torch import nn
    from yolo11_integration import ContextBeforeTower, wrap_detect_p3_classification

    class FakeDetect(nn.Module):
        def __init__(self):
            super().__init__()
            self.cv2 = nn.ModuleList([nn.Identity(), nn.Identity(), nn.Identity()])
            self.cv3 = nn.ModuleList([nn.Identity(), nn.Identity(), nn.Identity()])

    detect = FakeDetect()
    old_p4 = detect.cv3[1]
    old_p5 = detect.cv3[2]
    wrap_detect_p3_classification(detect, "strip", channels=8)
    assert isinstance(detect.cv3[0], ContextBeforeTower)
    assert detect.cv3[1] is old_p4
    assert detect.cv3[2] is old_p5
    x = torch.randn(1, 8, 16, 16)
    torch.testing.assert_close(detect.cv3[0](x), x, rtol=0.0, atol=0.0)


def test_switch_phase5_to_deploy_materializes_reparam_modules():
    from torch import nn
    from yolo11_integration import switch_phase5_to_deploy

    model = nn.Sequential(UniRepLKControl(4, gamma_init=0.1), CEPConvLKC(4, gamma_init=0.1)).eval()
    x = torch.randn(1, 4, 20, 20)
    before = model(x)
    switch_phase5_to_deploy(model)
    after = model(x)
    assert model[0].context.deploy
    assert model[1].peripheral.deploy
    torch.testing.assert_close(before, after, rtol=2e-4, atol=3e-5)
