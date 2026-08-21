import torch

from downsample_modules import PartialPolyphaseRepDown, RepContextDown, SPDDown


def test_output_shapes():
    x = torch.randn(2, 64, 160, 160)
    for module in (
        RepContextDown(64, 128),
        SPDDown(64, 128),
        PartialPolyphaseRepDown(64, 128, 0.25),
    ):
        y = module(x)
        assert y.shape == (2, 128, 80, 80)


def test_reparam_equivalence():
    torch.manual_seed(79)
    module = RepContextDown(64, 128).eval()
    x = torch.randn(1, 64, 64, 64)
    with torch.no_grad():
        y_train_graph = module(x)
        module.switch_to_deploy()
        y_deploy_graph = module(x)
    assert torch.max(torch.abs(y_train_graph - y_deploy_graph)).item() < 1e-5


def test_pprd_deploy_equivalence():
    torch.manual_seed(79)
    module = PartialPolyphaseRepDown(64, 128, 0.25).eval()
    x = torch.randn(1, 64, 64, 64)
    with torch.no_grad():
        y_before = module(x)
        module.switch_to_deploy()
        y_after = module(x)
    assert torch.max(torch.abs(y_before - y_after)).item() < 1e-5
