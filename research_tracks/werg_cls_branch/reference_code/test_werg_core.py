import torch

from werg_core import (
    AnnularWaterResidual,
    GainInvariantDeformationEvidence,
    WERGConfig,
    WERGLogitCorrection,
    normalized_structure_components,
    shape_distance_sq,
)


def test_water_residual_is_finite_nonnegative_and_detects_impulse():
    cfg = WERGConfig(detach_statistics=True)
    module = AnnularWaterResidual(4, cfg)
    x = torch.zeros(1, 4, 25, 25)
    flat = module(x)["z_w"]
    assert torch.isfinite(flat).all()
    assert (flat >= 0).all()
    x[:, :, 12, 12] = 3.0
    out = module(x)["z_w"]
    assert out[0, 0, 12, 12] > flat.mean() + 0.1


def test_quadratic_surface_has_small_center_residual_interior():
    cfg = WERGConfig(poly_ridge=1e-8, detach_statistics=True)
    module = AnnularWaterResidual(1, cfg)
    yy, xx = torch.meshgrid(torch.linspace(-1, 1, 41), torch.linspace(-1, 1, 41), indexing="ij")
    field = 0.7 + 0.2 * xx - 0.3 * yy + 0.4 * xx.square() + 0.1 * xx * yy - 0.2 * yy.square()
    x = field[None, None]
    aux = module(x, return_aux=True)
    residual = aux["residual"][..., 5:-5, 5:-5].abs().max()
    assert residual < 1e-4


def test_logit_correction_is_exact_identity_at_initialization():
    corr = WERGLogitCorrection(gamma=1.5)
    logits = torch.randn(2, 5, 16, 16)
    zw = torch.rand(2, 1, 16, 16)
    zg = torch.rand(2, 1, 16, 16)
    corrected, delta = corr(logits, zw, zg)
    assert torch.equal(corrected, logits)
    assert torch.count_nonzero(delta) == 0


def test_shape_distance_zero_for_isotropic_gain_positive_for_anisotropy():
    one = torch.ones(1, 1, 1, 1)
    zero = torch.zeros_like(one)
    q1 = normalized_structure_components(one, zero, one, rho=0.0, eps=1e-8)[:3]
    q2 = normalized_structure_components(4.0 * one, zero, 4.0 * one, rho=0.0, eps=1e-8)[:3]
    d_iso = shape_distance_sq(q2, q1)
    assert d_iso.max() < 1e-5

    a = 2.0 * one
    b = 0.5 * one
    c = 0.75 * one
    qa = normalized_structure_components(a, b, c, rho=0.0, eps=1e-8)[:3]
    d_aniso = shape_distance_sq(qa, q1)
    assert d_aniso.min() > 0.05


def test_geometry_evidence_is_approximately_affine_intensity_invariant_without_clipping():
    cfg = WERGConfig(geometry_scale=1, detach_statistics=True)
    module = GainInvariantDeformationEvidence(cfg)
    yy, xx = torch.meshgrid(torch.linspace(0, 1, 64), torch.linspace(0, 1, 64), indexing="ij")
    gray = 0.25 + 0.15 * torch.sin(9.0 * xx + 3.0 * yy) + 0.1 * torch.cos(5.0 * yy)
    rgb = gray[None, None].repeat(1, 3, 1, 1)
    rgb2 = 0.6 * rgb + 0.2
    z1 = module(rgb, (32, 32))["z_g"]
    z2 = module(rgb2, (32, 32))["z_g"]
    diff = (z1[..., 4:-4, 4:-4] - z2[..., 4:-4, 4:-4]).abs().mean()
    assert diff < 2e-2
