from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn
import torch.nn.functional as F


@dataclass(frozen=True)
class WERGConfig:
    ring_outer: int = 3
    ring_inner: int = 1
    poly_ridge: float = 1.0e-4
    variance_shrinkage: float = 0.10
    variance_floor: float = 1.0e-4
    tangent_ridge_scale: float = 1.0e-2
    tangent_ridge_floor: float = 1.0e-5
    geometry_scale: int = 2
    tensor_window: int = 3
    tensor_rho: float = 1.0e-2
    tensor_eps: float = 1.0e-6
    gradient_confidence_floor: float = 1.0e-3
    correction_gamma: float = 1.5
    detach_statistics: bool = True
    force_fp32_statistics: bool = True


def _annulus_mask(outer: int, inner: int, *, dtype: torch.dtype = torch.float64) -> Tensor:
    if outer < 1:
        raise ValueError("outer must be >= 1")
    if inner < 0 or inner >= outer:
        raise ValueError("inner must satisfy 0 <= inner < outer")
    k = 2 * outer + 1
    mask = torch.zeros((k, k), dtype=dtype)
    for iy, dy in enumerate(range(-outer, outer + 1)):
        for ix, dx in enumerate(range(-outer, outer + 1)):
            if max(abs(dx), abs(dy)) <= outer and max(abs(dx), abs(dy)) > inner:
                mask[iy, ix] = 1.0
    return mask


def build_annular_polynomial_kernels(
    outer: int = 3,
    inner: int = 1,
    ridge: float = 1.0e-4,
    *,
    dtype: torch.dtype = torch.float64,
) -> tuple[Tensor, Tensor]:
    """Return six fixed polynomial-continuation kernels and a uniform annular kernel.

    Basis order: [1, x, y, x^2, x*y, y^2], where x/y are normalized by ``outer``.
    The central square |dx|<=inner, |dy|<=inner is excluded from the fit.
    """
    mask = _annulus_mask(outer, inner, dtype=dtype)
    coords: list[tuple[int, int]] = []
    for iy, dy in enumerate(range(-outer, outer + 1)):
        for ix, dx in enumerate(range(-outer, outer + 1)):
            if mask[iy, ix] > 0:
                coords.append((dy, dx))

    rows = []
    scale = float(outer)
    for dy, dx in coords:
        x = dx / scale
        y = dy / scale
        rows.append([1.0, x, y, x * x, x * y, y * y])
    xmat = torch.tensor(rows, dtype=dtype)
    penalty = torch.diag(torch.tensor([0.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=dtype))
    gram = xmat.T @ xmat + float(ridge) * penalty
    coeff = torch.linalg.solve(gram, xmat.T)  # [6, M]

    k = 2 * outer + 1
    kernels = torch.zeros((6, 1, k, k), dtype=dtype)
    n = 0
    for iy in range(k):
        for ix in range(k):
            if mask[iy, ix] > 0:
                kernels[:, 0, iy, ix] = coeff[:, n]
                n += 1

    ring = (mask / mask.sum()).view(1, 1, k, k)
    return kernels, ring


def _depthwise_fixed_conv(x: Tensor, kernels: Tensor, groups_per_channel: int, pad: int) -> Tensor:
    channels = x.shape[1]
    weight = kernels.to(device=x.device, dtype=x.dtype).repeat(channels, 1, 1, 1)
    x_pad = F.pad(x, (pad, pad, pad, pad), mode="replicate")
    y = F.conv2d(x_pad, weight, groups=channels)
    b, _, h, w = y.shape
    return y.view(b, channels, groups_per_channel, h, w)


class AnnularWaterResidual(nn.Module):
    """Water-explainable residual statistic on a P3-like feature map.

    It fits a fixed annular quadratic continuation independently per channel, whitens the
    center innovation by local ring variance, then removes the component explainable by the
    local first/second-order water tangent basis.
    """

    def __init__(self, channels: int, config: WERGConfig | None = None) -> None:
        super().__init__()
        self.channels = int(channels)
        self.config = config or WERGConfig()
        kernels, ring = build_annular_polynomial_kernels(
            self.config.ring_outer,
            self.config.ring_inner,
            self.config.poly_ridge,
            dtype=torch.float32,
        )
        self.register_buffer("poly_kernels", kernels, persistent=True)
        self.register_buffer("ring_kernel", ring, persistent=True)

    def forward(self, feature: Tensor, *, return_aux: bool = False) -> dict[str, Tensor]:
        if feature.ndim != 4:
            raise ValueError(f"feature must be BCHW, got {tuple(feature.shape)}")
        if feature.shape[1] != self.channels:
            raise ValueError(f"expected {self.channels} channels, got {feature.shape[1]}")

        x = feature.detach() if self.config.detach_statistics else feature
        if self.config.force_fp32_statistics and x.dtype in (torch.float16, torch.bfloat16):
            x = x.float()

        pad = self.config.ring_outer
        coeff = _depthwise_fixed_conv(x, self.poly_kernels, 6, pad)  # B,C,6,H,W
        predicted = coeff[:, :, 0]
        tangent = coeff[:, :, 1:]

        ring_weight = self.ring_kernel.to(device=x.device, dtype=x.dtype).repeat(self.channels, 1, 1, 1)
        x_pad = F.pad(x, (pad, pad, pad, pad), mode="replicate")
        ring_mean = F.conv2d(x_pad, ring_weight, groups=self.channels)
        ring_second = F.conv2d(x_pad * x_pad, ring_weight, groups=self.channels)
        ring_var = (ring_second - ring_mean.square()).clamp_min(0.0)
        global_local_var = ring_var.mean(dim=1, keepdim=True)
        sigma2 = (
            (1.0 - self.config.variance_shrinkage) * ring_var
            + self.config.variance_shrinkage * global_local_var
            + self.config.variance_floor
        )
        precision = sigma2.reciprocal()

        residual = x - predicted
        d = tangent.permute(0, 3, 4, 1, 2).contiguous()  # B,H,W,C,5
        r = residual.permute(0, 2, 3, 1).contiguous()  # B,H,W,C
        wdiag = precision.permute(0, 2, 3, 1).contiguous()  # B,H,W,C

        g = torch.einsum("bhwci,bhwcj,bhwc->bhwij", d, d, wdiag)
        q = torch.einsum("bhwci,bhwc,bhwc->bhwi", d, r, wdiag)
        rwr = torch.einsum("bhwc,bhwc,bhwc->bhw", r, r, wdiag)

        trace = torch.diagonal(g, dim1=-2, dim2=-1).sum(dim=-1)
        lam = (
            self.config.tangent_ridge_scale * trace / float(g.shape[-1])
            + self.config.tangent_ridge_floor
        )
        eye = torch.eye(g.shape[-1], device=g.device, dtype=g.dtype)
        hmat = g + lam[..., None, None] * eye
        alpha = torch.linalg.solve(hmat, q.unsqueeze(-1)).squeeze(-1)
        explained = (q * alpha).sum(dim=-1)
        s_w = (rwr - explained).clamp_min(0.0)

        dof = float(max(self.channels - 5, 1))
        t_w = s_w / dof
        z_w = torch.log1p(t_w).unsqueeze(1)
        result = {"z_w": z_w, "s_w": s_w.unsqueeze(1)}
        if return_aux:
            result.update(
                {
                    "predicted": predicted,
                    "residual": residual,
                    "ring_variance": ring_var,
                    "tangent": tangent,
                }
            )
        return result


def _gaussian_kernel(window: int, sigma: float | None = None) -> Tensor:
    if window % 2 == 0 or window < 1:
        raise ValueError("window must be positive and odd")
    sigma = float(sigma if sigma is not None else max(window / 3.0, 0.8))
    radius = window // 2
    x = torch.arange(-radius, radius + 1, dtype=torch.float32)
    yy, xx = torch.meshgrid(x, x, indexing="ij")
    kernel = torch.exp(-(xx.square() + yy.square()) / (2.0 * sigma * sigma))
    return (kernel / kernel.sum()).view(1, 1, window, window)


def normalized_structure_components(
    a: Tensor,
    b: Tensor,
    c: Tensor,
    *,
    rho: float,
    eps: float,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Normalize a 2x2 SPD tensor by sqrt(det), removing multiplicative contrast scale."""
    trace = (a + c).clamp_min(0.0)
    reg = rho * trace * 0.5 + eps
    ar = a + reg
    cr = c + reg
    det = (ar * cr - b.square()).clamp_min(eps)
    scale = det.sqrt()
    return ar / scale, b / scale, cr / scale, trace


def shape_distance_sq(
    center: tuple[Tensor, Tensor, Tensor],
    ring: tuple[Tensor, Tensor, Tensor],
    *,
    eps: float = 1.0e-6,
) -> Tensor:
    """Affine-invariant distance between determinant-normalized 2x2 SPD tensors.

    Both inputs are component triples (a, b, c) for [[a,b],[b,c]]. When det ~= 1,
    D^2 = 2 * acosh(0.5 * tr(Q_ring^-1 Q_center))^2.
    """
    ca, cb, cc = center
    ra, rb, rc = ring
    det_r = (ra * rc - rb.square()).clamp_min(eps)
    relative_trace = (rc * ca + ra * cc - 2.0 * rb * cb) / det_r
    z = (0.5 * relative_trace).clamp_min(1.0)
    x = (z - 1.0).clamp_min(0.0)
    small = 4.0 * x - (2.0 / 3.0) * x.square()
    regular = 2.0 * torch.acosh(z).square()
    return torch.where(x < 1.0e-4, small.clamp_min(0.0), regular)


class GainInvariantDeformationEvidence(nn.Module):
    """Low-level geometry evidence from RGB via gain-invariant structure-tensor shape."""

    def __init__(self, config: WERGConfig | None = None) -> None:
        super().__init__()
        self.config = config or WERGConfig()
        sobel_x = torch.tensor(
            [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]], dtype=torch.float32
        ) / 8.0
        sobel_y = sobel_x.T.contiguous()
        self.register_buffer("sobel_x", sobel_x.view(1, 1, 3, 3), persistent=True)
        self.register_buffer("sobel_y", sobel_y.view(1, 1, 3, 3), persistent=True)
        self.register_buffer(
            "smooth_kernel", _gaussian_kernel(self.config.tensor_window), persistent=True
        )
        _, ring = build_annular_polynomial_kernels(
            self.config.ring_outer,
            self.config.ring_inner,
            self.config.poly_ridge,
            dtype=torch.float32,
        )
        self.register_buffer("ring_kernel", ring, persistent=True)

    def _smooth(self, x: Tensor) -> Tensor:
        pad = self.config.tensor_window // 2
        xp = F.pad(x, (pad, pad, pad, pad), mode="replicate")
        return F.conv2d(xp, self.smooth_kernel.to(device=x.device, dtype=x.dtype))

    def _ring_average(self, x: Tensor) -> Tensor:
        pad = self.config.ring_outer
        xp = F.pad(x, (pad, pad, pad, pad), mode="replicate")
        return F.conv2d(xp, self.ring_kernel.to(device=x.device, dtype=x.dtype))

    def forward(self, rgb: Tensor, target_hw: tuple[int, int], *, return_aux: bool = False) -> dict[str, Tensor]:
        if rgb.ndim != 4 or rgb.shape[1] != 3:
            raise ValueError(f"rgb must be Bx3xHxW, got {tuple(rgb.shape)}")
        x = rgb.detach() if self.config.detach_statistics else rgb
        if self.config.force_fp32_statistics and x.dtype in (torch.float16, torch.bfloat16):
            x = x.float()

        th, tw = int(target_hw[0]), int(target_hw[1])
        gh = max(th * self.config.geometry_scale, th)
        gw = max(tw * self.config.geometry_scale, tw)
        gray = 0.2989 * x[:, :1] + 0.5870 * x[:, 1:2] + 0.1140 * x[:, 2:3]
        if gray.shape[-2:] != (gh, gw):
            gray = F.interpolate(gray, size=(gh, gw), mode="area")

        gpad = F.pad(gray, (1, 1, 1, 1), mode="replicate")
        gx = F.conv2d(gpad, self.sobel_x.to(device=x.device, dtype=x.dtype))
        gy = F.conv2d(gpad, self.sobel_y.to(device=x.device, dtype=x.dtype))
        a = self._smooth(gx.square())
        b = self._smooth(gx * gy)
        c = self._smooth(gy.square())

        ra = self._ring_average(a)
        rb = self._ring_average(b)
        rc = self._ring_average(c)

        ca_n, cb_n, cc_n, center_trace = normalized_structure_components(
            a, b, c, rho=self.config.tensor_rho, eps=self.config.tensor_eps
        )
        ra_n, rb_n, rc_n, ring_trace = normalized_structure_components(
            ra, rb, rc, rho=self.config.tensor_rho, eps=self.config.tensor_eps
        )
        d2 = shape_distance_sq(
            (ca_n, cb_n, cc_n), (ra_n, rb_n, rc_n), eps=self.config.tensor_eps
        )

        floor = self.config.gradient_confidence_floor
        center_conf = center_trace / (center_trace + floor)
        ring_conf = ring_trace / (ring_trace + floor)
        confidence = (center_conf * ring_conf).clamp(0.0, 1.0).sqrt()
        d2 = d2 * confidence

        if d2.shape[-2:] != (th, tw):
            d2 = F.interpolate(d2, size=(th, tw), mode="area")
            confidence = F.interpolate(confidence, size=(th, tw), mode="area")
        z_g = torch.log1p(d2)
        result = {"z_g": z_g, "d_g2": d2}
        if return_aux:
            result.update({"geometry_confidence": confidence, "gray": gray})
        return result


class WERGLogitCorrection(nn.Module):
    """Shared six-parameter bounded correction for P3 classification logits."""

    def __init__(self, gamma: float = 1.5) -> None:
        super().__init__()
        self.gamma = float(gamma)
        self.coeff = nn.Parameter(torch.zeros(6, dtype=torch.float32))

    def forward(self, logits: Tensor, z_w: Tensor, z_g: Tensor) -> tuple[Tensor, Tensor]:
        if logits.ndim != 4:
            raise ValueError("logits must be BCHW")
        if z_w.shape != z_g.shape or z_w.ndim != 4 or z_w.shape[1] != 1:
            raise ValueError("z_w and z_g must both be Bx1xHxW")
        if logits.shape[0] != z_w.shape[0] or logits.shape[-2:] != z_w.shape[-2:]:
            raise ValueError("evidence maps must align with logits")

        zw = z_w.to(dtype=torch.float32)
        zg = z_g.to(dtype=torch.float32)
        phi = torch.cat(
            [
                torch.ones_like(zw),
                zw,
                zg,
                zw.square(),
                zw * zg,
                zg.square(),
            ],
            dim=1,
        )
        raw = torch.einsum("bkhw,k->bhw", phi, self.coeff).unsqueeze(1)
        delta = self.gamma * torch.tanh(raw / self.gamma)
        corrected = logits + delta.to(dtype=logits.dtype)
        return corrected, delta


class WERGReference(nn.Module):
    """Framework-agnostic WERG v0 reference: P3 feature + RGB + P3 cls logits."""

    def __init__(self, p3_channels: int, config: WERGConfig | None = None) -> None:
        super().__init__()
        self.config = config or WERGConfig()
        self.water = AnnularWaterResidual(p3_channels, self.config)
        self.geometry = GainInvariantDeformationEvidence(self.config)
        self.correct = WERGLogitCorrection(self.config.correction_gamma)

    def forward(
        self,
        p3_feature: Tensor,
        p3_logits: Tensor,
        rgb: Tensor,
        *,
        return_aux: bool = False,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        water = self.water(p3_feature, return_aux=return_aux)
        geometry = self.geometry(rgb, p3_feature.shape[-2:], return_aux=return_aux)
        corrected, delta = self.correct(p3_logits, water["z_w"], geometry["z_g"])
        evidence: dict[str, Tensor] = {
            "z_w": water["z_w"],
            "z_g": geometry["z_g"],
            "delta_logit": delta,
        }
        if return_aux:
            evidence.update({f"water_{k}": v for k, v in water.items() if k != "z_w"})
            evidence.update({f"geom_{k}": v for k, v in geometry.items() if k != "z_g"})
        return corrected, evidence


@torch.no_grad()
def synthetic_sanity_check(device: str = "cpu") -> dict[str, Any]:
    """Cheap deterministic sanity check useful before wiring the module into YOLO."""
    cfg = WERGConfig(detach_statistics=True)
    water = AnnularWaterResidual(8, cfg).to(device)
    feature = torch.zeros(1, 8, 32, 32, device=device)
    base = water(feature)["z_w"].mean().item()
    feature[:, :, 16, 16] = 5.0
    impulse = water(feature)["z_w"][0, 0, 16, 16].item()

    corr = WERGLogitCorrection(cfg.correction_gamma).to(device)
    logits = torch.randn(1, 3, 8, 8, device=device)
    zeros = torch.zeros(1, 1, 8, 8, device=device)
    corrected, _ = corr(logits, zeros, zeros)
    exact_zero_init = bool(torch.equal(logits, corrected))
    return {"flat_water_z": base, "impulse_z": impulse, "zero_init_exact": exact_zero_init}
