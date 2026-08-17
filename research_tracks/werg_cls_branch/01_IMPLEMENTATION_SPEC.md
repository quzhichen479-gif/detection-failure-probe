# WERG v0 — Implementation Specification

## 1. Frozen architectural scope

WERG v0 is inserted only into the **accepted DBRA P3 classification path**.

```text
accepted P3 cls feature (after accepted DBRA point)
          │
          ├─ parent cls predictor ──────────────── p3_logits
          │
          └─ AnnularWaterResidual ─────────────── z_w

same normalized RGB detector input ─ GeometryEvidence ─ z_g

[p3_logits, z_w, z_g] ─ six-parameter bounded correction ─ corrected P3 logits
```

P4/P5 classification paths are unchanged. Every regression path is unchanged. DFL/TAL/box loss is unchanged.

**SRB-IoU is explicitly prohibited. It has already been falsified and is not part of WERG.**

## 2. v0 default constants

Reference defaults are intentionally conservative and must be frozen for the first registered probe/training:

```text
ring_outer = 3                 # 7x7 support
ring_inner = 1                 # exclude center 3x3 -> 40 ring cells
poly_ridge = 1e-4
variance_shrinkage = 0.10
variance_floor = 1e-4
tangent_ridge_scale = 1e-2
tangent_ridge_floor = 1e-5
geometry_scale = 2             # compute geometry at 2x P3 then area reduce
tensor_window = 3
tensor_rho = 1e-2
tensor_eps = 1e-6
gradient_confidence_floor = 1e-3
correction_gamma = 1.5
detach_statistics = True
force_fp32_statistics = True
```

Do not parameter-sweep these values before the mechanism probe. If v0 fails, do not rescue it with a search.

## 3. Annular kernels

Use basis

```text
[1, x, y, x^2, x*y, y^2]
```

with coordinates normalized by `ring_outer`. Build the fixed ridge solution once and register its six 7x7 kernels as non-trainable buffers. Use replicate padding to avoid a zero-padding anomaly ring at feature-map borders.

The annulus is the 7x7 square excluding the central 3x3 square. This is a deliberate anti-contamination choice for tiny objects that can occupy more than one P3 cell.

## 4. Water residual numerical implementation

For each P3 location:

1. compute the six polynomial coefficients via grouped fixed convolutions;
2. predict the center by coefficient 0;
3. compute annular per-channel variance via fixed uniform annular convolution;
4. apply shrinkage and a positive floor;
5. construct the five local tangent vectors from coefficients 1..5;
6. assemble a 5x5 Gram matrix and 5-vector `q`;
7. solve `(G + lambda I) alpha = q` using `torch.linalg.solve`, never explicit inverse;
8. clamp only the final tiny floating-point negative residue to zero;
9. normalize by `max(C-5,1)` and use `log1p`.

The reference implementation uses full 5x5 projection. Do not silently replace it with a diagonal approximation in the first integration. A cheaper approximation can only be introduced later as an attribution/efficiency experiment after the full statistic is validated.

## 5. Geometry branch numerical implementation

1. consume the same normalized RGB tensor that enters the detector;
2. convert to fixed luminance `0.2989 R + 0.5870 G + 0.1140 B`;
3. area-resize to `geometry_scale * P3_size`;
4. fixed Sobel gradients;
5. local Gaussian smoothing of `(gx^2, gx*gy, gy^2)`;
6. compute center and annular-average 2x2 tensors;
7. regularize by `rho * trace/2 + eps`;
8. determinant-normalize each tensor;
9. compute the 2x2 closed-form affine-invariant distance;
10. multiply by low-gradient confidence;
11. area-reduce to P3 and `log1p`.

Do not use a learned RGB stem in v0. The point of `z_g` is to retain a testable low-level physical statistic.

## 6. Gradient boundary

v0 defaults to

```text
z_w = stop-gradient statistic
z_g = stop-gradient statistic
```

Only the six correction coefficients train. This prevents the backbone from learning to manipulate the diagnostic statistic itself and makes the first positive result much harder to explain by added capacity.

An end-to-end WERG version is a later experiment only if the detached version first demonstrates conditional information.

## 7. Logit correction

Use exactly six coefficients:

```text
[1, z_w, z_g, z_w^2, z_w*z_g, z_g^2]
```

and

```text
delta = gamma * tanh(raw / gamma)
corrected_p3_logits = parent_p3_logits + delta
```

The same scalar `delta` is broadcast to all classes at the same P3 location. Initialize all coefficients to zero and verify bitwise identity against the parent P3 logits.

## 8. Required unit/integration invariants

Before training:

```text
[ ] quadratic feature field -> near-zero interior continuation residual
[ ] flat feature -> finite non-negative z_w
[ ] center impulse -> z_w increases
[ ] isotropic structure-tensor scaling -> D_G^2 approximately zero
[ ] anisotropic tensor deformation -> D_G^2 positive
[ ] I -> aI+b without clipping -> z_g approximately invariant
[ ] zero correction coefficients -> bitwise parent P3 logits
[ ] P4/P5 logits exactly unchanged
[ ] regression tensors exactly unchanged
[ ] all WERG evidence finite under AMP parent inputs (statistics may upcast to fp32)
[ ] backward shows gradients only in the six WERG coefficients when detach_statistics=True
```

## 9. Complexity reporting

Report separately:

- fixed-convolution cost for six annular polynomial kernels;
- 5x5 local projection solve cost;
- RGB geometry sidecar cost;
- six-parameter correction cost;
- measured batch-1 latency P50/P95 and VRAM, not FLOPs alone.

The full statistic is intentionally more expensive than a trivial attention gate. If the probe passes but runtime is unacceptable, optimize only after equivalence is demonstrated.
