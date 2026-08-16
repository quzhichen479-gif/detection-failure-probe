# Round-3 Candidate 2 — Slide Local Preservation + DBRA

> Priority: **2 / Round-3**  
> Primary idea: **parallel local / routed-context evidence**, not attention stacking in series  
> Base: DBRA at P3-Cls-Mid  
> Source: Slide-Transformer, CVPR 2023  
> Official code: https://github.com/LeapLabTHU/Slide-Transformer

## 1. Why the previous serial idea is revised

A naive design would be:

```text
cls block-0 -> Slide -> DBRA -> cls block-1
```

This is no longer the primary design.

Reason: DBRA P3-mid already has positive small-object evidence (`APs +0.0153`, `ARs +0.0075` vs baseline under paper-aligned COCOeval). The main new concern is that `APm/ARm` are lower. If Slide is placed *before* DBRA, it modifies the very input distribution of the mechanism that already works.

A safer and more informative design is:

```text
                       -> DBRA routed-context -----
U = cls block-0(P3) ---                              +-> residual fusion -> block-1
                       -> Slide local evidence -----
```

This preserves DBRA as an explicit branch while asking whether local contiguous evidence contributes complementary information.

## 2. What Slide Attention actually does

Slide Attention is local self-attention. For every query position it attends only to a `k x k` neighborhood.

The official implementation forms Q/K/V, then constructs local K/V neighborhoods using depthwise convolutions. One depthwise kernel is initialized as fixed shifts and frozen; another parallel depthwise kernel is learnable. Their outputs are added, which relaxes the rigid neighborhood shifts while keeping ordinary convolution primitives.

For each position:

\[
a_{p,j}=\operatorname{softmax}_j\left(\frac{q_p^T k_{p,j}}{\sqrt d}+b_j\right)
\]

for local neighbors `j in N_k(p)`, then

\[
y_p=\sum_{j\in N_k(p)}a_{p,j}v_{p,j}.
\]

The official code commonly uses `ka=3`, i.e. a 3x3 local neighborhood.

## 3. Project adaptation: dynamic-resolution NCHW Slide

The official Slide-Swin code stores a fixed `input_resolution`. That is unnecessary and brittle for YOLO, where inference shapes may vary.

The project implementation should infer `H,W` at runtime and use NCHW 1x1 convolutions for per-pixel QKV/projection. A 1x1 Conv2d is equivalent to applying a shared linear projection to every spatial token.

Clean project implementation skeleton:

```python
from __future__ import annotations

import math
import torch
import torch.nn as nn


class SlideAttention2d(nn.Module):
    """Dynamic-resolution NCHW implementation of Slide-style local attention."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        kernel_size: int = 3,
        qkv_bias: bool = True,
        proj_bias: bool = True,
    ):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}")
        if kernel_size % 2 != 1:
            raise ValueError("kernel_size must be odd")

        self.dim = int(dim)
        self.num_heads = int(num_heads)
        self.head_dim = dim // num_heads
        self.k = int(kernel_size)
        self.k2 = self.k * self.k
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Conv2d(dim, 3 * dim, 1, bias=qkv_bias)
        self.proj = nn.Conv2d(dim, dim, 1, bias=proj_bias)

        # Official Slide idea: fixed shift extractor + learnable deformation path.
        self.fixed_shift = nn.Conv2d(
            self.head_dim,
            self.k2 * self.head_dim,
            self.k,
            padding=self.k // 2,
            groups=self.head_dim,
            bias=False,
        )
        self.learned_shift = nn.Conv2d(
            self.head_dim,
            self.k2 * self.head_dim,
            self.k,
            padding=self.k // 2,
            groups=self.head_dim,
            bias=True,
        )

        self.relative_bias = nn.Parameter(
            torch.zeros(1, self.num_heads, 1, self.k2, 1, 1)
        )
        self._init_fixed_shift()

    def _init_fixed_shift(self) -> None:
        kernel = torch.zeros(self.k2, self.k, self.k)
        for i in range(self.k2):
            kernel[i, i // self.k, i % self.k] = 1.0

        # Conv weight layout: [head_dim*k2, 1, k, k]
        weight = kernel.unsqueeze(1).repeat(self.head_dim, 1, 1, 1)
        with torch.no_grad():
            self.fixed_shift.weight.copy_(weight)
        self.fixed_shift.weight.requires_grad_(False)

    def _localize(self, z: torch.Tensor, b: int, h: int, w: int) -> torch.Tensor:
        # z: [B, C, H, W]
        z = z.reshape(b * self.num_heads, self.head_dim, h, w)
        z = self.fixed_shift(z) + self.learned_shift(z)
        return z.reshape(
            b, self.num_heads, self.head_dim, self.k2, h, w
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=1)

        q = q.reshape(b, self.num_heads, self.head_dim, h, w)
        q = q.mul(self.scale).unsqueeze(3)

        k_local = self._localize(k, b, h, w) + self.relative_bias
        v_local = self._localize(v, b, h, w)

        attn = (q * k_local).sum(dim=2, keepdim=True)
        attn = torch.softmax(attn, dim=3)

        y = (attn * v_local).sum(dim=3)
        y = y.reshape(b, c, h, w)
        return self.proj(y)
```

### Important source-fidelity note

This is a project NCHW/dynamic-resolution adaptation of the published mechanism, not a byte-for-byte copy of upstream code. Codex must compare it against the pinned official `slide_swin.py::SlideAttention` semantics before use.

Do not call a plain depthwise 3x3 convolution “Slide Attention.”

## 4. Primary position — parallel Slide / DBRA at P3-Cls-Mid

Graph:

```text
P3 --------------------------------------------------------> box branch
 |
 +-> cls block-0 -> U
                   |\
                   | -> existing DBRA adapter ------ D
                   |
                   + -> SlideAttention2d ----------- L

                   Y = D + beta_local * L
                   |
                   -> cls block-1 -> predictor
```

Because the existing DBRA adapter itself is near-identity, the DBRA branch remains the anchor.

```python
class SlideDBRAParallel(nn.Module):
    def __init__(
        self,
        dbra: nn.Module,
        slide: nn.Module,
        local_scale_init: float = 1e-3,
    ):
        super().__init__()
        self.dbra = dbra
        self.slide = slide
        self.local_scale = nn.Parameter(
            torch.tensor(float(local_scale_init))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        routed = self.dbra(x)
        local = self.slide(x)
        return routed + self.local_scale * local
```

At `local_scale=0`, this model is **exactly the DBRA model**, assuming identical DBRA weights.

Default first-round values:

```text
kernel_size = 3
num_heads   = largest simple divisor <= 4 of active dim
local_scale_init = 1e-3
```

Do not start with 5x5/7x7 neighborhoods; the point of this branch is local preservation, not another large receptive field.

## 5. Secondary position/design — local preconditioner before DBRA

Only after the parallel model has been understood:

```text
P3 -> cls block-0 -> U
                    |
                    -> U_local = U + beta * Slide(U)
                    -> DBRA(U_local)
                    -> cls block-1
```

```python
class SlideThenDBRA(nn.Module):
    def __init__(self, dbra: nn.Module, slide: nn.Module, local_scale_init=1e-3):
        super().__init__()
        self.dbra = dbra
        self.slide = slide
        self.local_scale = nn.Parameter(torch.tensor(float(local_scale_init)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_local = x + self.local_scale * self.slide(x)
        return self.dbra(x_local)
```

At `local_scale=0`, it is exactly DBRA.

This secondary experiment asks whether DBRA routing itself benefits from a locally refined input. It is riskier because it changes DBRA's input distribution.

## 6. Why not use P4

Round-2 already shows DBRA P4-mid is materially worse than P3-mid. Round-3 is testing a companion mechanism to the successful P3 path, so both Slide designs remain P3-only.

Do not use a P4 Slide branch to “repair medium objects” without new evidence. APm is a scale metric, not proof that the defect physically resides in the P4 feature map.

## 7. Mandatory controls if this candidate is positive

A positive Slide+DBRA result does not prove local self-attention is necessary.

Run one simple matched-site control:

```text
DBRA + depthwise 3x3 local residual
```

with comparable local-branch width/cost.

If plain depthwise local evidence matches Slide, the claim should be “local evidence preservation” rather than “Slide Attention is required.”

Also report standalone Slide at the same P3-mid site only if needed for mechanism attribution; do not expand into a large standalone module sweep.

## 8. What to log

```text
local_scale trajectory
DBRA alpha trajectory
local branch activation RMS
routed branch activation RMS
cosine similarity(local, routed)
AP/APs/APm
ARs/ARm
FP @ matched Recall
latency / VRAM
```

A useful diagnostic is whether local and routed branches become nearly identical. If cosine similarity is persistently ~1, the extra branch may be redundant.

## 9. Desired signature

The strongest result is not merely higher overall AP.

Desired pattern:

```text
AP >= DBRA
APs / ARs retained
APm and/or ARm recover toward baseline
matched-recall FP does not worsen
```

If APm recovers but APs collapses, the design has traded away the main DBRA benefit and is not a clean improvement.

## 10. Failure modes

Stop or demote if:

- local branch mainly increases Recall and FP together;
- APm/ARm do not recover and APs/ARs decline;
- learned local scale stays effectively zero;
- the 3x3 DWConv control matches it;
- runtime cost is disproportionate to any gain.

## 11. Research interpretation

Slide itself is existing CVPR 2023 work and is not novelty.

The potentially interesting project-level question is broader:

> Can a classification head preserve local contiguous evidence while separately routing non-local context, instead of forcing one representation to serve both roles?

Only repeated evidence for that local/global complementarity would justify designing a water-specific dual-path module later.
