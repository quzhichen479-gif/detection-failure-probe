# Round-3 Candidate 3 — DBRA + Focal Modulation Lite

> Priority: **3 / conditional Round-3**  
> Purpose: test query-conditioned context modulation that is complementary to routing  
> Base: DBRA P3-Cls-Mid  
> Source: Focal Modulation Networks / FocalNet, NeurIPS 2022  
> Official code: https://github.com/microsoft/FocalNet

## 1. Why this candidate is deliberately lower priority

Focal Modulation is attractive because it does not perform pairwise self-attention. It builds hierarchical context with depthwise convolutions, uses content-dependent gates to aggregate context, then modulates the query feature.

However, this project already has direct negative evidence against indiscriminate large-context enhancement:

- LSK is strongly negative;
- CAA does not beat baseline overall;
- DBRA already supplies non-local contextual selection.

Therefore the claim

> “DBRA still needs more context”

is not justified.

Focal Modulation is retained only as a **conditional complement test** because its context is gated per query instead of being a fixed large receptive field.

## 2. Original mechanism

The official FocalNet implementation projects each token into:

```text
q
context feature
focal-level gates + global-context gate
```

For focal levels `l`:

\[
C_l = DWConv_l(C_{l-1})
\]

and gated aggregation:

\[
C_{agg}=\sum_l g_l(X)\odot C_l + g_g(X)\odot C_{global}.
\]

A 1x1 projection produces the modulator:

\[
M=h(C_{agg})
\]

and output:

\[
Y=q(X)\odot M.
\]

The official implementation supports `normalize_modulator`, which divides the aggregated context by the number of focal/global levels.

## 3. Project adaptation policy

Round-3 must **not** use a large FocalNet-style stack.

First implementation is deliberately constrained:

```text
focal_level = 1
focal_window = 3
focal_factor = 2
normalize_modulator = True
```

This means one local depthwise contextualization level plus the original global-context gate.

Do not begin with focal levels 2/3 or large kernels. That would recreate the already-risky “more context” search space.

## 4. Dynamic NCHW implementation skeleton

```python
from __future__ import annotations

import torch
import torch.nn as nn


class FocalModulation2dLite(nn.Module):
    """NCHW adaptation of FocalNet's focal modulation operator."""

    def __init__(
        self,
        dim: int,
        focal_window: int = 3,
        focal_level: int = 1,
        focal_factor: int = 2,
        normalize_modulator: bool = True,
    ):
        super().__init__()
        self.dim = int(dim)
        self.focal_level = int(focal_level)
        self.normalize_modulator = bool(normalize_modulator)

        self.pre = nn.Conv2d(
            dim,
            2 * dim + self.focal_level + 1,
            1,
            bias=True,
        )

        layers = []
        for level in range(self.focal_level):
            k = focal_window + focal_factor * level
            if k % 2 != 1:
                raise ValueError("focal kernels must be odd")
            layers.append(
                nn.Sequential(
                    nn.Conv2d(
                        dim,
                        dim,
                        kernel_size=k,
                        padding=k // 2,
                        groups=dim,
                        bias=False,
                    ),
                    nn.GELU(),
                )
            )
        self.focal_layers = nn.ModuleList(layers)

        self.modulator_proj = nn.Conv2d(dim, dim, 1, bias=True)
        self.out_proj = nn.Conv2d(dim, dim, 1, bias=True)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        packed = self.pre(x)
        q, ctx, gates = torch.split(
            packed,
            [self.dim, self.dim, self.focal_level + 1],
            dim=1,
        )

        ctx_all = torch.zeros_like(ctx)
        current = ctx

        for level, layer in enumerate(self.focal_layers):
            current = layer(current)
            ctx_all = ctx_all + current * gates[:, level : level + 1]

        ctx_global = self.act(current.mean(dim=(2, 3), keepdim=True))
        ctx_all = ctx_all + ctx_global * gates[:, self.focal_level :]

        if self.normalize_modulator:
            ctx_all = ctx_all / float(self.focal_level + 1)

        modulator = self.modulator_proj(ctx_all)
        y = q * modulator
        return self.out_proj(y)
```

This is a clean NCHW adaptation of the operator structure. Codex must compare it against the pinned official `classification/focalnet.py::FocalModulation` before use.

## 5. Primary design — parallel weak FocalMod branch with DBRA

Do **not** make the primary model:

```text
DBRA -> full FocalMod -> classifier
```

That serial design applies a second context transformation to every routed feature and makes it difficult to preserve the DBRA behavior that already works.

Primary graph:

```text
P3 -> cls block-0 -> U
                    |\
                    | -> DBRA ------------------- D
                    |
                    + -> FocalModLite ----------- M

                    Y = D + beta_focal * M
                    -> cls block-1 -> predictor
```

```python
class FocalDBRAParallel(nn.Module):
    def __init__(
        self,
        dbra: nn.Module,
        focal: nn.Module,
        focal_scale_init: float = 1e-3,
    ):
        super().__init__()
        self.dbra = dbra
        self.focal = focal
        self.focal_scale = nn.Parameter(
            torch.tensor(float(focal_scale_init))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        routed = self.dbra(x)
        modulation = self.focal(x)
        return routed + self.focal_scale * modulation
```

At `focal_scale=0`, it exactly reproduces DBRA for identical DBRA weights.

Why parallel is preferred:

- DBRA remains an explicit anchor;
- the Focal branch can be learned as a small correction;
- it tests complementary context rather than replacing/rerouting the representation;
- it is easier to ablate and interpret.

## 6. Secondary design — weak post-DBRA focal modulation

If the parallel design gives a real validation signal, the second predefined design is:

```text
U -> DBRA -> D
D -> D + beta * FocalModLite(D)
  -> cls block-1
```

```python
class DBRAThenFocal(nn.Module):
    def __init__(self, dbra: nn.Module, focal: nn.Module, focal_scale_init=1e-3):
        super().__init__()
        self.dbra = dbra
        self.focal = focal
        self.focal_scale = nn.Parameter(torch.tensor(float(focal_scale_init)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        routed = self.dbra(x)
        return routed + self.focal_scale * self.focal(routed)
```

This is explicitly higher risk because FocalMod contextualizes a feature that has already undergone routed context selection.

## 7. Why Focal-before-DBRA is not a preferred position

```text
FocalMod -> DBRA
```

would alter DBRA's input using local/global context before routing. Given the project's LSK/CAA failures, this direction has a higher chance of contaminating the routing input with structured water context.

It is not one of the two predefined Round-3 positions.

## 8. Mandatory monitoring

```text
focal_scale trajectory
per-level gate mean/std
ratio of global-context gate to local gate
modulation RMS
DBRA/focal branch cosine similarity
AP/APs/APm
ARs/ARm
FP @ matched Recall
latency / VRAM
```

The global-context gate is especially important. If it dominates immediately, the model may be recreating the broad-context behavior already warned against by earlier experiments.

## 9. Success condition

Focal+DBRA is only interesting if it improves DBRA without sacrificing its small-object signature.

Desired:

```text
AP > DBRA
APs >= DBRA - small tolerance
ARs >= DBRA - small tolerance
APm and/or matched-recall FP improves
```

A gain only on APl/ARl is insufficient because DBRA already improved large-object metrics while the current research target is not large-object specialization.

## 10. Stop conditions

Stop if:

- `focal_scale` stays effectively zero;
- APs/ARs fall while only large-object metrics rise;
- global gate dominates and water-background FP increases;
- result is no better than a matched-cost 3x3 depthwise residual branch;
- Slide/GRN already solves the observed deficit more simply;
- latency becomes disproportionate to the gain.

## 11. Execution priority

Implement this module together with the other Round-3 candidates so engineering is ready, but do **not** automatically spend a full training run on it first.

Recommended training priority:

```text
1. DBRA + GRN
2. Slide-parallel + DBRA
3. FocalMod-parallel + DBRA only if still justified by validation evidence/budget
```

If both higher-priority hypotheses fail strongly, do not assume FocalMod will rescue DBRA simply because it is a different module.

## 12. Novelty boundary

Focal Modulation is existing NeurIPS 2022 work. `DBRA + FocalMod` is a combination experiment, not a paper contribution by itself.
