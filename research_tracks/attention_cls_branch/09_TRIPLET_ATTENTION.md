# Triplet Attention for YOLO11 Classification Branch

> Candidate priority: **3 / Round-2**  
> Source: Rotate to Attend: Convolutional Triplet Attention Module, WACV 2021  
> Official repo: https://github.com/LandskapeAI/triplet-attention  
> Paper: https://openaccess.thecvf.com/content/WACV2021/html/Misra_Rotate_to_Attend_Convolutional_Triplet_Attention_Module_WACV_2021_paper.html

## 1. Why this is a useful counterfactual

Round-1 results are:

```text
LSK: strong degradation
CAA: AP75 gain but overall mAP loss
BRA: best candidate, higher Recall/AP50/AP75 but lower Precision
```

The project therefore needs a candidate that **does not deliberately enlarge context**. Triplet Attention is useful because it focuses on cross-dimensional interaction among channel and spatial axes using lightweight convolutional gates.

The falsifiable question is:

> Is long-range/global context actually necessary, or can better C-H / C-W / H-W feature selection already improve classification purity?

If Triplet performs competitively with BRA at much lower cost, the project should not over-interpret BRA's routing as the essential mechanism.

## 2. Core mechanism

Triplet Attention constructs three attention branches by permuting dimensions so that channel-spatial dependencies become spatial attention problems.

A simple channel pooling operator is:

\[
Z(X)=\left[\max_c X,\operatorname{mean}_c X\right].
\]

A spatial gate then computes:

\[
G(X)=X\odot\sigma\left(Conv_{k\times k}(Z(X))\right).
\]

The three branches capture roughly:

- channel-height interaction;
- channel-width interaction;
- ordinary height-width spatial interaction.

The branch outputs are rotated back and averaged.

For this project use a near-identity interpolation:

\[
Y=X+\alpha(TA(X)-X).
\]

## 3. Two insertion positions

### Position 1 — P3-Cls-Pre **(first priority)**

```text
P3 neck feature
   |-----------------------------> box tower unchanged
   |
   -> Triplet Attention
       -> complete YOLO cls tower
           -> class predictor
```

Why first:

- Triplet does not require a very large receptive field;
- directly tests whether raw P3 channel-spatial interactions can suppress misleading water responses before classification;
- gives a mechanism clearly different from BRA/DBRA/SHSA;
- leaves box regression untouched.

### Position 2 — P3-Cls-Mid

```text
P3
 |-----------------------------> box tower unchanged
 |
 -> cls block-0
      -> Triplet Attention
          -> cls block-1
              -> class predictor
```

Why second:

- tests whether the raw P3 feature is too noisy for cross-dimensional gating;
- preserves the same P3 target scale while moving attention after local semantic mixing.

Unlike DBRA/SHSA, P4 is not the preferred second position because Triplet's main value here is specifically to test **P3 feature selection without larger/global context**.

## 4. Project implementation

```python
from __future__ import annotations

import torch
import torch.nn as nn


class ZPool(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat(
            [
                torch.amax(x, dim=1, keepdim=True),
                torch.mean(x, dim=1, keepdim=True),
            ],
            dim=1,
        )


class TripletSpatialGate(nn.Module):
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        if kernel_size % 2 != 1:
            raise ValueError("kernel_size must be odd")
        self.compress = ZPool()
        self.conv = nn.Conv2d(
            2,
            1,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            bias=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = torch.sigmoid(self.conv(self.compress(x)))
        return x * scale


class TripletAttentionCore(nn.Module):
    """NCHW Triplet Attention using three cross-dimensional branches."""

    def __init__(self, kernel_size: int = 7, no_spatial: bool = False):
        super().__init__()
        self.cw = TripletSpatialGate(kernel_size)
        self.hc = TripletSpatialGate(kernel_size)
        self.hw = None if no_spatial else TripletSpatialGate(kernel_size)
        self.no_spatial = no_spatial

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # C-W interaction: N,C,H,W -> N,H,C,W
        x_perm1 = x.permute(0, 2, 1, 3).contiguous()
        y1 = self.cw(x_perm1)
        y1 = y1.permute(0, 2, 1, 3).contiguous()

        # C-H interaction: N,C,H,W -> N,W,H,C
        x_perm2 = x.permute(0, 3, 2, 1).contiguous()
        y2 = self.hc(x_perm2)
        y2 = y2.permute(0, 3, 2, 1).contiguous()

        if self.no_spatial:
            return 0.5 * (y1 + y2)

        y3 = self.hw(x)
        return (y1 + y2 + y3) / 3.0


class TripletAttentionLite(nn.Module):
    def __init__(
        self,
        dim: int,
        kernel_size: int = 7,
        no_spatial: bool = False,
        alpha_init: float = 1e-3,
    ):
        super().__init__()
        # dim kept for the common attention factory API.
        self.dim = dim
        self.core = TripletAttentionCore(
            kernel_size=kernel_size,
            no_spatial=no_spatial,
        )
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.core(x)
        return x + self.alpha * (y - x)
```

This is a project-adapted implementation of the published three-branch idea. Before finalizing, Codex should compare branch permutations and gating details with the pinned official repository and document any differences.

## 5. Initial configuration

Use:

```text
kernel_size = 7
no_spatial = false
alpha_init = 1e-3
```

Do not start with kernel sweeps.

If the candidate is positive, one later ablation can compare:

```text
full 3-branch Triplet
vs
no_spatial=true
```

but this is not part of the first training round.

## 6. Integration

Attention factory:

```python
if kind == "triplet":
    return TripletAttentionLite(dim=dim, **cfg)
```

YAML concepts:

```yaml
# Position 1: P3-Cls-Pre
attn_type: triplet
levels: [0]
site: pre
attn_cfg:
  kernel_size: 7
  no_spatial: false
  alpha_init: 0.001

# Position 2: P3-Cls-Mid
attn_type: triplet
levels: [0]
site: mid
attn_cfg:
  kernel_size: 7
  no_spatial: false
  alpha_init: 0.001
```

Reuse existing classification-only `AttnDetect`; box/DFL branch must remain bitwise/near-numerically identical under the same feature inputs.

## 7. Mandatory tests

1. shape preservation;
2. finite forward/backward;
3. permutation inverse correctness;
4. alpha=0 baseline equivalence;
5. changing Triplet attention changes scores but not raw boxes;
6. common pretrained weight transfer;
7. 1-epoch smoke train;
8. validation smoke;
9. Params/GFLOPs/VRAM/latency.

A dedicated permutation unit test should use a small deterministic tensor and verify that each `permute(...).permute(...)` path restores NCHW ordering.

## 8. Required comparison

Primary:

```text
Baseline
BRA-Lite
Triplet Attention
```

If Triplet is positive, add a simple gate control at the same location, e.g. a plain H-W spatial gate with similar 7x7 cost.

The mechanism claim only survives if cross-dimensional interaction beats a simpler spatial-only gate.

## 9. Success / failure signature

A particularly useful result would be:

```text
Precision >= BRA
mAP50-95 >= BRA or very close
runtime << BRA
```

Even if Recall is lower than BRA, Triplet can still be informative if it substantially restores classification purity.

Stop if:

- it simply reproduces CAA-like near-zero/negative behavior;
- Precision does not improve and mAP drops;
- a plain spatial gate matches it;
- the second insertion position does not resolve a clearly diagnosed pre-feature-noise problem.

## 10. Research status

Triplet Attention is an existing WACV 2021 module. It is included as a **low-cost mechanistic counterfactual**, not as project novelty.

Its role is to answer:

> Does the task actually need non-local context routing, or only better cross-dimensional P3 feature selection?
