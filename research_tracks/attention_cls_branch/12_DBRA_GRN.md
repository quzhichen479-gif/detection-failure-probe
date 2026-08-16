# Round-3 Candidate 1 — DBRA + GRN

> Priority: **1 / Round-3**  
> Purpose: post-routing channel-response competition, **not another spatial attention**  
> Base module: `DBRA P3-Cls-Mid`  
> Source mechanism: ConvNeXt V2, CVPR 2023  
> Official code: https://github.com/facebookresearch/ConvNeXt-V2

## 1. Why GRN is being tested

DBRA P3-mid already improves AP/APs/ARs, but the Ultralytics operating-point Precision remains below baseline.

A tempting but incorrect statement would be:

> “GRN suppresses water clutter and therefore should restore Precision.”

That is **not established** by ConvNeXt V2. GRN was proposed to enhance inter-channel feature competition. It computes spatial response magnitude per channel, normalizes those responses across channels, and uses learnable affine residual modulation. It has no explicit foreground/background semantics.

For water-surface P3, the spatial norm can even be dominated by background because most pixels are water. Therefore GRN is a low-cost **probe of post-routing response competition**, not a guaranteed background suppressor.

## 2. Original GRN formulation

For a channel-last tensor in the original implementation:

\[
G_c = \lVert X_c \rVert_2
\]

over spatial dimensions, then

\[
N_c = \frac{G_c}{\operatorname{mean}_{j}(G_j)+\epsilon}
\]

and

\[
Y = X + \gamma \odot (X \odot N) + \beta.
\]

The official ConvNeXt V2 implementation initializes both `gamma` and `beta` to zero, so GRN starts as exact identity.

For NCHW YOLO tensors, use the equivalent axes:

```python
Gx = torch.norm(x, p=2, dim=(2, 3), keepdim=True)
Nx = Gx / (Gx.mean(dim=1, keepdim=True) + eps)
y = x + gamma * (x * Nx) + beta
```

## 3. Project implementation

```python
from __future__ import annotations

import torch
import torch.nn as nn


class GRN2d(nn.Module):
    """NCHW equivalent of ConvNeXt-V2 Global Response Normalization."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = float(eps)
        self.gamma = nn.Parameter(torch.zeros(1, dim, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gx = torch.norm(x, p=2, dim=(2, 3), keepdim=True)
        nx = gx / (gx.mean(dim=1, keepdim=True) + self.eps)
        return x + self.gamma * (x * nx) + self.beta
```

Do not add BatchNorm, sigmoid gating, ECA, or another attention around this GRN in the first experiment.

## 4. Primary model position — DBRA then GRN at P3-Cls-Mid

Recommended graph:

```text
P3 neck feature
  |-------------------------------------------> box branch (unchanged)
  |
  +-> cls block-0
        -> DBRA
        -> GRN2d
        -> cls block-1
        -> class predictor
```

Formal form:

\[
U = H^{cls}_{0}(F_3)
\]

\[
D = A_{DBRA}(U)
\]

\[
\hat D = GRN(D)
\]

\[
z = H^{cls}_{1:}(\hat D).
\]

Why this is primary:

1. DBRA first performs the mechanism that already has positive evidence.
2. GRN only recalibrates the routed representation.
3. `gamma=beta=0` means the new model can initialize **exactly as DBRA**, not merely approximately as baseline.
4. It does not add any new spatial receptive field.

### Composite module

```python
class DBRAGRNPost(nn.Module):
    def __init__(self, dim: int, dbra: nn.Module, eps: float = 1e-6):
        super().__init__()
        self.dbra = dbra
        self.grn = GRN2d(dim, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.grn(self.dbra(x))
```

In the real implementation, `dbra` should be constructed by the already-fixed Round-2 DBRA factory/config, not by inventing a second DBRA variant.

## 5. Secondary model position — DBRA at mid, GRN immediately before predictor

Graph:

```text
P3
 -> cls block-0
 -> DBRA
 -> cls block-1
 -> GRN2d
 -> class predictor
```

This asks a different question:

> Should response competition operate on the routed semantic feature immediately, or on the final hidden feature directly feeding class logits?

The secondary site is closer to score calibration but has less opportunity for the existing YOLO classifier block to re-integrate the normalized feature.

### Required head support

The existing `AttnDetect` likely supports `pre` and `mid` only. Do **not** hack `cv3` indices in YAML to get this site.

Extend classification hook semantics cleanly:

```text
pre      : before cls block-0
mid      : after cls block-0
pre_pred : after cls block-1, before predictor
```

Add regression tests proving all three positions leave `cv2` raw box outputs unchanged.

## 6. Stronger initialization test

Round-3 has a better identity test than previous rounds.

Given the same DBRA weights:

```text
DBRA model
DBRA + GRN model with gamma=beta=0
```

must satisfy:

```python
torch.testing.assert_close(
    cls_hidden_dbra,
    cls_hidden_dbra_grn,
    atol=1e-6,
    rtol=1e-5,
)
```

and full class scores should also match before training.

This proves that any later difference is learned by GRN rather than an accidental architecture/load mismatch.

## 7. What to monitor during training

Log at least:

```text
mean(abs(gamma))
max(abs(gamma))
mean(abs(beta))
channel-response coefficient of variation before/after GRN
```

If `gamma` remains essentially zero throughout training, GRN is unused and should not be credited for any noise-level metric difference.

If `gamma` grows extremely large early, check for activation-scale instability.

## 8. Evaluation hypothesis

Primary comparison:

```text
DBRA P3-mid
vs
DBRA + GRN-post P3-mid
```

Desired signature:

```text
AP >= DBRA
APs approximately retained
ARs approximately retained
FP @ matched Recall decreases
Precision may rise, but raw Precision alone is insufficient
APm/ARm must not worsen materially
```

Suggested project screening constraints, not universal laws:

```text
Delta AP vs DBRA > 0
Delta APs >= -0.003
Delta ARs >= -0.003
and one of:
    matched-recall FP improves materially
    APm partially recovers
    raw Precision rises without AP loss
```

## 9. Failure interpretation

If GRN hurts APs/ARs:

- global spatial response competition may be dominated by water background;
- channel energy is not a reliable purity signal for this task.

If GRN changes Precision but not AP / matched-recall FP:

- likely score calibration / operating-point movement rather than better classification ranking.

If GRN is neutral:

- do not tune epsilon/gamma init repeatedly;
- its low cost makes the negative result informative enough.

## 10. Source fidelity

The NCHW implementation above is an axis-transposed equivalent of official GRN, not a new attention mechanism.

Do not claim `DBRA+GRN` as novelty. Its role is a controlled test of whether post-routing inter-channel competition complements DBRA.
