# LSK-Lite for YOLO11 Classification Branch

## 1. Motivation

Large Selective Kernel (LSK) style modules are attractive here because they model **large spatial context with selective multi-scale responses** while remaining convolutional. This is relevant when the target itself is tiny but its surrounding water pattern helps distinguish it from specular highlights, foam, ripples, or shoreline clutter.

The objective is not to reproduce the full LSKNet backbone. LSK-Lite keeps only the core idea: two depthwise spatial receptive fields, lightweight spatial selection, and residual modulation.

## 2. Core formulation

Given

\[
X \in \mathbb{R}^{B\times C\times H\times W},
\]

compute two depthwise spatial branches:

\[
U_1 = \operatorname{DWConv}_{k_1}(X),
\]

\[
U_2 = \operatorname{DWConv}_{k_2,d}(U_1),
\]

where the second branch uses a larger effective receptive field via dilation.

Compress channel information to spatial descriptors:

\[
S_{avg}=\operatorname{Mean}_C([U_1,U_2]),
\qquad
S_{max}=\operatorname{Max}_C([U_1,U_2]).
\]

A small spatial selector predicts branch weights:

\[
W=\sigma(\operatorname{Conv}([S_{avg},S_{max}])).
\]

Use the learned spatial weights to form

\[
U = W_1\odot U_1 + W_2\odot U_2,
\]

then project and apply residual modulation:

\[
Y = X + \alpha\, X\odot \phi(U).
\]

Again, \(\alpha\) should initialize at zero or near zero.

## 3. Suggested implementation skeleton

```python
import torch
import torch.nn as nn


class LSKLite(nn.Module):
    def __init__(self, channels, alpha_init=0.0):
        super().__init__()
        self.dw5 = nn.Conv2d(
            channels, channels, 5,
            padding=2, groups=channels, bias=False,
        )
        self.dw7d3 = nn.Conv2d(
            channels, channels, 7,
            padding=9, dilation=3,
            groups=channels, bias=False,
        )
        self.selector = nn.Conv2d(4, 2, 7, padding=3, bias=True)
        self.proj = nn.Conv2d(channels, channels, 1, bias=True)
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))

    def forward(self, x):
        u1 = self.dw5(x)
        u2 = self.dw7d3(u1)

        avg1 = u1.mean(dim=1, keepdim=True)
        max1 = u1.amax(dim=1, keepdim=True)
        avg2 = u2.mean(dim=1, keepdim=True)
        max2 = u2.amax(dim=1, keepdim=True)

        w = torch.sigmoid(self.selector(torch.cat([avg1, max1, avg2, max2], dim=1)))
        u = u1 * w[:, 0:1] + u2 * w[:, 1:2]
        g = torch.sigmoid(self.proj(u))
        return x + self.alpha * (x * g)
```

This is an intentionally simplified project implementation, not a line-for-line reproduction of LSKNet.

## 4. Insertion positions

### Priority 1 — P3 classification branch, between class feature transforms

```text
P3 feature
  -> cls transform 1
  -> LSK-Lite
  -> cls transform 2
  -> class predictor
```

Rationale:

- LSK-Lite is primarily a representation reshaper rather than merely a final score gate.
- Placing it in the middle allows the second classification transform to consume the selectively enlarged spatial context.
- P3 retains enough spatial resolution for meaningful large-kernel context around tiny objects.

### Priority 2 — P3 classification branch, immediately before class predictor

```text
P3 feature
  -> cls transforms
  -> LSK-Lite
  -> class predictor
```

Rationale:

- lower intervention depth;
- cleaner attribution;
- useful if the mid-branch variant changes optimization too strongly.

## 5. First-round configuration

Use exactly one preset:

```text
branch 1: DWConv 5x5
branch 2: DWConv 7x7, dilation=3
selector: Conv 7x7, 4 -> 2 spatial maps
alpha_init = 0.0
scale = P3 only
position = mid-cls branch
```

Do not start by searching 9x9/11x11/23x23 kernels or multiple dilation patterns. If the fixed mechanism cannot survive the first gate, kernel search is unlikely to provide strong scientific evidence.

## 6. Engineering requirements

- Maintain input/output shape.
- No resize/interpolation in the module.
- No modification to P3 feature used by regression.
- The feature tensor must be forked logically so only the class path receives LSK-Lite.
- Provide independent enable/disable flag.
- Disabled state must produce the original YOLO11 Detect behavior.
- Check FLOPs and latency because dilated large kernels can become costly despite depthwise grouping.

## 7. Diagnostic hypotheses

### Expected positive pattern

If LSK-Lite is useful, the most plausible signature is:

- fewer water-texture false positives;
- Precision improves or remains stable;
- Recall does not collapse;
- mAP75 remains stable because regression features are untouched;
- improvements concentrate on tiny/hard clutter cases.

### Negative patterns

- Precision improves but Recall falls sharply: context gate is too conservative.
- mAP50 improves but mAP75 falls: ranking is favoring boxes with poorer localization.
- latency overhead is high with tiny AP gain: mechanism is not Pareto-competitive.
- gains appear only after stacking across P3/P4/P5: attribution becomes weak; do not accept this as first-round evidence.

## 8. Minimal tests

```python
x = torch.randn(2, 128, 80, 80)
m = LSKLite(128)
y = m(x)
assert y.shape == x.shape
assert torch.isfinite(y).all()
loss = y.square().mean()
loss.backward()
```

Also validate AMP, export, deterministic shape behavior, and the exact P3-only branch wiring.
