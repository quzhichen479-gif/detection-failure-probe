# CAA-Lite for YOLO11 Classification Branch

## 1. Motivation

Context Anchor Attention (CAA) comes from the PKINet line for remote-sensing detection. Its useful idea here is not “attention in general”, but **long-range directional context aggregation** around local anchors. Water-surface floating objects are often tiny and weakly textured, while wave edges, reflections, foam, and shoreline structures create confusing local patterns. A lightweight directional context gate can increase contextual discrimination without changing the box-regression branch.

CAA-Lite is intentionally smaller than the original module. The goal is to preserve the mechanism while minimizing compute and integration risk.

## 2. Core formulation

Let input feature map be

\[
X \in \mathbb{R}^{B\times C\times H\times W}.
\]

First obtain a local anchor feature using average pooling and a pointwise/depthwise projection:

\[
A = \phi_{loc}(\operatorname{AvgPool}_{k\times k}(X)).
\]

Then aggregate long-range horizontal and vertical context using strip depthwise convolutions:

\[
H_c = \operatorname{DWConv}_{1\times k_h}(A),
\]

\[
V_c = \operatorname{DWConv}_{k_v\times 1}(H_c).
\]

Generate a bounded gate:

\[
G = \sigma(\phi_g(V_c)),
\]

and apply residual modulation:

\[
Y = X \odot (1 + \alpha G),
\]

where \(\alpha\) is a small learnable or fixed scale. Recommended initialization is \(\alpha=0\) or a very small positive value so the module starts close to identity.

This residual gate is preferred over pure multiplication \(X\odot G\), because the latter can strongly suppress weak tiny-object features early in training.

## 3. Suggested lightweight implementation

```python
import torch
import torch.nn as nn


class CAALite(nn.Module):
    def __init__(self, channels, pool_kernel=3, strip_kernel=11, alpha_init=0.0):
        super().__init__()
        p = pool_kernel // 2
        s = strip_kernel // 2

        self.pool = nn.AvgPool2d(pool_kernel, stride=1, padding=p)
        self.pre = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
        )
        self.h = nn.Conv2d(
            channels, channels,
            kernel_size=(1, strip_kernel),
            padding=(0, s),
            groups=channels,
            bias=False,
        )
        self.v = nn.Conv2d(
            channels, channels,
            kernel_size=(strip_kernel, 1),
            padding=(s, 0),
            groups=channels,
            bias=False,
        )
        self.gate = nn.Conv2d(channels, channels, 1, bias=True)
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))

    def forward(self, x):
        z = self.pre(self.pool(x))
        z = self.h(z)
        z = self.v(z)
        g = torch.sigmoid(self.gate(z))
        return x * (1.0 + self.alpha * g)
```

This code is a project-specific lightweight re-expression of the mechanism, not a verbatim copy of the upstream implementation.

## 4. Insertion positions

### Priority 1 — P3 classification branch, immediately before final predictor

Conceptually:

```text
P3 feature
  -> Detect.cv3[0] feature transforms
  -> CAA-Lite
  -> final class conv
```

Why first:

- P3 has the highest spatial resolution among standard detect scales and is the most relevant to tiny objects.
- Late insertion changes class scoring while preserving most of the existing feature extractor.
- It minimizes interference with the sibling regression branch.

### Priority 2 — P3 classification branch, between the two class feature transforms

Conceptually:

```text
P3 feature
  -> cls transform 1
  -> CAA-Lite
  -> cls transform 2
  -> final class conv
```

Why second:

- gives the module more capacity to reshape the classification representation;
- still isolated from box regression;
- slightly higher risk because downstream class transforms adapt to attention-modulated features.

## 5. Implementation requirements

Codex must inspect the exact `ultralytics==8.4.113` `Detect` implementation before deciding indices. Do not assume current upstream line numbers.

Preferred integration pattern:

1. Add `CAALite` as a standalone module in a detector-specific experimental package outside this repository's probe `src/`.
2. Add an explicit constructor/config flag, e.g. `attention_cls='caa_lite'` and `attention_scale=0` for P3.
3. Default behavior must remain bitwise/structurally equivalent to baseline when disabled.
4. Do not monkey-patch unrelated Ultralytics functions if a local subclass or registered module is feasible.
5. Preserve ONNX/TorchScript-friendly operations.

## 6. First ablation settings

Recommended fixed values for the first screen:

```text
pool_kernel = 3
strip_kernel = 11
alpha_init = 0.0
scale = P3 only
position = pre-predictor
```

Do not grid-search kernels in the first round. The objective is to test the mechanism, not tune it until it wins.

## 7. Failure signatures to watch

- Precision rises but Recall collapses: gate may suppress ambiguous true positives.
- Recall rises while mAP75 drops: class confidence ranking may be promoting poorly localized boxes.
- No effect and alpha remains near zero: optimizer may reject the mechanism, which is useful negative evidence.
- Strong gain only when alpha grows very large: inspect feature amplification and calibration before trusting the result.

## 8. Unit tests

At minimum:

```python
x = torch.randn(2, 128, 80, 80)
m = CAALite(128)
y = m(x)
assert y.shape == x.shape
assert torch.isfinite(y).all()
```

Also test:

- CPU forward
- CUDA forward if available
- AMP forward
- backward gradient finite
- export path used by the project
- disabled configuration reproduces baseline graph/output shape
