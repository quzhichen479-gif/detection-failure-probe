# SHSA — Single-Head Self-Attention for YOLO11 Classification Branch

> Candidate priority: **2 / Round-2**  
> Source: SHViT, CVPR 2024  
> Official repo: https://github.com/ysj9909/SHViT  
> Paper: https://openaccess.thecvf.com/content/CVPR2024/html/Yun_SHViT_Single-Head_Vision_Transformer_with_Memory_Efficient_Macro_Design_CVPR_2024_paper.html

## 1. Why SHSA is useful after BRA

BRA-Lite suggests adaptive non-local context has signal, but it is relatively complex and loses Precision. SHSA tests a simpler alternative:

> Is explicit sparse region routing necessary, or is a lightweight single-head global correction on a partial channel subset sufficient?

The official SHViT implementation uses single-head self-attention on only part of the channels and leaves the remaining channels as a parallel local path. This makes it a useful control between pure convolutional attention and BRA-style routing.

The SHViT authors also report that attention in earlier stages can often be replaced by convolution and that several attention heads in later stages are redundant. This supports using SHSA **after a local classification block**, not directly on raw shallow features.

## 2. Official SHSA structure

For input:

\[
X\in\mathbb{R}^{B\times C\times H\times W},
\]

split channels into an attended subset and untouched subset:

\[
X=[X_a, X_b],
\]

where \(X_a\in\mathbb{R}^{B\times C_a\times H\times W}\).

The official implementation normalizes the partial channels, forms single-head Q/K/V using a 1x1 projection, flattens the spatial grid, and computes:

\[
A = \operatorname{softmax}\left(\frac{Q^TK}{\sqrt{d_q}}\right).
\]

Then:

\[
Y_a = VA^T.
\]

The attended partial channels are concatenated with untouched channels and projected:

\[
Y=P([Y_a,X_b]).
\]

For this project, wrap it as a near-identity adapter:

\[
\hat Y=X+\alpha(Y-X).
\]

## 3. Two insertion positions

### Position 1 — P3-Cls-Mid **(first priority)**

```text
P3
 |-----------------------------> box tower unchanged
 |
 -> cls block-0
      -> SHSA
          -> cls block-1
              -> class predictor
```

Why first:

- respects SHViT's own local-before-attention design logic;
- P3 is the feature level most relevant to tiny targets;
- provides a direct low-complexity counterfactual to BRA;
- avoids changing regression geometry.

### Position 2 — P4-Cls-Mid

```text
P4
 |-----------------------------> box tower unchanged
 |
 -> cls block-0
      -> SHSA
          -> cls block-1
              -> class predictor
```

Why second:

- quadratic spatial attention is substantially cheaper at P4 than P3;
- P4 semantics may reject pixel-level glitter/wave responses better;
- tests whether the useful context is tiny-detail-local or higher-level semantic.

Do not enable P3 and P4 SHSA together in the first ablation.

## 4. Project implementation

The following is a compact NCHW adaptation of the official SHSA idea. It is not claimed to be the author's exact module wrapper because the original SHViT block includes surrounding residual convolution and FFN. The core single-head partial-channel attention is preserved.

```python
from __future__ import annotations

import torch
import torch.nn as nn


class SHSALite(nn.Module):
    """Partial-channel single-head self-attention for YOLO classification features."""

    def __init__(
        self,
        dim: int,
        partial_dim: int | None = None,
        qk_dim: int = 16,
        alpha_init: float = 1e-3,
    ):
        super().__init__()

        if partial_dim is None:
            partial_dim = min(max(dim // 4, 16), 64)

        if not (0 < partial_dim <= dim):
            raise ValueError((dim, partial_dim))

        self.dim = dim
        self.partial_dim = partial_dim
        self.qk_dim = qk_dim
        self.scale = qk_dim ** -0.5

        self.norm = nn.GroupNorm(1, partial_dim)
        self.qkv = nn.Conv2d(
            partial_dim,
            2 * qk_dim + partial_dim,
            kernel_size=1,
            bias=False,
        )
        self.proj = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(dim, dim, kernel_size=1, bias=False),
        )

        self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))

    def core(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        xa, xb = torch.split(
            x,
            [self.partial_dim, c - self.partial_dim],
            dim=1,
        )

        xa = self.norm(xa)
        qkv = self.qkv(xa)
        q, k, v = qkv.split(
            [self.qk_dim, self.qk_dim, self.partial_dim],
            dim=1,
        )

        q = q.flatten(2)       # B,d,N
        k = k.flatten(2)       # B,d,N
        v = v.flatten(2)       # B,Ca,N

        attn = (q.transpose(-2, -1) @ k) * self.scale  # B,N,N
        attn = attn.softmax(dim=-1)

        ya = (v @ attn.transpose(-2, -1)).reshape(
            b, self.partial_dim, h, w
        )

        y = torch.cat([ya, xb], dim=1)
        return self.proj(y)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.core(x)
        return x + self.alpha * (y - x)
```

## 5. Important cost warning

At 640 input, P3 is roughly 80x80, so spatial token count is about 6400. Even single-head attention has an \(N\times N\) attention matrix.

Therefore the first P3 experiment must log:

```text
peak VRAM
batch-1 latency P50/P95
training iteration time
```

If P3 SHSA becomes impractical, do **not** silently introduce spatial reduction or windowing while keeping the same experiment name. That would be a new variant. Run the predefined P4 second position instead.

## 6. Initial parameters

Use one fixed first configuration:

```text
partial_dim = min(max(C / 4, 16), 64)
qk_dim = 16
alpha_init = 1e-3
```

No multi-head variant in Round-2. Multi-head attention would defeat the purpose of this control.

## 7. Integration

Add to the existing attention factory:

```python
if kind == "shsa":
    return SHSALite(dim=dim, **cfg)
```

YAML concepts:

```yaml
# Position 1
attn_type: shsa
levels: [0]
site: mid
attn_cfg:
  qk_dim: 16
  alpha_init: 0.001

# Position 2
attn_type: shsa
levels: [1]
site: mid
attn_cfg:
  qk_dim: 16
  alpha_init: 0.001
```

Reuse the existing `AttnDetect` classification-only path. Do not modify `cv2`, DFL, loss, assigner, image size or training recipe.

## 8. Mandatory tests

Before full training:

1. output shape equals input shape;
2. all outputs finite;
3. gradients finite;
4. attention parameters get gradient with non-zero alpha;
5. alpha=0 reproduces baseline path;
6. raw box branch invariant when alpha changes;
7. common baseline weights transfer;
8. P3 memory/latency check;
9. 1-epoch smoke train;
10. validation smoke.

## 9. Required comparisons

Primary:

```text
Baseline vs BRA-Lite vs SHSA
```

If SHSA is positive, add a simple control:

```text
P3-Cls-Mid + same-cost 1x1/3x3 local block
```

The question is whether global single-head interaction is responsible, not merely an extra projection.

## 10. Success / failure signature

Useful outcome:

```text
mAP50-95 >= BRA or close to BRA
Precision > BRA
Recall >= baseline or near BRA
latency materially lower than BRA/DBRA
```

Especially valuable:

> SHSA preserves most of BRA's Recall/AP gains while recovering Precision with lower runtime cost.

Stop if:

- P3 memory/latency is unacceptable;
- Precision still drops similarly to BRA without better mAP;
- P4 is also negative;
- a simple local block matches the result.

## 11. Research status

SHSA is an existing CVPR 2024 mechanism and is not project novelty. It is a controlled test of whether **simple partial-channel single-head global context** is enough to explain the signal previously seen with BRA.
