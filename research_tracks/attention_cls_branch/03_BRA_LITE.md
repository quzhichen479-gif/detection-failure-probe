# BRA-Lite for YOLO11 Classification Branch

## 1. Motivation

Bi-Level Routing Attention (BRA) introduces dynamic sparse routing between regions before token-level attention. The potentially useful part for water-surface detection is **conditional non-local context**: a tiny candidate may benefit from evidence in selected surrounding regions instead of uniformly aggregating all water texture.

This is also the riskiest candidate in this track. Full BRA can add implementation complexity, memory traffic, export difficulty, and latency. Therefore this document specifies a deliberately reduced BRA-Lite intended only as a controlled screening experiment.

## 2. Reduced formulation

Let feature map

\[
X\in\mathbb{R}^{B\times C\times H\times W}
\]

be partitioned into \(N\) regions. For each region \(r_i\), compute pooled query/key summaries:

\[
q_i = W_q\operatorname{Pool}(r_i),\qquad
k_i = W_k\operatorname{Pool}(r_i).
\]

Construct region affinity:

\[
A_{ij}=\frac{q_i^T k_j}{\sqrt{d}}.
\]

For each source region, retain only Top-K routed regions:

\[
\mathcal{R}_i = \operatorname{TopK}_j(A_{ij}, K).
\]

Token/value aggregation is then restricted to routed regions rather than all regions:

\[
Y_i = \operatorname{Attn}(Q_i,K_{\mathcal{R}_i},V_{\mathcal{R}_i}).
\]

For the first screening implementation, full token-to-token BRA is optional. A safer BRA-Lite approximation is acceptable:

1. region pooling;
2. top-k region routing;
3. gather routed region context vectors;
4. project aggregated routed context back to the spatial feature;
5. residual gate the original classification feature.

This preserves the key hypothesis — dynamic selective non-local context — while reducing engineering risk.

## 3. Recommended BRA-Lite approximation

Pseudo-code:

```python
class BRALite(nn.Module):
    def __init__(self, channels, region_size=10, topk=4, hidden_ratio=0.25, alpha_init=0.0):
        super().__init__()
        # 1x1 projections for region descriptors
        # adaptive region pooling / reshape
        # q,k,v on pooled region tokens
        # top-k routing from region affinity
        # gather routed v and weighted aggregate
        # broadcast/project context to region pixels
        # residual gate with alpha initialized near identity
        ...

    def forward(self, x):
        # x: [B,C,H,W]
        # pad only internally if H/W are not divisible by region size,
        # then crop back exactly to original H/W.
        ...
        return y
```

Codex should prefer simple PyTorch gather/topk/matmul operations and avoid custom CUDA kernels in the first version.

## 4. Identity-safe output

Use

\[
Y=X+\alpha\,\phi(C_{route}),
\]

or

\[
Y=X\odot(1+\alpha\sigma(\phi(C_{route})))
\]

with \(\alpha=0\) initialization. This ensures the newly inserted module starts close to baseline behavior.

## 5. Insertion positions

### Priority 1 — P3 classification branch, between class feature transforms

```text
P3 feature
  -> cls transform 1
  -> BRA-Lite
  -> cls transform 2
  -> class predictor
```

Reason:

- P3 is the relevant tiny-object scale;
- mid-branch placement allows downstream class features to adapt to routed context;
- regression path remains untouched.

### Priority 2 — P4 classification branch, between class feature transforms

```text
P4 feature
  -> cls transform 1
  -> BRA-Lite
  -> cls transform 2
  -> class predictor
```

Reason:

- lower spatial resolution makes region routing much cheaper;
- tests whether broader context helps classification without the large P3 cost;
- useful as a compute-vs-accuracy comparison if P3 is too expensive.

Do **not** enable P3 and P4 BRA-Lite simultaneously during first screening.

## 6. First-round preset

Use one conservative configuration chosen after inspecting actual P3/P4 feature sizes:

```text
topk = 4
hidden_ratio = 0.25
alpha_init = 0.0
single scale only
no custom CUDA
no global full attention fallback
```

Region partition should target roughly 8–10 regions along each spatial axis at the target scale, but Codex must compute a shape-safe configuration from the actual YOLO11n feature dimensions instead of hard-coding assumptions.

## 7. Engineering risks

BRA-Lite must be rejected early if:

- ONNX/TorchScript/export path breaks and cannot be fixed simply;
- TopK/gather creates unstable dynamic-shape behavior;
- memory or latency is materially worse than CAA-Lite/LSK-Lite;
- padding/cropping changes alignment;
- numerical instability occurs under AMP.

Do not spend extensive engineering effort reproducing full BiFormer if the lightweight routed-context hypothesis already fails.

## 8. Diagnostic hypothesis

A credible positive result would look like selective suppression of distant-but-similar clutter and improved classification confidence on true tiny targets, without AP75 degradation.

A suspicious result would be:

- Recall gain with Precision collapse;
- gains only from large latency increases;
- improvements only after multiple scales are stacked;
- unstable results depending strongly on region partition;
- large confidence changes without better localization-conditioned ranking.

## 9. Unit/integration tests

Must test multiple spatial sizes, including non-divisible shapes:

```python
for hw in [(80, 80), (40, 40), (79, 81)]:
    x = torch.randn(2, 128, *hw)
    y = BRALite(128)(x)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()
```

Also test backward, AMP, export, and disabled baseline behavior.
