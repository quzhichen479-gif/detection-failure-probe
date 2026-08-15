# DBRA — Deformable Agent Bi-level Routing Attention for YOLO11 Classification Branch

> Candidate priority: **1 / Round-2**  
> Source: DeBiFormer, ACCV 2024  
> Upstream: https://github.com/maclong01/DeBiFormer  
> Paper: https://openaccess.thecvf.com/content/ACCV2024/html/BaoLong_DeBiFormer_Vision_Transformer_with_Deformable_Agent_Bi-level_Routing_Attention_ACCV_2024_paper.html

## 1. Why this module follows BRA

Round-1 BRA-Lite gives the most informative pattern:

```text
Recall     +0.016510
mAP50      +0.005732
mAP75      +0.001601
Precision  -0.013542
mAP50-95   -0.000555
```

This suggests a useful but incomplete mechanism: adaptive context routing appears able to recover additional positives, yet classification purity is still worse than baseline.

DeBiFormer explicitly motivates DBRA by pointing out a limitation of BiFormer/BRA: routed key-value pairs may still be influenced by irrelevant queries. DBRA introduces deformable/agent-based selection to improve key-value selection. That makes DBRA a **mechanistically continuous experiment after BRA**, rather than another unrelated attention module.

The falsifiable project question is:

> Can DBRA retain BRA's Recall/AP50/AP75 gains while recovering part of the lost Precision?

## 2. High-level formulation

Let input feature be:

\[
X\in\mathbb{R}^{B\times C\times H\times W}.
\]

BRA first constructs query-aware routed regions. A simplified region routing score is:

\[
A_r = \frac{Q_rK_r^T}{\sqrt d},
\]

and only top-k routed regions are retained:

\[
I_r=\operatorname{TopK}(A_r,k).
\]

DBRA extends this idea by using deformable/agent-based query-key/value selection before/within the routed attention computation. The exact operator should be taken from the pinned official DeBiFormer source rather than re-derived from memory.

For YOLO integration, use a near-identity adapter:

\[
Y = X + \alpha\left(DBRA(X)-X\right),
\]

with small non-zero initial \(\alpha\), e.g. `1e-3`.

At `alpha=0`, the adapter is exactly identity. At `alpha=1`, it reduces to the wrapped DBRA output.

## 3. Two insertion positions

### Position 1 — P3-Cls-Mid **(first priority)**

Conceptual path:

```text
P3 neck feature
   |---------------------------> box tower (unchanged)
   |
   -> YOLO cls block-0
       -> DBRA adapter
           -> YOLO cls block-1
               -> class predictor
```

Formal form:

\[
U_3 = H^{cls}_{3,0}(F_3),
\]

\[
\tilde U_3 = A_{DBRA}(U_3),
\]

\[
z^{cls}_3 = H^{cls}_{3,1:}(\tilde U_3).
\]

Why first:

- preserves the already promising BRA insertion logic;
- gives DBRA semantically processed P3 features rather than raw low-level glitter/wave gradients;
- directly targets tiny-object classification;
- leaves the complete `cv2` box/DFL branch untouched.

### Position 2 — P4-Cls-Mid

```text
P4 neck feature
   |---------------------------> box tower (unchanged)
   |
   -> YOLO cls block-0
       -> DBRA adapter
           -> YOLO cls block-1
               -> class predictor
```

Why second:

- lower spatial resolution reduces routing/attention cost;
- features are more semantic and may be less dominated by pixel-level water texture;
- tests whether background rejection needs P3 detail or higher-level contextual semantics.

Do **not** enable P3 and P4 DBRA simultaneously in the first ablation.

## 4. Implementation strategy

### 4.1 Do not hand-reimplement full DBRA first

DBRA is substantially more complex than CAA/LSK/Triplet. For the first faithful experiment:

1. Pin one upstream DeBiFormer commit.
2. Record repository URL, commit SHA, retrieval date and license.
3. Vendor only the minimal DBRA dependencies needed for forward execution.
4. Preserve upstream computation before project adaptation.
5. Wrap it with a YOLO NCHW adapter and the near-identity interpolation.

Suggested local path in the YOLO engineering repository:

```text
ultralytics/nn/modules/third_party/debiformer/
    dbra_upstream.py
    SOURCE.md
```

`SOURCE.md` must record:

```text
upstream_repo: https://github.com/maclong01/DeBiFormer
upstream_commit: <PINNED_SHA>
license: upstream license
local_changes:
  - import cleanup
  - dependency isolation
  - no algorithmic change unless documented
```

### 4.2 Project adapter skeleton

The constructor names/arguments below are intentionally **not guessed**. Codex must inspect the pinned upstream class and fill the exact call.

```python
class DBRALiteAdapter(nn.Module):
    def __init__(self, dim: int, alpha_init: float = 1e-3, **dbra_cfg):
        super().__init__()

        # Import the exact DBRA class from the pinned vendored source.
        DBRAUpstream = _resolve_dbRA_upstream_class()

        self.core = DBRAUpstream(
            # map dim / heads / windows / top-k / deformable parameters
            # from the pinned upstream signature
            **_build_exact_upstream_kwargs(dim, dbra_cfg)
        )
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.core(x)
        if y.shape != x.shape:
            raise RuntimeError(f"DBRA shape changed: {x.shape} -> {y.shape}")
        return x + self.alpha * (y - x)
```

The helper names above are placeholders for explicit local code, not runtime reflection magic. Once upstream is pinned, Codex should replace them with normal imports and a concrete constructor.

## 5. Parameter policy

The first experiment must use **one fixed DBRA configuration** selected from a small/efficient upstream variant compatible with the actual P3/P4 channel count.

Do not launch a grid over:

```text
heads × topk × windows × deformable points × offset range
```

before establishing a positive mechanism signal.

The first implementation must report:

- attention/routing window count;
- top-k;
- number of heads;
- deformable/agent settings;
- active channel dimension;
- parameters;
- GFLOPs;
- batch-1 P50/P95 latency;
- VRAM.

## 6. Integration with existing AttnDetect

Reuse the existing classification-only head infrastructure. Add `dbra` to the attention factory:

```python
if kind == "dbra":
    return DBRALiteAdapter(dim=dim, **cfg)
```

Recommended YAML concept:

```yaml
# Position 1: P3-Cls-Mid
attn_type: dbra
levels: [0]
site: mid

# Position 2: P4-Cls-Mid
attn_type: dbra
levels: [1]
site: mid
```

Do not change the box tower, DFL, loss, assignment, image size, augmentation or training recipe.

## 7. Mandatory tests

Before training:

1. shape preservation;
2. finite forward;
3. finite backward;
4. gradient reaches DBRA parameters;
5. `alpha=0` equals baseline cls path;
6. changing DBRA/alpha changes class output but **not raw box output**;
7. pretrained baseline weights transfer to all common YOLO parameters;
8. 1-epoch smoke train;
9. validation smoke test;
10. latency and memory test.

If upstream DBRA needs custom CUDA or unsupported ops, document that before adding new dependencies. Do not silently replace the operator with a simpler attention and still call it DBRA.

## 8. Success / failure signature

The main comparison is not just baseline vs DBRA. It is:

```text
Baseline
BRA-Lite
DBRA
```

Desired mechanism signature:

```text
DBRA Recall     ~= BRA Recall
DBRA mAP50      ~= or > BRA mAP50
DBRA mAP75      >= BRA mAP75
DBRA Precision  >  BRA Precision
DBRA mAP50-95   >  BRA mAP50-95
```

The strongest evidence would be recovering a meaningful fraction of BRA's `-0.013542` Precision delta without giving back its Recall gain.

Stop DBRA if:

- Precision falls further while Recall rises further;
- mAP50-95 stays below BRA with materially higher latency;
- routing/agent maps collapse to near-constant patterns;
- P3 implementation cost makes deployment obviously dominated by BRA/SHSA;
- the result can be reproduced by a substantially simpler local/global attention.

## 9. Research status

DBRA itself is an existing ACCV 2024 module and must **not** be presented as project novelty. Its role here is a mechanism probe:

> Does more selective BRA-style K/V/context selection correct the Precision cost observed in BRA-Lite?

Only if that hypothesis is supported should the project consider designing a water-clutter-specific routing mechanism.
