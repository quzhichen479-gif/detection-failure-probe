# Round-3 Candidate 2 — Slide Local Preservation + DBRA

> Priority: **2 / Round-3**  
> Primary idea: **parallel local / routed-context evidence**, not attention stacking in series  
> Base: DBRA at P3-Cls-Mid  
> Source: Slide-Transformer, CVPR 2023  
> Official code: https://github.com/LeapLabTHU/Slide-Transformer
> Canonical project reference code: `reference_code/round3_companion_modules.py::SlideAttention2d`

## 1. Why the previous serial idea is revised

A naive design would be:

```text
cls block-0 -> Slide -> DBRA -> cls block-1
```

This is no longer the primary design.

DBRA P3-mid already has positive small-object evidence (`APs +0.0153`, `ARs +0.0075` vs baseline under paper-aligned COCOeval), while `APm/ARm` are lower. Putting Slide before DBRA would modify the input distribution of the mechanism that already works.

The primary design is therefore:

```text
                       -> DBRA routed-context -----
U = cls block-0(P3) ---                              +-> residual fusion -> block-1
                       -> Slide local evidence -----
```

This preserves DBRA as an explicit anchor and asks whether local contiguous evidence is complementary.

## 2. What Slide Attention actually does

Slide Attention is local self-attention. Each query attends only to a `k x k` neighborhood.

The official Slide-Swin implementation uses:

```text
qkv projection
-> reshape to per-head packed q/k/v feature
-> designed depthwise shift + learnable depthwise shift for K/V
-> relative local bias
-> softmax over k^2 local neighbors
-> weighted local V aggregation
-> output projection
```

For each position, conceptually:

\[
a_{p,j}=\operatorname{softmax}_j\left(\frac{q_p^T k_{p,j}}{\sqrt d}+b_j\right)
\]

for `j` in the local neighborhood, then

\[
y_p=\sum_j a_{p,j}v_{p,j}.
\]

The paper/official repository describes two shift paths during training: one designed/fixed shift kernel and one learnable path, merged by re-parameterization for inference. The released `SlideAttention` commonly uses `ka=3`.

## 3. Source-fidelity correction made during this design review

An earlier draft of this document used a conventional:

```python
q, k, v = qkv.chunk(3, dim=channel)
# then split each into heads
```

layout and omitted the fixed-shift convolution bias. After re-reading the official `slide_swin.py::SlideAttention`, that is not an exact port of the released implementation.

The official code instead projects to `3*C`, permutes to NCHW, and directly reshapes to:

```text
[B * num_heads, 3 * head_dim, H, W]
```

before slicing q/k/v. It also freezes only the designed shift **weight**; the convolution bias remains present/trainable, and the relative-position bias is initialized with truncated normal (`std=.02`).

Therefore the canonical project reference has been corrected to preserve those semantics while changing only:

```text
fixed input_resolution -> runtime H/W
BHWC token I/O         -> NCHW I/O
Linear projection      -> equivalent 1x1 Conv2d
```

Codex must use:

```text
reference_code/round3_companion_modules.py::SlideAttention2d
```

as the implementation reference and compare it against a pinned upstream commit before porting it to the actual YOLO repository.

This correction is important: a cleaner-looking conventional head split would create a **different module**, so it must not silently be called a faithful Slide implementation.

## 4. Primary position — parallel Slide / DBRA at P3-Cls-Mid

Graph:

```text
P3 --------------------------------------------------------> box branch
 |
 +-> cls block-0 -> U
                   |\
                   | -> existing fixed DBRA adapter ------ D
                   |
                   + -> SlideAttention2d ----------------- L

                   Y = D + beta_local * L
                   |
                   -> cls block-1 -> predictor
```

Reference wrapper:

```python
class SlideDBRAParallel(nn.Module):
    def __init__(self, dbra, slide, local_scale_init=1e-3):
        super().__init__()
        self.dbra = dbra
        self.slide = slide
        self.local_scale = nn.Parameter(torch.tensor(float(local_scale_init)))

    def forward(self, x):
        routed = self.dbra(x)
        local = self.slide(x)
        return routed + self.local_scale * local
```

At `local_scale=0`, this is exactly the DBRA parent for identical DBRA weights.

First fixed config:

```text
kernel_size      = 3
num_heads        = largest simple divisor <= 4 of active dim
local_scale_init = 1e-3
```

Do not begin with 5x5/7x7 neighborhoods; the point is local evidence preservation, not another large-context search.

## 5. Why parallel is preferred over serial

The primary hypothesis is not “Slide makes DBRA stronger.” It is:

> DBRA may benefit tiny-object detection through routed non-local evidence while losing some local/contiguous evidence relevant to other cases.

A parallel branch isolates that question better because:

- DBRA receives the same hidden feature as before;
- the new branch can learn a small correction;
- disabling the branch exactly reproduces DBRA;
- local/global branch activations can be compared directly;
- a failure does not ambiguously mean that Slide corrupted DBRA input.

## 6. Secondary design — local preconditioner before DBRA

Only after the parallel model has produced an interpretable validation signal:

```text
P3 -> cls block-0 -> U
                    |
                    -> U_local = U + beta * Slide(U)
                    -> DBRA(U_local)
                    -> cls block-1
```

Reference:

```python
class SlideThenDBRA(nn.Module):
    def forward(self, x):
        x_local = x + self.local_scale * self.slide(x)
        return self.dbra(x_local)
```

At `local_scale=0`, it is exactly DBRA.

This is riskier because it changes the routing input, but it directly tests whether DBRA routing itself benefits from locally refined evidence.

## 7. Why not use P4

Round-2 already shows DBRA P4-mid is materially worse than P3-mid, and SHSA P4-mid fails catastrophically. Round-3 is testing a companion mechanism to the successful P3 route, so both Slide designs remain P3-only.

Do not infer from lower `APm` that the physical defect “must be P4.” COCO size bins describe object size, not which pyramid level caused the error.

## 8. Mandatory control if Slide+DBRA is positive

A positive result does not prove local self-attention is necessary.

Run a matched-site simple control:

```text
DBRA + depthwise 3x3 local residual
```

with comparable local-branch width/cost.

If it matches Slide, the supported conclusion is:

> local evidence preservation complements DBRA

not:

> Slide Attention specifically is required.

## 9. Required diagnostics

Log:

```text
local_scale trajectory
DBRA alpha trajectory
local branch activation RMS
routed branch activation RMS
cosine similarity(local, routed)
AP / APs / APm
ARs / ARm
FP @ matched Recall
latency / VRAM
```

If local/routed branch cosine similarity stays near 1, the extra branch may be redundant.

## 10. Desired signature

The strongest mechanism pattern is:

```text
AP >= DBRA
APs / ARs retained
APm and/or ARm recover toward baseline
FP @ matched Recall does not worsen
```

If APm recovers only by sacrificing the main DBRA APs/ARs benefit, this is a scale trade-off rather than a clean improvement.

## 11. Stop conditions

Stop or demote if:

- local branch increases Recall and FP together;
- APm/ARm do not recover and APs/ARs decline;
- learned local scale stays effectively zero;
- the 3x3 DWConv control matches it;
- runtime cost is disproportionate to gain.

## 12. Novelty boundary

Slide is existing CVPR 2023 work and is not project novelty.

The research-level question worth preserving is:

> Can a detection classification head preserve local contiguous evidence while separately routing non-local context, instead of forcing one representation to serve both roles?

Only repeated evidence for that local/global complementarity would justify a later water-clutter-specific dual-path design.
