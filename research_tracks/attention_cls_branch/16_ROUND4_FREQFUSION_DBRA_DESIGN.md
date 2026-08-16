# Round-4 — FreqFusion + DBRA for YOLO11 Water-Surface Detection

> Status: **implementation-ready engineering design / not yet validated**  
> Target: YOLO11n, Ultralytics 8.4.113, frozen PoTATO protocol  
> Parent detector: **Round-2 DBRA P3-Cls-Mid**  
> Goal: add one neck-side feature-fusion mechanism while keeping the validated DBRA classification mechanism fixed.  
> Loss work is explicitly deferred to a later round.

---

## 1. Why this Round-4 model exists

The current project already has a useful parent model:

```text
YOLO11n + DBRA @ P3-Cls-Mid
```

Under the paper-aligned PoTATO COCO evaluation, the accepted DBRA parent improved overall AP and small-object AP/AR over the frozen YOLO11n baseline. The subsequent GRN and Slide companion experiments did not improve the DBRA parent. Therefore Round-4 should **not** add another attention beside DBRA.

The new engineering objective is to improve a different stage of the detector:

```text
Backbone -> neck cross-scale reconstruction -> P3 -> DBRA classification routing
```

FreqFusion is selected because it targets the **feature-fusion / upsampling boundary itself**, rather than adding another generic attention block.

Project interpretation:

```text
FreqFusion: improve how P4 semantic information is reconstructed and fused with high-resolution P3 detail.
DBRA:       improve how the resulting P3 classification feature selects useful contextual evidence.
```

The two mechanisms therefore operate at different locations and answer different questions.

---

## 2. Source facts that constrain the implementation

Primary source:

- Paper: *Frequency-aware Feature Fusion for Dense Image Prediction*, TPAMI 2024.
- Official repository: `https://github.com/Linwei-Chen/FreqFusion`
- Pinned repository revision for this implementation specification:
  `3fb0c70637a3c194fb74294d3ce4681958b26241`
- Pinned clean source blob for root `FreqFusion.py` at that revision:
  `b8fa94d418c3094a8d6653712b65037f70daccec`

The official implementation exposes:

```python
FreqFusion(
    hr_channels,
    lr_channels,
    scale_factor=1,
    lowpass_kernel=5,
    highpass_kernel=3,
    up_group=1,
    encoder_kernel=3,
    encoder_dilation=1,
    compressed_channels=64,
    align_corners=False,
    upsample_mode='nearest',
    feature_resample=False,
    feature_resample_group=4,
    comp_feat_upsample=True,
    use_high_pass=True,
    use_low_pass=True,
    hr_residual=True,
    semi_conv=True,
    hamming_window=True,
    feature_resample_norm=True,
)
```

The official README defines the intended calling pattern as:

```python
_, hr_feat_refined, lr_feat_up = ff(hr_feat=hr_feat, lr_feat=lr_feat)
```

and assumes the spatial size of `hr_feat` is twice that of `lr_feat`.

This matters for YOLO integration: **FreqFusion is not a single-input upsampler**. It jointly uses the high-resolution low-level feature and the low-resolution high-level feature to generate adaptive filtering / reconstruction.

### 2.1 Core mechanisms

The published / official implementation combines:

1. **ALPF — Adaptive Low-Pass Filter generator**
   - predicts spatially varying low-pass kernels;
   - reconstructs the low-resolution semantic feature at high resolution;
   - aims to reduce high-frequency inconsistency inside semantic regions.

2. **AHPF — Adaptive High-Pass Filter generator**
   - predicts high-pass filtering for the high-resolution low-level feature;
   - preserves / restores detailed boundaries.

3. **Optional local-similarity-guided resampling**
   - controlled by `feature_resample`;
   - uses similarity-guided offsets and `grid_sample` to refine the reconstructed low-resolution feature.

The official clean code defaults `feature_resample=False`. Round-4 respects that default in the primary model instead of silently enabling an additional mechanism.

---

## 3. Important correction: do NOT replace only `nn.Upsample`

Official YOLO11 top-down P3 construction is conceptually:

```text
layer 13: fused P4 feature
     |
layer 14: nearest-neighbor x2 upsample
     |
layer 15: concat with backbone P3 (layer 4)
     |
layer 16: C3k2 -> P3 detection feature
```

A naive patch would be:

```text
P4 -> FreqFusion -> concat P3
```

but a normal one-input `FreqFusion` layer cannot do that because FreqFusion needs both features at the same time.

The correct Round-4 graph is:

```text
backbone P3 (high-resolution, layer 4) ----\
                                          -> FreqFusionConcat -> C3k2 -> P3
fused P4 (low-resolution, layer 13) ------/
```

`FreqFusionConcat` performs:

```python
_, hr_refined, lr_up = freqfusion(hr_feat=backbone_p3, lr_feat=fused_p4)
fused = torch.cat([hr_refined, lr_up], dim=1)
```

So it replaces **both** the second top-down nearest upsample and the following concat node.

This is the primary architecture decision for Round-4.

---

## 4. Why only P4 -> P3 is changed first

The first implementation intentionally leaves the P5 -> P4 top-down path unchanged.

Reasons:

1. DBRA's accepted location is P3 classification mid.
2. Existing project evidence shows small-object gains are a meaningful target.
3. The P4 -> P3 fusion is the last top-down reconstruction before the high-resolution detection feature.
4. Changing both top-down fusion sites immediately would prevent attribution.
5. A positive single-site result can later justify a two-site FreqFusion ablation.

Primary model:

```text
R4-FD1 = YOLO11n + FreqFusion(P4->P3) + DBRA(P3-Cls-Mid)
```

Deferred secondary model, only after a positive R4-FD1 validation result:

```text
R4-FD2 = YOLO11n + FreqFusion(P5->P4) + FreqFusion(P4->P3) + DBRA(P3-Cls-Mid)
```

Do not prepare R4-FD2 as a rescue model after looking at test results.

---

## 5. Primary FreqFusion configuration

Use the official clean-code defaults unless a compatibility constraint requires a documented change:

```yaml
compressed_channels: 64
lowpass_kernel: 5
highpass_kernel: 3
up_group: 1
encoder_kernel: 3
encoder_dilation: 1
feature_resample: false
feature_resample_group: 4
comp_feat_upsample: true
use_high_pass: true
use_low_pass: true
hr_residual: true
semi_conv: true
hamming_window: true
feature_resample_norm: true
```

### Why `feature_resample=False` in the primary model

The official clean implementation defaults to `False`. Keeping it off in R4-FD1 gives a cleaner first experiment:

```text
ALPF + AHPF fusion effect
without simultaneously adding offset-guided resampling
```

If R4-FD1 is positive and diagnostic visualizations suggest remaining boundary displacement / spatial mismatch, a later registered ablation may enable:

```text
feature_resample=true
```

That should be treated as a separate mechanism experiment, not an unreported tuning switch.

---

## 6. DBRA is frozen as the parent mechanism

Round-4 does **not** redesign DBRA.

Reuse exactly the accepted Round-2 DBRA P3-mid implementation:

```text
P3 -> classification block-0 -> DBRA -> classification block-1 -> predictor
```

Do not change:

```text
DBRA upstream revision
heads
top-k / routing settings
window settings
deformable / agent settings
adapter alpha initialization
DBRA insertion site
```

The only intended architectural difference between DBRA parent and R4-FD1 is the P4 -> P3 fusion path.

This is essential for attribution.

---

## 7. Combined model graph

```text
                              YOLO11 backbone
                    P3(layer4)   P4(layer6)   P5(layer10)
                         |            |             |
                         |            +------ top-down ------+
                         |                                  |
                         |                    original nearest x2
                         |                                  |
                         |                              fused P4(layer13)
                         |                                  |
                         +---------- FreqFusionConcat <-----+
                                      |
                                    C3k2
                                      |
                                     P3'
                         +------------+-------------+
                         |                          |
                    box/reg branch             cls block-0
                    unchanged                      |
                                                 DBRA
                                                   |
                                              cls block-1
                                                   |
                                                predictor

P3', P4, P5 continue through the original YOLO11 PAN / Detect graph.
```

FreqFusion changes the feature reaching **both** P3 box and P3 classification branches because it is a neck module. DBRA remains classification-only.

This distinction must be stated correctly in the paper and implementation report.

---

## 8. Expected YAML graph after node replacement

Because the original layer 14 (`Upsample`) and layer 15 (`Concat`) become one layer, subsequent indices shift by -1.

Conceptual modified head:

```yaml
head:
  - [-1, 1, nn.Upsample, [None, 2, nearest]]      # 11
  - [[-1, 6], 1, Concat, [1]]                     # 12
  - [-1, 2, C3k2, [512, False]]                   # 13 fused P4

  # replaces original layers 14 + 15
  - [[4, 13], 1, FreqFusionConcat, []]             # 14 fused HR/LR concat at P3 resolution
  - [-1, 2, C3k2, [256, False]]                   # 15 P3

  - [-1, 1, Conv, [256, 3, 2]]                    # 16
  - [[-1, 13], 1, Concat, [1]]                    # 17
  - [-1, 2, C3k2, [512, False]]                   # 18 P4
  - [-1, 1, Conv, [512, 3, 2]]                    # 19
  - [[-1, 10], 1, Concat, [1]]                    # 20
  - [-1, 2, C3k2, [1024, True]]                   # 21 P5

  # final head must reuse the project's existing DBRA-enabled AttnDetect API.
  # Inputs change from [16,19,22] in stock YOLO11 to [15,18,21].
  - [[15, 18, 21], 1, AttnDetect, <EXISTING_DBRA_ARGS>]
```

`<EXISTING_DBRA_ARGS>` is intentionally not reinvented here. Codex must read the actual working DBRA YAML / head implementation and preserve its constructor/API exactly.

---

## 9. Parser support

Ultralytics `parse_model()` already handles list-input modules, e.g. `Concat`, by receiving a Python list of tensors at runtime.

Add a dedicated branch for the custom wrapper:

```python
elif m is FreqFusionConcat:
    if not isinstance(f, list) or len(f) != 2:
        raise ValueError("FreqFusionConcat requires [hr_source, lr_source]")
    hr_c, lr_c = (ch[x] for x in f)
    args = [hr_c, lr_c, *args]
    c2 = hr_c + lr_c
```

The forward input order is fixed:

```text
f[0] = high-resolution feature
f[1] = low-resolution feature
```

For R4-FD1 that means:

```yaml
from: [4, 13]
```

Do not reverse these sources.

---

## 10. Pretrained-weight behavior

The module changes the graph indices after layer 13, so blind index-based YOLO pretrained transfer can become fragile.

Required strategy:

1. Prefer starting from the already-working project DBRA model-definition code and insert FreqFusion while preserving module ordering/names as much as practical.
2. Audit every loaded key.
3. Backbone and layers before the replacement site should transfer exactly.
4. Existing DBRA parameters should transfer if the head state-dict naming remains unchanged.
5. New FreqFusion parameters are expected to initialize from their official initialization.
6. If widespread downstream YOLO weights fail to transfer only because numeric `model.<index>` keys shifted, do not silently accept it. Implement an explicit remap or choose a graph-preserving integration wrapper.

### Recommended graph-preserving alternative if weight-index transfer becomes a problem

Instead of deleting two YAML nodes, Codex may implement a small custom neck wrapper that preserves downstream module indices, **provided the computation is exactly equivalent and documented**. Do not preserve indices by keeping a useless duplicate nearest-upsample computation in the active path.

The implementation report must state which strategy was used.

---

## 11. Dependency / source policy

Do not paste an untracked copy of upstream code into the YOLO repository.

Preferred approach:

```text
ultralytics/nn/modules/third_party/freqfusion/
  freqfusion_upstream.py
  SOURCE.md
```

`SOURCE.md` must record:

```text
upstream_repo: https://github.com/Linwei-Chen/FreqFusion
upstream_commit: 3fb0c70637a3c194fb74294d3ce4681958b26241
upstream_file: FreqFusion.py
upstream_blob: b8fa94d418c3094a8d6653712b65037f70daccec
retrieval_date: <actual date>
license_status: <verified by implementer before redistribution>
local_changes:
  - import/dependency adaptation only, or exact list of semantic modifications
```

Important: the repository root did not expose a simple `LICENSE` file at the pinned revision when this specification was prepared. Codex must verify redistribution/licensing terms before vendoring upstream source. If unclear, keep upstream source external and commit only the project adapter / acquisition instructions.

The official clean code includes a self-implemented CARAFE fallback when MMCV is unavailable. Prefer the pinned upstream fallback first so the YOLO environment does not gain a heavy MMCV dependency unless profiling shows it is necessary.

---

## 12. Mandatory engineering gates

Before long training:

```text
[ ] import and YAML parse
[ ] model build
[ ] P3/P4 spatial ratio assertion passes
[ ] output fused channels = hr_channels + lr_channels
[ ] P3/P4/P5 Detect strides remain 8/16/32
[ ] existing DBRA configuration unchanged
[ ] common pretrained-weight transfer audit
[ ] finite forward
[ ] finite loss
[ ] finite backward
[ ] gradients reach ALPF/AHPF parameters
[ ] gradients reach DBRA parameters
[ ] 1-epoch smoke train
[ ] validation smoke
[ ] predict smoke
[ ] export smoke if export is a project requirement
[ ] Params/GFLOPs/VRAM/train-iteration/batch1 latency measured
```

### Shape test

For YOLO11n @ 640, expected qualitative relation is:

```text
backbone P3: H x W
fused P4:    H/2 x W/2
FreqFusion lr output: H x W
FreqFusion hr output: H x W
concat output: H x W
```

Do not hardcode `80x80` in production code; assert the 2:1 relationship dynamically.

---

## 13. Mechanism diagnostics

A positive AP result is useful, but the module should also be checked for the mechanism it claims to provide.

Log / visualize on frozen validation images:

```text
ALPF kernel entropy / spatial variation
AHPF kernel entropy / spatial variation
mean absolute change: hr_refined - hr_input
mean absolute change: lr_up - nearest(lr_input)
feature cosine similarity inside matched GT regions
feature cosine similarity across GT boundary
small/medium/large AP and AR
water-clutter FP categories if available
```

The strongest expected signature is:

```text
AP >= DBRA parent
APs retained or improved
AP75 not degraded materially
APm does not worsen further
```

If FreqFusion improves only very large objects while reducing APs, it is not serving the intended P3 reconstruction role.

---

## 14. Required ablation order

Do not train many FreqFusion variants immediately.

```text
A0 frozen YOLO11n baseline          (reuse)
A1 DBRA P3-mid parent               (reuse)
A2 DBRA + FreqFusion P4->P3         (train)
```

Only after A2 is positive on the frozen validation protocol may one of the following be registered:

```text
A3 feature_resample=True
or
A3 two-site FreqFusion
```

Do not run both simultaneously as the first follow-up.

---

## 15. What not to change in Round-4

Do not simultaneously change:

```text
loss / IoU / DFL
TAL assignment
input resolution
augmentation
optimizer / LR schedule
dataset split
DBRA parameters / insertion site
P2 detection head
another attention module
```

The purpose of this round is to isolate **FreqFusion + fixed DBRA**.

---

## 16. Paper-level wording boundary

Allowed engineering contribution framing if validated:

> We integrate frequency-aware cross-scale reconstruction into the final top-down P4-to-P3 fusion of YOLO11 and combine it with a P3 classification-only DBRA branch, separating cross-scale feature reconstruction from contextual classification routing.

Do not claim:

```text
"we invent FreqFusion"
"we are the first to use frequency-aware fusion in YOLO"
"FreqFusion proves water clutter is high-frequency noise"
```

FreqFusion is an existing TPAMI method. Project novelty, if any, lies in the task-driven architecture combination, placement, later loss design, and experimental evidence.
