# Round-4 — FreqFusion + DBRA for YOLO11 Water-Surface Detection

> Status: **implementation-ready engineering design / not yet validated**  
> Target: YOLO11n, Ultralytics 8.4.113, frozen PoTATO protocol  
> Parent detector: **accepted DBRA P3-Cls-Mid**  
> New change: **FreqFusion only at the final top-down P4 -> P3 fusion**  
> Loss redesign is deferred to a later round.

---

## 1. Design objective

Round-4 intentionally stops adding attention beside DBRA. The accepted DBRA parent already supplies content-dependent contextual routing in the P3 classification branch; GRN/Slide companion experiments did not improve that parent. The next engineering contribution should therefore operate at a different stage:

```text
Backbone -> cross-scale neck reconstruction -> P3 -> DBRA classification routing
```

The intended division of labor is:

```text
FreqFusion -> reconstruct / align P4 semantics with high-resolution P3 detail
DBRA       -> select useful context inside the resulting P3 classification representation
```

Headline candidate:

```text
R4-FD1 = YOLO11n + FreqFusion(P4->P3) + fixed DBRA(P3-Cls-Mid)
```

No loss, TAL, DFL, P2, resolution, augmentation or optimizer change belongs in this round.

---

## 2. Pinned FreqFusion source

Primary source:

- Paper: *Frequency-aware Feature Fusion for Dense Image Prediction*, TPAMI 2024.
- Official repository: `https://github.com/Linwei-Chen/FreqFusion`
- Pinned repository revision: `3fb0c70637a3c194fb74294d3ce4681958b26241`
- Root clean implementation: `FreqFusion.py`
- Pinned blob: `b8fa94d418c3094a8d6653712b65037f70daccec`

The public operator consumes two feature maps:

```python
_, hr_refined, lr_reconstructed = ff(hr_feat=hr_feat, lr_feat=lr_feat)
```

with the high-resolution feature spatially 2x the low-resolution feature.

FreqFusion contains three relevant mechanisms:

1. **ALPF** — adaptive low-pass filtering / content-aware reconstruction of the low-resolution semantic feature;
2. **AHPF** — adaptive high-pass refinement of the high-resolution feature;
3. **Local-similarity-guided feature resampling** — optional offset-based resampling of the reconstructed low-resolution branch.

### Detection-profile correction

The root clean class has `feature_resample=False` as a constructor default. However, the official Faster R-CNN/COCO FreqFusion configuration at the pinned revision explicitly uses:

```text
use_high_pass=True
use_low_pass=True
lowpass_kernel=5
highpass_kernel=3
compress_ratio=8
feature_resample=True
semi_conv=True
feature_resample_group=4
```

Therefore the primary YOLO detection transfer **must follow the detection-validated profile**, not mechanically inherit the clean-demo constructor default.

For R4-FD1:

```yaml
compress_ratio: 8
lowpass_kernel: 5
highpass_kernel: 3
up_group: 1
encoder_kernel: 3
encoder_dilation: 1
feature_resample: true
feature_resample_group: 4
comp_feat_upsample: true
use_high_pass: true
use_low_pass: true
hr_residual: true
semi_conv: true
hamming_window: true
feature_resample_norm: true
```

Set:

```text
compressed_channels = (hr_channels + lr_channels) // compress_ratio
```

rather than hard-coding `64`. This follows the official detection FPN wrapper, which derives compressed width from the two fusion inputs.

For YOLO11n P4->P3 the active scaled channels should be read from the actual parsed model; do not hard-code a model-scale-specific channel count in the implementation.

A later **core-only control** may set:

```text
feature_resample=False
```

but that is not the headline R4-FD1 model.

---

## 3. Critical integration rule: FreqFusion is not an `Upsample` replacement

Stock YOLO11 constructs the top-down P3 feature conceptually as:

```text
fused P4
   -> nearest x2
   -> concat(backbone P3)
   -> C3k2
   -> P3 detection feature
```

FreqFusion requires both the high-resolution low-level feature and low-resolution high-level feature simultaneously. Therefore the correct graph is:

```text
backbone P3 (HR) ---------\
                           -> FreqFusionConcat -> C3k2 -> P3'
fused P4 (LR) ------------/
```

Project wrapper:

```python
_, hr_refined, lr_reconstructed = self.freqfusion(
    hr_feat=backbone_p3,
    lr_feat=fused_p4,
)
fused = torch.cat((hr_refined, lr_reconstructed), dim=1)
```

So `FreqFusionConcat` replaces the **second top-down `Upsample + Concat` pair together**.

### Why concatenate instead of add

The official MMDetection `FreqFusionCARAFEFPN` refines the two lateral branches and then adds them, because FPN itself uses additive lateral fusion.

YOLO11's native top-down neck instead uses concatenation before `C3k2`. R4 preserves the native YOLO fusion topology by concatenating FreqFusion's two refined outputs. This is a deliberate **YOLO adaptation**, not a byte-for-byte transplant of the official FPN wrapper.

That distinction must be documented in the implementation report and paper.

---

## 4. Why only P4 -> P3 first

R4-FD1 modifies only the final top-down fusion.

Reasons:

1. DBRA's accepted site is P3 classification-mid;
2. P4->P3 is the last reconstruction step before the high-resolution detection feature;
3. changing P5->P4 simultaneously would destroy attribution;
4. one-site fusion is cheaper and produces a clean ablation.

Deferred only after positive validation evidence:

```text
R4-FD2 = FreqFusion(P5->P4) + FreqFusion(P4->P3) + DBRA
```

Do not authorize R4-FD2 from test-set feedback.

---

## 5. DBRA parent remains fixed

Reuse the exact accepted Round-2 DBRA implementation/configuration:

```text
P3 -> cls block-0 -> DBRA -> cls block-1 -> predictor
```

Do not change:

```text
DBRA upstream revision
heads / windows / top-k
agent/deformable settings
adapter alpha initialization
P3 level
cls-mid insertion site
```

FreqFusion is a neck module, so it changes the P3 feature seen by both P3 box and classification towers. DBRA itself remains classification-only.

---

## 6. Combined model graph

```text
                              YOLO11 backbone
                    P3            P4             P5
                     |             |              |
                     |             +--- top-down -+
                     |                    |
                     |             stock P5->P4 fusion
                     |                    |
                     |                 fused P4
                     |                    |
                     +--- FreqFusionConcat+
                               |
                              C3k2
                               |
                              P3'
                    +----------+-----------+
                    |                      |
               box/reg tower          cls block-0
                                           |
                                          DBRA
                                           |
                                      cls block-1
                                           |
                                        predictor
```

The remainder of the bottom-up PAN and P3/P4/P5 Detect topology stays unchanged except for unavoidable YAML index shifts.

---

## 7. Conceptual YAML

Starting from stock-like YOLO11 indexing, the original layer 14 `Upsample` and layer 15 `Concat` become one node:

```yaml
head:
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]   # 11
  - [[-1, 6], 1, Concat, [1]]                    # 12
  - [-1, 2, C3k2, [512, False]]                  # 13 fused P4

  - [[4, 13], 1, FreqFusionConcat, []]            # 14 HR=P3, LR=P4
  - [-1, 2, C3k2, [256, False]]                  # 15 P3

  - [-1, 1, Conv, [256, 3, 2]]                   # 16
  - [[-1, 13], 1, Concat, [1]]                   # 17
  - [-1, 2, C3k2, [512, False]]                  # 18 P4
  - [-1, 1, Conv, [512, 3, 2]]                   # 19
  - [[-1, 10], 1, Concat, [1]]                   # 20
  - [-1, 2, C3k2, [1024, True]]                  # 21 P5

  - [[15, 18, 21], 1, AttnDetect, <EXACT_EXISTING_DBRA_ARGS>]
```

This is conceptual only. Codex must start from the **actual accepted DBRA parent YAML** and derive the real indices/API from that source of truth.

Input order is mandatory:

```text
f[0] = HR backbone P3
f[1] = LR fused P4
```

---

## 8. `parse_model()` support

Add a dedicated multi-input branch:

```python
elif m is FreqFusionConcat:
    if not isinstance(f, list) or len(f) != 2:
        raise ValueError("FreqFusionConcat requires [hr_source, lr_source]")
    hr_c, lr_c = (ch[x] for x in f)
    args = [hr_c, lr_c, *args]
    c2 = hr_c + lr_c
```

Do not register this as a normal single-input base module or repeat module.

The wrapper must dynamically assert:

```text
H_hr == 2 * H_lr
W_hr == 2 * W_lr
```

and output:

```text
[B, C_hr + C_lr, H_hr, W_hr]
```

---

## 9. Source/dependency policy

Do not paste an untracked copy of upstream code into the actual YOLO repository.

Preferred structure if redistribution is verified:

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
license_status: <verified before redistribution>
local_changes: <exact list>
```

At specification time, a simple repository-root `LICENSE` file was not resolved at the pinned revision. Codex must verify redistribution terms before vendoring.

### CARAFE backend

The clean upstream file contains a non-MMCV fallback implementation. That fallback uses unfold/interpolation and must be profiled for memory/latency. The pinned fallback also contains debug `print(...)` statements around its CARAFE tensor shapes; if this fallback is vendored, remove those prints as a documented **semantic-neutral integration patch**.

Do not add MMCV automatically. Prefer the pinned implementation first, then benchmark if another backend is necessary.

---

## 10. Weight-transfer risk

Replacing two YAML nodes with one shifts downstream numeric `model.<index>` keys. A generic partial-load message is insufficient.

Required audit:

1. enumerate parent DBRA state-dict keys;
2. enumerate R4-FD1 keys;
3. identify exact semantic modules whose prefixes shifted only because of graph indexing;
4. explicitly remap those keys where safe;
5. confirm all DBRA parent weights load into the same DBRA module;
6. list genuinely new FreqFusion parameters separately.

If graph-index shifting causes a large unintended random portion of YOLO, stop before training.

A graph-preserving integration is allowed if it preserves the intended computation without executing an unused nearest-upsample branch; document the strategy.

---

## 11. Mandatory engineering gates

Before long training:

```text
[ ] source/provenance verified
[ ] import / YAML parse / model build
[ ] HR/LR source order verified
[ ] exact 2:1 spatial relation verified dynamically
[ ] output channels == C_hr + C_lr
[ ] Detect strides remain 8/16/32
[ ] DBRA source/config/site unchanged
[ ] full parent-to-R4 weight-transfer audit
[ ] finite FP32 forward / loss / backward
[ ] AMP forward/backward if training uses AMP
[ ] ALPF gradients finite
[ ] AHPF gradients finite
[ ] resampler/offset gradients finite (primary profile enables it)
[ ] DBRA gradients finite
[ ] 1-epoch smoke train
[ ] smoke val / predict
[ ] export smoke if required
[ ] Params / GFLOPs / VRAM / iteration time / latency P50-P95
```

---

## 12. Mechanism diagnostics

On frozen validation images, record where practical:

```text
ALPF kernel entropy / spatial variation
AHPF kernel entropy / spatial variation
offset magnitude / spatial variation from the resampler
mean |hr_refined - hr_input|
mean |lr_reconstructed - nearest(lr_input)|
feature similarity inside GT regions
feature similarity across GT boundaries
AP/APs/APm/APl and ARs/ARm/ARl
water-clutter FP taxonomy if available
```

Desired signature:

```text
AP > DBRA parent
APs retained or improved
AP75 not materially degraded
APm does not worsen further
cost increase acceptable
```

---

## 13. Ablation order

First round:

```text
A0 frozen YOLO11n baseline                 reuse
A1 DBRA P3-mid parent                      reuse
A2 DBRA + detection-profile FreqFusion     train
```

If A2 is positive on frozen validation, register one follow-up at a time:

```text
A3-core: feature_resample=False
```

This separates ALPF+AHPF from the local-similarity resampler and is an attribution control.

Only after that, if justified:

```text
A4: two-site FreqFusion
```

Do not grid-search kernels/compressed width from test feedback.

---

## 14. Round-4 exclusions

Do not simultaneously change:

```text
loss / IoU / DFL / TAL
input resolution
augmentation
optimizer / LR / epochs
dataset split
DBRA parameters or insertion site
P2
another attention/fusion module
```

Loss innovation is explicitly deferred to the next research stage.

---

## 15. Paper wording boundary

If validated, a defensible engineering framing is:

> We adapt frequency-aware cross-scale reconstruction to the final P4-to-P3 top-down fusion of YOLO11 while retaining a classification-only DBRA branch at P3, separating cross-scale feature reconstruction from contextual classification routing.

Do not claim invention of FreqFusion or DBRA, and do not reduce the mechanism to “water clutter is high frequency.”
