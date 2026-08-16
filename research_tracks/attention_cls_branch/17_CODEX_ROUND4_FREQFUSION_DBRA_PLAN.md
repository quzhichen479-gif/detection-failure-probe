# Codex Round-4 Implementation Plan — FreqFusion + Fixed DBRA

> **Primary execution plan**  
> Target: YOLO11n / Ultralytics 8.4.113 / frozen PoTATO protocol  
> Parent: accepted **DBRA P3-Cls-Mid**  
> New component: **FreqFusion at P4 -> P3 only**  
> Loss redesign: **deferred**

---

## 0. Read in this order

```text
research_tracks/attention_cls_branch/05_BASELINE_AND_TRAINING_PROTOCOL.md
research_tracks/attention_cls_branch/07_DBRA.md
research_tracks/attention_cls_branch/11_ROUND2_TEST_EVIDENCE_AND_ROUND3_HYPOTHESES.md
research_tracks/attention_cls_branch/16_ROUND4_FREQFUSION_DBRA_DESIGN.md
research_tracks/attention_cls_branch/reference_code/freqfusion_yolo_adapter.py
research_tracks/attention_cls_branch/reference_code/test_freqfusion_yolo_adapter.py
research_tracks/attention_cls_branch/17_CODEX_ROUND4_FREQFUSION_DBRA_PLAN.md
```

Then inspect the actual accepted DBRA implementation/YAML in the real YOLO engineering repository. That working implementation is the source of truth for DBRA class names, args and state-dict layout.

---

## 1. Build exactly one headline candidate

```text
R4-FD1 = YOLO11n
       + FreqFusion at final top-down P4->P3 fusion
       + existing DBRA at P3 classification-mid
```

Do not change:

```text
loss / IoU / DFL / TAL
DBRA settings or site
P2
imgsz
augmentation
optimizer / LR / epochs
dataset split
another attention/fusion module
```

---

## 2. Lock FreqFusion source/provenance

Use:

```text
upstream: https://github.com/Linwei-Chen/FreqFusion
commit:   3fb0c70637a3c194fb74294d3ce4681958b26241
file:     FreqFusion.py
blob:     b8fa94d418c3094a8d6653712b65037f70daccec
```

Before vendoring upstream code:

```text
[ ] verify redistribution/license terms
[ ] record retrieval date
[ ] record exact source commit/blob
[ ] list every local modification
```

If redistribution terms are unclear, keep upstream outside the repository and commit only project-authored adapter/instructions.

Suggested source record if vendoring is permitted:

```text
ultralytics/nn/modules/third_party/freqfusion/
  freqfusion_upstream.py
  SOURCE.md
```

If the pinned clean fallback CARAFE is used, remove its tensor-shape debug `print(...)` statements as a documented semantic-neutral patch. Do not add MMCV automatically; benchmark the fallback first.

---

## 3. Primary FreqFusion profile = official object-detection profile

Do **not** use the root class's `feature_resample=False` default as the headline detector setting.

The official Faster R-CNN/COCO config uses:

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

Therefore R4-FD1 uses:

```text
compress_ratio          = 8
compressed_channels     = (C_hr + C_lr) // 8
lowpass_kernel          = 5
highpass_kernel         = 3
up_group                = 1
encoder_kernel          = 3
encoder_dilation        = 1
feature_resample        = True
feature_resample_group  = 4
comp_feat_upsample      = True
use_high_pass           = True
use_low_pass            = True
hr_residual             = True
semi_conv               = True
hamming_window          = True
feature_resample_norm   = True
```

`feature_resample=False` is reserved as a later **core-only attribution control** if R4-FD1 is positive.

No first-pass kernel/grid search.

---

## 4. Port `FreqFusionConcat`

Reference:

```text
research_tracks/attention_cls_branch/reference_code/freqfusion_yolo_adapter.py
```

Suggested actual YOLO path:

```text
ultralytics/nn/modules/freqfusion_yolo.py
```

Contract:

```text
input[0] = HR backbone P3
input[1] = LR fused P4
output   = cat(hr_refined, lr_reconstructed), dim=1
```

The wrapper must assert dynamically:

```text
H_hr == 2 * H_lr
W_hr == 2 * W_lr
```

and output:

```text
[B, C_hr + C_lr, H_hr, W_hr]
```

### Why concat

Official FreqFusionCARAFEFPN adds the refined branches because FPN uses additive lateral fusion. YOLO11 natively concatenates top-down and backbone features. Keep YOLO's concat semantics after refining both branches.

---

## 5. Integrate imports and `parse_model()`

Expose `FreqFusionConcat` via the normal Ultralytics module import chain and import it in `tasks.py`.

Add an explicit parser branch:

```python
elif m is FreqFusionConcat:
    if not isinstance(f, list) or len(f) != 2:
        raise ValueError("FreqFusionConcat requires [hr_source, lr_source]")
    hr_c, lr_c = (ch[x] for x in f)
    args = [hr_c, lr_c, *args]
    c2 = hr_c + lr_c
```

Do not add it to unrelated base/repeat module sets.

Add a parser unit test proving source order and output channel bookkeeping.

---

## 6. Build YAML from the accepted DBRA parent, not from memory

Procedure:

1. copy the actual accepted DBRA P3-mid YAML;
2. leave all DBRA args unchanged;
3. replace only the final P4->P3 `Upsample + Concat` pair with `FreqFusionConcat`;
4. update all downstream indices carefully;
5. generate a parent-vs-R4 structural diff.

Stock-like conceptual graph:

```yaml
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]   # 11 P5->P4 stays stock
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
  - [[15, 18, 21], 1, AttnDetect, <EXACT_PARENT_DBRA_ARGS>]
```

If actual DBRA-parent indices differ, derive from the real graph rather than forcing these numbers.

Suggested candidate YAML:

```text
ultralytics/cfg/models/11/yolo11-dbra-freqfusion-p3.yaml
```

---

## 7. Mandatory parent-identity audit

Round-4 changes only the neck fusion path. Generate:

```text
implementation/ROUND4_PARENT_CONFIG_DIFF.md
```

Allowed differences:

```text
FreqFusion source/adapter/parser plumbing
P4->P3 fusion node replacement
necessary graph-index updates
run/output naming
```

Forbidden differences:

```text
DBRA class/source/config/site
loss/TAL/DFL
training recipe
input resolution
other model blocks
```

---

## 8. Weight-transfer audit is mandatory

Replacing two nodes with one can shift numeric `model.<index>` prefixes.

Before training:

```text
[ ] enumerate DBRA-parent keys
[ ] enumerate R4 keys
[ ] match semantic modules before/after index shift
[ ] explicitly remap safe index-only changes
[ ] confirm DBRA weights load into the same semantic DBRA module
[ ] list genuinely new FreqFusion tensors separately
```

Do not accept a generic `Transferred X/Y items` line as proof.

If a large portion of downstream YOLO becomes randomly initialized only because numeric indices shifted, stop and fix the transfer path.

---

## 9. Tests

Use/adapt:

```text
reference_code/test_freqfusion_yolo_adapter.py
```

Required tests:

```text
A. parser: [HR,LR] order and c2=C_hr+C_lr
B. shape: HR HxW + LR H/2xW/2 -> fused HxW
C. invalid spatial ratio rejected
D. compressed_channels derived from compress_ratio=8
E. resampler grouping / normalization compatibility
F. finite gradients through ALPF/AHPF/resampler
G. full model still has Detect strides 8/16/32
H. DBRA class/config/site identical to parent
```

Also run with the actual pinned FreqFusion operator, not only a dummy stand-in.

---

## 10. Smoke and cost gates

Run in order:

```text
1 import
2 YAML parse
3 model build
4 parent->R4 weight-transfer audit
5 FP32 forward
6 AMP forward/backward if used
7 loss forward/backward
8 one-epoch smoke train
9 smoke val
10 smoke predict
11 export smoke if required
12 Params/GFLOPs/VRAM/train-step/batch1 latency P50/P95
```

Primary profile has offset-based resampling, so explicitly check:

```text
offset/resampler gradients
finite grid_sample outputs
AMP stability
VRAM impact
```

---

## 11. Formal training fairness

Reuse the frozen DBRA-parent training recipe and allowed initialization source.

Do not:

```text
retrain baseline merely to start R4
load a fully-trained DBRA best.pt and give R4 another full schedule as headline comparison
change seed policy or training budget
```

Reuse existing baseline/DBRA results as comparators.

Architecture advancement is decided from the frozen validation protocol, not repeated test feedback.

---

## 12. First ablation table

```text
A0 YOLO11n baseline                                 reuse
A1 YOLO11n + DBRA P3-mid                           reuse
A2 YOLO11n + DBRA + detection-profile FreqFusion  train
```

If A2 is positive, the first attribution control is:

```text
A3-core = same model, feature_resample=False
```

This tests whether ALPF+AHPF alone explain the gain or whether local-similarity-guided resampling contributes materially.

Only later, if justified, consider two-site FreqFusion.

---

## 13. Evaluation and diagnostics

Headline:

```text
Precision / Recall / mAP50 / mAP75 / mAP50-95
```

Paper-aligned where authorized:

```text
AP / APs / APm / APl / ARs / ARm / ARl
```

Diagnostics where practical:

```text
ALPF/AHPF kernel variation
offset magnitude/distribution
|hr_refined-hr_input|
|lr_reconstructed-nearest(lr_input)|
P3 feature similarity inside GT and across boundaries
water-clutter FP categories
```

Desired signature:

```text
AP > DBRA parent
APs retained/improved
AP75 not materially lower
APm not further degraded
acceptable latency/VRAM cost
```

If negative, diagnose implementation, transfer and mechanism before tuning kernels or widths.

---

## 14. Required outputs

Before long training:

```text
ROUND4_FREQFUSION_DBRA_INTEGRATION_REPORT.md
```

Include:

1. YOLO repo commit/dirty status;
2. Ultralytics version;
3. exact DBRA parent source/config;
4. FreqFusion repo/commit/blob/license status;
5. local FreqFusion patches, including fallback debug-print removal if applicable;
6. final wrapper constructor/profile;
7. actual YAML graph;
8. parent config diff;
9. state-dict transfer audit;
10. unit/smoke/gradient results;
11. Params/GFLOPs/VRAM/latency;
12. deviations from this plan.

After formal validation:

```text
ROUND4_FREQFUSION_DBRA_VALIDATION_REPORT.md
round4_freqfusion_dbra_summary.csv
round4_freqfusion_dbra_summary.json
```

---

## 15. Short Codex invocation

```text
Read research_tracks/attention_cls_branch/05_BASELINE_AND_TRAINING_PROTOCOL.md, 07_DBRA.md, 11_ROUND2_TEST_EVIDENCE_AND_ROUND3_HYPOTHESES.md, 16_ROUND4_FREQFUSION_DBRA_DESIGN.md, reference_code/freqfusion_yolo_adapter.py, reference_code/test_freqfusion_yolo_adapter.py, and 17_CODEX_ROUND4_FREQFUSION_DBRA_PLAN.md. Then inspect the actual accepted DBRA P3-mid implementation/YAML in the YOLO11 engineering repo. Implement exactly one Round-4 candidate: fixed DBRA P3-Cls-Mid plus FreqFusion only at the final P4->P3 top-down fusion. Treat FreqFusion as a two-input operator replacing the second Upsample+Concat pair: FreqFusionConcat([backbone_P3, fused_P4]); preserve YOLO concat semantics after refining both branches. Pin upstream to commit 3fb0c70637a3c194fb74294d3ce4681958b26241, verify license/provenance before vendoring, and use the official object-detection profile: compress_ratio=8, lowpass=5, highpass=3, feature_resample=True, feature_resample_group=4, semi_conv=True, high/low-pass enabled. Do not modify DBRA, loss, TAL, DFL, imgsz, augmentation, optimizer, schedule, dataset split, or baseline. Add module/parser support, exact YAML, explicit state-dict remapping/audit for graph-index shifts, unit/gradient/shape tests, smoke train/val/predict and cost profiling. Generate ROUND4_FREQFUSION_DBRA_INTEGRATION_REPORT.md before long training. Train only after all gates pass, using the frozen parent protocol and validation-side selection.
```
