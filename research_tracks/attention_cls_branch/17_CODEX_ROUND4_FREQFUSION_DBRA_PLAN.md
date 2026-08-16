# Codex Round-4 Implementation Plan — FreqFusion + Fixed DBRA

> **Primary execution entry for Round-4**  
> Target: YOLO11n / Ultralytics 8.4.113 / frozen PoTATO training protocol  
> Parent: accepted **DBRA P3-Cls-Mid** model  
> New component: **FreqFusion only at P4 -> P3 top-down fusion**  
> Loss redesign: **deferred; do not touch in this round**

---

## 0. Read order

Before editing the real YOLO repository, read these files in order:

```text
research_tracks/attention_cls_branch/05_BASELINE_AND_TRAINING_PROTOCOL.md
research_tracks/attention_cls_branch/07_DBRA.md
research_tracks/attention_cls_branch/11_ROUND2_TEST_EVIDENCE_AND_ROUND3_HYPOTHESES.md
research_tracks/attention_cls_branch/16_ROUND4_FREQFUSION_DBRA_DESIGN.md
research_tracks/attention_cls_branch/reference_code/freqfusion_yolo_adapter.py
research_tracks/attention_cls_branch/17_CODEX_ROUND4_FREQFUSION_DBRA_PLAN.md
```

Also inspect the **actual working Round-2 DBRA implementation and YAML** in the detector engineering repository. The research repository does not replace that source of truth.

---

## 1. Goal

Build exactly one headline Round-4 architecture:

```text
R4-FD1 = YOLO11n
       + FreqFusion at P4->P3 top-down fusion
       + existing DBRA at P3 classification-mid
```

No other architectural or training change is permitted in the headline comparison.

The intended division of labor is:

```text
FreqFusion -> cross-scale feature reconstruction / fusion
DBRA       -> P3 classification context routing
```

Do not add GRN, Slide, FocalMod, P2, a new IoU loss, a new DFL loss, or a new assigner in Round-4.

---

## 2. Lock upstream FreqFusion provenance first

Pinned source specification:

```text
repository: https://github.com/Linwei-Chen/FreqFusion
commit:     3fb0c70637a3c194fb74294d3ce4681958b26241
file:       FreqFusion.py
blob:       b8fa94d418c3094a8d6653712b65037f70daccec
```

Before redistribution/vendor:

```text
[ ] inspect upstream repository terms/license
[ ] record retrieval date
[ ] record whether source is copied, patched, or imported externally
[ ] record all local semantic changes
```

The specification author did not find a simple root `LICENSE` file at the pinned revision. Do not silently copy upstream source without checking redistribution terms.

Recommended real-YOLO layout if vendoring is permitted:

```text
ultralytics/nn/modules/third_party/freqfusion/
    freqfusion_upstream.py
    SOURCE.md
```

`SOURCE.md` must contain the pinned repo/commit/blob and a local modification list.

### Dependency policy

The official clean root `FreqFusion.py` contains a fallback CARAFE implementation when `mmcv.ops.carafe` is unavailable. For the first YOLO integration:

1. prefer the pinned official fallback path;
2. do not add MMCV merely because upstream historically used it;
3. profile memory/latency;
4. only introduce MMCV if the fallback is functionally insufficient and the dependency cost is accepted.

Do not write a third unrelated CARAFE implementation unless necessary.

---

## 3. Port the project adapter

Port:

```text
research_tracks/attention_cls_branch/reference_code/freqfusion_yolo_adapter.py
```

into the actual detector repository, suggested path:

```text
ultralytics/nn/modules/freqfusion_yolo.py
```

The production class is:

```python
FreqFusionConcat
```

Its contract is fixed:

```text
input[0] = high-resolution low-level feature
input[1] = low-resolution high-level feature
output   = concat(refined_hr, reconstructed_lr), dim=1
```

Primary YOLO sources:

```text
input[0] = backbone P3, original YOLO11 layer 4
input[1] = fused P4, original YOLO11 layer 13
```

Do not reverse them.

The wrapper must assert dynamically:

```text
H_hr == 2 * H_lr
W_hr == 2 * W_lr
```

and must not hardcode 80x80 / 40x40.

---

## 4. Use the official-default FreqFusion profile first

Primary R4-FD1 config:

```text
compressed_channels      = 64
lowpass_kernel           = 5
highpass_kernel          = 3
up_group                 = 1
encoder_kernel           = 3
encoder_dilation         = 1
feature_resample         = False
feature_resample_group   = 4
comp_feat_upsample       = True
use_high_pass            = True
use_low_pass             = True
hr_residual              = True
semi_conv                = True
hamming_window           = True
feature_resample_norm    = True
```

Do not tune these before the first frozen-validation result.

`feature_resample=False` is deliberate because it matches the official clean-code default and isolates ALPF/AHPF behavior. Offset-guided resampling can be a registered follow-up only if R4-FD1 is already justified.

---

## 5. Ultralytics module export / import

Expose the class through the existing module import chain.

Typical steps:

```python
# ultralytics/nn/modules/__init__.py
from .freqfusion_yolo import FreqFusionConcat
```

and ensure `ultralytics/nn/tasks.py` imports it into the namespace used by `parse_model()`.

Follow the local repository's existing style; do not create a second parser or monkey-patch globals at runtime.

---

## 6. `parse_model()` special case

Ultralytics runtime already passes list inputs to multi-source modules. Add one explicit branch near `Concat` handling:

```python
elif m is FreqFusionConcat:
    if not isinstance(f, list) or len(f) != 2:
        raise ValueError("FreqFusionConcat requires [hr_source, lr_source]")

    hr_c, lr_c = (ch[x] for x in f)
    args = [hr_c, lr_c, *args]
    c2 = hr_c + lr_c
```

Notes:

- `f[0]` is HR, `f[1]` is LR.
- output channels equal the concat role it replaces.
- do not treat it as a normal single-input base module.
- do not add it to an unrelated repeat-module set.

Add a parser unit test for source order and output-channel bookkeeping.

---

## 7. Build the modified YAML from the actual working DBRA YAML

**Do not start from stock YOLO11 and reconstruct DBRA from memory.**

Procedure:

1. copy the exact working DBRA P3-mid model YAML used for the accepted parent;
2. keep all DBRA head args unchanged;
3. replace only the P4->P3 `Upsample + Concat` pair with `FreqFusionConcat`;
4. update downstream graph indices carefully;
5. diff the final YAML against the working DBRA parent.

If the parent YAML follows stock YOLO11 indexing, the conceptual new head is:

```yaml
head:
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]    # 11
  - [[-1, 6], 1, Concat, [1]]                     # 12
  - [-1, 2, C3k2, [512, False]]                   # 13 fused P4

  - [[4, 13], 1, FreqFusionConcat, []]             # 14 replaces old 14+15
  - [-1, 2, C3k2, [256, False]]                   # 15 P3

  - [-1, 1, Conv, [256, 3, 2]]                    # 16
  - [[-1, 13], 1, Concat, [1]]                    # 17
  - [-1, 2, C3k2, [512, False]]                   # 18 P4
  - [-1, 1, Conv, [512, 3, 2]]                    # 19
  - [[-1, 10], 1, Concat, [1]]                    # 20
  - [-1, 2, C3k2, [1024, True]]                   # 21 P5

  - [[15, 18, 21], 1, AttnDetect, <EXACT_EXISTING_DBRA_ARGS>]
```

If the actual DBRA repository has different indices/API, the actual working parent wins. The semantic requirement is what must be preserved.

Suggested output YAML name:

```text
ultralytics/cfg/models/11/yolo11-dbra-freqfusion-p3.yaml
```

---

## 8. Protect DBRA identity

Round-4 must not create a new DBRA variant.

Generate an automated comparison between the parent DBRA config and R4-FD1 config. Allowed architecture changes:

```text
new FreqFusion-related source files
new FreqFusion parser/import plumbing
P4->P3 fusion node replacement
necessary downstream YAML index updates
run name / output path
```

Forbidden differences:

```text
DBRA constructor args
DBRA source code
DBRA insertion level/site
Detect loss configuration
TAL / DFL / CIoU
P2/P4/P5 attention changes
imgsz
training hyperparameters
```

Write the diff to:

```text
implementation/ROUND4_PARENT_CONFIG_DIFF.md
```

---

## 9. Weight-transfer strategy

This is a high-risk engineering detail because deleting one YAML node can shift numeric `model.<index>` state-dict prefixes.

Required audit:

```text
[ ] enumerate parent DBRA state_dict keys
[ ] enumerate R4-FD1 state_dict keys
[ ] report exact/common-shape matches
[ ] identify keys changed only because module indices shifted
[ ] explicitly remap such keys if safe
[ ] identify genuinely new FreqFusion parameters
[ ] verify DBRA weights load into the same semantic DBRA module
```

Do not accept a log such as:

```text
Transferred 2xx/4xx items
```

without checking which tensors were missed.

### Preferred implementation choice

If possible, design the model graph so pretrained semantic modules keep stable state-dict names. If index shifting is unavoidable, write an explicit, tested remap for the known downstream layers rather than relying on accidental partial loading.

The integration report must include a key-by-key summary.

---

## 10. Unit tests

Port / adapt the reference tests in:

```text
research_tracks/attention_cls_branch/reference_code/test_freqfusion_yolo_adapter.py
```

At minimum add tests for:

### A. parser bookkeeping

```text
from=[hr, lr]
constructor receives hr_c, lr_c in that order
c2 == hr_c + lr_c
```

### B. shape contract

```text
hr: [B, C1, H, W]
lr: [B, C2, H/2, W/2]
out: [B, C1+C2, H, W]
```

### C. invalid ratio

Reject non-2:1 feature sizes.

### D. gradients

After one backward pass:

```text
ALPF/content encoder receives finite gradients
AHPF/content encoder2 receives finite gradients
DBRA parameters receive finite gradients in the full model
```

### E. box/head graph

Verify Detect still consumes exactly three levels with strides:

```text
8 / 16 / 32
```

### F. DBRA identity

Verify the same DBRA class/config/site is present as in the parent model.

---

## 11. Smoke gates before formal training

Run in this order:

```text
1. import
2. YAML parse
3. model build
4. parent -> R4 weight-transfer audit
5. dummy FP32 forward
6. dummy AMP forward if training uses AMP
7. loss forward
8. backward
9. 1-epoch smoke train
10. smoke val
11. smoke predict
12. TorchScript/ONNX/export smoke only if required by the project
```

Stop on NaN/Inf or shape mismatch. Do not rescue by changing FreqFusion hyperparameters before determining whether the problem is implementation or optimization.

---

## 12. Cost audit

On the same hardware and settings used for DBRA parent, record:

```text
Params
GFLOPs
peak VRAM
training iteration time
batch-1 inference latency P50 / P95
batch-8 inference throughput if relevant
```

FreqFusion's fallback CARAFE uses unfold/interpolation and may have memory cost. Measure it rather than assuming it is lightweight.

If MMCV and fallback implementations are both tested, report them as **backend implementations**, not separate architecture candidates.

---

## 13. Formal training protocol

Reuse the exact frozen DBRA-parent training recipe.

Do not:

```text
retrain baseline just to start this round
continue-train the old DBRA best.pt for an extra full schedule and call that fair
change augmentation
change optimizer / LR
change epochs
change image size
change seed policy
```

The formal R4-FD1 candidate must receive the same allowed initialization source and same optimization budget as the architecture comparisons defined by the project protocol.

The previously trained DBRA result is reused as parent comparator.

---

## 14. Selection/evaluation boundary

The repeatedly used test set is not an iterative tuning set.

Round-4 architecture/config decisions are frozen before formal candidate training. Use the existing validation protocol to decide whether R4-FD1 advances.

Primary metrics:

```text
Precision
Recall
mAP50
mAP75
mAP50-95
```

Paper-aligned metrics when authorized:

```text
AP
APs APm APl
ARs ARm ARl
```

Mechanism diagnostics:

```text
nearest(P4) vs FreqFusion reconstructed P4 feature difference
P3 feature intra-region cosine similarity
boundary similarity margin
water-background FP taxonomy if available
```

---

## 15. Decision rule

The strongest desired result is:

```text
overall AP > DBRA parent
APs >= DBRA parent or only trivially lower
AP75 not materially lower
APm recovers or at least does not worsen
cost increase remains acceptable
```

If R4-FD1 is negative, do not immediately grid-search kernels/compressed width.

First inspect:

```text
weight-transfer correctness
ALPF/AHPF gradient health
feature-change magnitude
fallback CARAFE correctness/cost
whether high-pass refinement amplifies structured water clutter
```

Only a mechanism-supported diagnosis can authorize a second FreqFusion configuration.

---

## 16. Deferred follow-ups

These are **not part of the first implementation**:

### Follow-up A — offset-guided FreqFusion

```text
feature_resample=True
```

Only if the core fusion is positive and spatial mismatch remains plausible.

### Follow-up B — two-site FreqFusion

Add P5->P4 as well as P4->P3 only after single-site evidence.

### Follow-up C — new loss

Loss design is a separate research round. Do not couple it to the first FreqFusion result.

---

## 17. Required Codex outputs

Before long training, create in the real YOLO implementation workspace:

```text
ROUND4_FREQFUSION_DBRA_INTEGRATION_REPORT.md
```

It must include:

1. YOLO repository commit / dirty state;
2. Ultralytics version (`8.4.113` expected);
3. exact DBRA parent implementation/config reused;
4. FreqFusion upstream repo/commit/blob and license/provenance status;
5. files changed;
6. final FreqFusion wrapper constructor;
7. actual YAML graph and indices;
8. parent-vs-R4 config diff;
9. full weight-transfer audit;
10. unit/smoke test results;
11. gradient checks;
12. Params/GFLOPs/VRAM/latency;
13. deviations from this specification.

After formal validation training, create:

```text
ROUND4_FREQFUSION_DBRA_VALIDATION_REPORT.md
round4_freqfusion_dbra_summary.csv
round4_freqfusion_dbra_summary.json
```

Do not access the held/descriptive test set unless the project protocol explicitly authorizes it after validation-side selection is frozen.

---

## 18. Short Codex invocation prompt

```text
Read research_tracks/attention_cls_branch/05_BASELINE_AND_TRAINING_PROTOCOL.md, 07_DBRA.md, 11_ROUND2_TEST_EVIDENCE_AND_ROUND3_HYPOTHESES.md, 16_ROUND4_FREQFUSION_DBRA_DESIGN.md, reference_code/freqfusion_yolo_adapter.py, reference_code/test_freqfusion_yolo_adapter.py, and 17_CODEX_ROUND4_FREQFUSION_DBRA_PLAN.md. In the actual YOLO11 engineering repository, implement exactly one Round-4 candidate: the accepted DBRA P3-Cls-Mid parent plus FreqFusion only at the P4->P3 top-down fusion. Treat FreqFusion as a two-input fusion operator replacing the original second Upsample+Concat pair, not as a one-input upsampler. Pin upstream FreqFusion to commit 3fb0c70637a3c194fb74294d3ce4681958b26241 and verify redistribution/license terms before vendoring. Keep the official-default primary profile with feature_resample=False. Reuse DBRA implementation/config/site unchanged, do not modify loss/TAL/DFL/imgsz/augmentation/training hyperparameters, and do not retrain the frozen baseline. First complete parser/import integration, exact graph/YAML, parent-to-R4 weight-transfer audit, unit tests, gradient checks, smoke train/val/predict, and cost profiling. Generate ROUND4_FREQFUSION_DBRA_INTEGRATION_REPORT.md before any long training. Only after all gates pass, train R4-FD1 using the exact frozen parent protocol and select solely on the frozen validation protocol.
```
