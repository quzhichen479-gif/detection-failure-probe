# Round-4 FreqFusion + DBRA — Codex README

> **Use this file as the short entry point for the next implementation session.**  
> Target: YOLO11n / Ultralytics 8.4.113 / PoTATO  
> Model: **FreqFusion(P4->P3) + fixed DBRA(P3-Cls-Mid)**  
> Loss: unchanged in this round.

---

## 1. What to build

Build exactly:

```text
YOLO11n
  + FreqFusion at the final top-down P4->P3 fusion
  + the already-validated DBRA at P3 classification-mid
```

Do **not** add any other attention, P2 head, new loss, DFL change, TAL change, image-size change, or training-hyperparameter change.

The model responsibilities are:

```text
FreqFusion -> reconstruct / align high- and low-resolution neck features before P3
DBRA       -> route useful context only in the P3 classification branch
```

---

## 2. Required reading order

From repository root, read:

```text
research_tracks/attention_cls_branch/05_BASELINE_AND_TRAINING_PROTOCOL.md
research_tracks/attention_cls_branch/07_DBRA.md
research_tracks/attention_cls_branch/11_ROUND2_TEST_EVIDENCE_AND_ROUND3_HYPOTHESES.md
research_tracks/attention_cls_branch/16_ROUND4_FREQFUSION_DBRA_DESIGN.md
research_tracks/attention_cls_branch/reference_code/freqfusion_yolo_adapter.py
research_tracks/attention_cls_branch/reference_code/test_freqfusion_yolo_adapter.py
research_tracks/attention_cls_branch/17_CODEX_ROUND4_FREQFUSION_DBRA_PLAN.md
```

Then inspect the **actual working DBRA P3-mid implementation and YAML** in the real YOLO engineering repository. Preserve that DBRA implementation exactly.

---

## 3. Critical architecture rule

Do not implement FreqFusion as:

```text
P4 -> FreqFusion -> P3 concat
```

FreqFusion is a **two-input fusion operator**.

Correct graph:

```text
backbone P3 (HR) ---------\
                           -> FreqFusionConcat -> C3k2 -> P3 -> DBRA cls-mid
fused P4 (LR) ------------/
```

The wrapper performs:

```python
_, hr_refined, lr_up = freqfusion(hr_feat=backbone_p3, lr_feat=fused_p4)
out = torch.cat([hr_refined, lr_up], dim=1)
```

Therefore it replaces the stock YOLO11 **second `Upsample + Concat` pair together**.

Input order is mandatory:

```text
[HR, LR] = [backbone P3, fused P4]
```

---

## 4. FreqFusion source lock

Use / verify:

```text
upstream: https://github.com/Linwei-Chen/FreqFusion
commit:   3fb0c70637a3c194fb74294d3ce4681958b26241
file:     FreqFusion.py
blob:     b8fa94d418c3094a8d6653712b65037f70daccec
```

Before copying upstream code into another repository, verify redistribution/license terms and record provenance. The project reference adapter does **not** copy the upstream implementation.

Primary Round-4 profile follows the official clean-code defaults:

```text
compressed_channels=64
lowpass_kernel=5
highpass_kernel=3
feature_resample=False
comp_feat_upsample=True
use_high_pass=True
use_low_pass=True
hr_residual=True
semi_conv=True
hamming_window=True
```

Do not enable offset-guided `feature_resample` in the first candidate.

---

## 5. Expected custom module

Port:

```text
reference_code/freqfusion_yolo_adapter.py::FreqFusionConcat
```

Suggested production location:

```text
ultralytics/nn/modules/freqfusion_yolo.py
```

Add the class to the Ultralytics module import chain and `tasks.py` namespace.

`parse_model()` special case:

```python
elif m is FreqFusionConcat:
    if not isinstance(f, list) or len(f) != 2:
        raise ValueError("FreqFusionConcat requires [hr_source, lr_source]")
    hr_c, lr_c = (ch[x] for x in f)
    args = [hr_c, lr_c, *args]
    c2 = hr_c + lr_c
```

---

## 6. YAML rule

Start from the **actual accepted DBRA parent YAML**, not stock YOLO11.

Replace only the final top-down P4->P3 pair:

```text
nearest x2
+ concat backbone P3
```

with:

```yaml
[[backbone_p3_index, fused_p4_index], 1, FreqFusionConcat, []]
```

Then update downstream indices while keeping the existing DBRA head arguments untouched.

For stock-like indexing, the expected new Detect inputs become:

```text
P3/P4/P5 = [15, 18, 21]
```

instead of stock `[16, 19, 22]`.

Do not trust these numbers if the real working DBRA YAML differs; derive them from the actual graph.

---

## 7. Weight-transfer warning

Because replacing two YAML nodes with one can shift `model.<index>` keys, a partial weight-load log is not enough.

Before training, create:

```text
implementation/ROUND4_PARENT_CONFIG_DIFF.md
```

and audit:

```text
parent DBRA state_dict keys
R4 state_dict keys
semantic key remaps caused by index shift
new FreqFusion parameters
DBRA weight transfer
backbone / neck transfer
```

If downstream weights fail only because numeric indices shifted, explicitly remap them or use a graph-preserving integration strategy. Do not silently train with a large unintended random portion of YOLO.

---

## 8. Gates before long training

Required:

```text
[ ] source/provenance verified
[ ] module import
[ ] YAML parse
[ ] model build
[ ] 2:1 HR/LR spatial assertion
[ ] output channel assertion
[ ] Detect strides remain 8/16/32
[ ] DBRA config/site unchanged
[ ] parent-to-R4 weight-transfer audit
[ ] finite FP32 forward
[ ] finite AMP forward if applicable
[ ] finite loss/backward
[ ] FreqFusion ALPF/AHPF gradients finite
[ ] DBRA gradients finite
[ ] smoke train
[ ] smoke val
[ ] smoke predict
[ ] Params/GFLOPs/VRAM/latency measured
```

Before long training generate:

```text
ROUND4_FREQFUSION_DBRA_INTEGRATION_REPORT.md
```

---

## 9. Formal training rule

Reuse the frozen parent protocol exactly.

Do not retrain the frozen baseline merely to start this experiment. Do not give R4 an extra training budget by loading an already fully trained DBRA checkpoint and continuing for another full schedule as the headline comparison.

The fair comparison is:

```text
existing DBRA parent result
vs
R4-FD1 trained with the same allowed initialization and frozen optimization budget
```

Architecture selection is validation-side. Do not use the repeatedly evaluated test set to tune FreqFusion settings.

---

## 10. First ablation table

Only:

```text
A0 YOLO11n baseline                    reuse
A1 YOLO11n + DBRA P3-mid              reuse
A2 YOLO11n + FreqFusion P4->P3 + DBRA train
```

If A2 is positive on frozen validation, then separately register **one** follow-up:

```text
feature_resample=True
```

or

```text
two-site FreqFusion
```

not both at once.

---

## 11. Codex startup prompt

Copy this directly into Codex:

```text
Implement Round-4 FreqFusion + DBRA. First read research_tracks/attention_cls_branch/05_BASELINE_AND_TRAINING_PROTOCOL.md, 07_DBRA.md, 11_ROUND2_TEST_EVIDENCE_AND_ROUND3_HYPOTHESES.md, 16_ROUND4_FREQFUSION_DBRA_DESIGN.md, reference_code/freqfusion_yolo_adapter.py, reference_code/test_freqfusion_yolo_adapter.py, and 17_CODEX_ROUND4_FREQFUSION_DBRA_PLAN.md. Then inspect the actual accepted DBRA P3-mid implementation/YAML in the YOLO11 engineering repo. Build exactly one candidate: fixed DBRA P3-Cls-Mid plus FreqFusion only at the P4->P3 top-down fusion. FreqFusion is a two-input operator and must replace the second Upsample+Concat pair as FreqFusionConcat([backbone_P3, fused_P4]); do not implement it as a one-input upsampler. Pin upstream FreqFusion to commit 3fb0c70637a3c194fb74294d3ce4681958b26241 and verify redistribution/license terms before vendoring. Use the official-default first profile with feature_resample=False. Do not change DBRA, loss, TAL, DFL, imgsz, augmentation, optimizer, schedule, dataset split, or baseline. Add module import/parser support, build the exact YAML, audit all weight transfers caused by possible index shifts, run unit/gradient/shape tests and smoke train/val/predict, profile cost, and generate ROUND4_FREQFUSION_DBRA_INTEGRATION_REPORT.md. Only after all gates pass should you start formal training with the frozen parent protocol; select on validation only.
```
