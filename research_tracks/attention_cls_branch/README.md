# YOLO11 Water-Surface Attention Track — Codex Entry

> Updated: **2026-08-16**  
> Target: YOLO11n / Ultralytics 8.4.113 / PoTATO water-surface floating-object detection  
> Research rule: existing attention modules are **mechanism probes / engineering controls**, not project novelty by themselves.  
> Architecture rule: preserve the box/DFL/regression path unless a later experiment explicitly changes that hypothesis.

## 1. Repository boundary

This repository is primarily `detection-failure-probe`. Detector architecture specifications are isolated under:

```text
research_tracks/attention_cls_branch/
```

Codex must **not** place YOLO detector modules inside this repository's existing `src/` probe package. The actual implementation belongs in the real YOLO/Ultralytics engineering repository.

## 2. Baseline rule

The YOLO11n baseline already exists. Do not retrain or re-evaluate it merely to start another candidate.

Always read:

```text
05_BASELINE_AND_TRAINING_PROTOCOL.md
```

and recover the exact frozen recipe from real artifacts before formal candidate training.

## 3. Round-1 — CAA / LSK / BRA

Frozen matched-test results:

| Model | Precision | Recall | mAP50 | mAP75 | mAP50-95 |
|---|---:|---:|---:|---:|---:|
| Baseline | **0.902309** | 0.806146 | 0.870646 | 0.484366 | **0.484773** |
| CAA-Lite | 0.901744 | 0.806871 | 0.863882 | **0.488910** | 0.481432 |
| LSK-Lite | 0.857760 | 0.789229 | 0.845539 | 0.407234 | 0.437660 |
| BRA-Lite | 0.888767 | **0.822656** | **0.876378** | 0.485967 | 0.484218 |

Interpretation:

- BRA gave the first useful adaptive-routing signal but lost Precision.
- CAA was near baseline but did not improve overall AP.
- LSK clearly failed and is stopped.

Files:

```text
00_SCOPE_AND_EVIDENCE.md
01_CAA_LITE.md
02_LSK_LITE.md
03_BRA_LITE.md
04_CODEX_IMPLEMENTATION_PLAN.md
05_BASELINE_AND_TRAINING_PROTOCOL.md
06_ROUND1_TEST_EVIDENCE.md
```

## 4. Round-2 — DBRA / SHSA / Triplet

Round-2 tested more selective routing, lightweight global correction, and cross-dimensional local feature selection.

Matched-test ranking:

| Rank | Model / position | Precision | Recall | mAP50 | mAP75 | mAP50-95 |
|---:|---|---:|---:|---:|---:|---:|
| 1 | Triplet P3-pre | 0.896656 | **0.841059** | **0.887231** | 0.490605 | **0.491697** |
| 2 | **DBRA P3-mid** | 0.897511 | 0.835872 | 0.887000 | 0.485984 | 0.490352 |
| 3 | SHSA P3-mid | 0.889798 | 0.825070 | 0.868812 | **0.495884** | 0.485731 |
| — | Baseline | **0.902309** | 0.806146 | 0.870646 | 0.484366 | 0.484773 |
| 4 | Triplet P3-mid | 0.898863 | 0.805571 | 0.868523 | 0.483174 | 0.481905 |
| 5 | DBRA P4-mid | 0.875784 | 0.810446 | 0.855762 | 0.481646 | 0.474930 |
| 6 | SHSA P4-mid | 0.742158 | 0.658519 | 0.698702 | 0.147091 | 0.272280 |

Paper-aligned PoTATO `COCOeval(bbox), useCats=0` identifies **DBRA P3-mid** as the strongest cross-split-consistent parent candidate:

```text
DBRA P3-mid vs baseline:
AP   +0.0066
APs  +0.0153
ARs  +0.0075
APl  +0.0426
ARl  +0.0478
APm  -0.0209
ARm  -0.0198
```

Important interpretation:

- Triplet P3-pre is the single highest matched-test point but had validation/test split sensitivity.
- DBRA P3-mid ranked first on validation and second on test, and improves paper-aligned AP/APs/ARs; it is therefore the preferred **Round-3 parent**.
- DBRA/SHSA P4 variants directly argue against casually moving the next mechanism to P4.
- All results remain single-run point estimates unless separately repeated.

Round-2 files:

```text
07_DBRA.md
08_SHSA.md
09_TRIPLET_ATTENTION.md
10_CODEX_ROUND2_IMPLEMENTATION_PLAN.md
```

## 5. Test-use boundary

The PoTATO test has now been used repeatedly to compare and rank architecture families. It must not be used to tune Round-3 positions, module parameters, rescue variants, or candidate order.

Round-3 selection must use the frozen validation protocol. Prefer a new external/final confirmation set after the model family is frozen.

Also, raw reported Precision is operating-point dependent. Round-3 should emphasize:

```text
AP / PR behavior
FP per image @ matched Recall
APs / APm
ARs / ARm
```

in addition to the standard Ultralytics metrics.

## 6. Round-3 — DBRA companion mechanisms

Round-3 is **not** another generic-attention sweep. DBRA already supplies content-dependent non-local routing.

The remaining hypotheses are complementary:

| Priority | Candidate | Primary architecture | Hypothesis |
|---:|---|---|---|
| **1** | **DBRA + GRN** | `P3 block0 -> DBRA -> GRN -> block1` | Does post-routing inter-channel response competition improve purity without losing DBRA's small-object gain? |
| **2** | **Slide + DBRA** | **parallel** local Slide / routed DBRA at P3-mid | Can local contiguous evidence recover the APm/ARm deficit while retaining APs/ARs? |
| **3** | **FocalMod + DBRA** | weak **parallel** FocalMod / DBRA at P3-mid | Does query-conditioned gated context add information that routing misses? |

### Why the designs were corrected

During source-level review, two earlier intuitive designs were rejected:

1. `Slide -> DBRA` is no longer primary because it modifies the input of a DBRA mechanism that already works. The safer primary experiment keeps DBRA unchanged in one branch and adds local Slide evidence in parallel.
2. `DBRA -> full FocalMod` is no longer primary because the project already has negative evidence for indiscriminate context expansion. FocalMod is a weak parallel conditional branch first.

GRN also must not be described as a guaranteed “Precision fixer.” Its original purpose is inter-channel response competition; on water-dominated P3 features, global channel energy may or may not correspond to foreground purity.

## 7. Round-3 predefined positions/designs

### GRN

Primary:

```text
P3 -> cls block0 -> DBRA -> GRN -> cls block1 -> predictor
```

Secondary:

```text
P3 -> cls block0 -> DBRA -> cls block1 -> GRN -> predictor
```

The secondary requires a clean `pre_pred` classification hook; do not implement it by renaming/wrapping existing `cv3` blocks.

### Slide

Primary:

```text
                     -> DBRA ---------
P3 -> cls block0 -> U                 +-> residual fusion -> cls block1 -> predictor
                     -> Slide local ---
```

Secondary:

```text
P3 -> cls block0 -> [U + beta*Slide(U)] -> DBRA -> cls block1 -> predictor
```

### Focal Modulation

Primary:

```text
                     -> DBRA ------------
P3 -> cls block0 -> U                    +-> residual fusion -> cls block1 -> predictor
                     -> FocalModLite -----
```

Secondary:

```text
P3 -> cls block0 -> DBRA -> [D + beta*FocalMod(D)] -> cls block1 -> predictor
```

No Round-3 P4 companion position is pre-registered.

## 8. Round-3 files

Read in this order:

```text
11_ROUND2_TEST_EVIDENCE_AND_ROUND3_HYPOTHESES.md
12_DBRA_GRN.md
13_SLIDE_DBRA_LOCAL_GLOBAL.md
14_DBRA_FOCAL_MODULATION.md
15_CODEX_ROUND3_DBRA_COMPANION_PLAN.md
reference_code/round3_companion_modules.py
```

Purpose:

- `11` freezes current evidence, methodological limits, and revised hypotheses.
- `12` contains GRN formula, NCHW code and two insertion sites.
- `13` contains corrected Slide mechanism, official-source fidelity notes, parallel design and secondary preconditioner.
- `14` contains constrained Focal Modulation and the conditional parallel/post-DBRA designs.
- `15` is the **primary Codex Round-3 execution plan**.
- `reference_code/round3_companion_modules.py` is reference-only code for GRN2d, source-aligned dynamic NCHW Slide, FocalModulation2dLite, and DBRA composite wrappers.

## 9. Important Slide source correction

The released Slide-Swin implementation does not use the conventional “chunk q/k/v, then split heads” layout. It projects to `3*C`, then reshapes directly to:

```text
[B * num_heads, 3 * head_dim, H, W]
```

before q/k/v slicing. It also freezes only the designed shift convolution **weight**, retains its bias, and initializes relative bias with truncated normal.

The Round-3 reference code was corrected after direct source review. Codex must not silently substitute a cleaner but semantically different Slide implementation while retaining the same module name.

## 10. Round-3 fairness rule

Do not formally compare a composite model created by:

```text
load already-trained DBRA best.pt
-> add companion
-> continue another full training schedule
```

against the old DBRA. That grants extra optimization budget.

Formal comparison must use the same allowed common initialization and same frozen training budget. Existing DBRA result is reused as the parent comparator.

DBRA warm-start is allowed only as a clearly labeled engineering probe.

## 11. Identity and isolation gates

Round-3 has a stronger equivalence requirement than the earlier baseline gate.

For identical DBRA weights:

```text
GRN gamma=beta=0       -> exact DBRA
Slide local_scale=0    -> exact DBRA
Focal focal_scale=0    -> exact DBRA
```

Class outputs must match to strict numerical tolerance, and raw box outputs must remain invariant regardless of companion branch state.

Do not start long training before:

```text
import/build
weight-transfer audit
DBRA-parent equivalence
box-branch invariance
finite forward/backward
new branch gradient check
smoke train/val/predict
Params/GFLOPs/VRAM/latency
```

all pass.

## 12. Training priority

Prepare all primary implementations, but the default long-run order is:

```text
R3-G1: DBRA -> GRN
R3-S1: DBRA || Slide
R3-F1: DBRA || FocalMod only if still justified by validation evidence/budget
```

Do not automatically run all secondary variants.

From a research-value perspective, `DBRA || Slide` is the most interesting if it works because it tests a reusable local/global decomposition. `DBRA -> GRN` is first operationally because it is much cheaper and more falsifiable.

## 13. Mandatory simple controls after a positive result

If Slide+DBRA is positive:

```text
DBRA + matched-site depthwise 3x3 local residual
```

If Focal+DBRA is positive:

```text
DBRA + simple/matched-cost local residual
```

If the simple control matches the module, credit the simpler mechanism rather than the named attention/modulation module.

## 14. Multi-seed follow-up

For a Round-3 candidate that passes frozen validation, the relevant repeated comparison is **candidate vs its DBRA parent**.

Without retraining the original YOLO baseline merely for this purpose:

```text
reuse existing DBRA seed 42 result
train DBRA on two additional predefined validation seeds
train candidate on the same two seeds
compare paired candidate - DBRA deltas
```

Do not evaluate all repeated variants on the already-used test set.

## 15. External source references

- ConvNeXt V2 / GRN, CVPR 2023: https://github.com/facebookresearch/ConvNeXt-V2
- Slide-Transformer, CVPR 2023: https://github.com/LeapLabTHU/Slide-Transformer
- FocalNet / Focal Modulation, NeurIPS 2022: https://github.com/microsoft/FocalNet
- DeBiFormer / DBRA, ACCV 2024: https://github.com/maclong01/DeBiFormer

## 16. Codex invocation — Round-3

Use this short startup instruction:

```text
Read research_tracks/attention_cls_branch/README.md, then read 05_BASELINE_AND_TRAINING_PROTOCOL.md, 07_DBRA.md, and 11 through 15 in order, followed by reference_code/round3_companion_modules.py. Implement Round-3 DBRA companion mechanisms in the actual YOLO11 engineering repository. Reuse the exact fixed DBRA P3-mid implementation/config. Primary candidates are DBRA->GRN, parallel Slide+DBRA, and conditional parallel FocalModLite+DBRA. Preserve the classification-only design and do not modify box/DFL, DBRA hyperparameters, loss, assignment, imgsz, augmentation, or frozen training recipe. Treat the reference code as an implementation aid, but verify Slide/Focal/GRN semantics against pinned official upstream commits. Do not formally continue-train the old DBRA checkpoint. Before long training, pass DBRA-parent equivalence, raw-box invariance, weight-transfer, gradient, smoke train/val/predict, latency and VRAM gates. Do not use the repeatedly-evaluated test set for Round-3 selection or tuning.
```
