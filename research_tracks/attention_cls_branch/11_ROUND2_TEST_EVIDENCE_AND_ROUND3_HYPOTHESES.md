# Round-2 Test Evidence and Round-3 Hypotheses

> Frozen evidence date: 2026-08-16  
> Target: YOLO11n / Ultralytics 8.4.113 / PoTATO  
> Status: evidence ledger for **DBRA-based Round-3**, not a new effectiveness claim.

## 1. Frozen Round-2 results

The existing baseline was reused and was not retrained or re-evaluated. Each new candidate/position was evaluated once; there was no retry or rescue tuning.

### Ultralytics matched evaluator

| Model / position | Precision | Recall | mAP50 | mAP75 | mAP50-95 | vs baseline |
|---|---:|---:|---:|---:|---:|---:|
| Triplet P3-pre | 0.896656 | **0.841059** | **0.887231** | 0.490605 | **0.491697** | **+0.006925** |
| DBRA P3-mid | 0.897511 | 0.835872 | 0.887000 | 0.485984 | 0.490352 | +0.005579 |
| SHSA P3-mid | 0.889798 | 0.825070 | 0.868812 | **0.495884** | 0.485731 | +0.000958 |
| YOLO11n baseline | **0.902309** | 0.806146 | 0.870646 | 0.484366 | 0.484773 | — |
| Triplet P3-mid | 0.898863 | 0.805571 | 0.868523 | 0.483174 | 0.481905 | -0.002867 |
| DBRA P4-mid | 0.875784 | 0.810446 | 0.855762 | 0.481646 | 0.474930 | -0.009843 |
| SHSA P4-mid | 0.742158 | 0.658519 | 0.698702 | 0.147091 | 0.272280 | -0.212493 |

### PoTATO Table-3 / COCOeval(bbox), useCats=0

Author `test_rgb.json`: 2,000 images / 5,384 annotations.

| Model / position | AP | APs | APm | APl | ARs | ARm | ARl |
|---|---:|---:|---:|---:|---:|---:|---:|
| YOLO11n baseline | 0.4862 | 0.4256 | **0.6585** | 0.6507 | 0.5052 | **0.7144** | 0.7006 |
| **DBRA P3-mid** | **0.4928** | **0.4409** | 0.6377 | 0.6933 | **0.5128** | 0.6947 | 0.7484 |
| SHSA P3-mid | 0.4870 | 0.4318 | 0.6447 | 0.6829 | 0.5078 | 0.6996 | **0.7599** |
| Triplet P3-pre | 0.4909 | 0.4379 | 0.6502 | 0.6333 | 0.5124 | 0.7098 | 0.6841 |
| DBRA P4-mid | 0.4764 | 0.4212 | 0.6296 | 0.6833 | 0.4982 | 0.6903 | 0.7510 |
| SHSA P4-mid | 0.2755 | 0.2300 | 0.4812 | 0.2909 | 0.3722 | 0.6263 | 0.4777 |
| Triplet P3-mid | 0.4845 | 0.4240 | 0.6545 | **0.6976** | 0.5011 | 0.7075 | 0.7535 |

DBRA P3-mid relative to baseline under the paper-aligned evaluator:

```text
AP   +0.0066
APs  +0.0153
ARs  +0.0075
APl  +0.0426
ARl  +0.0478
APm  -0.0209
ARm  -0.0198
```

## 2. What is actually supported

Supported:

1. P3 is the productive level for DBRA/SHSA in this experiment; their P4 variants are worse.
2. DBRA P3-mid is the best cross-split-consistent Round-2 candidate: validation rank 1, test rank 2, and paper-aligned AP/APs/ARs are above baseline.
3. DBRA particularly improves small-object AP/AR on this single run.
4. DBRA still has lower point Precision than baseline in the Ultralytics report.
5. DBRA has lower APm/ARm than baseline under COCOeval.

Not supported yet:

- that DBRA is statistically superior across seeds;
- that its Precision loss is caused by a specific type of water clutter;
- that APm/ARm loss is definitely caused by non-local routing;
- that GRN will recover Precision;
- that a local-attention branch will recover APm/ARm;
- that adding more context after DBRA is beneficial.

Those are Round-3 hypotheses, not conclusions.

## 3. Methodological correction: Precision is not enough

The reported `Precision` is operating-point dependent. A module that changes score calibration can alter reported Precision without changing ranking quality.

Round-3 must therefore emphasize:

```text
FP / image @ matched baseline Recall
PR curve / AP
background-FP taxonomy if available
```

Raw Precision remains reported, but no module should be accepted solely because the single reported Precision number rises.

## 4. Test-use boundary

The current PoTATO test has now been used repeatedly to compare and rank architectural candidates. Therefore it must not drive Round-3 hyperparameter selection, ordering, rescue tuning, or position changes.

Round-3 model selection must use the frozen validation protocol. Prefer a new external/final confirmation set after the family is frozen.

## 5. Round-3 target is not “stronger attention”

DBRA already provides content-dependent non-local routing. Adding another global/deformable attention is mechanism duplication.

Round-3 should instead test three complementary hypotheses:

### H1 — post-routing response competition

```text
DBRA -> GRN
```

Question: after DBRA has routed contextual evidence, can zero-init inter-channel response competition improve feature purity without removing the small-object gain?

### H2 — preserve local evidence next to routed context

```text
            -> DBRA ---------
P3 cls-mid                   +-> residual fusion
            -> Slide local --
```

Question: is the APm/ARm drop associated with loss/overwriting of local contiguous evidence, and can a local dynamic branch recover it while retaining DBRA's small-object gain?

### H3 — content-gated multiscale modulation as a weak parallel correction

```text
            -> DBRA ---------
P3 cls-mid                   +-> residual fusion
            -> FocalMod -----
```

Question: does a query-conditioned context modulator add information that DBRA routing misses?

Because project evidence already warns against indiscriminate large-context enhancement, H3 is intentionally lower priority.

## 6. Revised candidate ranking

| Priority | Candidate | Primary design | Why |
|---:|---|---|---|
| 1 | DBRA + GRN | post-DBRA GRN inside P3 cls-mid | lowest risk, exact/near DBRA initialization, tests response competition rather than more spatial context |
| 2 | Slide + DBRA | **parallel local/global residual fusion** at P3 cls-mid | directly targets local-evidence preservation and APm/ARm hypothesis |
| 3 | FocalMod + DBRA | weak **parallel** modulation branch at P3 cls-mid | tests content-gated context but has overlap with previously risky large-context directions |

## 7. Important training fairness rule

Do **not** take the already-trained DBRA checkpoint and simply continue training after adding a new module for the formal comparison. That would grant the composite model extra optimization budget relative to baseline/DBRA.

Formal Round-3 comparison should:

1. instantiate the composite architecture;
2. load the same allowed YOLO pretrained/common initialization used by the frozen protocol;
3. initialize DBRA and the new companion according to the fixed module rules;
4. train for the exact same recipe/budget as the existing candidate protocol;
5. reuse existing baseline and DBRA results as comparators.

A DBRA-checkpoint warm-start may be used only as a clearly labeled engineering probe, never as the headline fair comparison.

## 8. P4 is not a Round-3 priority

Round-2 gives direct evidence that DBRA P4-mid and SHSA P4-mid are inferior to their P3 counterparts, with SHSA P4-mid catastrophic.

Therefore Round-3 companion modules are kept on **P3 classification only** unless a later mechanism audit provides a new reason to revisit P4.
