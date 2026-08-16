# Codex Round-3 Plan — DBRA Companion Mechanisms

> Primary Codex entry for Round-3  
> Target: YOLO11n / Ultralytics 8.4.113 / frozen PoTATO training protocol  
> Parent model: **DBRA P3-Cls-Mid**  
> Read first: `05`, `07`, `11`, then `12`–`14`.

## 0. Goal

Do not search for a stronger generic attention.

Round-3 asks whether the current DBRA representation can be improved by one of three **complementary** mechanisms:

1. `DBRA -> GRN`: post-routing channel-response competition;
2. `DBRA || Slide`: preserve local evidence next to routed non-local context;
3. `DBRA || FocalModLite`: weak query-conditioned multiscale modulation.

The regression/DFL branch remains unchanged.

## 1. Evidence lock

Before implementation, read:

```text
05_BASELINE_AND_TRAINING_PROTOCOL.md
07_DBRA.md
11_ROUND2_TEST_EVIDENCE_AND_ROUND3_HYPOTHESES.md
```

Do not recompute the frozen baseline simply to start Round-3.

Headline parent evidence:

```text
DBRA P3-mid vs baseline, paper-aligned COCOeval:
AP   +0.0066
APs  +0.0153
ARs  +0.0075
APl  +0.0426
ARl  +0.0478
APm  -0.0209
ARm  -0.0198
```

The current test has already influenced model search. **Do not use it for Round-3 architecture/hyperparameter selection.**

## 2. Corrected Round-3 model set

### R3-G1 — DBRA + GRN, primary

```text
P3 -> cls block0 -> DBRA -> GRN -> cls block1 -> pred
```

### R3-G2 — DBRA mid + GRN pre-predictor, secondary

```text
P3 -> cls block0 -> DBRA -> cls block1 -> GRN -> pred
```

### R3-S1 — Slide / DBRA parallel, primary

```text
                     -> DBRA ---------
P3 -> cls block0 -> U                 +-> residual fusion -> cls block1 -> pred
                     -> Slide local ---
```

### R3-S2 — Slide local preconditioner then DBRA, secondary

```text
P3 -> cls block0 -> [U + beta*Slide(U)] -> DBRA -> cls block1 -> pred
```

### R3-F1 — FocalMod / DBRA parallel, primary

```text
                     -> DBRA ------------
P3 -> cls block0 -> U                    +-> residual fusion -> cls block1 -> pred
                     -> FocalModLite -----
```

### R3-F2 — weak FocalMod after DBRA, secondary

```text
P3 -> cls block0 -> DBRA -> [D + beta*FocalMod(D)] -> cls block1 -> pred
```

All six are P3 classification-only. No P4 Round-3 companion experiment is pre-registered.

## 3. Priority and training budget

Implementation can prepare all candidates, but long training order is:

```text
1. R3-G1  DBRA -> GRN
2. R3-S1  DBRA || Slide
3. R3-F1  DBRA || FocalMod only if still justified/budgeted
```

Secondary variants are not automatically run. Run a secondary only if the primary produces a mechanism-relevant validation signal or a diagnostic specifically motivates the second placement.

Do not run all six because they exist.

## 4. Important fairness correction — do not continue-train the winner

For the headline comparison, do not:

```text
load trained DBRA best.pt
add GRN/Slide/Focal
continue training another full schedule
compare to old DBRA
```

That gives the composite additional optimization budget.

Formal Round-3 training must use the same allowed initial source and the same frozen schedule as the prior architecture comparisons.

A warm-start from DBRA may be run only as an explicitly labeled engineering probe and must never replace the fair-from-common-init result.

## 5. Reuse the existing DBRA implementation exactly

Do not create `DBRA_v2`, change heads/top-k/windows, or retune DBRA while adding a companion.

The Round-3 DBRA component must use the exact fixed Round-2 config that produced the accepted P3-mid parent result.

Record in the integration report:

```text
upstream DeBiFormer commit
local DBRA wrapper hash
DBRA constructor args
DBRA trainable parameter count
```

## 6. Proposed engineering files in the real YOLO repository

```text
ultralytics/nn/modules/water_attention_round3.py

# reuse existing Round-2 DBRA implementation
ultralytics/nn/modules/third_party/debiformer/...

tests/test_water_attention_round3.py
tests/test_round3_cls_hooks.py

ultralytics/cfg/models/11/
  yolo11-dbra-grn-p3-mid.yaml
  yolo11-dbra-grn-p3-prepred.yaml
  yolo11-slide-dbra-parallel-p3-mid.yaml
  yolo11-slide-then-dbra-p3-mid.yaml
  yolo11-focal-dbra-parallel-p3-mid.yaml
  yolo11-dbra-then-focal-p3-mid.yaml
```

## 7. Factory design

Prefer composite adapters over fragile manual tower surgery.

```python
def build_round3_adapter(kind: str, dim: int, cfg: dict):
    kind = kind.lower()

    if kind == "dbra_grn_post":
        return DBRAGRNPost(
            dim=dim,
            dbra=build_fixed_dbra(dim, cfg["dbra"]),
            eps=cfg.get("eps", 1e-6),
        )

    if kind == "slide_dbra_parallel":
        return SlideDBRAParallel(
            dbra=build_fixed_dbra(dim, cfg["dbra"]),
            slide=SlideAttention2d(
                dim=dim,
                num_heads=cfg.get("num_heads", 4),
                kernel_size=cfg.get("kernel_size", 3),
            ),
            local_scale_init=cfg.get("local_scale_init", 1e-3),
        )

    if kind == "slide_then_dbra":
        ...

    if kind == "focal_dbra_parallel":
        ...

    if kind == "dbra_then_focal":
        ...

    raise ValueError(kind)
```

`build_fixed_dbra()` must call the same DBRA implementation/config as Round-2.

## 8. Head integration

### 8.1 Primary candidates

R3-G1, R3-S1, R3-S2, R3-F1, R3-F2 can all be represented as a **single composite adapter at existing `P3-Cls-Mid`**:

```text
P3 -> cv3[0] -> composite_adapter -> cv3[1] -> predictor
```

Therefore reuse the existing `AttnDetect` mid hook whenever possible.

Do not alter `cv2`.

### 8.2 GRN secondary requires a clean `pre_pred` hook

R3-G2 needs:

```text
P3 -> cv3[0] -> DBRA -> cv3[1] -> GRN -> predictor
```

If/when R3-G2 is actually authorized, extend the classification hook semantics cleanly:

```python
def _cls_forward(...):
    y = tower[0](x)

    if mid_adapter is not None:
        y = mid_adapter(y)

    y = tower[1](y)

    if pre_pred_adapter is not None:
        y = pre_pred_adapter(y)

    y = tower[2](y)
    return y
```

Keep `tower[0]`, `tower[1]`, `tower[2]` as the original `cv3` modules so pretrained key names stay compatible.

Do not embed the original cls block into a new wrapper that renames its state-dict keys.

## 9. YAML concepts

Use the actual existing AttnDetect positional/API convention. The following are semantic configs, not permission to create a second incompatible parser.

### R3-G1

```yaml
level: 0
site: mid
adapter: dbra_grn_post
adapter_cfg:
  dbra: <exact frozen DBRA cfg>
  eps: 1.0e-6
```

### R3-S1

```yaml
level: 0
site: mid
adapter: slide_dbra_parallel
adapter_cfg:
  dbra: <exact frozen DBRA cfg>
  kernel_size: 3
  num_heads: 4   # adjust only to a divisor of active dim
  local_scale_init: 0.001
```

### R3-F1

```yaml
level: 0
site: mid
adapter: focal_dbra_parallel
adapter_cfg:
  dbra: <exact frozen DBRA cfg>
  focal_level: 1
  focal_window: 3
  focal_factor: 2
  normalize_modulator: true
  focal_scale_init: 0.001
```

No grid search is permitted in the first pass.

## 10. Round-3 identity/equivalence gates

Previous rounds used `alpha=0 -> baseline` as an important test. Round-3 can do something stronger:

### DBRA-parent equivalence

For a fixed DBRA model state:

```text
R3-G1 with GRN gamma=beta=0
R3-S1 with local_scale=0
R3-S2 with local_scale=0
R3-F1 with focal_scale=0
R3-F2 with focal_scale=0
```

must reproduce the corresponding DBRA hidden/class output to strict numerical tolerance.

Test:

```python
torch.testing.assert_close(
    scores_dbra,
    scores_round3_disabled,
    atol=1e-6,
    rtol=1e-5,
)
```

Raw boxes must be identical regardless of companion scale because all changes are cls-only.

This is mandatory before smoke training.

## 11. Gradient gates

A common mistake with zero-initialized residual scales is accidentally starving the inner branch of gradients.

For train-time defaults:

```text
GRN: gamma/beta = 0 is original and acceptable; gamma receives gradient immediately.
Slide: local_scale_init = 1e-3, not 0, so Slide parameters receive gradients from step 1.
Focal: focal_scale_init = 1e-3, not 0, so Focal parameters receive gradients from step 1.
```

Tests must confirm non-None finite gradients inside the new branch after one backward pass.

## 12. Slide implementation checks

Read `13_SLIDE_DBRA_LOCAL_GLOBAL.md`.

Verify against pinned official `SlideAttention`:

```text
Q/K/V projection
fixed shift depthwise kernel
learnable shift/deformation depthwise kernel
local k^2 attention softmax
relative bias
projection
```

Project changes allowed:

```text
NCHW instead of token-last
runtime H/W instead of fixed input_resolution
Conv2d(1x1) instead of per-token Linear
near-DBRA outer residual scale
```

Do not silently omit the learned shift path.

## 13. Focal implementation checks

Read `14_DBRA_FOCAL_MODULATION.md`.

First config is fixed:

```text
level=1
window=3
normalize=true
```

Verify:

```text
pre projection -> q / ctx / gates
DW contextualization
local gated aggregation
global context gate
modulator projection
q * modulator
output projection
```

Do not increase focal levels because the first validation result is mediocre.

## 14. Required unit/smoke gates

For every primary candidate:

```text
[ ] import
[ ] model build
[ ] pretrained/common weight transfer
[ ] shape preservation
[ ] finite forward
[ ] finite loss
[ ] finite backward
[ ] new branch receives gradient
[ ] DBRA-parent equivalence when companion disabled
[ ] raw box branch invariant
[ ] 1-epoch smoke train
[ ] validation smoke
[ ] predict smoke
[ ] Params/GFLOPs/VRAM/latency P50/P95
```

Do not begin long training if any gate fails.

## 15. Validation selection metrics

Headline:

```text
Precision
Recall
mAP50
mAP75
mAP50-95
```

Paper-aligned / mechanism metrics where available:

```text
AP
APs APm APl
ARs ARm ARl
FP/image @ matched Recall
```

Raw Precision is not sufficient because it is threshold/calibration dependent.

## 16. Candidate-specific hypotheses

### R3-G1 — DBRA + GRN

Desired:

```text
AP >= DBRA
APs/ARs retained
matched-recall FP decreases OR APm recovers
```

If only point Precision changes while AP and matched-recall FP do not, treat it as calibration movement.

### R3-S1 — parallel Slide + DBRA

Desired:

```text
AP >= DBRA
APs/ARs retained
APm and/or ARm moves toward baseline
```

This is the most informative candidate for local/global complementarity.

### R3-F1 — parallel FocalMod + DBRA

Desired:

```text
AP > DBRA
small-object signature retained
no global-gate-dominated FP increase
```

Because of previous context failures, neutral/slightly negative Focal should be stopped rather than rescued with larger focal windows.

## 17. Mechanism controls after a positive result

### GRN positive

No extra generic attention control is needed. Inspect gamma/beta and matched-recall FP. If gamma remains ~0, do not attribute the result to GRN.

### Slide+DBRA positive

Mandatory control:

```text
DBRA + matched-site depthwise 3x3 residual branch
```

If it matches Slide, the mechanism is local preservation, not local self-attention specifically.

### Focal+DBRA positive

Mandatory control:

```text
DBRA + simple depthwise 3x3 residual branch
```

If matched, do not claim gated hierarchical modulation is necessary.

## 18. Multi-seed policy without retraining the baseline

The parent comparison for Round-3 is DBRA, not the original baseline.

If a Round-3 candidate passes validation screening and is going toward a final claim, use **paired DBRA-vs-composite validation seeds**:

```text
existing DBRA seed 42 can be reused
train DBRA on two additional predefined seeds
train the candidate on the same additional seeds
compare paired candidate - DBRA deltas
```

This avoids needing to retrain the original YOLO baseline simply to test whether the companion consistently improves its DBRA parent.

Do not evaluate all seed variants on the repeatedly-used test set.

## 19. Why Triplet + DBRA is not promoted ahead of these three

Triplet P3-pre has the highest current test mAP50-95, so `Triplet-pre -> DBRA-mid` is an obvious temptation.

It is intentionally **not** the first Round-3 combination because:

1. Triplet showed validation/test split sensitivity;
2. choosing the combination mainly from the current test would deepen test-driven architecture selection;
3. both Triplet P3-pre and DBRA strongly raise Recall, while the remaining DBRA weakness includes Precision/purity;
4. the current three hypotheses more directly probe response competition and local-evidence preservation.

Keep Triplet+DBRA as a later registered control only if validation-side analysis motivates it. Do not run it opportunistically after seeing Round-3 test results.

## 20. Stop rule for the whole companion-module route

If R3-G1 and R3-S1 both fail clearly on frozen validation, and diagnostics do not support their hypothesized mechanisms, do not automatically cycle through many more attention/normalization modules.

At that point, the evidence would suggest that DBRA's remaining error is not cheaply fixed by post-routing calibration or local-companion features, and the project should return to error taxonomy / data-conditional analysis.

FocalMod is not a mandatory rescue run.

## 21. Required Codex outputs

Before long training:

```text
ROUND3_DBRA_COMPANION_INTEGRATION_REPORT.md
```

Must include:

1. repository commit / dirty status;
2. Ultralytics version;
3. exact DBRA source/config reused;
4. files changed;
5. final class signatures;
6. computation graph for each enabled YAML;
7. DBRA-parent equivalence tests;
8. box invariance tests;
9. pretrained weight-transfer audit;
10. gradients/smoke tests;
11. Params/GFLOPs/latency/VRAM;
12. deviations from this specification.

After frozen-validation training:

```text
ROUND3_DBRA_COMPANION_VALIDATION_REPORT.md
round3_dbra_companion_summary.csv
round3_dbra_companion_summary.json
```

## 22. Short Codex invocation prompt

```text
Read research_tracks/attention_cls_branch/README.md and then read 05_BASELINE_AND_TRAINING_PROTOCOL.md, 07_DBRA.md, and 11 through 15 in order. Implement Round-3 DBRA companion mechanisms in the actual YOLO11 engineering repository, reusing the exact existing DBRA P3-mid implementation/config and the classification-only head path. Primary candidates are DBRA->GRN, parallel Slide+DBRA, and conditional parallel FocalModLite+DBRA. Do not change the box/DFL branch, DBRA hyperparameters, image size, loss, assignment, augmentation, or frozen training recipe. Do not formally continue-train the old DBRA checkpoint; fair comparison must use the same allowed common initialization and training budget. Before long training, pass DBRA-parent equivalence, raw-box invariance, weight-transfer, forward/backward/gradient, smoke train/val/predict, latency and VRAM gates. Do not use the repeatedly-evaluated test set for Round-3 selection or tuning.
```
