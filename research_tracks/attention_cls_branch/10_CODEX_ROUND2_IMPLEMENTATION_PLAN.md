# Codex Round-2 Implementation Plan — DBRA / SHSA / Triplet Attention

> Entry status: **implementation specification, not an effectiveness claim**  
> Target: YOLO11n / Ultralytics 8.4.113 / existing frozen water-surface detection protocol  
> Read first: `05_BASELINE_AND_TRAINING_PROTOCOL.md` and `06_ROUND1_TEST_EVIDENCE.md`

## 0. Goal

Implement three new classification-only attention candidates, each with exactly two predefined insertion positions:

| Candidate | Primary insertion | Secondary insertion |
|---|---|---|
| DBRA | P3-Cls-Mid | P4-Cls-Mid |
| SHSA | P3-Cls-Mid | P4-Cls-Mid |
| Triplet Attention | P3-Cls-Pre | P3-Cls-Mid |

Do not stack modules. Do not modify the regression branch. Do not change loss/DFL/assigner/image size/training recipe in the first round.

## 1. Evidence that constrains the implementation

Reuse existing Round-1 test evidence; do not recompute baseline:

```text
Baseline mAP50-95  0.484773
CAA                0.481432
LSK                0.437660  -> stop
BRA                0.484218  -> strongest candidate
```

BRA relative to baseline:

```text
Precision  -0.013542
Recall     +0.016510
mAP50      +0.005732
mAP75      +0.001601
mAP50-95   -0.000555
```

Round-2 target:

```text
retain useful Recall/context signal
while recovering classification Precision
```

Do not use the current test to tune Round-2 modules. Selection must use the frozen validation protocol.

## 2. Reuse existing infrastructure

The repository already documents an `AttnDetect(Detect)` strategy for classification-only attention. Reuse the same design in the actual YOLO engineering repository.

Conceptual sites:

### `pre`

```text
F_i -----------------------------> cv2[i] box branch
 |
 +-> attention -> entire cv3[i] classification tower
```

### `mid`

```text
F_i -----------------------------> cv2[i] box branch
 |
 +-> cv3[i][0] -> attention -> cv3[i][1] -> predictor
```

For YOLO11 Detect input levels:

```text
level 0 = P3 / stride 8
level 1 = P4 / stride 16
level 2 = P5 / stride 32
```

Codex must inspect the installed Ultralytics **8.4.113** `head.py` before indexing `cv3`, because implementation structure is version-sensitive.

## 3. Files to add in the YOLO engineering repository

Suggested structure:

```text
ultralytics/nn/modules/water_attention_round2.py
ultralytics/nn/modules/third_party/debiformer/
    dbra_upstream.py
    SOURCE.md

tests/test_water_attention_round2.py

ultralytics/cfg/models/11/
    yolo11-dbra-p3-clsmid.yaml
    yolo11-dbra-p4-clsmid.yaml
    yolo11-shsa-p3-clsmid.yaml
    yolo11-shsa-p4-clsmid.yaml
    yolo11-triplet-p3-clspre.yaml
    yolo11-triplet-p3-clsmid.yaml
```

Do not place detector code inside this `detection-failure-probe` repository's existing `src/`. This repository is the research/specification store; the real detector implementation belongs in the YOLO codebase.

## 4. Module implementation tasks

### 4.1 DBRA

Read `07_DBRA.md`.

Implementation rule:

- clone/read the official `maclong01/DeBiFormer` source;
- pin an exact commit SHA;
- isolate the minimum DBRA dependencies;
- do not invent a simplified operator and still name it DBRA;
- add a project wrapper:

```python
Y = X + alpha * (DBRA(X) - X)
```

with `alpha_init=1e-3`.

Primary site:

```text
P3 / cls-mid
```

Secondary site:

```text
P4 / cls-mid
```

### 4.2 SHSA

Read `08_SHSA.md`.

Implement partial-channel single-head attention based on official SHViT:

```text
split attended partial channels / untouched channels
-> GroupNorm on partial channels
-> 1x1 qkv
-> single-head spatial attention
-> concatenate untouched channels
-> 1x1 projection
-> near-identity interpolation
```

Default:

```text
qk_dim=16
partial_dim=min(max(C//4, 16), 64)
alpha_init=1e-3
```

Primary:

```text
P3 / cls-mid
```

Secondary:

```text
P4 / cls-mid
```

### 4.3 Triplet Attention

Read `09_TRIPLET_ATTENTION.md`.

Implement three branches:

```text
C-W interaction
C-H interaction
H-W spatial interaction
```

using max/mean channel compression + 7x7 spatial gate.

Default:

```text
kernel_size=7
no_spatial=false
alpha_init=1e-3
```

Primary:

```text
P3 / cls-pre
```

Secondary:

```text
P3 / cls-mid
```

## 5. Attention factory

Extend the existing attention factory without breaking Round-1 modules:

```python
def build_water_attention(kind: str, dim: int, cfg=None):
    cfg = dict(cfg or {})
    kind = kind.lower()

    # existing kinds remain unchanged
    if kind == "dbra":
        return DBRALiteAdapter(dim=dim, **cfg)
    if kind == "shsa":
        return SHSALite(dim=dim, **cfg)
    if kind == "triplet":
        return TripletAttentionLite(dim=dim, **cfg)

    ...
```

Do not rename or alter existing CAA/LSK/BRA behavior while adding Round-2 modules.

## 6. YAML mapping

Use the existing `AttnDetect` argument convention from the Round-1 implementation.

Conceptual examples:

```yaml
# DBRA P3-mid
attn_type: dbra
levels: [0]
site: mid
attn_cfg:
  alpha_init: 0.001

# DBRA P4-mid
attn_type: dbra
levels: [1]
site: mid
attn_cfg:
  alpha_init: 0.001

# SHSA P3-mid
attn_type: shsa
levels: [0]
site: mid
attn_cfg:
  qk_dim: 16
  alpha_init: 0.001

# SHSA P4-mid
attn_type: shsa
levels: [1]
site: mid
attn_cfg:
  qk_dim: 16
  alpha_init: 0.001

# Triplet P3-pre
attn_type: triplet
levels: [0]
site: pre
attn_cfg:
  kernel_size: 7
  no_spatial: false
  alpha_init: 0.001

# Triplet P3-mid
attn_type: triplet
levels: [0]
site: mid
attn_cfg:
  kernel_size: 7
  no_spatial: false
  alpha_init: 0.001
```

If the actual Round-1 `AttnDetect` API differs, adapt the YAML to the existing implementation rather than creating a second incompatible attention head.

## 7. Mandatory engineering gates

Each module must pass all gates before long training.

### Gate A — import/build

```text
module import
YAML parse
model build
pretrained weight load
```

### Gate B — numerical identity

With attention `alpha=0` and identical common weights:

```text
raw boxes ~= baseline raw boxes
class scores ~= baseline class scores
```

Tolerance should be strict (`atol~1e-6`, `rtol~1e-5`) unless a documented backend op makes that impossible.

### Gate C — branch isolation

Set attention alpha to a visibly non-zero value.

Expected:

```text
classification scores change
raw box outputs do NOT change
```

If boxes change, stop: classification-only isolation is broken.

### Gate D — trainability

```text
finite forward
finite loss
finite backward
module parameters receive gradient
1-epoch smoke train
validation smoke
predict smoke
```

### Gate E — cost

Measure on the same device/config:

```text
Params
GFLOPs
VRAM
batch-1 latency P50/P95
training iteration time
```

SHSA P3 and DBRA P3 must receive special memory/latency scrutiny.

## 8. Training order

Do not train all six insertion variants immediately.

First-round Round-2 training:

```text
R2-0 existing baseline result (reuse; no retraining)
R2-1 DBRA @ P3-Cls-Mid
R2-2 SHSA @ P3-Cls-Mid
R2-3 Triplet @ P3-Cls-Pre
```

Only a module that passes the predefined validation gate may receive its second insertion experiment:

```text
DBRA -> P4-Cls-Mid
SHSA -> P4-Cls-Mid
Triplet -> P3-Cls-Mid
```

Do not use the current test set to decide whether to run the second position.

## 9. Validation metrics

At minimum report:

```text
Precision
Recall
mAP50
mAP75
mAP50-95
```

If available, additionally report:

```text
AP_tiny / AP_small
AR_tiny
FP per image
FP @ matched baseline Recall
water-background FP taxonomy
```

For this round, `FP @ matched Recall` is especially important because BRA's current weakness is Precision loss with Recall gain.

## 10. Round-2 decision logic

### DBRA

Primary hypothesis:

```text
recover BRA Precision while preserving BRA Recall/AP gains
```

Fail if it merely pushes Recall higher while Precision falls further.

### SHSA

Primary hypothesis:

```text
a simpler partial-channel global correction can match most BRA benefits
with less cost and/or better Precision
```

Fail if it is both less accurate and not materially cheaper.

### Triplet

Primary hypothesis:

```text
cross-dimensional P3 feature selection is sufficient;
non-local routing is not necessary
```

Fail if it does not improve classification purity or is matched by a simpler spatial-only gate.

## 11. Simple controls after a candidate passes

Do not jump directly to multi-seed after a tiny gain without a mechanism control.

DBRA positive:

```text
compare against BRA at matched/near-matched runtime if possible
```

SHSA positive:

```text
compare against a simple same-site local projection/block
```

Triplet positive:

```text
compare against spatial-only 7x7 gate
```

Only after surviving its simple control should a candidate enter multi-seed validation.

## 12. What must not happen

Do not:

```text
stack DBRA + SHSA
stack BRA + DBRA
add P2 at the same time
change IoU/DFL/TAL
change imgsz from the frozen protocol
retune augmentation for a candidate
retrain baseline unnecessarily
select new hyperparameters from the current test set
claim any of these existing attention modules as project novelty
```

## 13. Required Codex output

After implementation but before long training, generate:

```text
ROUND2_ATTENTION_INTEGRATION_REPORT.md
```

It must include:

1. exact source commit / dirty status;
2. Ultralytics version;
3. upstream DBRA commit and local modifications;
4. files changed;
5. final module constructor signatures;
6. exact insertion graph for each YAML;
7. alpha=0 equivalence results;
8. box-branch isolation results;
9. weight-transfer audit;
10. unit/smoke test results;
11. Params/GFLOPs/latency/VRAM;
12. any deviations from this specification.

Then, after validation training, generate:

```text
ROUND2_ATTENTION_VALIDATION_REPORT.md
round2_attention_summary.csv
round2_attention_summary.json
```

## 14. Codex invocation prompt

Use this as the short startup instruction:

```text
Read research_tracks/attention_cls_branch/README.md, then read 05_BASELINE_AND_TRAINING_PROTOCOL.md, 06_ROUND1_TEST_EVIDENCE.md, 07_DBRA.md, 08_SHSA.md, 09_TRIPLET_ATTENTION.md, and 10_CODEX_ROUND2_IMPLEMENTATION_PLAN.md in order. Implement Round-2 DBRA, SHSA, and Triplet Attention in the actual YOLO11 engineering repository using the existing classification-only AttnDetect infrastructure. Do not retrain the existing baseline. First complete only the engineering gates, unit tests, alpha=0 baseline-equivalence audit, box-branch isolation audit, pretrained-weight transfer audit, and smoke train/val/predict. Do not start long training until these gates pass. Preserve the frozen baseline training protocol and do not use the current test set for iterative selection.
```
