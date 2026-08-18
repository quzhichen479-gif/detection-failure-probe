# Project Implementation Contract: L1-L4 on YOLO11n / Ultralytics 8.4.113

This document defines how to integrate the four bbox-loss experiments into the existing YOLO project without changing the frozen training protocol.

## 1. Scope

The only intended algorithmic change is the IoU-style bbox regression term inside the active YOLO11 detection criterion.

The formal experiment set is:

```text
L1 = EIoU
L2 = Inner-EIoU, ratio=1.2
L3 = Inner-EIoU, ratio=0.8
L4 = SQA-Inner-EIoU, eta=0.20, detached quality
```

The existing frozen YOLO11n baseline must be reused and must not be retrained.

## 2. Inspect the exact local source before patching

The project is pinned to Ultralytics 8.4.113, but the local research tree may contain custom detection heads or other project modifications. Locate the active detection loss path first.

At minimum inspect:

```text
ultralytics/utils/loss.py
BboxLoss
v8DetectionLoss or the active YOLO11 criterion
bbox_decode
TaskAlignedAssigner call
DFL computation
```

Record in the experiment manifest:

```text
ultralytics version
git commit / working tree status
active BboxLoss path
pred_bboxes shape and coordinate system
target_bboxes shape and coordinate system
fg_mask shape
target_scores shape
```

Do not infer the active code path from class names alone.

## 3. Integration location

The original BboxLoss commonly contains logic conceptually equivalent to:

```python
weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
iou = bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, CIoU=True)
loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum
```

The project patch should replace only this IoU regression calculation with the shared adapter:

```python
loss_iou = weighted_loss_for_ultralytics(
    loss_id=loss_id,
    pred_bboxes=pred_bboxes,
    target_bboxes=target_bboxes,
    target_scores=target_scores,
    target_scores_sum=target_scores_sum,
    fg_mask=fg_mask,
)
```

Use exactly the same foreground mask, target-score weights, and normalization as the frozen implementation.

DFL must continue unchanged immediately after this computation.

## 4. Coordinate units

Unlike SRB-IoU, this family has no pixel-resolution floor.

EIoU terms are normalized by enclosing-box geometry and Inner-IoU uses a dimensionless ratio. Therefore there is no per-stride `delta` conversion.

The only requirement is:

```text
pred_bboxes and target_bboxes must be expressed in the same coordinate system.
```

Do not add stride-dependent scaling solely for L1-L4.

## 5. One code path for all experiments

Do not create four independently patched copies of `loss.py`.

Prefer one experiment selector, for example:

```text
bbox_loss_variant: L1 | L2 | L3 | L4
```

that dispatches to `loss_by_id()`.

This is important for attribution: all four runs should share an identical integration path, with only the selected loss definition changing.

If the existing project configuration system cannot pass a custom loss ID cleanly, a small project-local constant or CLI/config extension is acceptable, but document the exact mechanism.

## 6. L4 controller contract

The formal L4 controller is frozen as:

```math
q=stopgrad(IoU(B,G)),
```

```math
g(q)=3q^2-2q^3,
```

```math
r(q)=1+0.20[1-2g(q)].
```

Therefore:

```text
q=0.0 -> r=1.20
q=0.5 -> r=1.00
q=1.0 -> r=0.80
```

Do not:

- remove the detach;
- replace smoothstep with a linear map;
- change eta from 0.20 before the preregistered L4 run;
- add epoch-dependent scheduling;
- add a second quality weight to the final loss.

Any such change is a new ablation, not L4.

## 7. Frozen components

The following must remain identical across L1-L4 and the frozen baseline protocol:

```text
model scale and architecture
initialization / pretrained checkpoint policy
training dataset and official split
imgsz
batch size
epoch count
optimizer
learning-rate schedule
augmentations
seed policy
TAL / TaskAlignedAssigner
classification loss
DFL and DFL gain
box loss gain
evaluator and evaluation thresholds
```

The first-pass experiment must retain the existing YOLO `box` loss gain. Do not rescale L1-L4 to make their raw magnitudes numerically match CIoU.

If a loss has an obviously pathological scale, record that as evidence rather than silently retuning the gain.

## 8. Build and numerical gate before training

Before any full training run, execute:

```bash
python research_tracks/sqa_inner_eiou/test_reference.py
```

Then patch the local YOLO tree and verify for every L1-L4:

```text
model build succeeds
one real batch forward succeeds
loss is finite
backward succeeds
AMP has no NaN/Inf
DFL output shape and code path are unchanged
TAL fg_mask is unchanged for identical model outputs
classification loss is unchanged for identical model outputs
```

For L4 additionally log one batch of:

```text
ratio min / mean / median / max
ratio p10 / p90
IoU mean
```

Expected ratio range is `[0.8, 1.2]`.

If ratio exits that range, stop and fix the implementation before training.

## 9. Formal training order

Run:

```text
L1 -> L2 -> L3 -> L4
```

using the same frozen training recipe.

Do not retrain the baseline.

Do not use test results to select among L1-L4. Use frozen validation for mechanism comparison and model selection. Test is reserved for later confirmation of a selected candidate.

## 10. Required reporting table

At minimum report:

| ID | Loss | AP | AP50 | AP75 | APs | APm | APl | ARs | ARm | ARl |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Baseline | frozen CIoU | reuse | reuse | reuse | reuse | reuse | reuse | reuse | reuse | reuse |
| L1 | EIoU | | | | | | | | | |
| L2 | Inner-EIoU 1.2 | | | | | | | | | |
| L3 | Inner-EIoU 0.8 | | | | | | | | | |
| L4 | SQA-Inner-EIoU | | | | | | | | | |

For L4 also report the validation distribution of the detached controller ratio.

## 11. Interpretation rules

The experiment should be interpreted as a mechanism tree, not a leaderboard-only sweep.

```text
L1 vs baseline -> value of EIoU geometry
L2 vs L1       -> value of broad auxiliary overlap
L3 vs L1       -> value of fine auxiliary overlap
L4 vs L2/L3    -> value of adaptive coarse-to-fine geometry
```

A convincing L4 result should ideally outperform both fixed-ratio controls, not merely the baseline.

If one fixed ratio wins and L4 does not exceed it, prefer the mature fixed Inner-EIoU result and do not force a custom-method claim.

## 12. DBRA combination gate

DBRA is not part of L1-L4.

Only after the loss-only comparison is complete should the selected loss be combined with DBRA, and only if it is credible on frozen validation.

Do not run four DBRA combinations. Select at most one justified loss candidate after the L1-L4 mechanism comparison.

## 13. Patch hygiene

Recommended local project changes:

```text
ultralytics/utils/loss.py                 minimal selector/call-site patch
ultralytics/utils/sqa_inner_eiou.py       copied/adapted loss implementation
project config/CLI                         one loss-ID selector if needed
project-local tests                        integration tests
```

Do not replace shared global IoU utilities used by assignment, NMS, metrics, or unrelated tracks.

Every deviation from `losses.py` in this reference track must be documented in the experiment report before results are interpreted.
