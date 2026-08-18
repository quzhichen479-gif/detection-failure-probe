# Project implementation contract: OPC-Inner-EIoU

This document defines how to integrate **L5 OPC-Inner-EIoU** into the existing
YOLO11n / Ultralytics 8.4.113 project without changing the frozen experiment
protocol.

## Scope

Current best loss control:

```text
L3 = Inner-EIoU, fixed ratio r=0.8
```

New candidate:

```text
L5 = OPC-Inner-EIoU, contraction floor r0=0.8
```

L5 is not a new architecture, assignment strategy, or detection head. It is a
bbox-regression loss-only experiment.

## Core mechanism

For each aligned positive prediction/target pair, compute

```text
u_x = 2 * abs(cx-cx_g) / (w+w_g+eps)
u_y = 2 * abs(cy-cy_g) / (h+h_g+eps)
u   = stop_gradient(max(u_x,u_y))
```

Use

```text
r = 0.8,                                  u <= 0.8
z = (u-0.8)/0.2,                          0.8 < u < 1
r = 0.8 + 0.2*(2z-z^2),                   0.8 < u < 1
r = 1.0,                                  u >= 1
```

Then compute

```text
L5 = 1 - IoU(pred^r, target^r) + EIoU_center + EIoU_width + EIoU_height
```

where `pred^r` and `target^r` are scaled about their own centers.

## Required invariants

The implementation must satisfy all of the following:

1. `r0` is exactly `0.8` for the first registered experiment.
2. `r` is always in `[0.8, 1.0]`.
3. `r` is never greater than 1.
4. `u` and `r` are detached from autograd.
5. For `u <= 0.8`, L5 must be numerically identical to L3 up to floating-point tolerance.
6. For `0.8 < u < 1`, the controller must satisfy `r > u`.
7. For `u >= 1`, use `r=1`, not an enlarged auxiliary box.
8. EIoU center/width/height terms remain unchanged from the L1/L3 implementation.
9. No additional focal/sample/scale weighting is introduced.

## Ultralytics integration point

Inspect the frozen local Ultralytics 8.4.113 tree and locate the same BboxLoss
path used by the already validated L3 experiment. Reuse that exact integration
point.

Replace only the CIoU/EIoU-style per-positive box regression term with L5.
Preserve the existing foreground selection and score weighting. The provided
`ultralytics_adapter.py` shows the intended weighting contract:

```text
per_box_loss
  -> multiply by target_scores.sum(-1)[fg_mask]
  -> sum
  -> divide by target_scores_sum
```

Do not alter DFL.

## Frozen components

The following must remain identical to the frozen L3 training protocol:

- dataset and split;
- initialization checkpoint;
- YOLO11n architecture;
- input resolution;
- batch size;
- optimizer;
- learning rate and scheduler;
- augmentation;
- epoch count;
- seed;
- TAL/TaskAlignedAssigner;
- DFL;
- classification loss;
- evaluator and confidence/NMS settings.

DBRA is not part of L5. Do not enable it for this experiment.

## Mixed precision

The reference implementation promotes bbox geometry to float32 for fp16/bfloat16
inputs. Preserve this behavior or provide an equivalent stable implementation.
The final loss may therefore be float32 under AMP; gradients still propagate to
the original prediction tensors through the cast.

## Pre-training audit

Before launching L5, export matched positive prediction/GT boxes from the frozen
L3 validation evaluation into CSV with columns:

```text
pred_x1,pred_y1,pred_x2,pred_y2,gt_x1,gt_y1,gt_x2,gt_y2
```

Run `audit_overlap_risk.py` and save the JSON report. The report must be attached
to the experiment artifacts.

Interpretation:

- if `0.8<u<1` is essentially absent, stop and report that OPC has little
  opportunity to differ from L3;
- if the band is material, continue to the implementation gate;
- do not tune `r0` from this audit.

The preregistered suggested engineering gate is 1% of matched positives in the
transition band. This is only a compute-allocation rule.

## Implementation gate before full training

Run all of the following before the full L5 training:

1. `python research_tracks/opc_inner_eiou/test_reference.py`
2. model build;
3. one training batch forward;
4. assert finite total/bbox/DFL/classification losses;
5. backward;
6. assert finite gradients;
7. AMP forward/backward;
8. verify runtime logging identifies L5 and `r0=0.8`;
9. verify DFL and TAL code paths are unchanged.

If any item fails, fix compatibility only. Do not change the mathematical
experiment definition to make the gate pass.

## Full training

Historical Baseline/L1/L2/L3/L4/DBRA runs are controls and must not be retrained.
L5 itself **must be trained from the same original initialization source used by
the frozen protocol**. Do not fine-tune from the converged L3 checkpoint.

Correct structure:

```text
same frozen initialization
        |
        +-- historical L3 result   [reuse]
        |
        +-- new L5 OPC run         [full training]
```

## Evaluation

Use frozen validation for model comparison. Primary comparison is L5 versus L3.
At minimum report:

```text
Precision
Recall
mAP50
mAP75
mAP50-95
COCO AP / APs / APm / APl / ARs / ARm / ARl when available under the same protocol
```

Also report controller diagnostics on validation positives:

```text
fraction u<=0.8
fraction 0.8<u<1
fraction u>=1
ratio percentiles
count/fraction of fixed-L3 inner-overlap collapse cases
count/fraction of OPC inner-overlap collapse cases
```

Test remains external confirmation only and must not be used to tune `r0`, the
transition polynomial, or future combinations.

## Decision rule

The central question is not whether L5 beats baseline. L3 already does that.
The required question is:

```text
Does repairing the fixed-r0 overlap-collapse region improve over L3 itself?
```

If L5 <= L3, accept fixed `r=0.8` as the better engineering solution and stop.
If L5 > L3 while retaining L3's mAP75 advantage, OPC has positive evidence.

Do not automatically launch another ratio/controller search after either result.
