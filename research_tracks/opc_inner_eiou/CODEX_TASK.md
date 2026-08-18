# Codex task: implement and validate L5 OPC-Inner-EIoU

Read this directory completely before changing the local YOLO project:

```text
research_tracks/opc_inner_eiou/
```

Priority documents:

1. `README.md`
2. `PROJECT_IMPLEMENTATION.md`
3. `experiment_plan.yaml`
4. `opc_inner_eiou.py`
5. `ultralytics_adapter.py`

## Goal

Implement exactly one new bbox-loss experiment in the frozen
YOLO11n / Ultralytics 8.4.113 project:

```text
L5 = OPC-Inner-EIoU(r0=0.8)
```

Current best historical loss control is:

```text
L3 = Inner-EIoU(r=0.8)
```

Do not retrain historical Baseline/L1/L2/L3/L4/DBRA. Do not enable DBRA in L5.

## Phase A — cheap geometric opportunity audit

Using existing frozen-L3 validation predictions, obtain aligned matched positive
prediction/GT boxes without rerunning training. Export:

```text
pred_x1,pred_y1,pred_x2,pred_y2,gt_x1,gt_y1,gt_x2,gt_y2
```

Run:

```bash
python research_tracks/opc_inner_eiou/audit_overlap_risk.py \
  --input <matched_val_boxes.csv> \
  --output <artifact_dir>/opc_overlap_audit.json
```

Record:

```text
fraction u<=0.8
fraction 0.8<u<1
fraction u>=1
fixed-L3 auxiliary-overlap collapse fraction
OPC auxiliary-overlap collapse fraction
ratio percentiles
```

If the `0.8<u<1` band is below the preregistered 1% engineering gate, STOP and
report that OPC changes too few matched positives to justify a full run. Do not
change r0 to force a larger band.

## Phase B — reference and integration gate

Run:

```bash
python research_tracks/opc_inner_eiou/test_reference.py
```

Then integrate `L5` at the exact BboxLoss location already used for L3.

Required mathematical identity:

```text
u_x = 2*abs(cx-cx_g)/(w+w_g+eps)
u_y = 2*abs(cy-cy_g)/(h+h_g+eps)
u   = stop_gradient(max(u_x,u_y))

r = 0.8                                 if u<=0.8
z = (u-0.8)/0.2                         if 0.8<u<1
r = 0.8 + 0.2*(2z-z^2)                  if 0.8<u<1
r = 1.0                                 if u>=1
```

Final loss:

```text
L5 = 1 - IoU(pred^r, target^r)
     + EIoU_center
     + EIoU_width
     + EIoU_height
```

Hard invariants:

```text
0.8 <= r <= 1.0
r never > 1
u and r detached
u<=0.8 => L5 numerically equals L3
0.8<u<1 => r>u
```

Do not add any NWD, WIoU, Focal/Focaler, Shape-IoU, sample weighting, or new
training schedule.

Preserve:

```text
TAL / TaskAlignedAssigner
fg_mask
target-score weighting
target_scores_sum normalization
DFL
classification loss
model architecture
optimizer/scheduler
augmentations
dataset split
evaluator
```

Before full training pass:

```text
model build
one-batch forward
finite total/bbox/DFL/classification losses
backward
finite gradients
AMP forward/backward
runtime confirmation that L5 and r0=0.8 are active
```

## Phase C — one new full training run

L5 must be trained from the same original initialization source as the frozen
protocol. Do NOT fine-tune from the converged L3 checkpoint.

Run exactly one new full training experiment:

```text
YOLO11n + OPC-Inner-EIoU(r0=0.8)
```

Use the same frozen seed, epochs, imgsz, batch size, optimizer, LR schedule,
augmentation, dataset split, and evaluator as L3.

Do not search:

```text
r0=0.75/0.85/etc.
other transition functions
DBRA+L5
SQA variants
additional loss stacks
```

## Phase D — frozen evaluation and report

Compare existing L3 with new L5 under the same frozen validation protocol.
Report at minimum:

```text
Precision
Recall
mAP50
mAP75
mAP50-95
COCO AP/APs/APm/APl/ARs/ARm/ARl when protocol-compatible
```

Also repeat OPC controller diagnostics on validation positives.

Generate:

```text
OPC_INNER_EIOU_REPORT.md
```

The report must include:

1. code/config/checkpoint commit SHA or hashes;
2. exact L5 integration point;
3. frozen-protocol consistency table;
4. pretraining overlap-risk audit;
5. reference/integration gate results;
6. L3 vs L5 validation table;
7. controller u/ratio distribution;
8. whether L5 preserves L3 mAP75 while improving recall/mAP50/mAP50-95;
9. test result only if the frozen protocol requires final external confirmation, clearly marked as not used for selection;
10. one of these conclusions:

```text
A. OPC supported: L5 > L3 and L3 high-IoU localization advantage is retained.
B. No added value: L5 <= L3; fixed Inner-EIoU(r=0.8) remains the preferred loss.
C. Trade-off only: L5 repairs overlap coverage but loses too much high-IoU localization.
```

After the report, STOP. Do not automatically start another training run.
