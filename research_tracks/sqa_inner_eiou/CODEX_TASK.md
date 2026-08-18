# Codex Task: Implement and Run the L1-L4 BBox-Loss Study

Read these files first:

```text
research_tracks/sqa_inner_eiou/README.md
research_tracks/sqa_inner_eiou/PROJECT_IMPLEMENTATION.md
research_tracks/sqa_inner_eiou/experiment_plan.yaml
research_tracks/sqa_inner_eiou/losses.py
research_tracks/sqa_inner_eiou/ultralytics_adapter.py
```

## Goal

Integrate the preregistered loss family into the existing local YOLO11n / Ultralytics 8.4.113 project and prepare the four controlled experiments:

```text
L1 EIoU
L2 Inner-EIoU ratio=1.2
L3 Inner-EIoU ratio=0.8
L4 SQA-Inner-EIoU eta=0.20
```

## Hard constraints

- Do not retrain the frozen baseline.
- Do not change the dataset split.
- Do not use the test set for model selection.
- Keep the frozen baseline training recipe unchanged.
- Change only the IoU-style bbox regression term.
- Keep TAL, fg_mask generation, target-score weighting, classification loss and DFL unchanged.
- Keep the existing box loss gain unchanged for the first comparison.
- Do not add NWD, WIoU, Focaler, Focal-EIoU, Shape-IoU, SRB, or any extra weighting mechanism.
- Do not combine DBRA with these losses until the loss-only L1-L4 comparison is complete.
- L2/L3 are fixed mechanism controls, not hyperparameter-search values.
- L4 must use `eta=0.20`, smoothstep and detached quality exactly as implemented.

## Step 1: Locate the actual project loss path

Inspect the exact local Ultralytics 8.4.113 working tree and record:

```text
version
commit / working tree status
active BboxLoss path
pred_bboxes coordinate system and shape
target_bboxes coordinate system and shape
fg_mask shape
target_scores shape
DFL call path
TAL call path
```

Do not guess from upstream source if the local project has modifications.

## Step 2: Run reference tests

Run:

```bash
python research_tracks/sqa_inner_eiou/test_reference.py
```

Fix only genuine implementation/compatibility problems. Do not alter the mathematical definitions to make tests pass.

## Step 3: Integrate one shared selector

Copy/adapt the reference implementation into the local YOLO project and expose one selector:

```text
L1 | L2 | L3 | L4
```

Use the same patched BboxLoss code path for all four experiments.

Preserve the original TAL weight and `target_scores_sum` normalization.

Do not globally replace generic IoU helpers used by assignment, NMS or metrics.

## Step 4: Pre-training integration gate

For each L1-L4 run one real batch and confirm:

```text
build PASS
forward PASS
loss finite
backward PASS
AMP finite
TAL fg_mask unchanged for identical outputs
DFL code path unchanged
classification loss unchanged for identical outputs
```

For L4 additionally save:

```text
IoU mean
ratio min
ratio p10
ratio p50
ratio p90
ratio max
```

Require:

```text
0.8 <= ratio <= 1.2
```

If this gate fails, stop before training and report the failure.

## Step 5: Formal loss-only runs

After the gate passes, run in this order with the exact frozen training recipe:

```text
L1 -> L2 -> L3 -> L4
```

Do not rerun baseline.

If the project already has a formal experiment launcher, use it rather than introducing a second training system.

## Step 6: Evaluation and report

Evaluate on frozen validation using the same evaluator and produce one table containing:

```text
AP, AP50, AP75, APs, APm, APl, ARs, ARm, ARl
```

Include the frozen baseline row by reusing the existing result.

Also report:

```text
L1 - baseline
L2 - L1
L3 - L1
L4 - L2
L4 - L3
```

Interpretation must answer:

```text
Does EIoU itself help?
Does broad-basin Inner geometry help?
Does fine-regression Inner geometry help?
Does SQA outperform both fixed controls?
```

For L4 include the observed quality/ratio distribution.

## Stop condition

After the L1-L4 loss-only report is complete, stop.

Do not automatically launch DBRA + loss training. The next combination must be selected only after reviewing the mechanism results.
