# Codex Task — SRB-IoU Validation and YOLO11 Integration

Read `README.md` and `ULTRALYTICS_8_4_113_INTEGRATION.md` completely before editing the YOLO11 project.

## Phase 1 — reference validation only

1. Use `srb_iou.py` as the frozen algorithmic reference.
2. Run:

```bash
python research_tracks/srb_iou_loss/test_srb_iou_reference.py
python research_tracks/srb_iou_loss/loss_surface_probe.py --delta 1.0
```

3. Inspect translation, scale, curvature and stationary-scan outputs.
4. Do not train a detector if the mathematical/numerical gates in `README.md` fail.
5. Do not modify the SRB formula simply to make a plot look better.

## Phase 2 — local Ultralytics 8.4.113 integration

Only after Phase 1 passes:

1. locate the exact active `BboxLoss` and detection criterion in the local YOLO11 tree;
2. verify whether decoded `pred_bboxes` / `target_bboxes` are in input-pixel or feature-grid coordinates;
3. copy/adapt `srb_iou.py` into an isolated local loss helper;
4. if boxes are in grid units, pass the per-anchor stride and use

```text
delta_grid = 1.0 / stride
```

for every positive candidate;
5. replace only the existing CIoU regression term with SRB-IoU;
6. preserve the existing TAL foreground mask, target-score weighting and normalization;
7. leave DFL, classification loss, DBRA, assignment and all training parameters unchanged;
8. add project-local unit/integration tests for forward, backward, AMP, stride conversion and DFL/TAL isolation.

## Phase 3 — stop before formal training unless explicitly requested

After integration tests pass, report:

```text
files changed
exact BboxLoss call path
box coordinate system
stride_tensor layout
resolved delta by stride
reference test results
loss-surface gate results
one-batch forward/backward/AMP results
```

Do not automatically launch a full training run from this task.

## Hard constraints

- do not retrain the existing baseline as part of implementation;
- do not use test for design/tuning;
- do not modify DFL or TAL;
- do not add NWD, Inner-IoU, WIoU focusing or another attention module;
- do not tune `delta` from detection performance in this first implementation;
- keep `delta_pixel=1.0` and `lambda_edge=1.0` as the preregistered defaults;
- do not warm-start a formal comparison by continuing the already-trained DBRA checkpoint;
- document any deviation from the reference formula before running training.
