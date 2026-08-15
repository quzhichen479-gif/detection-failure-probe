# Scope, Evidence Boundary, and Stop Rules

## 1. Research question

Can a **classification-only spatial/context attention module** improve YOLO11n water-surface floating-object detection without damaging localization quality or inflating the model enough to erase the YOLO11n efficiency advantage?

This is deliberately narrower than “does attention help?”. The project already has substantial evidence that generic feature enhancement and module stacking often fail to transfer.

## 2. Confirmed project evidence that constrains this track

Under the project's matched PoTATO independent-test protocol (2000 images, imgsz=640, batch=16, seed=42, validation-selected `best.pt`, Ultralytics 8.4.113):

- YOLO11n baseline: Precision 0.9023, Recall 0.8061, mAP50 0.8706, mAP75 0.4844, mAP50-95 0.4848.
- B3 TE-TG: Precision 0.8810, Recall 0.8235, mAP50 0.8674, mAP75 0.4777, mAP50-95 0.4790.
- Therefore TE-TG increased ordinary recall but did **not** improve total AP or high-IoU localization.
- Earlier generic/feature modules also produced many negative or unstable outcomes.
- E2+ECA had only a small single-seed matched-test gain (~+0.00237 mAP50-95 over its matched baseline). This is an engineering-control signal, not proof that attention is a publishable contribution.

Implication: a new attention experiment must preserve the regression path and answer whether classification/context discrimination can improve precision/recall balance **without perturbing localization**.

## 3. Hard constraints

1. No knowledge distillation, teacher/student, pseudo-label teacher, or teacher feature matching.
2. Do not use ECA/CBAM/CoordAtt/EMA/SKA as the proposed innovation.
3. Do not modify box regression, DFL, anchors/assignment, IoU loss, or NMS in the first attention round.
4. Do not enable more than one candidate module in a model.
5. Do not enable a candidate at more than one insertion point in the first screening round.
6. Do not touch the repository's existing `src/` probe package.
7. Do not use the independent test set for iterative tuning.
8. Preserve the frozen baseline recipe except for the module under test.

## 4. Why classification-only first

YOLO11 Detect has sibling box and class paths. Existing failures show that improving recall while worsening AP75 is easy. A shared-neck attention block would simultaneously perturb class and localization features, making attribution weak.

The first round therefore places attention only on `Detect.cv3` (classification path). This creates a cleaner causal test:

- box/DFL feature path remains baseline-identical;
- classification feature path changes;
- if AP75 drops substantially, the cause is less likely to be direct regression-feature corruption and more likely to involve scoring/ranking/assignment interactions.

## 5. Screening metrics

Minimum report for every candidate:

- Precision
- Recall
- mAP50
- mAP75
- mAP50-95
- parameter count
- GFLOPs if available
- inference latency under the same benchmark environment

Preferred diagnostic additions:

- APs / APm / APl or project tiny-object bins
- FP/image
- classification score distributions for TP vs FP
- matched-object IoU distribution
- per-scale P3/P4/P5 contribution if available

## 6. Stop / advance rules

### Immediate stop

Stop a candidate if any of the following holds under the frozen validation protocol:

- mAP50-95 decreases by >= 0.005 absolute versus matched baseline;
- mAP75 decreases by >= 0.005 without a compensating and explainable gain in the primary objective;
- latency or compute cost is disproportionate to any gain;
- implementation changes the regression branch or unrelated training logic;
- build/export/runtime behavior is unstable.

### Candidate for second seed / deeper audit

A candidate may advance only if it shows all of:

- non-negative mAP50-95 delta or a clearly valuable precision/recall trade-off;
- no meaningful mAP75 regression;
- reproducible build/train/val/predict/export behavior;
- modest complexity overhead;
- a plausible mechanism visible in diagnostics, not only a lucky scalar metric.

## 7. Experimental order

Recommended first-screen order:

1. CAA-Lite @ P3 cls pre-predictor
2. LSK-Lite @ P3 cls mid-branch
3. BRA-Lite @ P3 cls mid-branch

Only after those three are separately completed should the second insertion position of the best surviving candidate be tested.
