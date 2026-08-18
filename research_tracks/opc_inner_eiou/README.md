# OPC-Inner-EIoU research track

This directory contains the isolated implementation and validation plan for
**Overlap-Preserving Contractive Inner-EIoU (OPC-Inner-EIoU)**.

The track starts from one empirical fact already established by the frozen
validation experiments: **Inner-EIoU with fixed ratio `r=0.8` (L3) is the best
current bbox-loss candidate**, while `r=1.2` and the previous SQA controller did
not improve over L3. Therefore OPC does not reopen the full dynamic-ratio search.
It keeps the successful contraction floor `r0=0.8` and only repairs one explicit
geometric failure mode of fixed contraction.

## Files

- `opc_inner_eiou.py` — self-contained L3 and L5 reference implementation.
- `test_reference.py` — numerical, gradient, controller, and overlap-preservation tests.
- `audit_overlap_risk.py` — frozen-validation audit for the `0.8 < u < 1` risk band.
- `ultralytics_adapter.py` — TAL-weighted adapter for Ultralytics 8.4.113.
- `experiment_plan.yaml` — preregistered experiment and stop rules.
- `PROJECT_IMPLEMENTATION.md` — project integration contract.
- `CODEX_TASK.md` — concise executable handoff for Codex.

## 1. Starting point: why retain `r0=0.8`

For equal-size boxes with horizontal center displacement `d`, fixed Inner-IoU
with auxiliary-box ratio `r` gives

```text
IoU_inner = (r w - d) / (r w + d),  0 <= d < r w
```

and the corresponding overlap loss has local slope

```text
|dL/dd|_(d->0+) = 2 / (r w).
```

Thus `r=0.8` increases the high-quality local overlap gradient by a factor
`1/r = 1.25` relative to ordinary IoU. Its local curvature magnitude scales as
`1/r^2 = 1.5625`. This is consistent with L3's observed advantage being concentrated
in high-IoU localization metrics rather than only coarse detection.

OPC does **not** try to make contraction stronger than 0.8. It also never uses
`r>1`, because the previous broad-basin (`r=1.2`) control did not provide a gain.

## 2. The fixed-contraction defect

Let prediction and ground truth be axis-aligned boxes with centers `(cx, cy)` and
`(cx_g, cy_g)` and dimensions `(w, h)` and `(w_g, h_g)`.

Define normalized center-separation states

```text
u_x = 2 |cx-cx_g| / (w+w_g+eps)
u_y = 2 |cy-cy_g| / (h+h_g+eps)
u   = max(u_x, u_y)
```

For positive-area boxes:

```text
u < 1
```

is equivalent to positive overlap in both axes. If both boxes are contracted by
ratio `r`, their auxiliary boxes overlap when

```text
u < r.
```

Therefore fixed `r=0.8` creates a specific risk region

```text
0.8 < u < 1.0
```

where the original boxes still overlap but the contracted auxiliary boxes can
already have zero overlap. This is the only behavior OPC is designed to repair.

## 3. OPC controller

The validated contraction floor is fixed:

```text
r0 = 0.8
```

The overlap state is detached from autograd:

```text
u = stop_gradient(max(u_x, u_y)).
```

The controller is

```text
r(u) = r0,                                           u <= r0
z    = (u-r0)/(1-r0),                                r0 < u < 1
r(u) = r0 + (1-r0)(2z-z^2),                          r0 < u < 1
r(u) = 1,                                            u >= 1.
```

Hence

```text
0.8 <= r(u) <= 1.0.
```

There is no auxiliary-box enlargement.

### Overlap-preservation property

Inside the transition band,

```text
u = r0 + (1-r0) z
r = r0 + (1-r0)(2z-z^2)
```

so

```text
r-u = (1-r0) z (1-z) > 0,  0 < z < 1.
```

Because contracted auxiliary boxes overlap whenever `u < r`, this gives the
central design guarantee:

> Whenever the original boxes still overlap (`u<1`), OPC does not let the
> contraction controller remove that overlap prematurely.

This is a geometric constraint, not a learned quality heuristic.

## 4. Loss definition

OPC keeps the EIoU center/width/height geometry terms unchanged and modifies only
the overlap geometry:

```text
L_OPC = 1 - IoU(B^r, G^r) + D_center + D_width + D_height
```

where `r=r(u)` and `B^r`, `G^r` are width/height-scaled around their own centers.

The EIoU geometry terms are

```text
D_center = ((cx-cx_g)^2 + (cy-cy_g)^2) / (cw^2 + ch^2 + eps)
D_width  = (w-w_g)^2 / (cw^2 + eps)
D_height = (h-h_g)^2 / (ch^2 + eps)
```

with `(cw,ch)` from the smallest enclosing box.

No NWD, WIoU, Focaler, Shape-IoU, extra sample weighting, or extra training
schedule is added.

## 5. Why detach the controller

The optimization should use the geometry selected by the current state, but the
network should not obtain a second gradient route

```text
pred -> u -> r -> IoU_inner.
```

Therefore `u`, `u_x`, `u_y`, and `r` are detached. Backpropagation flows only
through the final auxiliary-box IoU and the EIoU geometry terms for the selected
ratio.

This also means classical `gradcheck` should be run in a region where the
controller is locally constant (`u<0.8`). In the transition region, numerical
finite differences would include the recomputed detached control state while
autograd intentionally excludes that path.

## 6. Training-before-training audit

Before spending a full run on L5, use matched **frozen-validation** prediction/GT
pairs and run:

```bash
python research_tracks/opc_inner_eiou/audit_overlap_risk.py \
  --input matched_val_boxes.csv \
  --output outputs/opc_overlap_audit.json
```

Input CSV columns:

```text
pred_x1,pred_y1,pred_x2,pred_y2,gt_x1,gt_y1,gt_x2,gt_y2
```

The important statistics are:

- fraction with `u <= 0.8` — OPC is exactly L3 here;
- fraction with `0.8 < u < 1` — the region OPC can actually change;
- fraction with `u >= 1` — OPC falls back to `r=1`;
- fraction where original IoU is positive but L3 inner IoU has collapsed to zero;
- the same collapse statistic for OPC, which should be zero up to floating-point tolerance.

A suggested engineering gate in `experiment_plan.yaml` is a transition-band
fraction of at least 1%. This is **not** a statistical significance threshold.
It simply prevents an expensive run if OPC would modify almost no matched cases.

## 7. Reference tests

Run:

```bash
python research_tracks/opc_inner_eiou/test_reference.py
```

The tests cover:

- exact-match zero loss;
- exact L3 equivalence in the safe `u<=0.8` region;
- known controller values (`u=0.85 -> r=0.8875`, `u=0.9 -> r=0.95`);
- strict `0.8 <= r <= 1` contract;
- analytic transition property `r>u`;
- a concrete case where L3 loses auxiliary overlap but OPC preserves it;
- centered scale mismatch invariance of simultaneous contraction;
- detached controller;
- tiny and elongated boxes;
- fp16/bfloat16 finite behavior through fp32 geometry promotion;
- autograd gradcheck in the constant-ratio region.

## 8. Ultralytics 8.4.113 integration contract

`ultralytics_adapter.py` deliberately replaces only the per-positive IoU-style
bbox regression term. Preserve the existing:

- TaskAlignedAssigner/TAL;
- foreground mask;
- target-score weighting;
- target-score normalization;
- DFL branch;
- classification loss;
- optimizer, scheduler, augmentation, dataset split, seed, and evaluator.

Unlike SRB, OPC has no pixel-resolution `delta`; all controller quantities are
dimensionless. Prediction and target boxes only need to be valid aligned xyxy
boxes in the same coordinate system.

## 9. Experiment identity

Historical runs are not retrained. The only new full training candidate is:

```text
L5 = YOLO11n + OPC-Inner-EIoU(r0=0.8)
```

Primary comparison:

```text
L3 fixed Inner-EIoU(r=0.8)  vs  L5 OPC-Inner-EIoU
```

The experiment is successful only if L5 improves the frozen-validation primary
metric over L3 without simply erasing L3's high-IoU localization advantage.
The preferred pattern is to preserve mAP75 while recovering recall/mAP50 or
raising overall mAP50-95.

After L5 evaluation and report generation, stop. Do not automatically search
`r0`, modify the transition polynomial, revive `r>1`, or combine with DBRA based
on the result.
