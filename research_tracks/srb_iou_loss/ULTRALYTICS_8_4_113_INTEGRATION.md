# Ultralytics 8.4.113 / YOLO11 Integration Notes

This document is a **versioned integration contract**, not permission to edit the frozen training protocol.

## 1. Scope

Formal SRB-IoU v1 changes only the IoU regression term inside the YOLO11 bounding-box loss.

Must remain unchanged:

- model architecture, including DBRA;
- DFL formulation and weight;
- TaskAlignedAssigner / TAL;
- classification loss;
- pretrained initialization policy;
- optimizer, LR schedule, batch, epochs, augmentations and imgsz;
- train/val split and evaluator;
- baseline checkpoint/result reuse policy.

Do not continue training an existing DBRA checkpoint after changing the loss for a formal comparison. Instantiate the architecture under the frozen training recipe.

---

## 2. Before touching code: inspect the local frozen source

The project is pinned to Ultralytics 8.4.113, but local research branches may already contain custom Detect/AttnDetect code. Inspect the exact working tree first.

Locate:

```text
ultralytics/utils/loss.py
```

and identify:

```text
BboxLoss
v8DetectionLoss (or the active YOLO11 detection criterion)
bbox_decode
TaskAlignedAssigner call
```

Record in the experiment manifest:

```text
ultralytics version
git commit / working-tree status
BboxLoss file path
pred_bboxes coordinate system
target_bboxes coordinate system
stride_tensor shape
fg_mask shape
```

Do not guess coordinate units from variable names.

---

## 3. Expected integration pattern

A common YOLO loss path decodes candidate boxes in feature-grid units and later uses a per-anchor `stride_tensor` to map to/from input pixels. If the local 8.4.113 code follows that pattern, SRB needs one additional input at the BboxLoss call site:

```text
stride_tensor
```

because the preregistered resolution floor is

```math
\delta_{pixel}=1.0
```

and therefore, for a positive candidate at stride `s`,

```math
\delta_{grid}=\frac{1.0}{s}.
```

The helper in `ultralytics_adapter.py` implements this conversion while preserving the existing TAL target-score weighting.

If the local boxes are already in input-pixel coordinates, do **not** divide by stride. Use `delta=1.0` directly.

---

## 4. Minimal conceptual patch

The original BboxLoss usually has a branch conceptually equivalent to:

```python
weight = target_scores.sum(-1)[fg_mask]
loss_iou = weighted_original_iou_loss(...)
```

Replace only that IoU calculation with the SRB per-box loss, preserving the same positive mask, target-score weight and normalization:

```python
loss_iou = weighted_srb_for_ultralytics(
    pred_bboxes=pred_bboxes,
    target_bboxes=target_bboxes,
    target_scores=target_scores,
    target_scores_sum=target_scores_sum,
    fg_mask=fg_mask,
    stride_tensor=stride_tensor,
    delta_pixel=1.0,
    lambda_edge=1.0,
)
```

The exact call signature must be adapted to the frozen local code rather than forcing the local code to match this example.

DFL must execute exactly as before after the SRB IoU term is computed.

---

## 5. Do not change loss gain implicitly

SRB is not numerically guaranteed to have the same raw scale as CIoU. For the first mechanism run:

1. keep the existing YOLO `box` loss gain unchanged;
2. log the unscaled SRB mean and final weighted box-loss contribution;
3. log CIoU reference values on a small frozen batch if possible, without using them to tune a new gain;
4. do not immediately rescale SRB to make its magnitude look like CIoU.

If the raw scale causes obvious optimization failure, stop and report it as evidence. Any later loss-gain adjustment must be a separately labeled ablation.

---

## 6. Mixed precision

`srb_iou.py` promotes fp16/bfloat16 geometry to fp32 and uses a stable log-cosh identity. Keep this behavior.

Do not replace

```python\stable_log_cosh(x)
```

with

```python
torch.log(torch.cosh(x))
```

inside AMP code because very large residuals can overflow `cosh` before the logarithm is applied.

---

## 7. Coordinate-space sanity assertion

Before the first real training run, print or save for one batch:

```text
unique positive strides
median GT width/height in BboxLoss coordinate units
resolved delta for each stride
```

For a standard stride set `[8, 16, 32]` and feature-grid box coordinates, the expected one-input-pixel deltas are:

```text
stride 8  -> delta 0.125
stride 16 -> delta 0.0625
stride 32 -> delta 0.03125
```

If the resolved values are instead `1.0` on every level while boxes are grid-normalized, stop: the integration is wrong.

---

## 8. Required integration tests

Before training, add local tests around the actual patched loss path:

### Build / forward

- baseline model builds;
- DBRA model builds;
- one batch forward succeeds;
- one batch loss backward succeeds;
- no NaN/Inf under AMP.

### Identity/control

On a synthetic batch where decoded prediction equals target:

```text
SRB IoU term ~= 0
```

### DFL isolation

With the same prediction tensors, confirm the DFL computation code path and shapes are unchanged by the SRB patch.

### TAL isolation

Confirm positive assignments and `fg_mask` are computed before the SRB term and remain identical to the unmodified run for the same model outputs.

### Stride resolution

Confirm every foreground candidate uses `delta_pixel / stride` if boxes are in grid units.

---

## 9. Logging for the first run

At minimum log per epoch or at a reasonable interval:

```text
mean SRB total
mean SRB overlap component
mean SRB edge component
mean beta
beta p10/p50/p90
box-gradient norm if instrumentation is available
number of foreground candidates by stride
```

The diagnostic components should be detached. Do not add them to the objective twice.

---

## 10. Formal comparison order

The first formal model comparison should isolate the loss:

```text
existing YOLO11 baseline       reuse result if protocol is frozen
YOLO11 + SRB                   new training run
existing YOLO11 + DBRA         reuse result
YOLO11 + DBRA + SRB            only after SRB-alone mechanism is credible
```

Mature-loss controls such as EIoU and MPDIoU are useful, but they must use the same training budget and not trigger a hyperparameter grid.

Do not compare a newly trained SRB model with a DBRA checkpoint that received extra continuation epochs.

---

## 11. Acceptance / rejection signals

Positive signal:

- mAP50-95 improves without a material AP75 collapse;
- APs/ARs or equivalent small-target metrics improve;
- training remains numerically stable;
- gradient statistics become less extreme in tiny bins if measured;
- gains are not only a change in score calibration.

Reject or redesign if:

- AP75 consistently drops while AP50 alone rises;
- edge term dominates the total loss for most of training;
- `beta` is unexpectedly near 1 for ordinary boxes because coordinate units are wrong;
- large/medium localization degrades materially;
- the method requires immediate NWD/Inner/focusing additions to become viable.

---

## 12. Patch hygiene

Keep project edits minimal and auditable:

```text
ultralytics/utils/loss.py             minimal call-site change
ultralytics/utils/srb_iou.py          copied/adapted reference implementation
(optional) tests/test_srb_iou.py      project-local integration tests
```

Do not overwrite upstream IoU helpers globally unless the project explicitly wants every caller to use SRB. Prefer a dedicated function so CIoU/EIoU controls remain easy to run.

Every local algorithmic deviation from `research_tracks/srb_iou_loss/srb_iou.py` must be documented before training.
