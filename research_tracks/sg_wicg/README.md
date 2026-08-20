# SG-WICG v1 — frozen research contract

Target stack: **YOLO11 / Ultralytics 8.4.113**.

This track tests a box-regression hypothesis only. It must not alter TaskAlignedAssigner, DFL, classification loss, Detect heads, decoding, augmentation, NMS, dataset split, training recipe, or evaluation protocol.

## Frozen formulation

For foreground positive `i`, with the existing TAL soft weight

`a_i = sum_c target_scores[i, c]`,

use

`L_box = sum_i a_i * g_i * ((1-lambda_i) * L_InnerCIoU_i + lambda_i * L_GCD_i) / target_scores_sum`.

- `Inner-CIoU`: auxiliary boxes scaled about their centers with `r=1.25`; CIoU distance/aspect penalties remain on the original boxes.
- `GCD`: published symmetric Gaussian Combined Distance, `1-exp(-sqrt(D_gcd^2))`, evaluated in FP32.
- `Scale gate`: `lambda_i = sigmoid((12 - s_i)/2)`, where `s_i` is the GT short side in **current network-input pixels**.
- `Wise focus`: plain-IoU quality only, `q_i=stopgrad(1-IoU_i)`, WIoUv3 non-monotonic form with `alpha=1.7`, `delta=2.7`, EMA rate `0.01`.
- `Mean preservation`: normalize Wise gains so the TAL-weighted mean gain is 1. This redistributes box-regression gradient instead of changing its total gain.

No learnable gate, dynamic Inner ratio, extra GCD multiplier, additional IoU family, or new box gain is allowed in v1.

## Six-cell ablation

| ID | `reg_loss_mode` | Regression geometry | Frozen extra args |
|---|---|---|---|
| A0 | `ciou` | native Ultralytics CIoU | none |
| A1 | `inner_ciou` | Inner-CIoU | `inner_ratio=1.25` |
| A2 | `gcd` | GCD | none |
| A3 | `inner_gcd_fixed` | 0.5 Inner-CIoU + 0.5 GCD | `inner_ratio=1.25`, `gcd_fixed_weight=0.5` |
| A4 | `sg_icg` | scale-gated Inner-CIoU/GCD | `inner_ratio=1.25`, `sg_tau_px=12`, `sg_temp_px=2` |
| A5 | `sg_wicg` | A4 + Wise focus | A4 args + `wise_alpha=1.7`, `wise_delta=2.7`, `wise_ema_rate=0.01` |

**A0 must use the exact native 8.4.113 line `bbox_iou(..., CIoU=True)`**. Do not reimplement baseline CIoU through `sg_wicg.py`.

## Exact Ultralytics 8.4.113 hook

In `ultralytics/utils/loss.py`, `BboxLoss.forward()` already receives:

- `pred_bboxes`
- `target_bboxes`
- `target_scores`
- `target_scores_sum`
- `fg_mask`
- `imgsz`
- `stride_tensor`

The only regression line to replace for A1–A5 is the native CIoU calculation/reduction. DFL immediately below it remains untouched.

For positive samples:

```python
weight = target_scores[fg_mask].sum(-1, keepdim=True)
p = pred_bboxes[fg_mask]
t = target_bboxes[fg_mask]
stride_pos = foreground_strides(stride_tensor, fg_mask)
```

`t` is already in grid units because `v8DetectionLoss` calls `BboxLoss` with `target_bboxes / stride_tensor`. The scale gate converts GT size back to input pixels internally by `wh_grid * stride_pos`.

Recommended structure inside `BboxLoss`:

```python
if self.reg_loss_mode == "ciou":
    iou = bbox_iou(p, t, xywh=False, CIoU=True)
    loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum
else:
    loss_iou, self.sg_wicg_diag = self.sg_wicg(
        p,
        t,
        weight,
        stride_pos,
        target_scores_sum,
    )
```

DFL code after this block must be byte-for-byte behavior-equivalent to baseline.

## Config keys

Register these near `box`, `cls`, `dfl` in `ultralytics/cfg/default.yaml` so CLI validation accepts them:

```yaml
reg_loss_mode: ciou
inner_ratio: 1.25
gcd_fixed_weight: 0.50
sg_tau_px: 12.0
sg_temp_px: 2.0
wise_alpha: 1.70
wise_delta: 2.70
wise_ema_rate: 0.01
wise_mean_init: 1.0
```

`BboxLoss` must receive the model/train args or an explicit immutable SG-WICG config built from them.

## Wise state and resume

`WiseFocus.iou_mean` is training state, not a model parameter. Ultralytics 8.4.113 strips `criterion` from serialized EMA checkpoints, therefore criterion-local `iou_mean` is **not sufficient for resume correctness**.

Codex must persist a minimal checkpoint field such as:

```python
"sg_wicg_state": {"iou_mean": float(...)}
```

and restore it after the criterion is reconstructed. A5 is not complete until uninterrupted-vs-resume next-step equivalence is tested.

A0–A4 have no Wise EMA state.

## DDP scope

The v1 reference code deliberately implements **single-GPU Wise population statistics only**. This is intentional. Stock `v8DetectionLoss` calls `BboxLoss` only when the local rank has foreground positives, so putting a collective inside A5 can deadlock when another rank has zero positives.

**For the first six-cell ablation, run A5 on one GPU.** If multi-GPU A5 is later required, move Wise population-stat synchronization to a code path executed by every rank every step, including ranks with zero positives, then add a dedicated DDP parity/deadlock test. Do not claim DDP support before that work is done.

## Required pre-training tests

Before launching the six full runs, pass all of the following:

1. `A0 parity`: native baseline output unchanged; ideally compare old/new branch with `reg_loss_mode=ciou` on fixed tensors and one short smoke train.
2. identical boxes: Inner-CIoU and GCD approximately 0.
3. GCD scale invariance for common coordinate scaling x2/x4.
4. non-overlap GCD finite with nonzero bbox gradient.
5. gate monotonicity and `lambda(short=12px)=0.5`.
6. Wise TAL-weighted mean gain approximately 1.
7. A1–A5 finite forward/backward under FP32 and AMP.
8. DFL parity: same tensors/targets give unchanged DFL result across A0 and A5.
9. empty-foreground behavior unchanged.
10. A5 save/resume equivalence for `iou_mean`, loss, and next-step gradients.

## Diagnostics to log

At minimum record, without feeding them back into optimization:

- `inner_loss_mean`
- `gcd_loss_mean`
- `plain_iou_mean`
- `sg_lambda_mean`, p10/p50/p90
- `target_short_px_mean`
- `wise_iou_mean`
- `wise_beta_mean`
- `wise_gain_mean` (must remain ~1 in TAL-weighted sense)
- `wise_gain_max`

Evaluation remains the project's frozen evaluator. Pay special attention to AP75, AP50-95, size buckets/tiny recall, and boundary/center error if already available.

## Interpretation rules

- A1 and A2 both fail vs A0: close the track; do not rescue it by adding more mechanisms.
- A2 > A3: Inner branch is likely contaminating useful GCD geometry.
- A3 ~= A4: no evidence for scale conditioning; remove the scale gate.
- A4 >= A5: Wise is redundant/harmful on top of TAL; remove Wise.
- Promote only a simpler winning prefix of the chain; `A5` is not privileged merely because it is the full name.
