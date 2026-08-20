# YOLO11-DDFL v1 — Densified DFL adapted to Ultralytics 8.4.113

Target: **YOLO11 / Ultralytics 8.4.113**, first used as a localization-representation experiment on IWHR.

This track adapts only the **Densified Distribution Focal Loss (D-DFL)** idea from DSDL. It is deliberately not a copy of the YOLOv9 implementation and not the full DSDL method.

## 1. Why an adaptation is required

Ultralytics 8.4.113 `Detect` uses `reg_max=16`, so each box side has 16 categorical logits and the box head emits `4*16=64` regression channels. The same `reg_max` is also used when choosing the hidden width of the regression tower. The native DFL integral uses support `0,1,...,15`.

A literal transplant of the DSDL configuration would change more than the representation we want to test. DSDL's published D-DFL uses a denser support near zero and a larger number of bins, while its full method can also combine D-TAL and Signed DFL.

YOLO11-DDFL v1 therefore makes four controlled changes:

1. **D-DFL only.** Native YOLO11 TAL requires candidate anchor points to lie inside GT boxes (`mask_pos = mask_topk * mask_in_gts * mask_gt`). Native positive `l,t,r,b` targets are consequently non-negative. Signed bins are not justified unless assignment is also changed to allow outside-GT positives, which is a different experiment.
2. **Preserve native range.** Standard YOLO11 represents distances up to about 15 feature units. DDFL v1 keeps support `[0,15]`; it does not import DSDL's extra endpoint 16.
3. **Preserve regression-tower hidden width.** We keep the original YOLO11 hidden-width calculation based on base `reg_max=16`. Only the final box-logit conv changes from 64 to 80 output channels.
4. **Add a matched-capacity control.** `uniform20` has the same 20 bins / 80 output channels as `ddfl20`, but distributes the 20 supports uniformly over `[0,15]`. This separates the effect of densifying near zero from simply increasing head capacity.

## 2. Frozen supports

### D0 — native standard16

```text
0, 1, 2, 3, ..., 15
```

16 bins, 64 regression output channels. This is the untouched Ultralytics baseline.

### D1 — uniform20 capacity control

```text
torch.linspace(0, 15, 20)
```

20 bins, 80 regression output channels, uniform spacing.

### D2 — YOLO11-DDFL20

```text
0,
0.25,
0.50,
0.75,
1.0,
1.5,
2.0,
3.0,
4.0,
5.0,
6.0,
7.0,
8.0,
9.0,
10.0,
11.0,
12.0,
13.0,
14.0,
15.0
```

20 bins, 80 regression output channels.

The dense part is intentionally limited to `[0,2]`. The first-round experiment must not tune these values.

## 3. Generalized DFL target interpolation

Let the strictly increasing support be

```text
R = {r_0, ..., r_(K-1)}.
```

For one continuous target distance `y`, find neighboring support values

```text
r_l <= y <= r_r.
```

Use

```text
w_r = (y - r_l) / (r_r - r_l)
w_l = 1 - w_r
```

and

```text
L_DDFL(y) = -w_l log p_l - w_r log p_r.
```

At an exact support point all target mass belongs to that exact bin. Targets are clamped to `[0, 14.99]`, preserving the native YOLO11 effective regression range.

`research_tracks/yolo11_ddfl/yolo11_ddfl.py` contains the framework-light implementation.

## 4. Generalized decode

For each box side,

```text
p = softmax(logits)
d_hat = sum_k p_k * r_k
```

The rest of `dist2bbox`, anchor geometry and stride scaling stays unchanged.

The same support buffer must be used by both the training criterion and the inference `Detect` head. A mismatch between training support and inference support is a hard error.

## 5. YOLO11 head contract

Do **not** simply set native `reg_max=20` and leave the rest untouched. In Ultralytics 8.4.113 that would also increase the hidden regression-tower width because native `c2` contains `self.reg_max * 4`.

For D1/D2 use conceptually:

```python
base_reg_max = 16                 # frozen native tower budget
reg_values = ...                  # uniform20 or ddfl20
num_bins = len(reg_values)        # 20

self.reg_max = num_bins
self.no = nc + 4 * num_bins

c2 = max((16, ch[0] // 4, base_reg_max * 4))  # NOT num_bins * 4
self.cv2 = ... nn.Conv2d(c2, 4 * num_bins, 1)
self.dfl = NonUniformDFLIntegral(reg_values)
```

This keeps the hidden tower identical to native YOLO11 and changes only its final categorical representation.

## 6. Pretrained-weight transfer

For D1/D2 load the same YOLO11 pretrained checkpoint as baseline.

Expected transfer behavior:

- backbone: transfer;
- neck: transfer;
- classification tower/head: transfer;
- box regression tower hidden convs: transfer because their widths remain native;
- only the last box-output conv for each detection level changes shape `64 -> 80` and is reinitialized.

Do **not** reuse the native 64-channel output weights under a different 16-bin semantic support. Tensor-shape compatibility would hide a semantic mismatch: old channel `k` means distance `k`, which is not true for a non-uniform 16-bin support. Reinitializing the changed final categorical layer is cleaner.

The implementation report must verify that unexpected missing/mismatched keys do not extend beyond the intended final box-output layers.

## 7. Loss integration contract

Keep native CIoU exactly:

```python
iou = bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, CIoU=True)
loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum
```

For D1/D2 only replace the DFL path:

```python
target_ltrb = bbox2dist(anchor_points, target_bboxes, reg_max=None)
target_pos = target_ltrb[fg_mask]
pred_pos = pred_dist[fg_mask].view(-1, 4, num_bins)
loss_dfl = nonuniform_dfl(pred_pos, target_pos) * weight
loss_dfl = loss_dfl.sum() / target_scores_sum
```

`v8DetectionLoss.bbox_decode()` must use the same support rather than `torch.arange(reg_max)`:

```python
p = pred_dist.view(B, A, 4, num_bins).softmax(3)
d = (p * reg_values.view(1, 1, 1, num_bins)).sum(3)
return dist2bbox(d, anchor_points, xywh=False)
```

No changes to:

- `TaskAlignedAssigner`;
- target-score normalization;
- CIoU;
- BCE classification loss;
- box / cls / dfl global gains;
- backbone / neck;
- augmentation;
- NMS / max_det;
- evaluation protocol.

## 8. Required three-cell experiment

Use exactly the frozen IWHR baseline training recipe, changing only the regression representation.

| ID | support | bins | output ch | purpose |
|---|---|---:|---:|---|
| D0 | native `0..15` | 16 | 64 | baseline |
| D1 | uniform `[0,15]` | 20 | 80 | matched-capacity control |
| D2 | densified near zero | 20 | 80 | YOLO11-DDFL |

Interpretation:

- `D2 <= D1`: densification itself is not supported; stop D-DFL even if both beat D0.
- `D1 > D0` and `D2 ~= D1`: increased categorical capacity, not D-DFL geometry, is the likely explanation.
- `D2 > D1 > D0`: both capacity and densification help; report both effects separately.
- `D2 > D0` and `D1 ~= D0`: strongest evidence that non-uniform density near zero matters.

Do not add Signed DFL, D-TAL, P2, new IoU or extra loss terms in this first experiment.

## 9. Engineering gates before training

All must pass:

1. standard16 generalized loss numerically matches native Ultralytics DFL on fixed tensors;
2. standard16 generalized integral matches `softmax @ arange(16)`;
3. exact support targets reduce to single-bin NLL;
4. targets between dense bins use correct linear interpolation;
5. D1 and D2 both produce `[B,4,A]` decoded distance tensors;
6. D1/D2 forward/backward finite in FP32 and AMP;
7. train-side bbox decode and inference-side head decode match for identical logits/support;
8. native D0 model remains byte/code-path compatible where possible;
9. D1/D2 pretrained transfer mismatches only intended final regression-output convs;
10. train / val / predict / TorchScript and ONNX smoke tests pass before full training.

## 10. Research boundary

This track is a direct YOLO11 adaptation of the D-DFL representation idea, not a claim that D-DFL was invented here. The potentially project-specific contribution would have to come later from task evidence or a further mechanism that is demonstrably distinct. First determine whether the published densification principle transfers to IWHR at all.
