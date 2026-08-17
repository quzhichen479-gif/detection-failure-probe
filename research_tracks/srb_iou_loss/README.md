# SRB-IoU — Smooth Resolution-Bounded IoU Loss

> Status: **reference implementation + pre-training validation track**  
> Target project: YOLO11 / Ultralytics 8.4.113 / PoTATO  
> Intended role: replace the **IoU regression term only**; keep DFL, TAL, classification loss, DBRA, data split, optimizer and training budget unchanged.  
> Default preregistered setting: `delta_pixel = 1.0`, `lambda_edge = 1.0`.

## 1. Why this loss exists

PoTATO is strongly pixel-limited at `imgsz=640`. The frozen train+val geometry audit contains 6,995 GT boxes and found:

- 47.48% have equivalent size `sqrt(area) < 8 px`;
- 83.92% have `sqrt(area) < 16 px`;
- 53.90% have short side `< 6 px`;
- median equivalent size is only `8.29 px`;
- a 1-pixel perturbation produces a mean loss jump of about `0.3263` for CIoU and `0.3709` for EIoU on `<4 px` targets, versus about `0.0194/0.0196` for targets `>=64 px`.

This motivates a narrower problem than generic "small-object localization": standard overlap normalization produces a very sharp loss response when the physical box itself occupies only a few input pixels.

SRB-IoU is therefore **not** another NWD/EIoU mixture. It directly changes the overlap normalization and its local influence function.

This also keeps the method distinct from the earlier DyNWD-EIoU work, whose central mechanism was target-scale-dependent mixing between NWD and EIoU. SRB-IoU has no Wasserstein term and no metric-switching gate.

---

## 2. Mathematical definition

For prediction

```text
B = (l, t, r, b)
```

and target

```text
G = (l_g, t_g, r_g, b_g),
```

let intersection and union areas be `I` and `U`. Define a resolution-boundary support area

```math
A_\delta = 2\delta(w_g+h_g) + 4\delta^2.
```

It equals the extra area produced by dilating the GT rectangle by `delta` on every side.

### 2.1 Resolution-bounded mismatch

Standard IoU loss can be written as

```math
1-IoU = \frac{U-I}{U}.
```

SRB first changes this to

```math
m = \frac{U-I}{U+A_\delta}.
```

For large boxes, `A_delta/U -> 0`, so this naturally approaches ordinary IoU mismatch. For pixel-limited boxes, the denominator has a finite resolution floor.

### 2.2 Geometry-derived smoothing factor

Define

```math
\beta_g = \frac{A_\delta}{A_g+A_\delta}.
```

`beta` is not a learned parameter and not a hand-written small/large threshold. It is determined by the fraction of the delta-expanded GT support occupied by the resolution boundary.

### 2.3 Smooth overlap term

```math
L_{overlap}^{SRB}
= \beta_g\log\cosh\left(\frac{m}{\beta_g}\right).
```

Near exact matching,

```math
L_{overlap}^{SRB}\approx \frac{m^2}{2\beta_g},
```

so the exact-match cusp of raw IoU/RB-v0 is converted into a locally quadratic basin with zero first derivative.

For larger mismatch,

```math
\beta\log\cosh(m/\beta) \approx m-\beta\log 2,
```

so medium/low-quality boxes remain close to the original resolution-bounded mismatch.

### 2.4 Bounded edge geometry

Overlap alone has no translation gradient when two boxes are fully disjoint. SRB therefore adds a four-edge robust term:

```math
s_x=w_g+2\delta,\qquad s_y=h_g+2\delta,
```

```math
z_l=\frac{l-l_g}{s_x},\quad
z_r=\frac{r-r_g}{s_x},\quad
z_t=\frac{t-t_g}{s_y},\quad
z_b=\frac{b-b_g}{s_y},
```

```math
L_{edge}=\frac14\sum_{e\in\{l,r,t,b\}}\log\cosh(z_e).
```

Final loss:

```math
\boxed{
L_{SRB}=L_{overlap}^{SRB}+\lambda_e L_{edge}
}
```

The preregistered first implementation uses

```text
delta_pixel = 1.0
lambda_edge = 1.0
```

with no hyperparameter grid before the mechanism is established.

---

## 3. Key mathematical behavior

For equal square boxes of side `w` with a horizontal offset `d`, standard IoU has

```math
\left.\frac{\partial L_{IoU}}{\partial d}\right|_{0^+}=\frac{2}{w}
```

and curvature magnitude proportional to `1/w^2`. Therefore the local conditioning becomes increasingly sharp as the target approaches a few pixels.

The resolution-bounded mismatch instead has

```math
\left.\frac{\partial m}{\partial d}\right|_{0^+}
=\frac{2w}{(w+2\delta)^2},
```

which stays finite and tends to zero as `w -> 0` for fixed positive `delta`.

The additional log-cosh smoothing gives

```math
\nabla L_{overlap}^{SRB}
=\tanh(m/\beta)\nabla m,
```

so the exact match has zero first derivative rather than a raw-IoU cusp.

The edge term satisfies, for example,

```math
\left|\frac{\partial L_{edge}}{\partial l}\right|
<\frac{1}{4(w_g+2\delta)}
\le\frac{1}{8\delta},
```

which supplies a bounded non-overlap gradient.

Important limitation: rectangle intersection uses `min/max/clamp`, so the complete loss remains piecewise smooth at topology changes such as exact edge contact. Do not claim the complete function is globally `C^2` or theoretically free of every non-GT local minimum.

---

## 4. What the dataset audit supports — and what it does not

The geometry audit supports the **relevance of the problem** and the use of `delta=1 px` as a conservative, non-trained default:

- `delta=1`: global median `beta = 0.3671`;
- `<4 px`: mean `beta ~= 0.588`;
- `8–16 px`: mean `beta ~= 0.308`;
- `32–64 px`: mean `beta ~= 0.097`;
- `>=64 px`: mean `beta ~= 0.056`;
- only 14.94% of GT has `beta >= 0.5`;
- no GT has `beta >= 0.8` at `delta=1` in the audit.

The 1-pixel audit also showed SRB's response is selectively reduced in the tiny regime while staying close to CIoU on large boxes.

This does **not** prove SRB improves AP, convergence, AP75, or gradient variance. Those are training/optimization claims and require later validation.

---

## 5. Repository layout

```text
research_tracks/srb_iou_loss/
├── README.md
├── srb_iou.py
├── test_srb_iou_reference.py
├── loss_surface_probe.py
├── ULTRALYTICS_8_4_113_INTEGRATION.md
└── CODEX_TASK.md
```

`SRB-IoU` is deliberately isolated from the package under `src/`. The main `detection-failure-probe` package remains lightweight and does not gain a hard PyTorch dependency.

---

## 6. Standalone use

```python
import torch

from srb_iou import srb_iou_loss

pred = torch.tensor([[10.0, 20.0, 18.0, 26.0]], requires_grad=True)
target = torch.tensor([[11.0, 20.0, 19.0, 26.0]])

loss = srb_iou_loss(
    pred,
    target,
    delta=1.0,
    lambda_edge=1.0,
    reduction="mean",
)
loss.backward()
```

For diagnostics:

```python
loss, parts = srb_iou_loss(
    pred,
    target,
    delta=1.0,
    return_components=True,
)

print(parts["overlap"])
print(parts["edge"])
print(parts["beta"])
print(parts["a_delta"])
```

---

## 7. Coordinate-space rule — critical for YOLO

`delta` is **not** a dimensionless hyperparameter. It is a spatial resolution in the same units as the boxes passed to the loss.

If decoded boxes are in input-pixel coordinates:

```text
delta = 1.0
```

means one input pixel.

If a level at stride `s` represents boxes in grid units, one input pixel becomes

```math
\delta_{grid}=\frac{1}{s}.
```

Therefore do not blindly use `delta=1` on P3/P4/P5 feature-grid boxes. That would mean 8/16/32 input pixels for strides 8/16/32 and would invalidate the mechanism.

Before integration, inspect the exact Ultralytics 8.4.113 `BboxLoss` path used by the local YOLO11 project and record the coordinate system of `pred_bboxes` and `target_bboxes`.

---

## 8. Mixed precision and numerical stability

The implementation intentionally:

1. promotes fp16/bfloat16 box geometry to fp32;
2. computes `log(cosh(x))` through the stable identity

```math
\log\cosh x=|x|+softplus(-2|x|)-\log2,
```

rather than calling `torch.cosh` directly;
3. clamps widths/heights only for numerical protection;
4. keeps a small `eps` in denominators;
5. returns the work-dtype loss rather than down-casting to fp16.

A decoded YOLO box should already satisfy `x2>x1` and `y2>y1`. If invalid boxes are appearing, fix the decoder/integration instead of relying on the clamp to hide them.

---

## 9. Ultralytics integration policy

Formal v1 changes **only the IoU regression term**.

Keep unchanged:

- DFL;
- TaskAlignedAssigner / TAL;
- classification loss;
- DBRA architecture and parameters;
- data split;
- optimizer and LR schedule;
- epochs / batch / imgsz / augmentation;
- evaluation protocol.

Conceptually change

```text
box_loss = CIoU
```

to

```text
box_loss = SRB-IoU
```

while leaving

```text
DFL = original DFL
```

unchanged.

See `ULTRALYTICS_8_4_113_INTEGRATION.md` before patching the local YOLO repository.

---

## 10. Pre-training gates

Do not start a full run just because the code imports.

### Gate M1 — exact match

```text
loss(B=G) == 0
finite gradient
zero/near-zero gradient at exact match
```

### Gate M2 — tiny local conditioning

For GT sizes

```text
4x4, 8x8, 16x16, 32x32, 64x64
```

scan one-axis translation and edge errors. Tiny-box gradient/curvature must not diverge as in raw IoU.

### Gate M3 — non-overlap guidance

For fully separated boxes, overlap gradient may be zero but the edge term must pull the box toward the target with finite gradient.

### Gate M4 — shape stress test

Include

```text
4x32, 8x32, 4x64
```

because PoTATO contains many elongated objects even though `AR>4` is rare.

### Gate M5 — numerical stationary scan

Randomly initialize `(cx, cy, logw, logh)` around tiny GTs and search for non-GT stationary basins. "None found" is acceptable evidence; do not upgrade this to a proof of global convexity.

Run `loss_surface_probe.py` plus the reference tests before training.

---

## 11. First training ablation — frozen ordering

Do not combine SRB with new modules or loss tricks in the first comparison.

Recommended order:

```text
L0  existing CIoU baseline result      (reuse, do not retrain if protocol is frozen)
L1  EIoU                               (mature geometric control)
L2  MPDIoU                             (mature point-distance control)
L3  previous DyNWD-EIoU                (previous-work control, if reproducible under protocol)
L4  RB-v0                              (mechanism ablation only)
L5  SRB-IoU                            (main proposal)
```

Then, only after SRB itself produces a credible signal:

```text
B + DBRA
B + SRB
B + DBRA + SRB
```

This isolates representation-side and localization-side contributions.

Do not rescue SRB by immediately adding NWD, Inner-IoU, WIoU focusing, a scale gate, or a second attention block.

---

## 12. Metrics that matter

The main question is not only mAP50. Track at least:

```text
mAP50-95
AP50
AP75
APs / ARs when evaluator supports them
Precision / Recall
box loss gradient norm statistics
box loss gradient variance
convergence curve
```

The main failure mode to watch is:

```text
tiny optimization becomes stable
but high-quality localization is over-smoothed
=> AP75 drops
```

If AP50/Recall increase but AP75 materially falls, inspect the high-quality loss surface before calling the method successful.

---

## 13. What may be claimed if later experiments succeed

Safe mechanism language:

> Standard IoU normalization produces scale-dependent sensitivity to fixed pixel boundary errors. SRB-IoU introduces a resolution-bounded symmetric-difference normalization and a locally smooth bounded influence function, while retaining robust geometric guidance for non-overlapping boxes.

Do **not** claim without additional evidence:

- water reflection is the cause of the localization problem;
- annotation noise is exactly one pixel;
- SRB proves globally convex optimization;
- SRB has no non-GT local minima;
- `delta=1` is globally optimal;
- SRB universally improves small-object detection beyond PoTATO.

---

## 14. Quick handoff to Codex

Use this short prompt after opening the branch:

```text
Read research_tracks/srb_iou_loss/README.md and CODEX_TASK.md completely.
Use srb_iou.py as the frozen reference implementation. First run the reference
unit tests and loss_surface_probe.py. If and only if the mathematical/numerical
gates pass, integrate SRB-IoU into the local Ultralytics 8.4.113 YOLO11 BboxLoss
by replacing only the CIoU term. Keep DFL/TAL/cls/data/training protocol intact.
Resolve delta=1 input pixel into the exact coordinate space used by BboxLoss.
Do not train or tune other modules as part of the integration step.
```
