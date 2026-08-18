# SQA-Inner-EIoU: L1-L4 Loss Experiment Track

This directory contains the implementation and preregistered experiment plan for a focused YOLO11n bounding-box loss study.

The track was created after SRB-IoU showed a clear negative result on the current PoTATO protocol. The purpose here is deliberately narrower: test whether a mature EIoU/Inner-IoU family can improve coarse-to-fine bounding-box regression without introducing resolution smoothing, NWD fusion, extra attention, or changes to assignment.

## 1. Experiments

| ID | Loss | Fixed definition | Question |
|---|---|---|---|
| L1 | EIoU | `1 - IoU + Dc + Dw + Dh` | Is explicit center/width/height geometry useful by itself? |
| L2 | Inner-EIoU broad | Inner ratio `r=1.2` | Does an enlarged auxiliary overlap basin help coarse / low-quality regression? |
| L3 | Inner-EIoU fine | Inner ratio `r=0.8` | Does a contracted auxiliary overlap region improve high-quality / fine regression? |
| L4 | SQA-Inner-EIoU | detached smooth quality controller, `eta=0.20` | Can one quality-adaptive geometry unify the broad and fine regimes? |

The frozen YOLO11n baseline is **not** L0 in this directory because its existing result must be reused rather than retrained.

## 2. Core geometry

For aligned prediction `B` and ground truth `G`, EIoU is implemented as

```math
L_{EIoU} = 1-IoU(B,G) + D_c + D_w + D_h
```

with

```math
D_c = \frac{(c_x-c_x^g)^2+(c_y-c_y^g)^2}{c_w^2+c_h^2+\epsilon},
```

```math
D_w = \frac{(w-w_g)^2}{c_w^2+\epsilon},
\qquad
D_h = \frac{(h-h_g)^2}{c_h^2+\epsilon},
```

where `cw,ch` are the width and height of the smallest enclosing box of prediction and target.

### Inner-EIoU

For an auxiliary ratio `r`, both boxes are resized about their own centers:

```math
B^r=(c_x,c_y,rw,rh),
\qquad
G^r=(c_x^g,c_y^g,rw_g,rh_g).
```

The loss becomes

```math
L_{Inner-EIoU}=1-IoU(B^r,G^r)+D_c+D_w+D_h.
```

Only the overlap term uses the auxiliary boxes. The EIoU center/width/height penalties are computed from the original prediction and target boxes.

## 3. L4: Smooth Quality-Adaptive Inner-EIoU

L4 uses the current standard IoU only as a detached control state:

```math
q = \operatorname{stopgrad}(IoU(B,G)).
```

The smoothstep controller is

```math
g(q)=3q^2-2q^3,
```

and the auxiliary ratio is

```math
r(q)=1+\eta[1-2g(q)].
```

The preregistered value is

```math
\eta=0.20,
```

so

```text
q=0.0 -> r=1.20
q=0.5 -> r=1.00
q=1.0 -> r=0.80
```

The final L4 objective is

```math
L_{SQA}=1-IoU(B^{r(q)},G^{r(q)})+D_c+D_w+D_h.
```

### Why detach quality?

The quality controller should select the local regression geometry for the current optimization step. It should not add a second gradient path

```text
box -> IoU -> ratio -> Inner-IoU.
```

Therefore the implementation explicitly detaches `q` before calculating `r`.

### Why smoothstep instead of a linear ratio?

The controller has zero slope at `q=0` and `q=1`, reducing ratio jitter at the lowest- and highest-quality endpoints while allowing the largest transition in the middle-quality regime. This is the only custom controller in the first formal experiment.

## 4. Files

```text
research_tracks/sqa_inner_eiou/
├── README.md
├── PROJECT_IMPLEMENTATION.md
├── CODEX_TASK.md
├── experiment_plan.yaml
├── losses.py
├── ultralytics_adapter.py
└── test_reference.py
```

`losses.py` contains all four implementations behind the same API. `loss_by_id("L1"..."L4")` is intended to keep the local YOLO patch identical across experiments.

## 5. Hard experiment boundaries

The first comparison changes **only** the IoU-style bbox regression term.

Do not change:

- YOLO11n architecture;
- dataset split;
- image size;
- pretrained initialization policy;
- optimizer or LR schedule;
- batch size / epochs / augmentations;
- TaskAlignedAssigner / TAL;
- positive assignments;
- DFL formulation or weight;
- classification loss;
- evaluator;
- frozen baseline result.

Do not add:

- NWD;
- WIoU/Focaler/Focal-EIoU weighting;
- Shape-IoU;
- another dynamic sample weight;
- DBRA during the initial loss-only comparison.

The four runs must answer one controlled question. Extra mechanisms destroy attribution.

## 6. Formal run order

Run exactly:

```text
L1  EIoU
L2  Inner-EIoU r=1.2
L3  Inner-EIoU r=0.8
L4  SQA-Inner-EIoU eta=0.20
```

L2 and L3 are **mechanism controls**, not two points in a hyperparameter search. Do not use their validation result to change the L4 formula before L4 is run.

Use the existing frozen baseline result as the external reference.

## 7. What each result means

### If L1 improves

EIoU's explicit center/width/height geometry is already beneficial. L2-L4 then test whether auxiliary overlap geometry adds further value.

### If L2 improves more than L1

There is evidence that a broadened overlap basin helps the current regression regime.

### If L3 improves more than L1

There is evidence that stronger fine-overlap sensitivity helps high-quality localization.

### If L2 and L3 show complementary strengths

This is the strongest mechanism justification for L4. A quality-adaptive ratio has a plausible reason to combine the two regimes in one training objective.

### If L4 exceeds both L2 and L3

The intended claim becomes credible: fixed Inner ratios specialize to one regime, whereas the detached smooth controller adapts coarse-to-fine geometry per positive candidate.

### If L4 does not exceed the best fixed-ratio control

Do not claim the controller helps. Keep the best mature loss as an engineering option and stop the custom-controller branch unless additional diagnostics provide a concrete mechanism.

## 8. Metrics

Primary:

```text
mAP50-95
```

Secondary:

```text
APs, ARs, AP75, APm, ARm
```

For L4 additionally log detached ratio statistics:

```text
mean / p10 / p50 / p90 of r(q)
```

These diagnostics must not be added to the objective.

Because the current task is dominated by small objects, an AP increase that is accompanied by a material collapse in APs/ARs should not be treated as a clean success.

## 9. Gate before DBRA combination

Do not immediately run `DBRA + L1/L2/L3/L4`.

Only consider the best loss with DBRA after the loss-alone experiment is credible on frozen validation:

- no material mAP50-95 degradation versus the frozen baseline;
- no material APs degradation;
- no numerical instability;
- preferably evidence that the selected loss improves the localization metric it was designed to address.

This prevents repeating the SRB outcome where an unvalidated loss erased an otherwise positive DBRA signal.

## 10. Reference tests

Inside the actual PyTorch environment run:

```bash
python research_tracks/sqa_inner_eiou/test_reference.py
```

The tests cover:

- exact-match zero loss;
- L1 dispatch equivalence;
- fixed L2/L3 ratios;
- SQA endpoint and midpoint ratios;
- detached controller;
- monotonic quality-to-ratio mapping;
- broad-basin behavior at low IoU;
- fine-overlap behavior at high IoU;
- finite gradients for tiny and elongated boxes;
- autograd gradcheck in a smooth overlap region.

Passing these tests verifies implementation consistency, not detector performance.

## 11. Implementation principle

The preferred local YOLO integration is a dedicated loss selector at the existing `BboxLoss` IoU term. Do not globally replace generic IoU helpers because that can silently affect assignment, metrics, NMS utilities, or unrelated research tracks.

See `PROJECT_IMPLEMENTATION.md` for the exact integration contract.
