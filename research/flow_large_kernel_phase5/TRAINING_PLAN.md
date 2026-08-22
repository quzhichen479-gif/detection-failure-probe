# FloW-Img Phase 5 — Large-Kernel Context Training Plan

## Goal

Run exactly three independent YOLO11n@640 scratch experiments on the frozen FloW-Img seed79 protocol:

1. `M1_unireplk_p3cls`
2. `M2_striplkc_p3cls`
3. `M3_cepconv_p3cls`

B0 is the existing frozen stock YOLO11n run and must not be rerun for model selection.

## Integration invariant

Only the P3 classification input is changed:

```text
stock P3 -> cv2[0]                          # untouched regression
stock P3 -> context_variant -> cv3[0]       # Phase-5 modification
P4/P5 -> stock cv2/cv3                      # untouched
```

The safest implementation is a `FlowLargeKernelDetect(Detect)` subclass that calls stock `Detect.__init__()` and wraps `self.cv3[0]` with `ContextBeforeTower`. Do not duplicate `Detect.forward()`.

## Frozen training protocol

```text
model scale: YOLO11n
initialization: scratch / pretrained=False
imgsz: 640
epochs: 300
batch: 32
optimizer: SGD
lr0: 0.01
momentum: 0.937
weight_decay: 0.0005
split seed: 79
train seed: 79
checkpoint selection: Val mAP50-95 best.pt
formal Test: once after training/selection
```

Keep all other stock/baseline hyperparameters exactly matched to the existing B0 run.

## Frozen module defaults

### M1

```text
UniRepLKControl
kernel=17
gamma_init=0
train branches=(5,d1),(9,d2),(3,d4),(3,d5),(3,d7)
```

### M2

```text
StripLKC
local=3
horizontal=1x17
vertical=17x1
gamma_init=0
```

### M3

```text
CEPConvLKC
local=3
peripheral=17
center exclusion=5
log-eccentricity parameter sharing
gamma_init=0
```

No kernel/gamma/position search.

## Pre-training acceptance

For each YAML:

1. build succeeds under the frozen Ultralytics 8.4.113 runtime;
2. parameter count is explainable;
3. `(1,3,640,640)` forward works;
4. one batch with GT and one empty-GT batch run forward/backward under FP32 and AMP;
5. at gamma=0, a weight-copied custom head matches stock Detect numerically because the context adapter is an identity;
6. P3 regression outputs are identical to stock for the same weights/input;
7. P4/P5 cls/reg outputs are identical to stock;
8. export smoke test succeeds after deploy materialization;
9. PyTorch pre/post materialization parity is within a documented numerical tolerance.

Failing build/gradient/parity/export is a No-Go and must be fixed before training.

## 75-epoch screen

The first 75 epochs are the beginning of the same 300-epoch run, not a separate hyperparameter trial. Continue only when there is no clear collapse and at least one task signature appears.

Preferred screen:

```text
Val mAP50-95 >= B0 same-epoch reference - 0.3 pp
AND no AP_small collapse > 1.5 pp
AND no Recall_small collapse > 2.0 pp
```

Positive context signal:

```text
AP_small +>=1.0 pp
OR Recall_small +>=2.0 pp
OR structured-water FP/image decreases >=10% with overall mAP non-negative
```

If the model clearly fails these signatures, stop and record it; do not tune kernel size on Val.

## 300-epoch promotion

Promote as a serious candidate when:

```text
Val mAP50-95 >= +0.7 pp
OR AP_small >= +1.5 pp
OR Recall_small >= +3.0 pp with mAP50-95 non-negative
```

For a method paper, approximately +1.5--2.0 pp mAP50-95 remains the preferable pure-accuracy target unless compensated by strong small-object, FP, latency or cross-dataset evidence.

## Mandatory reporting

For B0/M1/M2/M3 report:

```text
P, R, mAP50, AP75, mAP50-95
AP_small, Recall_small, AP_medium, AP_large
structured-water FP/image if the taxonomy pipeline is available
Params, GFLOPs, checkpoint size
batch-1 P50/P95 latency
peak VRAM
training wall time
```

M1/M3 additionally report train-form vs deploy-materialized parity and deployed latency.

## Interpretation boundary

These are three different causal probes, not a strict component ablation:

- M1 asks whether generic large RF helps;
- M2 asks whether directional long-range water context helps;
- M3 asks whether center-excluded surrounding context helps.

Do not infer that M2-M1 or M3-M1 isolates a single mathematical term.

## Stop rule

If all three are neutral/negative on overall and small-object metrics, close the large-kernel family. Do not respond by stacking them together or tuning larger kernels. Return to the independent assignment/ranking/resolution hypotheses.
