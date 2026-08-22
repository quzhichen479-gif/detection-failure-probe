# FloW-Img Large-Kernel Context Phase 5

Status: **design frozen / implementation-ready**.

Repository target: `https://github.com/quzhichen479-gif/detection-failure-probe`  
Branch: `research/flow-large-kernel-phase5`

This phase follows the completed P3/downsampling/upsampling/attention/MSCA experiments. The accumulated negative results lower the prior for another generic attention or feature-fusion block. Phase 5 isolates one still-open question:

> does large spatial context help FloW-Img tiny-object classification when it is added only to the P3 classification path, without contaminating box regression?

The three models are intentionally independent causal probes. They must not be combined in the first round.

## 1. Frozen project facts

Baseline protocol stays unchanged:

```text
FloW-Img Roboflow v2
1400 train / 200 val / 400 test
split seed = 79
train seed = 79
YOLO11n scratch
imgsz = 640
epochs = 300
batch = 32
optimizer = SGD
lr0 = 0.01
momentum = 0.937
weight_decay = 0.0005
best checkpoint selected by Val mAP50-95
Test used only after training/selection is complete
```

Frozen B0 Test reference:

```text
P = 90.58%
R = 82.19%
mAP50 = 87.20%
mAP50-95 = 45.22%
AP75 = 39.56%
AP_small = 32.61%
Recall_small = 42.43%
Params = 2.582 M
GFLOPs = 6.372
```

The project bottleneck remains small-object recovery. Previous modules repeatedly changed AP75 or precision without reliably improving AP_small / Recall_small. Therefore Phase 5 does not place large-kernel context in the shared neck feature and does not touch regression.

## 2. YOLO11 insertion decision

Stock YOLO11 Detect uses separate regression (`cv2`) and classification (`cv3`) towers for P3/P4/P5. Phase 5 inserts context **only before `cv3[0]`**, i.e. the P3/8 classification tower input.

```text
P3 feature -----------------------> cv2[0] -> box regression   (unchanged)
   |
   +-> Phase-5 context -> cv3[0] -> classification            (changed)

P4 feature -> cv2[1] / cv3[1]                                    unchanged
P5 feature -> cv2[2] / cv3[2]                                    unchanged
```

This placement is preferred over an inline neck block because:

1. large context can help decide `bottle vs reflection/wave/glitter`;
2. tiny-object box localization should retain the original local P3 feature;
3. P4/P5 are not the primary target and remain strict controls;
4. the final Detect output format, TAL/DFL/loss/NMS and downstream graph remain unchanged.

The reference helper `ContextBeforeTower` wraps only `Detect.cv3[0]`. The local Ultralytics port should implement a small `FlowLargeKernelDetect(Detect)` subclass that calls `super().__init__()` and then performs this wrapper. No `Detect.forward()` copy is required.

## 3. M1 — UniRepLK large-kernel control

Class: `UniRepLKControl`

Purpose: test whether a generic large receptive field is useful at all.

The core is a depthwise `17x17` branch plus the UniRepLKNet-style dilated training branches:

```text
(5, d=1)
(9, d=2)
(3, d=4)
(3, d=5)
(3, d=7)
```

All branches are summed during training. Before export they are exactly merged into one static depthwise `17x17` convolution.

Adapter:

```text
C = DilatedReparamDW17(X)
Y = X + gamma * PWConv(SiLU(C))
```

`gamma` is per-channel and initialized to zero, so the inserted block is an exact identity at initialization.

Interpretation:

- M1 positive: large receptive field itself has useful information;
- M1 neutral/negative but M2/M3 positive: generic large context is insufficient; task-structured context matters.

Primary reference:

- UniRepLKNet, CVPR 2024: https://openaccess.thecvf.com/content/CVPR2024/html/Ding_UniRepLKNet_A_Universal_Perception_Large-Kernel_ConvNet_for_Audio_Video_Point_CVPR_2024_paper.html
- official code: https://github.com/AILab-CVC/UniRepLKNet (Apache-2.0)

The implementation in this package is an independent minimal reimplementation of the dilated-reparameterization idea for the project contract; it does not copy the upstream source file.

## 4. M2 — Strip-LKC

Class: `StripLKC`

Purpose: test whether **directional water context** is more useful than isotropic large context.

FloW background is not random empty space. Wave bands, reflected highlights and shore/water structures can remain coherent over long horizontal/vertical spans. The module therefore estimates two directional contexts with trainable large depthwise strip kernels:

```text
L = DWConv3x3(X)
H = DWConv1x17(L)
V = DWConv17x1(L)
D_h = L - H
D_v = L - V
Y = X + gamma * PWConv(concat(L, D_h, D_v))
```

The strip kernels are initialized as directional averages but remain trainable. There is no sigmoid attention mask, selective-kernel routing or image-conditioned receptive-field choice.

This is deliberately different from the completed CC-MSCA route: Phase 4 used multi-scale fixed pooling/background descriptors in the backbone; Phase 5 uses one large learned directional scale, only inside the P3 classification path.

Primary reference:

- Strip R-CNN, AAAI 2026: https://ojs.aaai.org/index.php/AAAI/article/view/38217
- official code: https://github.com/HVision-NKU/Strip-R-CNN

The official Strip R-CNN repository is CC BY-NC 4.0. This package does not copy its implementation; it independently implements the project-specific directional-context idea from the published mechanism.

## 5. M3 — CE-PConv-LKC

Class: `CEPConvLKC`

Purpose: test whether useful context comes specifically from **separating the tiny target core from surrounding water**.

The peripheral branch is inspired by PeLK's parameter-sharing idea, but the project version adds a strict center exclusion:

```text
large kernel = 17x17
center exclusion = 5x5
```

All positions inside the center `5x5` are structurally zero in the peripheral kernel. Positions outside the center share depthwise coefficients according to logarithmic eccentricity groups: nearby peripheral positions are represented more finely, farther positions share more strongly.

```text
L = DWConv3x3(X)
P = CenterExcludedPeripheralConv17(L)
D = L - P
Y = X + gamma * PWConv(concat(L, P, D))
```

This makes the large branch a surrounding-context operator rather than another copy of the target-core feature. The learned shared kernel is materialized to one static depthwise `17x17` convolution before export.

Primary reference:

- PeLK, CVPR 2024: https://openaccess.thecvf.com/content/CVPR2024/html/Chen_PeLK_Parameter-efficient_Large_Kernel_ConvNets_with_Peripheral_Convolution_CVPR_2024_paper.html

No official PeLK codebase was used for this project implementation. CE-PConv-LKC is a paper-inspired, independently specified project adaptation rather than a reproduction claim.

## 6. Frozen defaults

No first-round grid search:

```text
M1 UniRepLKControl:
  kernel_size = 17
  gamma_init = 0.0

M2 StripLKC:
  local_kernel = 3
  strip_kernel = 17
  gamma_init = 0.0

M3 CEPConvLKC:
  local_kernel = 3
  peripheral_kernel = 17
  center_exclusion = 5
  peripheral sharing = logarithmic eccentricity bins
  gamma_init = 0.0
```

Why `17`:

P3 stride is 8, so a 17-cell span corresponds to roughly 136 input pixels. That is already many times larger than a typical FloW tiny bottle after 640 resize and is sufficient to test long-range context without jumping immediately to 31/51/101 kernels.

## 7. First-round experiment matrix

| ID | Module | Insertion | Question |
|---|---|---|---|
| M1 | `UniRepLKControl` | before `Detect.cv3[0]` | does generic large RF help? |
| M2 | `StripLKC` | before `Detect.cv3[0]` | does directional water context help? |
| M3 | `CEPConvLKC` | before `Detect.cv3[0]` | does center/surround separation help? |

All three retain stock:

- backbone and neck;
- P3 box tower;
- all P4/P5 towers;
- TAL / DFL / box loss / classification loss;
- augmentation;
- NMS/post-processing;
- 640 input and seed79 training protocol.

## 8. Expected metric signatures

Primary metrics:

```text
mAP50-95
AP_small
Recall_small
FP/image on structured-water negatives
P50/P95 batch-1 latency
```

Interpretation:

- `AP75 up but AP_small/Recall_small down`: repeat of earlier failure mode; do not call success.
- `P up but Recall_small down`: stronger suppression, not tiny-object recovery.
- `M1 > B0`: generic large RF remains viable.
- `M1 ~= B0, M2 > B0`: directional structure is the useful part.
- `M1 ~= B0, M3 > B0`: center/surround separation is the useful part.
- all <= B0: close the large-kernel family and return to assignment/ranking/high-resolution evidence tests.

Discovery promotion rule:

```text
Val mAP50-95 >= +0.7 pp
OR AP_small >= +1.5 pp
OR Recall_small >= +3.0 pp with overall mAP50-95 non-negative
```

A publication claim still needs stronger evidence than a single seed79 discovery run.

## 9. Deployment contract

- M1: call `switch_to_deploy()` to fuse all dilated branches into one static `17x17` depthwise conv.
- M2: already consists only of standard static convolutions.
- M3: call `switch_to_deploy()` to materialize shared peripheral weights into one static `17x17` depthwise conv.
- use `switch_phase5_to_deploy(model)` on a deep-copied trained model before ONNX/TensorRT export;
- verify pre/post materialization numerical parity before export;
- do not compare FLOPs only; report real batch-1 P50/P95 latency.

## 10. Prohibited first-round changes

Do not combine M1/M2/M3. Do not add:

- GPRA/RF-TAL/RS or any loss/assignment change;
- P2 detection head;
- higher training resolution;
- new augmentation;
- EMA/CAA/Triplet/MSCA/LSK attention;
- frequency/FFT/edge branch;
- DySample/downsampling replacement;
- multi-position stacking;
- Test-driven threshold/kernel/gamma tuning.

## 11. Files

- `large_kernel_context.py`: the three Phase-5 modules plus reparameterization operators;
- `yolo11_integration.py`: P3 classification-only wrapper, variant factory and deploy materialization helper;
- `test_large_kernel_context.py`: local PyTorch tests;
- `TRAINING_PLAN.md`: frozen execution contract;
- `phase5_manifest.yaml`: machine-readable experiment contract;
- `CODEX_TASK.md`: exact local Ultralytics port + training task.
