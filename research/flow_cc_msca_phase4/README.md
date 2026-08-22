# FloW-Img CC-MSCA Phase 4

Status: **design frozen / implementation-ready**. This package is based on the completed Phase-3 attention results and prepares a task-specific MSCA rewrite for the frozen FloW-Img YOLO11n protocol.

## 1. Why revisit MSCA after Phase 3?

The completed P3 attention controls did not solve the FloW bottleneck:

- EMA: Test mAP50-95 -0.873 pp vs B0;
- CAA: -1.492 pp;
- Triplet: -0.350 pp;
- none recovered the small-object AP/recall deficit.

That result lowers the prior for another generic late P3 reweighting block. It does **not** prove that all contextual modeling is useless. The remaining question is earlier and more specific:

> can shallow local evidence be contrasted against structured water context *before* it is compressed into the final detection feature?

This phase therefore moves the experiment into the backbone and changes the meaning of multi-scale context.

## 2. Source mechanism and collision boundary

SegNeXt MSCA (NeurIPS 2022) uses a local depthwise convolution plus multiple large separable strip-convolution branches, sums the branch responses, applies a pointwise mixing convolution and uses the result as spatial attention.

For FloW, directly interpreting long-range structure as useful context is questionable because waves, glare bands and bank/reflection patterns are themselves long-range structured signals. Recent floating-waste work also makes the generic design space crowded:

- YOLO11-MCN (Sustainability, 2026) already uses a multi-scale contextual attention module plus a P2 head for dynamic water surfaces;
- ESAK-YOLO (Scientific Reports, 2026) uses EMA + selective-kernel attention for floating waste;
- Float-DEIM (Marine Pollution Bulletin, 2026) uses partial efficient multi-scale attention to preserve small-object detail;
- large/separable-kernel attention for water-surface noise suppression has also appeared in recent YOLO variants.

Therefore this project must **not** claim novelty from "multi-scale attention", "dynamic receptive fields", "partial-channel attention", "large strip kernels" or "P2 + MSCA" alone.

The Phase-4 distinction is explicit:

> **multi-scale context aggregation -> multi-scale water-context contrast**.

## 3. Final CC-MSCA definition

Let the input feature be `X` and a trainable local identity-initialized depthwise 3x3 feature be

```text
L = DWConv3x3(X)
```

For each fixed odd context scale `k in {3,5,7}`, estimate directional background with **fixed average filters**, not learnable large kernels:

```text
H_k = AvgPool(1 x k)(L)
V_k = AvgPool(k x 1)(L)
```

Global learned scale logits are normalized once per forward pass:

```text
alpha = softmax(scale_logits)
```

They are scalar mixture weights shared by every image and position. This deliberately avoids image-conditioned selective-kernel behavior.

### Background estimate

```text
B = sum_k alpha_k * 0.5 * (H_k + V_k)
```

### Signed local-background residual

```text
R = L - B
```

`R` preserves contrast polarity. It is the feature actually injected back into the backbone.

### Polarity-invariant contrast

```text
D_h,k = abs(L - H_k)
D_v,k = abs(L - V_k)
C = sum_k alpha_k * 0.5 * (D_h,k + D_v,k)
```

### Directional anisotropy cue

```text
Q = sum_k alpha_k * abs(D_h,k - D_v,k)
```

`Q` is not hard-coded as a suppression mask. It is exposed to the gate as evidence that a response is strongly directional, which is common for elongated wave/reflection structures. A real bottle may also be anisotropic, so the network is allowed to decide rather than subtracting `Q` with a fixed coefficient.

### Gate and residual adapter

```text
A = sigmoid(PW2(SiLU(BN(PW1(concat(L, C, Q))))))
Y = X + Gamma * A * R
```

where `Gamma` is a per-channel learnable parameter initialized to exactly zero.

Consequences:

1. CC-MSCA is an **exact identity at initialization**;
2. constant features produce zero context residual and zero contrast initially;
3. the block can learn signed enhancement or suppression through `Gamma`;
4. long-range context is constrained to be a background reference, rather than a free large-kernel target feature;
5. context filters add no learned large-kernel parameters;
6. only the local depthwise 3x3, global scale mixture, and lightweight pointwise gate are learned.

The local 3x3 kernel is initialized as a depthwise identity kernel. The final gate projection is zero-initialized, so when the residual path first becomes active its gate is spatially neutral (`sigmoid(0)=0.5`).

## 4. Frozen defaults

CC-MSCA first-round defaults are frozen:

```text
context_kernels = (3, 5, 7)
reduction = 8
gamma_init = 0.0
scale_logits_init = equal
local_kernel = 3, identity initialization
context operators = fixed directional average pooling
```

No grid search is allowed in Phase 4. In particular, do not tune kernel sizes, reduction ratio, gamma initialization, scale weights or insertion positions against Val/Test.

## 5. M1-M3 experiment matrix

| ID | Module | Semantic insertion | Purpose |
|---|---|---|---|
| M1 | `SegNeXtMSCA` | backbone P2, immediately after the P2 C3k2 and before P3 stride-2 Conv | external mechanism control: does conventional MSCA help at the early high-resolution stage? |
| M2 | `ContextContrastMSCA` | same backbone P2 location as M1 | main hypothesis: background-reference contrast before P3 compression |
| M3 | `ContextContrastMSCA` | backbone P3, immediately after the backbone P3 C3k2 and before P4 stride-2 Conv | position control: is the contrast operation already too late at stride 8? |

M1 defaults follow the SegNeXt-style structure:

```text
local_kernel = 5
strip_kernels = (7, 11, 21)
```

M2/M3 use the identical CC-MSCA implementation and identical hyperparameters. Only insertion location changes.

### Important causal limitation

M1 vs M2 is **not** a strict one-factor ablation: M1 is a conventional MSCA mechanism control, while M2 deliberately changes context semantics, kernel policy and residual safety. Do not claim that an M2-M1 difference is caused by one isolated subcomponent.

M2 vs M3 is the clean first-round position comparison. If M2 becomes the winning method, later paper-stage ablations may decompose residual adapter / contrast / anisotropy, but those are not part of Phase 4 discovery.

## 6. YOLO11 semantic insertion contract

For stock YOLO11 the backbone semantics are approximately:

```text
P2/4 Conv
-> P2 C3k2          # insert M1 or M2 here
-> stride-2 Conv
-> P3/8 C3k2        # insert M3 here
-> stride-2 Conv
-> P4 ...
```

Resolve the actual local YAML by semantic feature level, not by blindly copying numeric layer indices. Because M1/M2/M3 are inline backbone blocks, downstream layer indices and head skip references must be updated consistently.

All modules are channel-preserving in Ultralytics `parse_model`:

```text
c1 = ch[f]
prepend c1 to YAML args
c2 = c1
```

Suggested YAML arguments after automatic channel injection:

```text
SegNeXtMSCA:       [5, [7, 11, 21]]
ContextContrastMSCA: [[3, 5, 7], 8, 0.0]
```

Register through the normal `ultralytics.nn.modules` export path and `tasks.py` globals. Do not hard-code YOLO11n-scaled channel counts.

## 7. Expected signatures and falsification

The primary project bottleneck remains small-object detection. B0 Test reference is:

```text
P = 90.58%
R = 82.19%
mAP50 = 87.20%
mAP50-95 = 45.22%
AP75 = 39.56%
AP_small = 32.61%
Recall_small = 42.43%
```

A useful CC-MSCA signature is not merely higher Precision. Prefer:

```text
AP_small up
Recall_small up
mAP50-95 non-negative/up
structured-water FP/image non-increasing
```

Interpretation rules:

- `P up, Recall_small down`: suppression became stronger; this does not support the tiny-evidence claim.
- `mAP50 up, mAP50-95 flat`: likely confidence/context discrimination gain, not proven localization improvement.
- `M2 positive, M3 neutral/negative`: supports acting before P3 compression.
- `M3 positive, M2 neutral/negative`: suggests very early contrast is too low-level/noisy; broader stride-8 context is more useful.
- `M1 positive, M2 negative`: conventional context aggregation works better than the background-contrast hypothesis; do not force the new mechanism.
- all M1-M3 neutral/negative: stop the MSCA/attention family and return to assignment/ranking or resolution diagnostics.

Phase-screening signal for M2/M3:

```text
mAP50-95 >= +0.7 pp
OR AP_small >= +1.5 pp
OR Recall_small >= +3.0 pp with overall mAP50-95 non-negative
```

This is a discovery threshold, not a publication claim. A pure accuracy paper still preferably needs about +1.5--2 pp mAP50-95 or a compensating small-object/FP/efficiency advantage under the project quality bar.

## 8. Prohibited Phase-4 changes

Do not add any of the following to M1-M3:

- P2 detection head;
- dynamic/selective kernels;
- partial-channel processing;
- FFT/frequency/edge branch;
- extra loss or assignment modification;
- new augmentation;
- DySample/PPRD/downsampling changes;
- CNAM/EMA/CAA/Triplet or any second attention;
- multi-position CC-MSCA stacking;
- Test-driven threshold or hyperparameter search.

## 9. Files

- `cc_msca.py`: clean PyTorch implementations of `SegNeXtMSCA` and `ContextContrastMSCA`;
- `test_cc_msca.py`: shape, identity, descriptor and gradient tests;
- `TRAINING_PLAN.md`: frozen M1-M3 execution contract;
- `phase4_manifest.yaml`: machine-readable experiment contract;
- `CODEX_TASK.md`: local Ultralytics port-and-train task.

## 10. Primary literature links

- SegNeXt / MSCA, NeurIPS 2022: https://proceedings.neurips.cc/paper_files/paper/2022/hash/08050f40fff41616ccfc3080e60a301a-Abstract-Conference.html
- YOLO11-MCN, Sustainability 2026: https://www.mdpi.com/2071-1050/18/10/5083
- ESAK-YOLO, Scientific Reports 2026: https://www.nature.com/articles/s41598-026-53167-2
- Float-DEIM, Marine Pollution Bulletin 2026: https://www.sciencedirect.com/science/article/pii/S0025326X26003796
