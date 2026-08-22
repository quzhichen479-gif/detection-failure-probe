# CC-MSCA Phase 4 — M1/M2/M3 training plan

## 1. Experimental policy

This phase is a **pre-registered three-model comparison**. Do not rerun B0 and do not use Test to decide whether M2/M3 should be trained.

All three variants are trained to the full frozen 300-epoch budget unless there is an engineering failure (NaN/Inf, invalid graph, OOM that cannot be solved without changing the frozen batch protocol, or export/build failure before training).

Train all M1-M3 first. Only after all three runs have finished and their configs/commits are frozen may Test be evaluated once for each `best.pt`.

This prevents sequential Test inspection from influencing later variants.

## 2. Frozen B0 protocol

Reuse the existing `flow_pbm_protocol_seed79` exactly.

```text
dataset: FloW-Img Roboflow v2
images / instances: 2000 / 5272
split: 1400 train / 200 val / 400 test
split seed: 79
train seed: 79
model scale: YOLO11n
initialization: scratch
pretrained: false
imgsz: 640
epochs: 300
batch: 32
optimizer: SGD
lr0: 0.01
momentum: 0.937
weight_decay: 0.0005
checkpoint rule: best validation mAP50-95
```

Use the saved B0 arguments/config as the source of truth for every other training option, including augmentation, warmup, workers, AMP, deterministic flags, close-mosaic behavior, NMS/evaluator settings and device policy. Do not reconstruct these from memory if the saved `args.yaml`/artifacts are available.

Frozen B0 Test reference:

```text
P 90.58
R 82.19
mAP50 87.20
mAP50-95 45.22
AP75 39.56
AP_small 32.61
Recall_small 42.43
Params 2.582 M
GFLOPs 6.372
batch-1 P50 5.615 ms
```

## 3. Local implementation target

Port `cc_msca.py` into the current Ultralytics 8.4.113 working tree, preferably as a dedicated module such as:

```text
ultralytics/nn/modules/cc_msca.py
```

Export/register `SegNeXtMSCA` and `ContextContrastMSCA` through the normal modules/tasks path and treat both as channel-preserving in `parse_model`.

Create three separate model YAMLs in the local working tree. Suggested names:

```text
yolo11n-flow-m1-msca-p2.yaml
yolo11n-flow-m2-ccmsca-p2.yaml
yolo11n-flow-m3-ccmsca-p3.yaml
```

Resolve insertion points semantically against the current YOLO11 YAML. Do not assume numeric layer indices have not changed.

## 4. Engineering gate before training

For both module classes and all three model YAMLs verify:

1. unit tests pass;
2. model YAML parses and builds;
3. inserted feature preserves NCHW shape and channel count;
4. all downstream Concat/Detect source indices resolve correctly after insertion;
5. one non-empty-GT and one empty-GT forward/backward are finite in FP32 and AMP;
6. a short train smoke test has finite cls/box/DFL losses;
7. val and predict run;
8. TorchScript and ONNX export succeed or the existing project export contract is met;
9. Params/GFLOPs and batch-1 latency can be measured with the existing harness;
10. for `ContextContrastMSCA`, a standalone default block is exact identity at initialization.

For the identity check, compare the module input/output directly. Do **not** demand bitwise equality between separately constructed stock and modified full models, because adding modules changes RNG consumption and downstream random initialization order.

## 5. M1 — SegNeXt-style MSCA at backbone P2

Run name:

```text
flow_yolo11n_m1_segnext_msca_p2_seed79
```

Change only:

```text
backbone P2 C3k2
-> SegNeXtMSCA(local_kernel=5, strip_kernels=(7,11,21))
-> original P3 stride-2 Conv
```

Purpose: external mechanism control. Test whether conventional multi-scale strip-context gating helps when applied before P3 compression.

Do not add a P2 detection head or any second attention.

## 6. M2 — CC-MSCA at backbone P2

Run name:

```text
flow_yolo11n_m2_ccmsca_p2_seed79
```

Change only:

```text
backbone P2 C3k2
-> ContextContrastMSCA(context_kernels=(3,5,7), reduction=8, gamma_init=0.0)
-> original P3 stride-2 Conv
```

Purpose: main method hypothesis. Estimate structured water background while shallow target geometry is still available, then inject only gated signed local-background residual.

Record after training:

```text
softmax(scale_logits)
mean / median / max abs(gamma)
fraction of channels with positive vs negative gamma
```

These are mechanism diagnostics, not tuning signals.

## 7. M3 — CC-MSCA at backbone P3

Run name:

```text
flow_yolo11n_m3_ccmsca_p3_seed79
```

Change only:

```text
original P3 stride-2 Conv
-> backbone P3 C3k2
-> ContextContrastMSCA(context_kernels=(3,5,7), reduction=8, gamma_init=0.0)
-> original P4 stride-2 Conv
```

Purpose: position control. M2 and M3 use the same module/config; only feature level changes.

Record the same scale/gamma diagnostics as M2.

## 8. Training order and Test isolation

Use this order only for operational convenience:

```text
M1 -> M2 -> M3
```

After each training run, save its final config, git commit, `results.csv`, `best.pt`, `last.pt`, Val metrics and build/latency metadata, but **do not run Test yet**.

After M3 training finishes:

1. verify that M1/M2/M3 configs and code commits are frozen;
2. evaluate each `best.pt` on the frozen 400-image Test exactly once;
3. do not modify kernels, gamma, reduction, location, thresholds, NMS or training settings after seeing Test.

## 9. Required report

Create one comparison report containing B0/M1/M2/M3:

```text
P
R
mAP50
mAP50-95
AP75
AP_small
Recall_small
AP_medium
AP_large
Params
GFLOPs
peak VRAM
training wall time
batch-1 P50
batch-1 P95 if the current harness supports it
best epoch
Val -> Test mAP50-95 delta
```

For M2/M3 also include the final scale weights and gamma diagnostics.

If the existing failure-analysis tooling can produce it without changing thresholds, also report structured-background FP/image. Do not create a new Test-specific taxonomy or threshold after seeing candidate Test predictions.

## 10. Interpretation contract

Discovery-positive signal for M2/M3:

```text
mAP50-95 >= B0 + 0.7 pp
OR AP_small >= B0 + 1.5 pp
OR Recall_small >= B0 + 3.0 pp with mAP50-95 >= B0
```

Important boundaries:

- Precision improvement alone is insufficient if Recall_small falls.
- AP75 improvement alone does not prove localization improvement.
- M1 vs M2 is not a one-factor causal ablation.
- M2 vs M3 is the primary position comparison.
- if all three are neutral/negative on small-object metrics, record the negative result and stop adding MSCA/attention variants.

Publication-stage evidence is stricter than this discovery gate. A pure accuracy claim should preferably approach +1.5--2 pp mAP50-95 or be compensated by strong small-object/FP/efficiency/generalization evidence.

## 11. Forbidden changes

During M1-M3 do not change:

```text
data split or seed
training seed
pretraining state
imgsz / epochs / batch / optimizer / LR / WD
augmentation
loss / TAL / DFL / NMS
P2 detection head
DySample / PPRD / downsampling
frequency / edge branches
partial-channel or dynamic-kernel variants
another attention module
multiple CC-MSCA placements
Test thresholds
```
