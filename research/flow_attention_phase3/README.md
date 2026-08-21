# FloW-Img Attention Phase 3

Status: implementation-ready research package. The frozen B0 baseline already exists and must not be retrained.

## Goal

Phase 1 (downsampling) and Phase 2 (DySample upsampling) did not improve the primary FloW-Img bottleneck: small-object AP/recall. Phase 3 therefore tests an orthogonal hypothesis:

> useful tiny-object evidence may already exist in the final fused P3 feature, but its saliency is poorly allocated against water texture/reflection background.

This phase does not modify spatial sampling. It changes only a single attention side branch feeding the P3 input of Detect.

## Frozen protocol

Reuse `flow_pbm_protocol_seed79` exactly:

- FloW-Img Roboflow v2: 2000 images / 5272 instances;
- 1400 train / 200 val / 400 test;
- split seed = train seed = 79;
- stock YOLO11n, scratch (`pretrained=False`);
- 640 input, 300 epochs, batch 32;
- SGD, `lr0=0.01`, momentum `0.937`, weight decay `0.0005`;
- choose `best.pt` only by validation mAP50-95;
- touch Test only after each training run completes.

Existing B0 Test reference:

- P 0.905838;
- R 0.821902;
- mAP50 0.872033;
- mAP50-95 0.452200;
- AP75 0.395600;
- AP_small 0.326085;
- Recall_small 0.424346.

Do not rerun B0 merely to launch this phase.

## Modules

### A1 — EMAAttention

Efficient Multi-Scale Attention (ICASSP 2023) implemented as a shape-preserving block. It groups channels, combines directional pooling with local 3x3 context, then forms cross-spatial weights.

Default: `factor=8`.

Purpose: cheap general-purpose attention sanity check.

### A2 — CAAAttention

Context Anchor Attention from the PKINet line (CVPR 2024), implemented with local average context, 1x1 mixing, and depthwise horizontal/vertical strip convolutions.

Default: `kernel_size=11`.

Purpose: test whether long-range water/background context helps suppress distractors while preserving P3 tiny-object evidence.

### A3 — TripletAttention

Triplet Attention (WACV 2021) with cross-dimension C-H, C-W and H-W interactions using three lightweight attention gates.

Default: spatial branch enabled, `kernel_size=7`.

Purpose: very small, mature attention control. If heavier/contextual modules cannot beat this control, there is little reason to design a more complicated attention block.

## Important insertion design: Detect-only P3 side branch

Do **not** simply insert attention inline after the final P3 C3k2 with `from=-1` for every following layer.

In stock YOLO11, the final fused P3 output is used twice:

1. as the P3 input to Detect;
2. as the source of the following stride-2 Conv that constructs the bottom-up P4/P5 PAN path.

An inline attention block would therefore alter not only P3 detection but also the later P4/P5 features, confounding the experiment.

Instead create a side branch:

```text
final fused P3 C3k2 (original)
       |-------------------------------> original stride-2 Conv -> P4/P5 PAN (unchanged)
       |
       +-> Attention -> Detect P3 input only
```

This makes the causal question clean: does attention on the small-object detection-scale feature help by itself?

### Stock YOLO11 semantic location

In the standard YOLO11 YAML, the final top-down P3 C3k2 is the layer commonly indexed `16`, followed by the bottom-up stride-2 Conv commonly indexed `17`, and Detect consumes `[16, 19, 22]`.

After inserting one attention side-branch row, use the semantic equivalent of:

```yaml
# existing final fused P3
- [-1, 2, C3k2, [256, false]]       # old 16

# new P3 Detect-only branch
- [16, 1, EMAAttention, [8]]         # A1; use CAA/Triplet for A2/A3

# preserve bottom-up PAN from the original P3, NOT the attention output
- [16, 1, Conv, [256, 3, 2]]
- [[-1, 13], 1, Concat, [1]]
- [-1, 2, C3k2, [512, false]]
- [-1, 1, Conv, [512, 3, 2]]
- [[-1, 10], 1, Concat, [1]]
- [-1, 2, C3k2, [1024, true]]

# Detect uses attended P3 plus otherwise-normal P4/P5
- [[17, 20, 23], 1, Detect, [nc]]
```

The numeric indices above are documentation for the stock layout; Codex must resolve them against the actual local YOLO11 YAML rather than blindly assuming no local changes. The invariant is semantic: **attention is only on the P3 branch presented to Detect; P4/P5 construction starts from the untouched original fused P3.**

## Ultralytics 8.4.113 integration

Copy the three classes into the actual Ultralytics working tree and register them through the normal `ultralytics.nn.modules` import/export path used by `tasks.py`.

Treat all three as channel-preserving modules in `parse_model`:

- `c1 = ch[f]`;
- prepend `c1` to YAML args;
- `c2 = c1`.

Suggested YAML args:

```yaml
EMAAttention:     [8]
CAAAttention:     [11]
TripletAttention: [false, 7]
```

Do not hard-code YOLO11n-scaled channel counts inside the modules.

## Experiment matrix

| ID | Change | Purpose |
|---|---|---|
| B0 | existing frozen YOLO11n | reference only, no rerun |
| A1 | EMA on Detect-only P3 side branch | cheapest general attention test |
| A2 | CAA on Detect-only P3 side branch | long-range context/background suppression |
| A3 | Triplet on Detect-only P3 side branch | minimal cross-dimension control |

Do not combine A1/A2/A3 in Phase 3.

## Engineering acceptance before full training

For each module:

1. unit tests pass;
2. YOLO YAML parses and model builds;
3. attention output shape exactly matches input P3 shape;
4. the original P3 tensor still feeds the bottom-up PAN path;
5. only Detect's P3 source uses the attention output;
6. forward/backward values are finite;
7. short training smoke test has finite losses;
8. val and predict run;
9. TorchScript or ONNX export runs;
10. Params, GFLOPs and batch-1 P50 are recorded with the existing harness.

## Screening

Primary metrics: mAP50-95, AP_small, Recall_small, AP75, Params, GFLOPs and batch-1 P50.

Useful first signal:

- mAP50-95 >= +0.5 pp and AP_small >= +1.0 pp versus B0; or
- Recall_small >= +2.0 pp while overall mAP50-95 is non-negative.

If all three variants are neutral/negative on small-object metrics, stop attention-module searching and move to assignment/head supervision diagnostics rather than adding more attention blocks.

## Source boundary

These files are clean PyTorch research implementations following the published module structures, not claims of byte-identical reproduction of third-party repositories. Final paper attribution should cite the original EMA (ICASSP 2023), PKINet/CAA (CVPR 2024), and Triplet Attention (WACV 2021) papers.
