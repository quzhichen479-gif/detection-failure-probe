# FloW-Img PPRD Phase 1

Status: implementation-ready research package; **do not rerun the frozen baseline**.

## Frozen experiment contract

This project is a first engineering-improvement stage on the already-frozen FloW-Img protocol.

- Dataset: FloW-Img Roboflow v2, 2000 images / 5272 bottle instances.
- Split: 1400 train / 200 val / 400 test.
- Split seed / train seed: 79 / 79.
- Baseline: stock YOLO11n, scratch (`pretrained=False`).
- Input: 640.
- Epochs / batch: 300 / 32.
- Optimizer: SGD, `lr0=0.01`, momentum `0.937`, weight decay `0.0005`.
- Checkpoint rule: select `best.pt` by validation mAP50-95; test is touched only after training.
- Existing baseline Test: P 90.58, R 82.19, mAP50 87.20, mAP50-95 45.22.
- Existing scale diagnosis: AP_small 32.61, Recall_small 42.43; small objects are the primary engineering target.

All Phase-1 variants must inherit the exact baseline data manifest, training budget, evaluator, checkpoint rule and seed. The baseline run is a fixed reference and must not be repeated merely to launch this project.

## Research question

Can YOLO11n preserve more small-object evidence during stride-2 spatial compression without paying the cost of full high-resolution processing?

The project deliberately isolates two mechanisms:

1. **polyphase preservation**: retain all 2x2 sampling phases for only part of the channels;
2. **cheap contextual downsampling**: process the remaining channels with a reparameterizable depthwise branch.

No attention, new loss, P2 detection head, augmentation change, training-budget change or confidence-threshold tuning is allowed in Phase 1.

## Implemented modules

`downsample_modules.py` contains:

- `RepContextDown`: RepDown-inspired 1x1 projection + parallel 7x7/3x3 depthwise stride-2 branches. The two depthwise branches can be fused into one 7x7 depthwise convolution for deployment. This is a paper-level engineering implementation, not a claim of byte-identical reproduction of external YOLO-ULM source code.
- `SPDDown`: full `PixelUnshuffle(2)` / space-to-depth followed by 1x1 projection.
- `PartialPolyphaseRepDown` (**PPRD**): a fixed `detail_ratio` of channels uses `SPDDown`; the rest uses `RepContextDown`; outputs are concatenated with no gate/attention so the mechanism remains auditable.

Default PPRD setting for the first experiment: `detail_ratio=0.25`.

## Phase-1 experiment matrix

The baseline is already available as `B0` and is not retrained.

| ID | Change from stock YOLO11n | Purpose |
|---|---|---|
| B0 | existing frozen baseline | fixed reference only |
| D1 | replace **P2/4 -> P3/8** stride-2 Conv with `RepContextDown` | test cheap contextual downsampling alone |
| D2 | replace **P2/4 -> P3/8** stride-2 Conv with `SPDDown` | test full polyphase-preservation upper direction |
| D3 | replace **P2/4 -> P3/8** stride-2 Conv with `PartialPolyphaseRepDown(rho=0.25)` | primary proposed mechanism |
| D4 | D3 plus replace **P3/8 -> P4/16** with PPRD (`rho=0.125` initially) | stage-extension only after D3 is healthy |

Do **not** run D4 before D1-D3 build/train/export checks are clean and D3 has a non-negative validation signal.

## Why the first insertion is P2/4 -> P3/8

In stock YOLO11 the backbone uses stride-2 convolutions at P1/2, P2/4, P3/8, P4/16 and P5/32. The first detector scale is P3/8, so the P2/4 -> P3/8 compression is the last major spatial reduction before features enter the shallow detection pyramid. This makes it the cleanest first location for a small-object information-survival experiment.

## Minimum engineering acceptance checks

Before any full training:

1. model YAML parses and model builds;
2. forward shape is unchanged relative to the replaced stride-2 layer;
3. parameter/GFLOPs summary is recorded;
4. one short train smoke test has finite losses;
5. validation runs;
6. predict runs;
7. TorchScript or ONNX export runs;
8. `RepContextDown.switch_to_deploy()` is numerically equivalent in eval mode within `1e-5` max absolute error.

## Screening metrics

Primary comparison against the existing B0 Test/Val records:

- mAP50-95;
- AP_small;
- Recall_small;
- AP75;
- Params / GFLOPs;
- batch-1 P50 latency under the same local timing harness.

First-stage Go signal (screening, not publication threshold):

- mAP50-95 >= +0.5 pp **and** AP_small >= +1.5 pp, or Recall_small >= +2 pp;
- no material medium/large collapse;
- GFLOPs and P50 preferably within +5% of baseline.

No-go examples:

- only full SPD works while PPRD is consistently neutral/negative;
- small-object gain is bought by clear overall AP collapse;
- latency rises >10% without a meaningful accuracy gain;
- D3 does not beat at least one of the single-mechanism controls D1/D2.

## Ultralytics integration contract

The actual YOLO11 working tree should copy/register the modules rather than importing from this audit repository. For Ultralytics 8.4.113:

- register the three module classes in the normal module export/import path used by `tasks.py`;
- teach `parse_model` to pass `(c1, c2, ...)` to these modules like a standard channel-changing block;
- duplicate the frozen YOLO11n YAML into dedicated research YAMLs; modify only the targeted stride-2 layer(s);
- preserve `nc`, scale `n`, head, loss, augmentations and all training arguments from the existing seed79 baseline.

Suggested semantic replacements in the stock backbone:

- layer `3` (`Conv [256, 3, 2]`, P2/4 -> P3/8) is the first target;
- layer `5` (`Conv [512, 3, 2]`, P3/8 -> P4/16) is used only by D4.

Exact channel values should continue to be resolved by Ultralytics compound scaling; do not hard-code YOLO11n-scaled channels inside the module implementation.
