# FloW-Img DySample Phase 2

Status: implementation-ready research package; **do not rerun the frozen B0 baseline**.

## Why Phase 2 changes direction

The frozen FloW-Img seed79 YOLO11n baseline has Test mAP50-95 `0.4522`, AP_small `0.3261`, Recall_small `0.4243`, and AP75 `0.3956`. Phase-1 downsampling variants reduced compute slightly but did not improve the small-object metrics. Phase 2 therefore stops changing stride-2 compression and tests the orthogonal question:

> Can content-aware neck upsampling recover a more useful high-resolution feature map for the P3 small-object branch than fixed nearest-neighbor interpolation?

Only the upsampling operator changes. Backbone downsampling, neck topology, heads, losses, augmentation, training budget, split, checkpoint selection, and evaluation protocol remain frozen.

## Source and implementation boundary

`dy_sample.py` implements **DySample** from:

- Wenze Liu, Hao Lu, Hongtao Fu, Zhiguo Cao, *Learning to Upsample by Learning to Sample*, ICCV 2023.
- Official MIT-licensed code: <https://github.com/tiny-smart/dysample>

The official point-sampling formulation is preserved. This repository copy only adds explicit validation, type hints, and explicit PyTorch `meshgrid` indexing. It is not presented as a new module.

The Phase-2 default is intentionally conservative:

```text
scale=2
style="lp"
groups=4
dyscope=False
```

No custom CUDA extension is required; the core operator uses standard `pixel_shuffle` and `grid_sample`.

## What DySample changes

Stock YOLO11 uses fixed nearest-neighbor interpolation in the top-down neck. DySample instead predicts small content-dependent sampling offsets and samples the low-resolution feature map on a learned sub-pixel grid. Channel count is preserved while spatial resolution is doubled.

For zero learned residual offset, the initialized regular grid is numerically equivalent to ordinary bilinear upsampling (`align_corners=False`); learning then perturbs sampling locations around that regular grid.

## Phase-2 experiment matrix

B0 is the already-saved baseline and must not be retrained.

| ID | Change from stock YOLO11n | Purpose |
|---|---|---|
| B0 | Existing frozen baseline | Reference only |
| U1 | Replace only **P4/16 -> P3/8** neck upsample with DySample | Primary small-object test |
| U2 | Replace only **P5/32 -> P4/16** neck upsample with DySample | Insertion-location control |
| U3 | Replace both neck x2 upsample layers with DySample | Combined test |

Run U1 first. U2/U3 are controls/extension; do not combine DySample with PPRD, attention, P2, a new loss, or augmentation changes during this screen.

## Ultralytics 8.4.113 integration contract

Port `DySample` into the actual YOLO11 working tree rather than importing this research directory at runtime.

1. Export/import `DySample` through the normal `ultralytics.nn.modules` path used by `tasks.py`.
2. In `parse_model`, treat DySample as a **channel-preserving** module:
   - infer `c1 = ch[f]`;
   - prepend `c1` to the YAML arguments passed to the constructor;
   - set `c2 = c1`.
3. Replace a stock YAML row of the form:

   ```yaml
   - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
   ```

   with:

   ```yaml
   - [-1, 1, DySample, [2, "lp", 4, false]]
   ```

4. Keep all Concat sources and downstream C3k2/Detect structure unchanged.
5. Let channels be inferred from the actual graph; do not hard-code YOLO11n-scaled channel counts in `DySample`.

In stock YOLO11 the first neck upsample is P5/32 -> P4/16 and the second is P4/16 -> P3/8. U1 targets the second one because it directly feeds the highest-resolution P3 detection branch after concatenation.

## Minimum engineering acceptance checks

Before full training:

1. module unit tests pass;
2. model YAML parses and model builds;
3. output channels remain unchanged and H/W exactly double;
4. forward/backward values are finite;
5. one short training smoke test has finite losses;
6. validation runs;
7. predict runs;
8. TorchScript or ONNX export runs;
9. Params/GFLOPs and batch-1 P50 are recorded with the existing local harness.

The standalone tests also verify that zeroing the offset predictor makes the LP implementation match regular bilinear interpolation within `1e-6`.

## Frozen experiment contract

Reuse the existing `flow_pbm_protocol_seed79` exactly:

- FloW-Img Roboflow v2: 2000 images / 5272 instances;
- 1400 Train / 200 Val / 400 Test;
- split seed = train seed = 79;
- YOLO11n scratch, 640, 300 epochs, batch 32;
- SGD, `lr0=0.01`, momentum `0.937`, weight decay `0.0005`;
- select `best.pt` only by Validation mAP50-95;
- touch Test only after each model has completed training.

See `phase2_manifest.yaml` for the machine-readable contract.

## Screening decision

Primary metrics are mAP50-95, AP_small, Recall_small, AP75 and batch-1 P50 latency.

A useful first signal is either:

- mAP50-95 at least `+0.5 pp` and AP_small at least `+1.0 pp` versus B0; or
- Recall_small at least `+2.0 pp` with non-negative overall mAP50-95.

This is a screening rule, not a publication claim. A single seed79 win is evidence to continue, not evidence of statistical significance.
