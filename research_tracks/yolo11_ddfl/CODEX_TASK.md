# CODEX TASK — integrate YOLO11-DDFL v1 into Ultralytics 8.4.113

Implement the frozen YOLO11-DDFL track in the actual YOLO11 / Ultralytics 8.4.113 worktree.

Read first:

1. `research_tracks/yolo11_ddfl/README.md`
2. `research_tracks/yolo11_ddfl/yolo11_ddfl.py`
3. `research_tracks/yolo11_ddfl/test_reference.py`

Do not redesign the support or mix in other losses.

## Implementation requirements

1. Add reusable non-uniform DFL integral + loss code based on `yolo11_ddfl.py`.
2. Provide three modes:
   - `standard16` = untouched native baseline;
   - `uniform20` = 20 uniformly spaced support values over `[0,15]`;
   - `ddfl20` = frozen support `[0,.25,.5,.75,1,1.5,2,3,...,15]`.
3. For `uniform20` / `ddfl20`, keep the native YOLO11 regression-tower hidden width based on base `reg_max=16`; only the final box-output conv changes `64 -> 80` channels.
4. Use the same support for:
   - training DFL target interpolation;
   - training bbox decode used by TAL/loss;
   - inference `Detect` DFL integral;
   - export.
5. Keep native TAL, CIoU, BCE, loss gains, backbone, neck, augmentation and NMS unchanged.
6. Load the same pretrained YOLO11 checkpoint. Verify that only the final box-output convs fail shape transfer for 20-bin modes; all earlier regression-tower weights must transfer.
7. Do not add Signed DFL or D-TAL in this track.

## Mandatory tests before training

Run the repository reference tests first, then add integration tests for:

- standard16 native-loss parity;
- standard16 native-decode parity;
- exact-support and between-support interpolation;
- D1/D2 FP32 + AMP forward/backward;
- train-side and inference-side decode equivalence;
- expected pretrained transfer keys only;
- output tensor shapes for train/val/predict;
- TorchScript export smoke;
- ONNX export smoke.

If any parity or support-consistency test fails, do not train.

## Training cells

Reuse the existing frozen IWHR baseline recipe exactly. Generate only these three commands:

- D0: `dfl_mode=standard16`
- D1: `dfl_mode=uniform20`
- D2: `dfl_mode=ddfl20`

Do not tune support values in the first round.

## Deliverables

Return:

- exact source diff;
- touched files;
- test output;
- pretrained transfer summary listing missing/mismatched keys;
- parameter/FLOPs delta D0 vs D1/D2;
- three runnable training commands;
- no formal training until all engineering gates pass.
