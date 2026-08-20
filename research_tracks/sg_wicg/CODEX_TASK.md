# CODEX TASK — integrate SG-WICG v1 into Ultralytics 8.4.113

Implement the frozen SG-WICG v1 loss track in the current YOLO11 / Ultralytics 8.4.113 worktree.

## Read first

Use these repository files as the specification:

1. `research_tracks/sg_wicg/README.md`
2. `research_tracks/sg_wicg/sg_wicg.py`
3. `research_tracks/sg_wicg/test_reference.py`

Do not redesign the method while implementing it.

## Required code changes

1. Copy/adapt `sg_wicg.py` into the Ultralytics source tree, preferably `ultralytics/utils/sg_wicg.py`.
2. Register SG-WICG config keys in `ultralytics/cfg/default.yaml`.
3. In `ultralytics/utils/loss.py`, extend `BboxLoss` so:
   - `reg_loss_mode=ciou` uses the original native `bbox_iou(..., CIoU=True)` path exactly.
   - A1–A5 use `SGWICGBoxLoss` only for the foreground box-regression term.
   - DFL, TAL, cls loss, head, decoding and normalization remain unchanged.
4. Extract positive strides from `stride_tensor` + `fg_mask` so the scale gate uses GT short-side pixels.
5. Persist/restore A5 `WiseFocus.iou_mean` in checkpoints because criterion is stripped from EMA serialization in 8.4.113.
6. Add mechanism diagnostics without changing optimization.

## Mandatory tests before training

Implement and run:

- native CIoU parity
- identical-box zero loss
- GCD scale invariance
- non-overlap finite/nonzero gradient
- gate monotonicity
- TAL-weighted Wise mean gain ~= 1
- A1–A5 FP32 + AMP forward/backward
- DFL parity
- empty-foreground parity
- uninterrupted vs save/resume A5 next-step equivalence

For the first ablation, use one GPU for A5 unless you additionally implement a rank-synchronous zero-positive-safe Wise population-stat update.

## Six training cells

Keep the current frozen IWHR baseline recipe exactly unchanged except the loss config:

- A0: `reg_loss_mode=ciou`
- A1: `reg_loss_mode=inner_ciou inner_ratio=1.25`
- A2: `reg_loss_mode=gcd`
- A3: `reg_loss_mode=inner_gcd_fixed inner_ratio=1.25 gcd_fixed_weight=0.5`
- A4: `reg_loss_mode=sg_icg inner_ratio=1.25 sg_tau_px=12 sg_temp_px=2`
- A5: `reg_loss_mode=sg_wicg inner_ratio=1.25 sg_tau_px=12 sg_temp_px=2 wise_alpha=1.7 wise_delta=2.7 wise_ema_rate=0.01`

Do not tune these values in the first six-cell experiment.

## Deliverables

- source diff
- tests and test output
- concise implementation note listing exact touched files
- six runnable commands generated from the existing frozen baseline command
- no training unless the engineering tests pass
