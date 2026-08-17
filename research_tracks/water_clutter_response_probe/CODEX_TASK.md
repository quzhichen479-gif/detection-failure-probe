# Codex execution task — Baseline vs DBRA zero-training water response probe

## Goal

Implement/run the already prepared probe in this directory. Do **not** invent a new model or
loss before the probe result is known.

Use the existing matched:

- YOLO11n baseline checkpoint;
- DBRA P3-mid checkpoint;
- frozen validation/audit split;
- same Ultralytics 8.4.113 inference protocol used for the DBRA comparison.

The baseline is already trained. The DBRA checkpoint is already trained. **Do not retrain either.**

## First actions

1. Read:
   - `README.md`
   - `probe_config.example.yaml`
   - `run_probe.py`
   - `probe_core.py`
   - `ultralytics_adapter.py`
   - `../attention_cls_branch/11_ROUND2_TEST_EVIDENCE_AND_ROUND3_HYPOTHESES.md`
2. In the YOLO11 engineering workspace, locate the exact matched baseline/DBRA checkpoint and
   frozen val protocol from manifests/logs. Do not guess from filenames if evidence exists.
3. Create a local config from `probe_config.example.yaml`; do not commit machine-specific paths
   unless explicitly useful.
4. Run a small sanity pass first (e.g. 5–20 val images) only to verify:
   - both checkpoints load;
   - no training path is entered;
   - null response is effectively zero;
   - GT-protection masks never alter protected pixels;
   - post-NMS correspondence works;
   - `pre_nms_delta` is populated when the final Detect hook is compatible.
5. Fix probe implementation bugs only. Do not tune DBRA or perturbations to obtain a preferred result.
6. Run the preregistered validation/audit probe.

## Hard constraints

- Zero training: no optimizer/backward/fit/train.
- Do not rerun baseline training.
- Do not use test. `run_probe.py` intentionally rejects `split: test`.
- Baseline and DBRA must consume identical cached/in-memory paired perturbations.
- Keep original TP/FP status frozen from `x`; never recompute populations independently on `xw`.
- Never compare NMS output lists by index.
- Missing corresponding detections remain in analysis with `score_xw=0`.
- Strict background FP means max IoU to all GT `<0.1`.
- Do not call strict background FP “water FP” without review/labels.
- Report both post-NMS paired response and pre-NMS fixed-candidate response when available.
- Null control failure invalidates the probe.
- No rescue tuning based on the result.
- Preserve the user's current dirty working tree; do not reset/restore unrelated files.

## Required final evidence

Return the run directory and summarize:

1. exact baseline/DBRA checkpoint + SHA256;
2. dataset/split/imgsz/conf/NMS settings;
3. number of images/GT/TP/strict FP;
4. null-control result;
5. Baseline and DBRA:
   - median `|delta_s|` TP;
   - median `|delta_s|` strict FP;
   - high-conf FP sensitivity;
   - `R_sens`;
   - TP-vs-strict-FP AUROC;
   - pre-NMS equivalents where available;
6. near vs far;
7. S1/S2/S3 dose response;
8. top sensitive FP manual-review manifest;
9. final decision from:
   - `PROBE_INVALID`
   - `NEED_MANUAL_WATER_FP_AUDIT`
   - `DO_NOT_PROCEED_WCR_YET`
   - `DO_NOT_PROCEED_WCR`

If strict-FP screening passes, stop at the manual FP audit boundary. Do **not** implement WCR-Loss
until water-clutter-specific sensitivity is confirmed.
