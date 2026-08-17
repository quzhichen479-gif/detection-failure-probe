# Water-Clutter Response Zero-Training Probe

Status: **mechanism probe only — no training, no new loss, no new module**

This track tests whether the existing YOLO11 baseline and the existing **DBRA P3-mid**
checkpoint respond differently when only target-external background pixels are perturbed.

The project evidence motivating this probe is the current DBRA pattern:

- DBRA P3-mid improves paper-aligned AP/APs/ARs over the matched YOLO11n baseline.
- Its single-point Precision remains below baseline.
- The cause of the Precision difference is **not established**.
- The current PoTATO test split has already been used for architectural comparisons and must
  not be used to tune the next direction.

See the frozen evidence in
[`../attention_cls_branch/11_ROUND2_TEST_EVIDENCE_AND_ROUND3_HYPOTHESES.md`](../attention_cls_branch/11_ROUND2_TEST_EVIDENCE_AND_ROUND3_HYPOTHESES.md).

## Research question

For original image `x` and a target-external nuisance intervention `xw`:

`delta_s_TP = s_TP(xw) - s_TP(x)`

`delta_s_FP = s_FP(xw) - s_FP(x)`

The candidate mechanism is supported only if true-positive response is relatively stable while
high-confidence/background false positives are materially more sensitive:

`abs(delta_s_TP) << abs(delta_s_FP)`

This is **not** assumed to be true. The probe is explicitly allowed to reject the WCR direction.

## Repository contents

- `run_probe.py` — end-to-end inference-only runner.
- `probe_core.py` — dataset parsing, GT protection, perturbations, matching and statistics.
- `ultralytics_adapter.py` — inference-only YOLO11 adapter with optional final-Detect raw capture.
- `probe_config.example.yaml` — local configuration template.
- `CODEX_TASK.md` — constraints, execution order and decision rules for Codex.

The scripts are intentionally isolated from the package in `src/failure_probe/`.
They should be run **inside the existing YOLO11 engineering environment**, where the exact
Ultralytics 8.4.113 + DBRA implementation/checkpoints already exist. This research repository
does not vendor the user's model weights or training tree.

## What the code controls

1. No optimizer, backward pass or training path exists.
2. Any split name containing `test` is rejected.
3. Baseline and DBRA consume the exact same generated `xw` arrays.
4. Every intervention excludes a dilated GT protection mask.
5. TP/FP status is frozen from predictions on original `x`.
6. Strict background FP requires max IoU to every GT `< 0.1`.
7. Post-NMS candidate lists are never compared by list index.
8. Original FPs are matched to perturbed predictions by class + box IoU.
9. Original TPs are followed through the same GT.
10. Missing perturbed detections get `score_xw=0` and are retained as disappearances.
11. A final-Detect forward hook attempts a secondary fixed raw-candidate score comparison.
12. Null perturbation is a hard validity control.
13. Strict background FP is **not renamed water FP**. A review CSV is generated for manual taxonomy.

## Run

From the YOLO11 workspace (or anywhere the paths resolve):

```bash
python path/to/detection-failure-probe/research_tracks/water_clutter_response_probe/run_probe.py \
  --config path/to/local_probe_config.yaml
```

Before running, copy `probe_config.example.yaml` and replace the checkpoint/dataset paths with
the **already existing matched Baseline and DBRA P3-mid checkpoint**. Do not retrain either model.

The `imgsz`, `conf`, `iou_nms`, preprocessing environment and validation split must be inherited
from the frozen DBRA comparison protocol. Do not accept the example numbers blindly if the
engineering manifest says otherwise.

## Output

The runner creates:

```text
outputs/water_response_probe/<run_id>/
├── probe_manifest.json
├── git_status.txt                 # when dirty
├── git_diff_stat.txt              # when dirty
├── perturbation_manifest.jsonl
├── analysis/
│   ├── per_detection_response.csv
│   ├── summary_by_condition.csv
│   ├── summary_by_condition.json
│   └── decision.json
├── sanity/intervention_examples/
├── fp_review_manifest.csv
└── REPORT.md
```

`per_detection_response.csv` contains both post-NMS paired response and, when capture succeeds,
`pre_nms_delta` for the same raw candidate index/class.

## Primary gates

The current **screening** gates are project decision rules, not field-wide constants:

- null median `abs(delta_s)` must stay within `null_tolerance`;
- DBRA `TP vs strict-background-FP` sensitivity AUROC should be at least `0.60`;
- median sensitivity ratio

`R_sens = median(abs(delta_s_FP)) / (median(abs(delta_s_TP)) + eps)`

should be at least `1.5`.

If these pass, the script deliberately returns `NEED_MANUAL_WATER_FP_AUDIT`, not a WCR success
claim. The top sensitive strict FPs must first be reviewed as reflection/wave/foam/shore/etc.

A stronger WCR claim should only be considered after confirmed water-clutter FP labels show the
same direction (preferred target: AUROC around/above `0.65`, with a meaningful dose response).

## Interpretation cases

- **A / later manual confirmation:** strong water-clutter-specific support -> proceed to WCR loss design.
- **B:** strict-background sensitivity passes, taxonomy unconfirmed -> manual FP audit first.
- **C:** baseline sensitivity is stronger and DBRA is lower -> DBRA may already suppress nuisance
  dependency; deprioritize WCR and inspect localization/assignment instead.
- **D:** both lack a stable response gap -> reject current WCR hypothesis.
- **E:** null/correspondence validity fails -> fix the probe before interpreting it.

## Explicit non-goals

Do **not** use this task to:

- train or fine-tune Baseline/DBRA;
- design WCR-Loss yet;
- modify DBRA;
- add an attention module;
- change TAL/DFL/IoU;
- search DBRA hyperparameters;
- use PoTATO test to choose a follow-up;
- relabel every strict FP as a water/reflection FP;
- claim that synthetic glare/ripple is a physically faithful water simulator.

The only question is whether background-only response is a real, reproducible failure signal
worth turning into a later training objective.
