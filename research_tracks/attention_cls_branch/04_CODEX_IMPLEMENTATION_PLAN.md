# Codex Implementation Plan — YOLO11 Attention Classification Track

## Mission

Implement and, when this task explicitly includes training, train the attention candidates described in this directory as a **separate YOLO11 detector experiment track**, without modifying or conflating the repository's existing detection-failure-probe package.

The project baseline already exists. **Never retrain the baseline.** Candidate runs must use the exact preceding-baseline training recipe and evaluator protocol, with the candidate model/module as the only substantive experimental change.

## Read-first order

1. `research_tracks/attention_cls_branch/README.md`
2. `research_tracks/attention_cls_branch/00_SCOPE_AND_EVIDENCE.md`
3. `research_tracks/attention_cls_branch/01_CAA_LITE.md`
4. `research_tracks/attention_cls_branch/02_LSK_LITE.md`
5. `research_tracks/attention_cls_branch/03_BRA_LITE.md`
6. `research_tracks/attention_cls_branch/05_BASELINE_AND_TRAINING_PROTOCOL.md`
7. this file

Before changing code, inspect the local project and the exact installed `ultralytics==8.4.113` implementation of YOLO11 `Detect`, especially `Detect.cv2`, `Detect.cv3`, model parsing/build logic, export behavior, and configuration path.

## Non-negotiable boundaries

- Existing `src/`, `scripts/`, tests and reports at repository root belong to detection-failure-probe.
- Do not place detector experiment modules inside that package.
- Do not modify regression/DFL branch for this task.
- Do not alter assigner, loss, NMS, data split, augmentation, optimizer, or baseline recipe.
- Do not implement ECA/CBAM/CoordAtt/EMA/SKA as new candidates.
- Do not stack attention modules.
- Do not use the independent test set for iterative tuning or candidate selection.
- Do not retrain, reproduce, or replace the existing baseline.
- Do not silently replace a paper mechanism with a materially different design; if a compatibility simplification is necessary, document it.

## Desired engineering layout

Create a self-contained detector experiment package under this research track, for example:

```text
research_tracks/attention_cls_branch/
  README.md
  00_SCOPE_AND_EVIDENCE.md
  01_CAA_LITE.md
  02_LSK_LITE.md
  03_BRA_LITE.md
  04_CODEX_IMPLEMENTATION_PLAN.md
  05_BASELINE_AND_TRAINING_PROTOCOL.md
  implementation/
    __init__.py
    attention.py
    detect_cls_attention.py
    SOURCE_AUDIT.md
    BASELINE_RECIPE_LOCK.md
    protocol_diffs/
    configs/
      yolo11n_clsattn_caa_p3.yaml
      yolo11n_clsattn_lsk_p3.yaml
      yolo11n_clsattn_bra_p3.yaml
    tests/
      test_attention_modules.py
      test_detect_wiring.py
      test_export.py
    README.md
```

Exact file names may differ if the repository structure demands it, but isolation must be preserved.

## Stage 0 — Source audit

Produce a short source audit before implementation:

- exact Ultralytics version;
- path and relevant structure of YOLO11 `Detect`;
- actual P3/P4/P5 class branch channel sizes for YOLO11n;
- safest insertion mechanism;
- whether subclassing, wrapper composition, YAML registration, or a small local fork is least invasive;
- export constraints.

Write the audit to:

`research_tracks/attention_cls_branch/implementation/SOURCE_AUDIT.md`

Do not modify detector code until this audit is written.

## Stage 0.5 — Freeze the existing baseline recipe

Locate the **already-completed preceding baseline run** and its primary artifacts. Do not launch any baseline training.

Recover the exact training and evaluation recipe from artifacts such as `args.yaml`, the original launch command/script, model/data YAML, logs, results files, checkpoint metadata, and evaluator configuration.

Write:

`research_tracks/attention_cls_branch/implementation/BASELINE_RECIPE_LOCK.md`

For every comparison-critical field, record:

- resolved value;
- source artifact/path;
- whether it is explicit or inferred.

Follow `05_BASELINE_AND_TRAINING_PROTOCOL.md` as the authority. If a critical field cannot be resolved unambiguously, engineering work may continue but candidate training must not start.

## Stage 1 — Common interface

Implement a common identity-safe interface:

```python
build_cls_attention(name, channels, **kwargs) -> nn.Module
```

Supported values:

```text
none
caa_lite
lsk_lite
bra_lite
```

Requirements:

- `none` returns `nn.Identity()` or exact baseline equivalent;
- every module preserves `[B,C,H,W]`;
- residual strength starts at or near identity;
- independent per-detection-scale configuration;
- no hidden global state.

## Stage 2 — CAA-Lite first

Implement only CAA-Lite and its tests first.

Primary wiring:

```text
P3 classification branch -> pre-predictor CAA-Lite
```

Do not proceed to its training until all of these pass:

- construction;
- shape tests;
- backward;
- AMP if CUDA exists;
- YOLO11n model build;
- one synthetic forward;
- disabled-mode baseline forward shape equivalence;
- project export path;
- static parameter/FLOPs sanity check;
- baseline-vs-CAA training configuration diff.

Commit CAA-Lite separately.

### CAA training gate

If training is part of the active task, generate:

`implementation/protocol_diffs/caa_p3_vs_baseline.md`

The diff must end with `PROTOCOL_MATCH: PASS` before launch.

Train CAA-Lite with the **exact existing-baseline recipe**. The only allowed experimental differences are model/module configuration, experiment name, and output path. Do not train a new baseline beside it.

After training, evaluate using the same validation/checkpoint-selection/evaluator protocol as the preceding baseline and record the delta against the existing baseline result.

## Stage 3 — LSK-Lite

Implement LSK-Lite after CAA-Lite plumbing is stable.

Primary wiring:

```text
P3 classification branch -> between class feature transforms
```

Use the fixed first-round configuration from `02_LSK_LITE.md`. Do not kernel-search.

Before any LSK training, generate `implementation/protocol_diffs/lsk_p3_vs_baseline.md`; require `PROTOCOL_MATCH: PASS`; then use the same frozen baseline recipe. Do not retrain baseline.

Commit separately.

## Stage 4 — BRA-Lite

Implement BRA-Lite last.

Primary wiring:

```text
P3 classification branch -> between class feature transforms
```

Requirements beyond common tests:

- non-divisible H/W shape test;
- padding is internal and cropped exactly;
- TopK/gather works under target export path or failure is explicitly documented;
- no custom CUDA;
- memory sanity check.

If export compatibility becomes disproportionately complex, stop and report rather than rewriting the whole detector.

Before any BRA training, generate `implementation/protocol_diffs/bra_p3_vs_baseline.md`; require `PROTOCOL_MATCH: PASS`; then use the same frozen baseline recipe. Do not retrain baseline.

Commit separately.

## Stage 5 — Configs

Provide at least these three isolated configurations:

```text
yolo11n_clsattn_caa_p3
yolo11n_clsattn_lsk_p3
yolo11n_clsattn_bra_p3
```

Also provide a `none`/baseline-equivalent configuration for wiring verification if needed. This `none` configuration is for build/identity checks only, **not for launching another baseline training run**.

Do not create stacked configs.

## Stage 6 — Static cost report

Report for the existing baseline model definition and each candidate:

- parameters;
- parameter delta;
- GFLOPs if the local tooling can measure it reliably;
- dummy-input inference timing, clearly marked as engineering timing rather than final benchmark;
- output shapes;
- export status.

Do not retrain baseline to obtain these values.

Write:

`research_tracks/attention_cls_branch/implementation/ENGINEERING_REPORT.md`

## Stage 7 — Candidate training and validation, when requested

When the active Codex task requests training, train the three candidate configurations one at a time in this order:

1. CAA-Lite P3
2. LSK-Lite P3
3. BRA-Lite P3

For every run:

1. load the frozen `BASELINE_RECIPE_LOCK.md`;
2. generate the candidate-vs-baseline diff;
3. require `PROTOCOL_MATCH: PASS`;
4. launch with the exact baseline recipe;
5. use the same checkpoint-selection rule;
6. evaluate under the same validation evaluator;
7. compare against the **existing baseline result**;
8. do not inspect the independent test set for selection/tuning.

Do not optimize one candidate's hyperparameters separately. This round is a controlled structural ablation, not per-module tuning.

If a candidate fails the stop rule in `00_SCOPE_AND_EVIDENCE.md`, record the failure; do not change its training recipe to rescue it during this screening round.

## Stage 8 — Final self-review

Before declaring completion, answer explicitly:

1. Did any change touch `Detect.cv2` / box regression / DFL?
2. Does disabled mode preserve baseline behavior?
3. Is each candidate independently selectable?
4. Are P3/P4/P5 choices explicit rather than hard-coded by accidental index?
5. Are there any modifications outside `research_tracks/attention_cls_branch/`? If yes, list and justify each one.
6. Was the existing baseline retrained? Expected answer: **no**.
7. Did each trained candidate pass a baseline-recipe diff with only allowed differences? Expected answer: **yes**.
8. Was the independent test set used for iterative selection/tuning? Expected answer: **no**.
9. Did you accidentally mix this work into the existing detection-failure-probe package? Expected answer: **no**.

## Deliverables

At completion, provide:

- changed-file list;
- commit hashes;
- source audit;
- frozen baseline recipe and provenance;
- protocol-diff reports;
- tests and results;
- engineering cost table;
- candidate training commands actually used, if training was requested;
- validation metrics and deltas versus the existing baseline, if training was requested;
- known limitations;
- exact command to instantiate each candidate;
- recommendation on which candidate, if any, survives the first screening round.

## Stop condition

If the active task is implementation-only, stop after build/test/export/static-cost validation.

If the active task explicitly includes candidate training, continue through Stage 7 using the frozen existing-baseline recipe. In either case, **never retrain the baseline and never use the independent test set for iterative tuning.**
