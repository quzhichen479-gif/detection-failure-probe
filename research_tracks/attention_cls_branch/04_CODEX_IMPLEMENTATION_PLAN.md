# Codex Implementation Plan — YOLO11 Attention Classification Track

## Mission

Implement the attention candidates described in this directory as a **separate YOLO11 detector experiment track**, without modifying or conflating the repository's existing detection-failure-probe package.

Do not train yet. This task ends at engineering validation unless the user explicitly requests a training phase later.

## Read-first order

1. `research_tracks/attention_cls_branch/README.md`
2. `research_tracks/attention_cls_branch/00_SCOPE_AND_EVIDENCE.md`
3. `research_tracks/attention_cls_branch/01_CAA_LITE.md`
4. `research_tracks/attention_cls_branch/02_LSK_LITE.md`
5. `research_tracks/attention_cls_branch/03_BRA_LITE.md`
6. this file

Before changing code, inspect the local project and the exact installed `ultralytics==8.4.113` implementation of YOLO11 `Detect`, especially `Detect.cv2`, `Detect.cv3`, model parsing/build logic, export behavior, and configuration path.

## Non-negotiable boundaries

- Existing `src/`, `scripts/`, tests and reports at repository root belong to detection-failure-probe.
- Do not place detector experiment modules inside that package.
- Do not modify regression/DFL branch for this task.
- Do not alter assigner, loss, NMS, data split, augmentation, optimizer, or baseline recipe.
- Do not implement ECA/CBAM/CoordAtt/EMA/SKA as new candidates.
- Do not stack attention modules.
- Do not use the independent test set.
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
  implementation/
    __init__.py
    attention.py
    detect_cls_attention.py
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

## Stage 0 — Baseline source audit

Produce a short source audit before implementation:

- exact Ultralytics version;
- path and relevant structure of YOLO11 `Detect`;
- actual P3/P4/P5 class branch channel sizes for YOLO11n;
- safest insertion mechanism;
- whether subclassing, wrapper composition, YAML registration, or a small local fork is least invasive;
- export constraints.

Write the audit to:

`research_tracks/attention_cls_branch/implementation/SOURCE_AUDIT.md`

Do not modify code until this audit is written.

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

Do not implement LSK-Lite/BRA-Lite until all of these pass:

- construction;
- shape tests;
- backward;
- AMP if CUDA exists;
- YOLO11n model build;
- one synthetic forward;
- disabled-mode baseline forward shape equivalence;
- project export path.

Commit CAA-Lite separately.

## Stage 3 — LSK-Lite

Implement LSK-Lite after CAA-Lite plumbing is stable.

Primary wiring:

```text
P3 classification branch -> between class feature transforms
```

Use the fixed first-round configuration from `02_LSK_LITE.md`. Do not kernel-search.

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

Commit separately.

## Stage 5 — Configs

Provide at least these three isolated configurations:

```text
yolo11n_clsattn_caa_p3
yolo11n_clsattn_lsk_p3
yolo11n_clsattn_bra_p3
```

Also provide a `none`/baseline-equivalent configuration for wiring verification if needed.

Do not create stacked configs.

## Stage 6 — Static cost report

Without training, report for baseline and each candidate:

- parameters;
- parameter delta;
- GFLOPs if the local tooling can measure it reliably;
- dummy-input inference timing, clearly marked as engineering timing rather than final benchmark;
- output shapes;
- export status.

Write:

`research_tracks/attention_cls_branch/implementation/ENGINEERING_REPORT.md`

## Stage 7 — Final self-review

Before declaring completion, answer explicitly:

1. Did any change touch `Detect.cv2` / box regression / DFL?
2. Does disabled mode preserve baseline behavior?
3. Is each candidate independently selectable?
4. Are P3/P4/P5 choices explicit rather than hard-coded by accidental index?
5. Are there any modifications outside `research_tracks/attention_cls_branch/`? If yes, list and justify each one.
6. Did you train or inspect independent test metrics? Expected answer for this task: **no**.
7. Did you accidentally mix this work into the existing detection-failure-probe package? Expected answer: **no**.

## Deliverables

At completion, provide:

- changed-file list;
- commit hashes;
- source audit;
- tests and results;
- engineering cost table;
- known limitations;
- exact command to instantiate each candidate;
- recommendation on which candidate is safe to train first, based only on engineering quality/cost, not invented accuracy expectations.

## Stop condition

Stop after build/test/export/static-cost validation. Do not start model training until explicitly requested by the user.
