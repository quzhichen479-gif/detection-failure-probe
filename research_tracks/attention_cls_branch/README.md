# YOLO11 Water-Surface Attention Track — Codex Entry

> Status: **candidate implementation package / not validated as effective**  
> Created: 2026-08-15  
> Target: YOLO11n, Ultralytics 8.4.113, PoTATO-style water-surface floating-object detection  
> Scope rule: **classification branch only; do not modify box/DFL/regression branch in the first implementation round.**

## Why this directory exists

This repository already contains `detection-failure-probe`, whose `src/`, `scripts/`, `tests/`, and existing reports belong to the probe/tooling project. The files in this directory are intentionally isolated as a **detector research track**. Codex must not treat them as an extension of the existing probe package and must not place detector modules under the repository's current `src/` unless explicitly instructed later.

This track is also **not a claim that attention is already effective**. Existing project evidence contains many negative or unstable feature-enhancement results. The only recent attention-like engineering control with a small positive matched-test delta was E2+ECA, and that gain was only about +0.00237 mAP50-95 on one seed, which is insufficient as an innovation claim. Therefore these three modules are exploratory candidates and require isolated ablation.

## Baseline rule

The preceding YOLO11n baseline already exists. **Do not retrain or reproduce it.** Reuse its existing checkpoint, metrics, run metadata, and evaluator outputs.

Before any candidate training, read `05_BASELINE_AND_TRAINING_PROTOCOL.md`, recover the exact recipe from the actual baseline artifacts, write `implementation/BASELINE_RECIPE_LOCK.md`, and run the required baseline-vs-candidate configuration diff. Candidate training may change only the module/model structure, run name, and output path. If a comparison-critical baseline parameter cannot be recovered, stop before training instead of guessing.

## Candidate order

1. **CAA-Lite** — first priority. Long-range axis/context aggregation with low implementation risk.
2. **LSK-Lite** — second priority. Large selective spatial context, motivated by tiny-object/context problems in remote sensing.
3. **BRA-Lite** — third priority. Dynamic sparse region routing; potentially useful but highest implementation and latency risk.

## First two insertion positions for each module

| Module | Priority-1 insertion | Priority-2 insertion | First-round restriction |
|---|---|---|---|
| CAA-Lite | P3 classification branch, immediately before final class predictor | P3 classification branch, between the two feature-transform blocks | P3 only |
| LSK-Lite | P3 classification branch, between the two feature-transform blocks | P3 classification branch, immediately before final class predictor | P3 only |
| BRA-Lite | P3 classification branch, between the two feature-transform blocks | P4 classification branch, between the two feature-transform blocks | never enable P3+P4 together in the first ablation |

For Ultralytics YOLO11, the relevant classification path is the `Detect.cv3` branch. Codex must inspect the installed **8.4.113** source before patching because exact container nesting may differ from current upstream. The conceptual positions are stable; line numbers are not.

## Directory map

- `00_SCOPE_AND_EVIDENCE.md` — evidence boundary, non-goals, experiment rules.
- `01_CAA_LITE.md` — CAA-Lite equations, reference implementation skeleton, insertion guidance.
- `02_LSK_LITE.md` — LSK-Lite equations, implementation skeleton, insertion guidance.
- `03_BRA_LITE.md` — BRA-Lite routing formulation, implementation skeleton, insertion guidance.
- `04_CODEX_IMPLEMENTATION_PLAN.md` — staged engineering task for Codex.
- `05_BASELINE_AND_TRAINING_PROTOCOL.md` — mandatory baseline reuse and matched-training protocol.

## Source papers / official implementations

- PKINet / Context Anchor Attention, CVPR 2024: https://openaccess.thecvf.com/content/CVPR2024/html/Cai_Poly_Kernel_Inception_Network_for_Remote_Sensing_Detection_CVPR_2024_paper.html
- PKINet official repository: https://github.com/PKINet/PKINet
- LSKNet, ICCV 2023: https://openaccess.thecvf.com/content/ICCV2023/html/Li_Large_Selective_Kernel_Network_for_Remote_Sensing_Object_Detection_ICCV_2023_paper.html
- LSKNet official repository: https://github.com/zcablii/LSKNet
- BiFormer / Bi-Level Routing Attention, CVPR 2023: https://openaccess.thecvf.com/content/CVPR2023/html/Zhu_BiFormer_Vision_Transformer_With_Bi-Level_Routing_Attention_CVPR_2023_paper.html
- BiFormer official repository: https://github.com/rayleizhu/BiFormer

## Codex reading rule

Codex should start from this README, then read all numbered files in order, including `05_BASELINE_AND_TRAINING_PROTOCOL.md`. It must implement **one candidate at a time**, with independent switches and no simultaneous module stacking. The existing baseline must not be retrained. Any candidate training must use the exact preceding-baseline protocol and pass the configuration-diff gate first. The independent test set must not be used for iterative selection or tuning.
