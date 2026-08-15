# YOLO11 Water-Surface Attention Track — Codex Entry

> Status: **candidate implementation package / not validated as effective**  
> Created: 2026-08-15  
> Target: YOLO11n, Ultralytics 8.4.113, PoTATO-style water-surface floating-object detection  
> Scope rule: **classification branch only; do not modify box/DFL/regression branch in the first implementation round.**

## Why this directory exists

This repository already contains `detection-failure-probe`, whose `src/`, `scripts/`, `tests/`, and existing reports belong to the probe/tooling project. The files in this directory are intentionally isolated as a **detector research track**. Codex must not treat them as an extension of the existing probe package and must not place detector modules under the repository's current `src/` unless explicitly instructed later.

This track is also **not a claim that attention is already effective**. Existing project evidence contains many negative or unstable feature-enhancement results. The only recent attention-like engineering control with a small positive matched-test delta was E2+ECA, and that gain was only about +0.00237 mAP50-95 on one seed, which is insufficient as an innovation claim. Therefore these modules are exploratory candidates and require isolated ablation.

## Baseline rule

The preceding YOLO11n baseline already exists. **Do not retrain or reproduce it.** Reuse its existing checkpoint, metrics, run metadata, and evaluator outputs.

Before any candidate training, read `05_BASELINE_AND_TRAINING_PROTOCOL.md`, recover the exact recipe from the actual baseline artifacts, write `implementation/BASELINE_RECIPE_LOCK.md`, and run the required baseline-vs-candidate configuration diff. Candidate training may change only the module/model structure, run name, and output path. If a comparison-critical baseline parameter cannot be recovered, stop before training instead of guessing.

## Round-1 candidates and original insertion plan

1. **CAA-Lite** — long-range axis/context aggregation with low implementation risk.
2. **LSK-Lite** — large selective spatial context.
3. **BRA-Lite** — dynamic sparse region routing.

| Module | Priority-1 insertion | Priority-2 insertion | First-round restriction |
|---|---|---|---|
| CAA-Lite | P3 classification branch, immediately before final class predictor | P3 classification branch, between the two feature-transform blocks | P3 only |
| LSK-Lite | P3 classification branch, between the two feature-transform blocks | P3 classification branch, immediately before final class predictor | P3 only |
| BRA-Lite | P3 classification branch, between the two feature-transform blocks | P4 classification branch, between the two feature-transform blocks | never enable P3+P4 together in the first ablation |

For Ultralytics YOLO11, the relevant classification path is the `Detect.cv3` branch. Codex must inspect the installed **8.4.113** source before patching because exact container nesting may differ from current upstream. The conceptual positions are stable; line numbers are not.

## Round-1 test evidence — frozen 2026-08-15

The baseline was not retrained or re-evaluated; its existing fixed evaluator result was reused.

| Model | Precision | Recall | mAP50 | mAP75 | mAP50-95 |
|---|---:|---:|---:|---:|---:|
| Baseline | **0.902309** | 0.806146 | 0.870646 | 0.484366 | **0.484773** |
| CAA-Lite | 0.901744 | 0.806871 | 0.863882 | **0.488910** | 0.481432 |
| LSK-Lite | 0.857760 | 0.789229 | 0.845539 | 0.407234 | 0.437660 |
| BRA-Lite | 0.888767 | **0.822656** | **0.876378** | 0.485967 | 0.484218 |

Interpretation boundary:

- **BRA** is the strongest Round-1 candidate: Recall/AP50/AP75 rise, but Precision falls about 1.35 percentage points and mAP50-95 remains 0.000555 below baseline.
- **CAA** is near baseline in Precision/Recall and raises AP75, but loses 0.003340 mAP50-95.
- **LSK** is clearly negative and is stopped.
- These are single-seed point estimates; no statistical significance claim is allowed.

The original protocol's predefined CAA second-insertion priority is not retroactively changed by the test ranking.

Because the test result has now influenced which new mechanisms are investigated, **the current test must not be used as the iterative selection/tuning set for Round-2 candidates**. Round-2 selection uses frozen validation; paper-level final confirmation should use a new final confirmation set or external data if required.

## Round-2 candidates

The new search is intentionally narrower: preserve the useful adaptive-context signal suggested by BRA while improving classification purity or testing whether complex routing is actually necessary.

| Module | Primary insertion | Secondary insertion | Mechanism question |
|---|---|---|---|
| **DBRA** | **P3-Cls-Mid** | **P4-Cls-Mid** | Can more selective deformable/agent K/V routing recover BRA's lost Precision while retaining Recall/AP gains? |
| **SHSA** | **P3-Cls-Mid** | **P4-Cls-Mid** | Can a simpler partial-channel single-head global correction provide most BRA benefits at lower cost / higher Precision? |
| **Triplet Attention** | **P3-Cls-Pre** | **P3-Cls-Mid** | Is non-local context necessary, or is lightweight cross-dimensional P3 feature selection sufficient? |

Do not enable both insertion positions simultaneously in the first ablation. A secondary position is run only if the primary position passes the predefined validation gate.

## Directory map

### Round-1

- `00_SCOPE_AND_EVIDENCE.md` — original evidence boundary, non-goals, experiment rules.
- `01_CAA_LITE.md` — CAA-Lite equations, reference implementation skeleton, insertion guidance.
- `02_LSK_LITE.md` — LSK-Lite equations, implementation skeleton, insertion guidance.
- `03_BRA_LITE.md` — BRA-Lite routing formulation, implementation skeleton, insertion guidance.
- `04_CODEX_IMPLEMENTATION_PLAN.md` — staged Round-1 engineering task for Codex.
- `05_BASELINE_AND_TRAINING_PROTOCOL.md` — mandatory baseline reuse and matched-training protocol.

### Round-2

- `06_ROUND1_TEST_EVIDENCE.md` — frozen Round-1 test results, interpretation limits, and new test-use boundary.
- `07_DBRA.md` — DBRA rationale, formulation, upstream-vendoring rule, adapter, P3/P4 cls-mid positions.
- `08_SHSA.md` — SHSA formulation, project PyTorch implementation, P3/P4 cls-mid positions, cost warning.
- `09_TRIPLET_ATTENTION.md` — Triplet Attention formulation, project implementation, P3 cls-pre/cls-mid positions.
- `10_CODEX_ROUND2_IMPLEMENTATION_PLAN.md` — **primary Codex execution entry for the new three modules**.

## Source papers / official implementations

### Round-1

- PKINet / Context Anchor Attention, CVPR 2024: https://openaccess.thecvf.com/content/CVPR2024/html/Cai_Poly_Kernel_Inception_Network_for_Remote_Sensing_Detection_CVPR_2024_paper.html
- PKINet official repository: https://github.com/PKINet/PKINet
- LSKNet, ICCV 2023: https://openaccess.thecvf.com/content/ICCV2023/html/Li_Large_Selective_Kernel_Network_for_Remote_Sensing_Object_Detection_ICCV_2023_paper.html
- LSKNet official repository: https://github.com/zcablii/LSKNet
- BiFormer / Bi-Level Routing Attention, CVPR 2023: https://openaccess.thecvf.com/content/CVPR2023/html/Zhu_BiFormer_Vision_Transformer_With_Bi-Level_Routing_Attention_CVPR_2023_paper.html
- BiFormer official repository: https://github.com/rayleizhu/BiFormer

### Round-2

- DeBiFormer / DBRA, ACCV 2024: https://openaccess.thecvf.com/content/ACCV2024/html/BaoLong_DeBiFormer_Vision_Transformer_with_Deformable_Agent_Bi-level_Routing_Attention_ACCV_2024_paper.html
- DeBiFormer official repository: https://github.com/maclong01/DeBiFormer
- SHViT / SHSA, CVPR 2024: https://openaccess.thecvf.com/content/CVPR2024/html/Yun_SHViT_Single-Head_Vision_Transformer_with_Memory_Efficient_Macro_Design_CVPR_2024_paper.html
- SHViT official repository: https://github.com/ysj9909/SHViT
- Triplet Attention, WACV 2021: https://openaccess.thecvf.com/content/WACV2021/html/Misra_Rotate_to_Attend_Convolutional_Triplet_Attention_Module_WACV_2021_paper.html
- Triplet Attention official repository: https://github.com/LandskapeAI/triplet-attention

## Codex reading rule

### For Round-2 implementation

Start from this README, then read in this exact order:

```text
05_BASELINE_AND_TRAINING_PROTOCOL.md
06_ROUND1_TEST_EVIDENCE.md
07_DBRA.md
08_SHSA.md
09_TRIPLET_ATTENTION.md
10_CODEX_ROUND2_IMPLEMENTATION_PLAN.md
```

Then implement the three candidates in the **actual YOLO11 engineering repository**, not under this repository's existing `src/` probe package.

Implement **one candidate at a time**, with independent switches and no simultaneous module stacking. Reuse the existing baseline; do not retrain it. Before long training, complete build/import, pretrained-weight transfer, alpha=0 baseline-equivalence, box-branch isolation, forward/backward, smoke train/val/predict and latency/VRAM gates.

The short Codex invocation is contained at the end of `10_CODEX_ROUND2_IMPLEMENTATION_PLAN.md`.
