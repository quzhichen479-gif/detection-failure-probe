# Codex task — FloW-Img CC-MSCA Phase 4

Read all files under `research/flow_cc_msca_phase4/` first and treat `phase4_manifest.yaml` plus `TRAINING_PLAN.md` as the frozen contract.

In the current local Ultralytics 8.4.113 YOLO11 working tree:

1. Port `SegNeXtMSCA` and `ContextContrastMSCA` from `cc_msca.py` into the normal `ultralytics.nn.modules` path, export/register them, and make `parse_model` treat them as channel-preserving (`c1=ch[f]`, prepend `c1`, `c2=c1`).
2. Create three YOLO11n YAMLs by semantic location, fixing all shifted layer references:
   - M1: `SegNeXtMSCA(5,[7,11,21])` after backbone P2 C3k2, before P3 downsample;
   - M2: `ContextContrastMSCA([3,5,7],8,0.0)` at the same P2 location;
   - M3: the identical CC-MSCA after backbone P3 C3k2, before P4 downsample.
3. Do not add a P2 detection head, second attention, dynamic/partial kernels, frequency/edge branch, new loss/assignment/augmentation, DySample/PPRD, or any other model change.
4. Before training run unit tests plus build/shape/index checks, non-empty and empty-GT FP32+AMP forward/backward, short train, val, predict, export, Params/GFLOPs and latency smoke tests. Verify standalone CC-MSCA is exact identity at default initialization.
5. Reuse the saved B0 `flow_pbm_protocol_seed79` arguments exactly: YOLO11n scratch, pretrained=False, 640, 300 epochs, batch 32, SGD, lr0=.01, momentum=.937, weight_decay=.0005, seed 79, same split/augmentation/evaluator/checkpoint policy. Do not rerun B0 and do not tune anything.
6. Train full M1 -> M2 -> M3 sequentially. Save config/commit/results/best.pt/last.pt/Val metrics for each. Do **not** evaluate candidate Test during this sequence.
7. After M3 finishes, freeze all three configs/commits, then evaluate each `best.pt` on the frozen Test exactly once. Report B0/M1/M2/M3 P, R, mAP50, mAP50-95, AP75, AP_small, Recall_small, AP_medium/large, Params, GFLOPs, VRAM, train time, P50/P95 latency, best epoch and Val->Test delta. For M2/M3 also report final softmax scale weights and gamma statistics.
8. Write one final Phase-4 report and stop. Do not alter hyperparameters or rerun variants after seeing Test. If M1-M3 are all neutral/negative on small-object metrics, record the negative result and close the MSCA/attention direction rather than inventing M4.
