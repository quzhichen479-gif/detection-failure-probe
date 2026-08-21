# Codex task — FloW-Img PPRD Phase 1

Use `research/flow_pprd_phase1/` as the implementation source. Port `RepContextDown`, `SPDDown`, and `PartialPolyphaseRepDown` into the actual Ultralytics 8.4.113 YOLO11 working tree, register them with model parsing, and create D1/D2/D3 YOLO11n YAML variants that replace only the P2/4 -> P3/8 stride-2 backbone Conv. D3 uses `detail_ratio=0.25`.

The FloW-Img seed79 protocol, split manifest, baseline, evaluator and training parameters are already frozen and available; **do not recreate the split, do not tune training parameters, and do not rerun B0 baseline**. Reuse the exact existing 1400/200/400 seed79 manifest and YOLO11n scratch 640/300ep/batch32/SGD settings from `phase1_manifest.yaml`.

First complete build/shape/finite-loss/val/predict/export/reparameterization checks. Then train D1-D3 under the frozen protocol, select each `best.pt` only by Val mAP50-95, and touch the 400-image Test only after training. Report deltas versus the saved B0 baseline for mAP50-95, AP_small, Recall_small, AP75, Params, GFLOPs and batch-1 P50. Do not run D4 unless D3 is non-negative and all engineering checks pass.
