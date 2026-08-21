# Codex task — FloW-Img DySample Phase 2

Use `research/flow_dysample_phase2/` as the implementation source. Port `DySample` into the actual Ultralytics 8.4.113 YOLO11 working tree, register it with `ultralytics.nn.modules` / `tasks.py`, and make `parse_model` treat it as channel-preserving (`c1=ch[f]`, constructor args `[c1, 2, "lp", 4, false]`, `c2=c1`).

Create three YOLO11n YAML variants under the existing frozen FloW-Img seed79 experiment structure: **U1** replaces only P4/16 -> P3/8 nearest upsample, **U2** replaces only P5/32 -> P4/16, and **U3** replaces both. Do not alter backbone, Concat routes, heads, losses, augmentations, split, evaluator, seed, or training arguments; do not rerun B0 and do not combine with PPRD.

First pass unit/build/shape/finite-forward-backward/short-train/val/predict/export checks. Then run U1 first under the frozen 1400/200/400 seed79, YOLO11n scratch, 640, 300ep, batch32, SGD protocol. Select `best.pt` only by Val mAP50-95 and touch Test only after training. Report deltas versus saved B0 for mAP50-95, AP_small, Recall_small, AP75, Params, GFLOPs, and batch-1 P50. Run U2/U3 only as insertion-location controls after U1 is engineering-clean.
