# Codex task — FloW-Img Attention Phase 3

Read `research/flow_attention_phase3/` and port `EMAAttention`, `CAAAttention`, and `TripletAttention` into the current Ultralytics 8.4.113 YOLO11 working tree.

Reuse the existing frozen `flow_pbm_protocol_seed79` and all saved B0 training/evaluation arguments exactly. **Do not rerun B0, repartition data, tune training parameters, or combine modules.**

Create three YAML variants using a **Detect-only P3 side branch** after the final fused P3 C3k2:

- A1: `EMAAttention(factor=8)`
- A2: `CAAAttention(kernel_size=11)`
- A3: `TripletAttention(no_spatial=False, kernel_size=7)`

The untouched original P3 must still feed the normal bottom-up P4/P5 PAN path; only Detect's P3 source should use the attended branch. Register all modules as channel-preserving in `parse_model` (`c1=ch[f]`, prepend `c1`, `c2=c1`).

Before full training verify build, exact P3 shape preservation, graph source indices, finite forward/backward, short train, val, predict, export, Params/GFLOPs and batch-1 P50. Then train A1/A2/A3 with the frozen 300-epoch seed79 protocol, select `best.pt` only by Val mAP50-95, and evaluate Test only after each run finishes.

Report versus saved B0: mAP50-95, AP_small, Recall_small, AP75, Params, GFLOPs and batch-1 P50. Do not add any other attention, P2, loss, DySample, PPRD or augmentation change.
