# Codex task — FloW-Img UR-P2 / SDAS / WCRF

Use branch `research/flow-urp2-sdas-wcrf` of `quzhichen479-gif/detection-failure-probe` as the design source. Read `research/flow_urp2_sdas_wcrf/README.md`, `modules.py`, `variants.py`, `variant_manifest.yaml`, and `seed79_protocol.yaml` first.

Port UR-P2, SDAS and WCRF into a copied Ultralytics 8.4.113 YOLO11 runtime; do not edit the frozen B0 runtime. Re-audit the local stock `yolo11n.yaml` anchors before patching. Implement all seven YAML/model variants: `U`, `S`, `W`, `US`, `UW`, `SW`, `USW`, following the fixed graph order in `variant_manifest.yaml`.

Critical integration rules:

- UR-P2: isolated `URP2Detect(Detect)`; use stock P2 detail + P3 raw DFL entropy; refine P3 raw box/class logits only; P4/P5 stay stock. V1 is dense soft routing and makes no sparse-speed claim.
- SDAS: training-only P2 GT-center auxiliary head/loss with `lambda_sdas=0.25`; supervise all GT centers; remove it from eval/export.
- WCRF: Detect-only P3 side branch after final fused P3; untouched P3 must still feed PAN P4/P5.
- With WCRF+UR-P2 the order is `WCRF(P3) -> UR-P2`; SDAS always acts on stock P2 during training.

Before training, run unit tests plus build/forward/backward/empty-GT/AMP/short-train/val/predict/TorchScript/ONNX checks for all seven variants. Verify U-containing variants start numerically equal to stock P3 raw logits because UR-P2 delta heads are zero initialized; verify S-containing exports contain no SDAS path; verify W-containing graphs keep the stock P3 PAN path untouched.

Then launch full training sequentially in this order: `U,S,W,US,UW,SW,USW`, all with the frozen `flow_pbm_protocol_seed79`: 1400/200/400 split, split/train seed 79, scratch YOLO11n, imgsz 640, 300 epochs, batch 32, SGD lr0=0.01 momentum=0.937 weight_decay=0.0005, and every other argument inherited exactly from the saved B0 seed79 run. Do not rerun B0 and do not tune on Test. Select `best.pt` only by Val mAP50-95; run Test once after each completed run.

Save per-run P/R/mAP50/mAP50-95, available AP75/scale metrics, Params/GFLOPs/VRAM/P50/P95; for UR-P2 also save mean DFL uncertainty, mean route gate and gate>=0.5 fraction; for SDAS save auxiliary-loss curves. Produce one final comparison table against the saved B0 and YOLO11s reference. If any variant fails build/export or diverges, stop that variant, record the failure, and continue to the next without changing defaults.
