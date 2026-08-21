# FloW-Img seed79 — UR-P2 / SDAS / WCRF research package

Status: implementation-ready research package for the frozen `flow_pbm_protocol_seed79`.

This package deliberately uses only the current verified project facts: the user's PBM-YOLO-aligned FloW-Img protocol and saved local B0 results. Earlier incompatible FloW size/statistics claims are not used to gate these modules.

## 1. Frozen protocol

See `seed79_protocol.yaml`. Required contract:

- FloW-Img Roboflow v2: 2000 images, 5272 bottle instances.
- split = 1400 train / 200 val / 400 test.
- split seed = 79; train seed = 79.
- stock YOLO11n, scratch, `pretrained=False`.
- imgsz 640, 300 epochs, batch 32.
- SGD, lr0 0.01, momentum 0.937, weight decay 0.0005.
- choose `best.pt` only by Validation mAP50-95.
- Test is run only after a completed run; no Test-driven retuning.
- do not rerun B0.
- all unspecified arguments must be inherited from the saved B0 seed79 run rather than replaced by current Ultralytics defaults.

Reference B0 Test: P 90.58, R 82.19, mAP50 87.20, mAP50-95 45.22, 2.582 M params, 6.372 GFLOPs.

## 2. Audited YOLO11 anchors

For the project's Ultralytics 8.4.113 stock YOLO11 graph, prior source audit recorded:

- backbone P2/4: layer 2 output;
- final fused P3/8 before PAN bottom-up: layer 16;
- final P4/16: layer 19;
- final P5/32: layer 22;
- stock Detect sources: `[16, 19, 22]`.

These are semantic insertion anchors. Codex must re-check the exact numeric indices in the local working tree before patching; do not blindly reuse indices if the local YAML differs.

## 3. Module A — UR-P2

**UR-P2 = Uncertainty-Routed P2 residual refinement.**

Goal: reuse shallow P2 detail without adding a fourth dense P2 detector. The stock P3 head first produces raw DFL regression logits and class logits. Their normalized DFL entropy defines localization uncertainty. P2 detail is downsampled to P3 resolution and mixed with P3 semantics; only the uncertainty-gated residual is added back to the P3 raw logits.

### V1 equations

For P3 raw DFL distribution `p_e(k)` for edge `e`:

`U = mean_e[-sum_k p_e(k) log p_e(k) / log(reg_max)]`.

Soft route:

`G = sigmoid((detach(U) - tau) / T)`.

Detail/context:

`D2 = proj_stride2(P2)`, `S3 = proj(P3)`, `R = phi([D2,S3,|D2-S3|,G])`.

Residual output:

`reg' = reg + G * 0.25 * tanh(delta_reg(R))`

`cls' = cls + G * delta_cls(R)`.

The delta heads are zero initialized, so insertion starts from the exact stock P3 logits.

### Insertion

Implement a small `Detect` subclass in the local Ultralytics runtime. It must:

1. receive stock P2 plus the normal P3/P4/P5 Detect inputs;
2. compute the stock P3 raw box/class logits once;
3. call `URP2Refiner(P2, P3, p3_reg, p3_cls)`;
4. replace only the P3 raw logits with refined logits;
5. leave P4/P5, decode, DFL, loss and NMS semantics stock.

V1 is **dense-compute soft routing**. It does not claim sparse runtime acceleration. Sparse gather/scatter is a later engineering stage only if the scientific mechanism is positive.

Expected primary signature: mAP50-95 and mAP50, with router statistics recorded. Recall may rise, but UR-P2 cannot guarantee recovery of targets for which stock P3 produces no useful candidate.

## 4. Module B — SDAS

**SDAS = Shallow Detail Auxiliary Supervision.**

Goal: give backbone P2 direct GT supervision during scratch training while preserving the stock inference graph.

Attach `SDASHead` to P2 only in training. It predicts a one-channel GT-center heatmap. Add:

`L_total = L_yolo + lambda_sdas * L_center`, default `lambda_sdas=0.25`.

V1 supervises **all GT centers**. There is intentionally no COCO-small/area gate because the earlier size statistics are not trusted under the current seed79 evidence policy.

At eval/export, the SDAS head must not participate in inference and must be removable from the exported graph. The stock Detect inputs remain P3/P4/P5.

Expected primary signature: Recall / mAP50 improvement with zero deployed cost.

## 5. Module C — WCRF

**WCRF = Water-Context Residual Fusion.**

Goal: use water-background context without copying MSCA/Inception-style parallel kernels or generic channel/spatial attention.

For input P3 feature `F`:

- `L = local(F)` from depthwise 3x3 + projection;
- `C = context(F)` from average context aggregation + dilated depthwise 3x3 + projection;
- `D = C - L`;
- `G = sigmoid(psi([L,C,|D|]))`;
- `Y = F + gamma * out(L + G*D)`.

`gamma` is initialized to zero so the block starts as an identity mapping.

### Insertion

Use **Detect-only P3 side branch** after the final fused P3 C3k2. The untouched original P3 must continue into the normal bottom-up PAN P4/P5 path. Detect receives `[WCRF(P3), P4, P5]`.

Expected signature: improved discrimination/quality on the P3 detection path without contaminating PAN features.

## 6. Seven structure variants

The exact contracts are machine-readable in `variants.py`.

| Key | Modules | Structure |
|---|---|---|
| U | UR-P2 | P2 detail + stock P3 DFL uncertainty -> P3 raw-logit residual refinement |
| S | SDAS | P2 train-only center supervision; stock inference |
| W | WCRF | WCRF only on Detect-side P3 branch |
| US | SDAS + UR-P2 | SDAS-supervised P2 also supplies UR-P2 detail |
| UW | WCRF + UR-P2 | WCRF(P3) semantic branch -> UR-P2, with stock P2 detail |
| SW | SDAS + WCRF | P2 aux supervision + Detect-only WCRF(P3) |
| USW | SDAS + WCRF + UR-P2 | SDAS(P2) during train; WCRF(P3) -> UR-P2 at Detect; P4/P5 stock |

### Combination order is fixed

- SDAS acts on stock P2 during training.
- WCRF acts on the Detect-only P3 branch.
- If UR-P2 and WCRF coexist, the order is `WCRF(P3) -> UR-P2`; UR-P2 receives stock P2 detail and WCRF-refined P3 semantics.
- P4/P5 are never modified by these modules in V1.

## 7. Required local Ultralytics files

Codex should port the research code into isolated files in the current Ultralytics 8.4.113 copy, e.g.:

- `ultralytics/nn/modules/flow_refine.py`: `URP2Refiner`, `WCRF`;
- `ultralytics/nn/modules/flow_aux.py`: `SDASHead` + target/loss helpers;
- `ultralytics/nn/modules/head.py`: add an isolated `URP2Detect(Detect)` subclass, do not alter stock `Detect` behavior;
- `ultralytics/nn/tasks.py`: register WCRF/URP2Detect and any required multi-input parse rule;
- `ultralytics/utils/loss.py`: add SDAS loss only when the model exposes SDAS training output;
- model YAML variants copied from the stock `yolo11n.yaml` and minimally patched.

Do not edit the frozen B0 runtime snapshot in place. Work in a copied runtime/branch.

## 8. Build gates before training

For every variant:

1. model build and parameter count;
2. random `(1,3,640,640)` forward;
3. one GT batch and one empty-GT batch forward/backward;
4. AMP finite;
5. graph-source audit (especially P3 PAN branch vs Detect-only branch);
6. 1-epoch/mini-dataset smoke train and Val;
7. predict;
8. TorchScript and ONNX export parity;
9. Params, GFLOPs, peak VRAM and batch-1 P50/P95.

Additional gates:

- U/US/UW/USW: at initialization P3 refined raw logits must equal stock P3 raw logits within numerical tolerance because the residual heads are zero initialized. Log mean DFL uncertainty, mean gate and gate>=0.5 fraction.
- S/US/SW/USW: exported inference graph must contain no SDAS output/head.
- W/UW/SW/USW: untouched P3 must still feed PAN P4/P5; only the Detect-side P3 branch uses WCRF.

## 9. Training order and governance

Run the three singles first: `U`, `S`, `W`.

Do not automatically launch combinations until all three single-module runs have completed and their Val/Test results are recorded. The pair/triple YAMLs should be built and smoke-tested in advance, but full combination training is a second step. This avoids spending four extra full runs before learning whether the individual mechanisms are viable.

When combination training is approved, use order: `US`, `UW`, `SW`, `USW`.

Each run uses exactly the frozen seed79 protocol. Do not tune module defaults on Test. Any later module hyperparameter change creates a new explicitly named experiment and must be selected from Train/Val only.

## 10. Metrics

At minimum report against saved B0:

- Precision, Recall;
- mAP50, mAP50-95;
- available AP75/scale metrics from the current evaluator (do not import old incompatible scale statistics as project facts);
- Params, GFLOPs, VRAM;
- batch-1 P50/P95 end-to-end latency;
- UR-P2 route statistics for U-containing variants;
- SDAS auxiliary loss curve for S-containing variants.

The research question is whether shallow evidence can be used more selectively and more efficiently than simply scaling YOLO11n to YOLO11s or copying an existing dense-P2 + attention stack.
