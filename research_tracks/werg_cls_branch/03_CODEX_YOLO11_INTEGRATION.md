# WERG v0 — Codex YOLO11 Integration Plan

## 0. Repository boundary

Research/spec source:

```text
https://github.com/quzhichen479-gif/detection-failure-probe
research_tracks/werg_cls_branch/
```

Implementation target:

```text
the existing local YOLO11 worktree that already contains the accepted DBRA P3-Cls-Mid experiment
```

Do not put production detector code into `detection-failure-probe/src/`.

## 1. Before touching code, record the accepted parent

Codex must write a preflight record containing:

```text
YOLO worktree absolute path
git branch
git commit
git dirty status
Ultralytics version (project target: 8.4.113)
accepted DBRA implementation path
accepted DBRA YAML path
accepted DBRA checkpoint path
accepted DBRA P3-Cls-Mid tensor location and channel count
accepted baseline/DBRA resolved training args source
accepted evaluator command/config
```

Do not guess DBRA APIs from the spec repository. Inspect the actual accepted implementation that produced the positive result.

## 2. Explicitly forbidden changes

For WERG v0, do not change:

```text
DBRA definition or insertion point
P4/P5 classification paths
any regression branch
DFL
TAL/assigner
box loss
classification loss definition
P2 head
neck topology
input resolution
data split
optimizer/scheduler
augmentation
training length/early-stop policy
pretrained initialization policy
```

**SRB-IoU is already falsified. Do not combine, revive, tune or compare a DBRA+WERG+SRB-IoU model.**

## 3. Production files to create in the YOLO worktree

Exact paths may follow the existing project conventions, but the implementation should separate the mathematical core from Ultralytics wiring. Suggested layout:

```text
ultralytics/nn/modules/werg.py
ultralytics/nn/modules/werg_detect.py          # only if a custom Detect subclass is required
ultralytics/cfg/models/11/<accepted-dbra>-werg.yaml
tests/test_werg_math.py
tests/test_werg_detect_integration.py
scripts/werg_probe_extract.py
```

Copy/adapt from this track's `reference_code/` rather than re-deriving formulas during implementation.

## 4. Raw RGB plumbing requirement

`z_g` must use the same normalized RGB tensor that enters the detector. The accepted Ultralytics Detect head normally receives feature maps, not raw input, so this requires an explicit and auditable side path.

Preferred integration hierarchy:

1. **custom DetectionModel/DetectWERG explicit input sidecar** that passes the original normalized RGB reference to the WERG P3 classification adapter;
2. if the local architecture already has a clean context carrier mechanism, reuse it;
3. do not use global mutable state, hidden forward hooks as production behavior, or a reconstructed image from P3 features.

The zero-training probe may use hooks to extract tensors, but the trained production graph must make the RGB dependency explicit.

If explicit RGB plumbing would require an unsafe rewrite of the accepted model graph, stop and report the integration blocker. Do not silently replace RGB `z_g` with a neural feature proxy in v0.

## 5. P3 classification wiring

Locate the exact accepted DBRA P3-Cls-Mid feature `f3` and its parent pre-sigmoid logits `l3`.

Pseudo-flow:

```python
l3_parent = parent_p3_cls_predictor(f3)

werg_out = self.werg_p3(
    p3_cls_feature=f3,
    p3_cls_logits=l3_parent,
    rgb=raw_normalized_rgb,
)

l3 = werg_out.logits
l4 = parent_l4
l5 = parent_l5
```

Do not place WERG before DBRA in v0. The mechanism probe and trained model must use the same P3 feature definition.

## 6. State-dict migration

Start from the accepted DBRA parent checkpoint.

Expected new trainable keys are only the six WERG correction coefficients (plus no trainable mathematical-statistic buffers). Fixed kernels are buffers and may appear as new state keys.

Perform a key-by-key migration audit:

```text
[ ] every parent trainable parameter loaded exactly
[ ] no parent key silently dropped
[ ] only documented WERG keys are new
[ ] WERG six coefficients initialize to zero
[ ] fixed annular/Sobel/Gaussian kernels have expected values/shapes
```

Immediately after load, before any optimizer step, compare parent vs WERG model on the same input:

```text
P3 logits: bitwise equal or exact within dtype semantics because delta=0
P4/P5 logits: exactly equal
box outputs: exactly equal
postprocess output: equal under deterministic settings
```

If zero-init equivalence fails, do not train.

## 7. Gradient audit

With `detach_statistics=True`:

```text
[ ] six correction coefficients receive gradients
[ ] WERG fixed kernels/buffers do not require grad
[ ] z_w/z_g do not send extra gradient into backbone/RGB path
[ ] accepted DBRA and parent YOLO parameters receive only their original loss gradients through the original logits
```

The corrected logits still affect the ordinary classification loss, so parent classifier gradients may change after WERG coefficients move away from zero. The important boundary is that the statistic computation itself is detached.

## 8. Long-training gates

All must pass before one full run:

```text
[ ] frozen-detector probe passes 02_ZERO_TRAINING_PROBE_PROTOCOL.md
[ ] reference unit tests ported and passing
[ ] build/forward finite
[ ] zero-init parent equivalence
[ ] gradient audit
[ ] AMP forward/backward finite
[ ] 1-epoch smoke train
[ ] smoke val
[ ] smoke predict
[ ] export behavior either validated or explicitly marked unsupported for v0
[ ] Params/GFLOPs/VRAM/batch-1 P50/P95 latency reported
[ ] resolved training args diff vs accepted DBRA contains only model/run/output differences
```

Write `WERG_V0_INTEGRATION_REPORT.md` before the full run.

## 9. Exactly one first registered training

Initial comparison:

```text
A0 accepted YOLO11n baseline                 reuse
A1 accepted YOLO11n + DBRA P3-Cls-Mid       reuse
A2 accepted DBRA + WERG-v0 P3 classification train exactly once
```

Do not retrain A0/A1 just to start WERG. Do not launch multi-seed or parameter search before seeing the registered A2 validation result.

Allowed differences from A1:

```text
model architecture/YAML necessary for WERG
run name
output directory
WERG fixed v0 constants
six new zero-initialized correction parameters
```

Everything else must be restored from accepted A1 artifacts and diffed structurally.

After launch, record exact command, resolved args, commit, source hashes, run dir and log path in `WERG_V0_TRAIN_LAUNCH.md`.

## 10. First-result decision

Primary WERG mechanism metrics are not only total mAP:

```text
water-clutter FP at fixed recall
Precision / Recall
AP_tiny / small if available
glare/reflection/wave/foam/shore FP groups
transparent subset if and only if reliably labeled
P3 correction distribution delta_logit
z_w / z_g distributions on TP and FP
latency / VRAM
```

If total mAP rises but water-clutter discrimination does not improve, the WERG mechanism claim is not supported.

If WERG is negative, stop the track and preserve it as a falsified mechanism. Do not combine it with SRB-IoU or rescue it by stacking another module.
