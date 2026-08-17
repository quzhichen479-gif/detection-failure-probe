# WERG v0 — Water-Explainable Residual Geometry for YOLO11 P3 Classification

> Status: **research implementation / falsification-first**  
> Target parent: accepted YOLO11n + accepted DBRA @ P3-Cls-Mid  
> Ultralytics target used by the project: 8.4.113  
> Dataset/protocol: reuse the already frozen project protocol; do not retrain baseline or accepted DBRA to start this track.

## 0. Hard boundary

**SRB-IoU has been falsified in the project and is explicitly excluded from WERG.**

Do not combine WERG with SRB-IoU. Do not reopen SRB-IoU, tune it, or use it as a comparison parent for this track. WERG v0 changes only the P3 classification side. Regression, DFL, TAL and box loss remain exactly as in the accepted parent model.

## 1. What WERG is trying to test

Wavelet/high-frequency assumptions are too weak for this task because floating objects, glitter, foam and fine waves can occupy overlapping frequency bands, while transparent bottles may mainly warp the background instead of introducing new spectral energy.

WERG tests a narrower hypothesis:

1. normal local water appearance can be approximately continued from an annular neighborhood by a low-order differential model;
2. water-like illumination/wave changes occupy a low-dimensional local nuisance tangent space in P3 features;
3. transparent/refractive objects can introduce gain-invariant local deformation-shape changes that are not equivalent to a scalar brightness gain;
4. these two statistics contain information conditional on the accepted semantic classifier, especially for water-clutter false positives and transparent-object true positives.

The two evidence maps are:

- `z_w`: water-explainable residual statistic in the accepted P3 classification feature;
- `z_g`: gain-invariant deformation-shape statistic computed from the normalized RGB input.

They do **not** hard-filter predictions. A six-parameter, zero-initialized bounded correction adjusts only P3 pre-sigmoid class logits.

## 2. Repository layout

```text
research_tracks/werg_cls_branch/
├── README.md
├── 00_THEORY_AND_DERIVATION.md
├── 01_IMPLEMENTATION_SPEC.md
├── 02_ZERO_TRAINING_PROBE_PROTOCOL.md
├── 03_CODEX_YOLO11_INTEGRATION.md
└── reference_code/
    ├── werg_core.py
    ├── werg_probe.py
    ├── werg_yolo_adapter.py
    └── test_werg_core.py
```

The files under `reference_code/` are a framework-agnostic PyTorch reference. They are intentionally not placed in `detection-failure-probe/src/`, because the actual detector implementation belongs in the already existing YOLO11 worktree that contains the accepted DBRA experiment.

## 3. Required execution order

```text
A. Read 00_THEORY_AND_DERIVATION.md
B. Read 01_IMPLEMENTATION_SPEC.md
C. Run/reference test_werg_core.py
D. Run the frozen-detector probe in 02_ZERO_TRAINING_PROBE_PROTOCOL.md
E. Only if the probe passes, read 03_CODEX_YOLO11_INTEGRATION.md and wire the P3 adapter into the accepted DBRA parent
F. Run integration gates and a single registered WERG training
```

Do not skip D. WERG is a mechanism hypothesis, not a module that is entitled to a full training run.

## 4. Reference-code quick check

In an environment that already has PyTorch:

```bash
cd research_tracks/werg_cls_branch/reference_code
pytest -q test_werg_core.py
python -c "from werg_core import synthetic_sanity_check; print(synthetic_sanity_check())"
```

Expected invariants:

- all statistics finite and non-negative;
- a center impulse produces larger `z_w` than flat water;
- determinant-normalized geometry is approximately invariant to `I -> a I + b` when no clipping occurs;
- anisotropic structure-tensor deformation gives positive `D_G^2`;
- zero-initialized WERG correction is bitwise identical to the parent logits.

## 5. Minimal calling interface

```python
from werg_core import WERGConfig
from werg_yolo_adapter import P3WERGClassificationAdapter

adapter = P3WERGClassificationAdapter(
    p3_channels=C_P3_CLS_MID,
    config=WERGConfig(detach_statistics=True),
)

out = adapter(
    p3_cls_feature=p3_mid,      # accepted DBRA P3 classification feature
    p3_cls_logits=p3_logits,    # pre-sigmoid P3 class logits
    rgb=normalized_input_rgb,   # same normalized RGB tensor entering YOLO
)

p3_logits = out.logits
```

P4/P5 logits are unchanged. The regression branch is unchanged.

## 6. Codex one-shot start prompt

```text
Use https://github.com/quzhichen479-gif/detection-failure-probe as the WERG research/spec source. Read research_tracks/werg_cls_branch/README.md first, then 00_THEORY_AND_DERIVATION.md, 01_IMPLEMENTATION_SPEC.md, 02_ZERO_TRAINING_PROBE_PROTOCOL.md and 03_CODEX_YOLO11_INTEGRATION.md in order, together with all files under reference_code/. The implementation target is the existing local YOLO11 worktree that already contains the accepted DBRA P3-Cls-Mid experiment; do not implement detector code into detection-failure-probe/src. First record YOLO worktree path, git branch/commit/dirty status, Ultralytics version, accepted DBRA module/YAML/checkpoint and frozen training/evaluation args. SRB-IoU is already falsified: do not combine, revive, tune or benchmark SRB-IoU in this track. Implement WERG v0 only on the P3 classification side: z_w from the accepted DBRA P3 classification feature via annular quadratic continuation + local variance whitening + tangent-space residual; z_g from the same normalized RGB input via determinant-normalized structure-tensor center/ring deformation distance; use the six-parameter zero-initialized bounded shared logit correction. P4/P5, regression, DFL, TAL and loss remain unchanged. Before any long training, run the frozen-detector WERG probe and the registered unit/shape/finite/identity/photometric-invariance tests. If the probe fails the pre-registered gates, stop the WERG track and write a negative report; do not rescue it with extra attention, P2, neck changes, new loss or parameter search. If it passes, perform the exact integration gates in 03_CODEX_YOLO11_INTEGRATION.md, diff resolved args against the accepted DBRA parent, then launch exactly one registered WERG-v0 training under the frozen protocol and write the integration/launch reports.
```

## 7. Non-goals

WERG v0 is not claimed to reconstruct water reflection, recover invisible transparent objects, segment water, or solve saturated glitter. Sharp saturated glitter and foam remain explicit hard negatives. If the frozen probe shows that `z_w`/`z_g` add no conditional information beyond the accepted semantic logit, the correct action is to stop this track.
