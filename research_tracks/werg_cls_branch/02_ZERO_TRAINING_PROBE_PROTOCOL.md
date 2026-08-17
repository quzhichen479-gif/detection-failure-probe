# WERG v0 — Frozen-Detector / Zero-Training Probe Protocol

## 1. Why this gate is mandatory

The project already has multiple examples where plausible modules did not transfer into stable detection gains. WERG must therefore prove that its statistics contain information not already captured by the accepted semantic classifier before a long training run is allowed.

This probe freezes the accepted DBRA detector. It does **not** use SRB-IoU and does not change any detector weight.

## 2. Candidate taxonomy

Build candidate records from the frozen accepted DBRA validation predictions and GT matching. At minimum distinguish:

```text
TP_opaque
TP_transparent_or_highly_transmissive   # manual subset if the dataset supports identification
FP_glare
FP_reflection
FP_wave
FP_foam
FP_shore_or_vegetation
FP_other
```

Do not silently infer `transparent` from class name if the dataset cannot support it. If transparent-object identity cannot be determined reliably, mark the dedicated transparent hypothesis as `not testable` rather than inventing labels.

## 3. Extraction bundle schema

The reference probe accepts a torch `.pt` dictionary:

```python
{
  "samples": [
    {
      "id": "image_or_frame_id",
      "p3_feature": Tensor[C,H3,W3],
      "rgb": Tensor[3,H,W],
      "semantic_logit": Tensor[H3,W3] or Tensor[nc,H3,W3],
      "candidates": [
        {
          "y": 17,
          "x": 31,
          "label": 1,                    # object=1, water/background=0
          "group": "TP_transparent",
          "probe_split": "fit"          # fit | eval | unassigned
        }
      ]
    }
  ]
}
```

`p3_feature` must be the exact accepted DBRA P3 classification feature selected for WERG insertion. `semantic_logit` must be the corresponding pre-sigmoid P3 class score map, not post-NMS confidence.

For multiclass logits, the reference script uses the max class logit at the candidate cell for the generic object-vs-water probe. Class-specific analysis can be added to the report but must not replace the generic gate.

## 4. Running the reference probe

```bash
cd research_tracks/werg_cls_branch/reference_code
python werg_probe.py \
  --bundle /ABS/PATH/werg_probe_bundle.pt \
  --out-dir /ABS/PATH/werg_probe_output \
  --device cuda
```

Outputs:

```text
werg_candidates.csv
werg_probe_summary.json
```

If candidates are explicitly split into `fit` and `eval`, the script also fits two tiny frozen-detector logistic probes:

```text
A: semantic_logit only
B: semantic_logit + z_w + z_g + z_w^2 + z_w*z_g + z_g^2
```

This probe fitting is not detector training. Evaluation must occur on the held-out `eval` candidates; do not fit and report on the same candidates.

## 5. Required visual audit

For at least 50 representative cases across the taxonomy, save overlays containing:

```text
RGB
accepted DBRA prediction / GT
z_w heatmap
z_g heatmap
candidate point
error group
```

Specific theoretical checks:

1. smooth water/waves should not produce a systematic high `z_w` everywhere;
2. uniform affine brightness change without clipping should leave `z_g` approximately stable;
3. if a reliable transparent subset exists, curved/edge regions should show more `z_g` evidence than flat transparent interiors and nearby water;
4. sharp saturated glitter is allowed to remain a hard negative and must be reported, not hidden.

## 6. Pre-registered pass gates

WERG may proceed to detector integration only if all applicable gates pass:

```text
Gate A — numerical sanity:
all evidence finite; invariance/unit tests pass.

Gate B — conditional discrimination:
on held-out eval candidates, semantic+WERG must reduce water-clutter FP at the same target recall by >= 10% relative to semantic-only,
OR improve candidate AUROC by >= 0.02 with no worse fixed-recall FP.

Gate C — cross-group usefulness:
the gain cannot come only from one tiny hand-picked FP subtype; at least two of glare/reflection/wave/foam/shore groups must show a non-adverse direction where sample size is adequate.

Gate D — transparent hypothesis (only if labelable):
z_g must add measurable discrimination for transparent TP vs local water/background. If not labelable, mark this gate N/A; do not claim transparent-object validation.
```

The thresholds are project decision rules, not universal constants.

## 7. Immediate stop conditions

Stop WERG v0 and write `WERG_ZERO_TRAINING_NEGATIVE_REPORT.md` if:

- `z_w` is dominated by ordinary water roughness after whitening/projection;
- `z_g` changes strongly under non-clipping `I -> aI+b` controls;
- semantic+WERG cannot beat semantic-only on held-out candidates;
- apparent improvement disappears when near-duplicate frames/scenes are grouped;
- gain depends on manually relabeling ambiguous negatives in WERG's favor.

Do not respond to a failed probe by adding P2, attention, FreqFusion, a new IoU, SRB-IoU, extra loss terms or a parameter sweep.

## 8. Leakage control

Candidate `fit`/`eval` division must be group-aware whenever source frames or scenes are correlated. The split unit should be sequence/scene/group rather than individual candidate cells whenever metadata permits.
