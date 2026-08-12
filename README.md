# Detection Failure Probe

Local-first dataset auditing and prediction failure analysis for YOLO object detection.

Detection Failure Probe helps researchers answer three practical questions before changing a
model: **is the dataset structurally sound, what kinds of detection failures occurred, and how
small do the labeled objects become at candidate input resolutions?** It runs on CPU and does not
upload images, labels, predictions, or reviewer notes.

> Resolution survival is a geometric diagnostic. It does **not** predict accuracy, recall, or
> model performance.

## Quick start: install → run demo → inspect

Python 3.10 or newer is required.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -e .
```

Run the complete demo analysis:

```bash
failure-probe analyze \
  --dataset demo/dataset/dataset.yaml \
  --predictions demo/predictions.json \
  --run-name demo_analysis
```

The command prints its run directory and a TP/FP/FN summary. Inspect the static report, or start
the loopback-only visual reviewer:

```bash
failure-probe report runs/demo_analysis
failure-probe review runs/demo_analysis
```

Open `runs/demo_analysis/report.html` directly for the static report. The review command opens a
local page with GT/prediction overlays, failure/class/confidence filters, audit flags, and reviewer
notes. Stop it with `Ctrl+C`.

The four tiny demo images are committed to the repository. To regenerate them deterministically:

```bash
python scripts/generate_demo.py
```

## Commands

```bash
# Dataset-only audit (also computes resolution diagnostics)
failure-probe audit dataset.yaml

# Dataset audit + prediction failure analysis
failure-probe analyze --dataset dataset.yaml --predictions predictions.json

# Local visual review; listens on 127.0.0.1 by default
failure-probe review runs/audit_YYYYMMDD_HHMMSS

# Rebuild a self-contained HTML summary
failure-probe report runs/audit_YYYYMMDD_HHMMSS
```

Useful options:

```bash
failure-probe audit dataset.yaml --resolutions 320 640 1024
failure-probe analyze --dataset dataset.yaml --predictions predictions.json \
  --match-iou 0.5 --localization-iou 0.1
failure-probe review runs/demo_analysis --no-browser --port 8765
```

Every invocation creates a new directory. If a requested `--run-name` exists, a suffix such as
`_01` is added; existing results are never silently overwritten.

## YOLO dataset input

The MVP reads standard five-column detection labels:

```text
class_id x_center y_center width height
```

Coordinates are normalized. The YAML can use `train`, `val`, and `test`, or the compact `images`
key. Image sources may be directories, individual supported images, or UTF-8 text file lists.

```yaml
path: .
train: images/train
val: images/val
names:
  0: person
  1: vehicle
```

For a path such as `images/train/frame.jpg`, the corresponding label is
`labels/train/frame.txt`. An optional `labels: labels` key can set one explicit label root.

For a safe default, every YAML-referenced path must stay under the directory containing the YAML.
Absolute external dataset roots and `..` traversal are rejected. Put the YAML at a common parent
of images and labels, or create a non-executable YAML wrapper there.

## Prediction JSON input

The documented format is intentionally small and explicit:

```json
{
  "bbox_format": "xywh",
  "normalized": false,
  "images": [
    {
      "image": "images/frame_001.jpg",
      "predictions": [
        {"class_id": 0, "bbox": [120, 80, 64, 96], "confidence": 0.91}
      ]
    }
  ]
}
```

- `bbox_format` is `xywh` (top-left, width, height) or `xyxy`.
- Coordinates are pixels unless `normalized` is `true`.
- `score` and `category_id` are accepted aliases for `confidence` and `class_id`.
- A flat list of prediction objects is also accepted when every object has an `image` field.
- Image references may be dataset-relative paths or unique basenames. Traversal is rejected.

The MVP does not load model checkpoints or pickle files. Convert framework-specific prediction
objects to JSON before analysis.

## What is measured

Dataset audit:

- class and per-image distribution;
- pixel width, height, area, aspect ratio, and COCO-style scale buckets;
- small-object ratio at native resolution (`area < 32²` pixels);
- malformed, non-finite, out-of-range, non-positive, and out-of-image boxes;
- exact and same-class near-duplicate annotations;
- missing labels, orphan labels, empty-label images, unreadable images;
- suspicious boxes narrower/shorter than 2 pixels or with aspect ratio at least 10:1.

Prediction analysis:

- TP, FP, and FN from confidence-ordered class-aware greedy IoU matching;
- localization errors, classification errors, duplicate detections, and background false positives;
- breakdowns by class, confidence bucket, and object scale.

An unmatched prediction is always an FP, even when assigned a specific failure subtype. An
unmatched GT is always an FN. This means a classification or localization error can contribute one
FP and one FN. The JSON artifact records the exact matching thresholds and method.

Resolution survival:

- estimated pixel width, height, and area after aspect-preserving resize into each square input;
- minimum-side buckets and survival ratios at 1, 2, 4, 8, and 16 pixels;
- per-object records for downstream research analysis.

Padding is assumed not to change object size. These values describe geometry only.

## Python API

Core logic is independent of the CLI:

```python
from failure_probe import (
    analyze_predictions,
    audit_dataset,
    load_dataset,
    resolution_survival,
)

dataset = load_dataset("dataset.yaml")
audit = audit_dataset(dataset)
failures = analyze_predictions(dataset, "predictions.json", match_iou=0.5)
survival = resolution_survival(dataset, [320, 640, 1024])
```

These functions return JSON-serializable dictionaries. `failure_probe.workflow.run_audit` and
`run_analysis` add run-directory persistence and report generation.

## Run artifacts

```text
runs/analysis_.../
├── .failure-probe-run
├── manifest.json
├── audit.json
├── analysis.json        # analysis runs only
├── resolution.json
├── reviewer_notes.json
└── report.html
```

## Privacy and security model

- No network client, telemetry, analytics, or upload path exists in the package.
- YAML uses `yaml.safe_load`; dataset content is never executed.
- Predictions use JSON only. Pickle and model checkpoint loading are out of scope.
- Referenced dataset paths are resolved beneath a trusted dataset root.
- Run names are allowlisted, runs are created exclusively, and artifacts are atomically written.
- The review UI binds only to loopback, uses a per-process random token, serves only allowlisted
  dataset images, and applies browser security headers.

The tool still reads user-selected local datasets and serves selected images to the same machine's
browser while review is running. See [SECURITY.md](SECURITY.md) for the precise attack surface and
reporting process.

## Current scope

This is a deliberately small alpha. It supports YOLO detection labels and the documented JSON
prediction format. It does not yet support segmentation/keypoints, remote URLs, archive extraction,
video, distributed datasets, COCO JSON import, dataset mutation, or model inference. Near-duplicate
**annotations** are detected; image-content deduplication is not part of the MVP.

## Development

```bash
pip install -e ".[dev]"
pytest
ruff check .
```

Contributions are welcome. Please read [CONTRIBUTING.md](CONTRIBUTING.md) and
[SECURITY.md](SECURITY.md).

## License

[MIT](LICENSE)
