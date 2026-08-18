"""Audit whether fixed Inner-EIoU(r=0.8) prematurely removes auxiliary overlap.

Input CSV must contain aligned matched prediction/GT boxes from frozen validation:

    pred_x1,pred_y1,pred_x2,pred_y2,gt_x1,gt_y1,gt_x2,gt_y2

Optional extra columns are preserved only in the source CSV and ignored here.
This script does not train, load a model, or alter predictions.

Example:

    python research_tracks/opc_inner_eiou/audit_overlap_risk.py \
        --input matched_val_boxes.csv \
        --output outputs/opc_overlap_audit.json
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch
from opc_inner_eiou import bbox_iou_xyxy, opc_ratio, scale_boxes_about_center


REQUIRED_COLUMNS = (
    "pred_x1",
    "pred_y1",
    "pred_x2",
    "pred_y2",
    "gt_x1",
    "gt_y1",
    "gt_x2",
    "gt_y2",
)


def _load_boxes(path: Path) -> tuple[torch.Tensor, torch.Tensor]:
    pred_rows: list[list[float]] = []
    gt_rows: list[list[float]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("input CSV has no header")
        missing = [name for name in REQUIRED_COLUMNS if name not in reader.fieldnames]
        if missing:
            raise ValueError(f"input CSV is missing required columns: {missing}")
        for row in reader:
            pred_rows.append([float(row[name]) for name in REQUIRED_COLUMNS[:4]])
            gt_rows.append([float(row[name]) for name in REQUIRED_COLUMNS[4:]])

    if not pred_rows:
        raise ValueError("input CSV contains no matched boxes")
    pred = torch.tensor(pred_rows, dtype=torch.float64)
    target = torch.tensor(gt_rows, dtype=torch.float64)
    return pred, target


def _fraction(mask: torch.Tensor) -> float:
    return float(mask.double().mean())


def _percentiles(values: torch.Tensor) -> dict[str, float]:
    q = torch.tensor([0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99], dtype=values.dtype)
    result = torch.quantile(values, q)
    names = ("p01", "p05", "p25", "p50", "p75", "p95", "p99")
    return {name: float(value) for name, value in zip(names, result, strict=True)}


def audit(pred: torch.Tensor, target: torch.Tensor, r0: float = 0.8) -> dict[str, object]:
    ratio, u, u_x, u_y = opc_ratio(pred, target, r0=r0)
    original_iou = bbox_iou_xyxy(pred, target)
    l3_inner_iou = bbox_iou_xyxy(
        scale_boxes_about_center(pred, r0),
        scale_boxes_about_center(target, r0),
    )
    opc_inner_iou = bbox_iou_xyxy(
        scale_boxes_about_center(pred, ratio),
        scale_boxes_about_center(target, ratio),
    )

    safe = u <= r0
    transition = (u > r0) & (u < 1.0)
    nonoverlap_state = u >= 1.0
    original_overlap = original_iou > 0.0
    l3_collapsed = original_overlap & (l3_inner_iou <= 0.0)
    opc_collapsed = original_overlap & (opc_inner_iou <= 0.0)

    return {
        "n_matched_boxes": int(pred.shape[0]),
        "r0": r0,
        "fractions": {
            "u_le_r0_safe": _fraction(safe),
            "r0_lt_u_lt_1_transition": _fraction(transition),
            "u_ge_1_original_nonoverlap_state": _fraction(nonoverlap_state),
            "original_positive_iou": _fraction(original_overlap),
            "l3_inner_overlap_collapsed_while_original_overlaps": _fraction(l3_collapsed),
            "opc_inner_overlap_collapsed_while_original_overlaps": _fraction(opc_collapsed),
        },
        "counts": {
            "u_le_r0_safe": int(safe.sum()),
            "r0_lt_u_lt_1_transition": int(transition.sum()),
            "u_ge_1_original_nonoverlap_state": int(nonoverlap_state.sum()),
            "l3_inner_overlap_collapsed_while_original_overlaps": int(l3_collapsed.sum()),
            "opc_inner_overlap_collapsed_while_original_overlaps": int(opc_collapsed.sum()),
        },
        "u_percentiles": _percentiles(u),
        "u_x_percentiles": _percentiles(u_x),
        "u_y_percentiles": _percentiles(u_y),
        "opc_ratio_percentiles": _percentiles(ratio),
        "mean_iou": {
            "original": float(original_iou.mean()),
            "l3_inner_r0": float(l3_inner_iou.mean()),
            "opc_inner": float(opc_inner_iou.mean()),
        },
        "interpretation_gate": {
            "recommended_min_transition_fraction_for_training": 0.01,
            "note": (
                "If the transition band is almost absent, OPC changes too few matched samples "
                "to justify another full training run. This is a preregistered engineering gate, "
                "not a statistical significance threshold."
            ),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--r0", type=float, default=0.8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pred, target = _load_boxes(args.input)
    report = audit(pred, target, r0=args.r0)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
