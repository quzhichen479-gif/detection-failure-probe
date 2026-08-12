"""Prediction matching and interpretable failure categorization."""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from failure_probe.dataset import load_dataset
from failure_probe.errors import DatasetFormatError
from failure_probe.geometry import Box, area, iou, scale_bucket, xywh_to_xyxy
from failure_probe.models import Dataset, ImageRecord
from failure_probe.paths import resolve_within

MAX_PREDICTIONS_BYTES = 100_000_000


@dataclass(frozen=True)
class Prediction:
    """Validated prediction in pixel xyxy coordinates."""

    class_id: int
    box: Box
    confidence: float


def analyze_predictions(
    dataset: str | Path | Dataset,
    predictions: str | Path,
    *,
    match_iou: float = 0.5,
    localization_iou: float = 0.1,
) -> dict[str, Any]:
    """Match predictions to GT and categorize common detection failures.

    TP/FP/FN use class-aware greedy matching. Every unmatched prediction is an
    FP and may additionally receive a more specific failure category. Every
    unmatched GT is an FN, including GT involved in a classification or
    localization error.
    """
    loaded = load_dataset(dataset) if not isinstance(dataset, Dataset) else dataset
    if not 0 <= localization_iou < match_iou <= 1:
        raise ValueError("Expected 0 <= localization_iou < match_iou <= 1")
    predictions_by_image = _load_predictions(loaded, predictions)

    totals: Counter[str] = Counter()
    failure_types: Counter[str] = Counter()
    class_metrics: dict[int, Counter[str]] = defaultdict(Counter)
    confidence_failures: dict[str, Counter[str]] = defaultdict(Counter)
    scale_metrics: dict[str, Counter[str]] = defaultdict(Counter)
    images_payload: list[dict[str, Any]] = []

    for image in loaded.images:
        if image.image_error:
            continue
        ground_truth = [annotation for annotation in image.annotations if annotation.valid]
        gt_boxes = [annotation.pixel_box(image.width, image.height) for annotation in ground_truth]
        image_predictions = predictions_by_image.get(image.relative_path, [])
        matches = _greedy_matches(ground_truth, gt_boxes, image_predictions, match_iou)
        matched_gt = {gt_index for gt_index, _ in matches}
        matched_predictions = {pred_index for _, pred_index in matches}
        pred_to_gt = {pred_index: gt_index for gt_index, pred_index in matches}

        serialized_gt: list[dict[str, Any]] = []
        for gt_index, annotation in enumerate(ground_truth):
            status = "tp" if gt_index in matched_gt else "fn"
            totals[status] += 1
            class_metrics[annotation.class_id][status] += 1
            bucket = scale_bucket(area(gt_boxes[gt_index]))
            scale_metrics[bucket][status] += 1
            serialized_gt.append(
                {
                    "class_id": annotation.class_id,
                    "line": annotation.line,
                    "bbox_xyxy": _round_box(gt_boxes[gt_index]),
                    "status": status,
                    "scale_bucket": bucket,
                }
            )

        serialized_predictions: list[dict[str, Any]] = []
        for pred_index, prediction in enumerate(image_predictions):
            if pred_index in matched_predictions:
                error_type = "tp"
                gt_index = pred_to_gt[pred_index]
                matched_iou = iou(prediction.box, gt_boxes[gt_index])
                totals["tp_predictions"] += 1
                class_metrics[prediction.class_id]["tp_predictions"] += 1
            else:
                totals["fp"] += 1
                class_metrics[prediction.class_id]["fp"] += 1
                error_type, gt_index, matched_iou = _failure_type(
                    prediction,
                    ground_truth,
                    gt_boxes,
                    matched_gt,
                    match_iou,
                    localization_iou,
                )
                failure_types[error_type] += 1
                confidence_failures[_confidence_bucket(prediction.confidence)][error_type] += 1
                scale_metrics[scale_bucket(area(prediction.box))][error_type] += 1
            serialized_predictions.append(
                {
                    "class_id": prediction.class_id,
                    "confidence": round(prediction.confidence, 6),
                    "bbox_xyxy": _round_box(prediction.box),
                    "status": error_type,
                    "matched_gt_index": gt_index,
                    "iou": round(matched_iou, 6),
                }
            )

        images_payload.append(
            {
                "image": image.relative_path,
                "width": image.width,
                "height": image.height,
                "ground_truth": serialized_gt,
                "predictions": serialized_predictions,
            }
        )

    per_class = []
    for class_id, name in loaded.names.items():
        metrics = class_metrics[class_id]
        tp = metrics["tp"]
        fp = metrics["fp"]
        fn = metrics["fn"]
        per_class.append(
            {
                "class_id": class_id,
                "name": name,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": round(tp / (tp + fp), 6) if tp + fp else None,
                "recall": round(tp / (tp + fn), 6) if tp + fn else None,
            }
        )
    tp = totals["tp"]
    fp = totals["fp"]
    fn = totals["fn"]
    return {
        "schema_version": 1,
        "method": {
            "matching": "confidence-ordered, class-aware greedy IoU matching",
            "match_iou": match_iou,
            "localization_iou": localization_iou,
            "note": (
                "Specific error categories annotate unmatched predictions; classification and "
                "localization errors can therefore contribute both one FP and one FN."
            ),
        },
        "summary": {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": round(tp / (tp + fp), 6) if tp + fp else None,
            "recall": round(tp / (tp + fn), 6) if tp + fn else None,
            "failure_types": dict(sorted(failure_types.items())),
        },
        "per_class": per_class,
        "confidence_failures": {
            bucket: dict(sorted(counts.items()))
            for bucket, counts in sorted(confidence_failures.items())
        },
        "scale_failures": {
            bucket: dict(sorted(counts.items())) for bucket, counts in sorted(scale_metrics.items())
        },
        "images": images_payload,
    }


def _load_predictions(dataset: Dataset, path: str | Path) -> dict[str, list[Prediction]]:
    prediction_path = Path(path).resolve(strict=True)
    if not prediction_path.is_file() or prediction_path.suffix.lower() != ".json":
        raise DatasetFormatError(f"Predictions must be an existing JSON file: {prediction_path}")
    if prediction_path.stat().st_size > MAX_PREDICTIONS_BYTES:
        raise DatasetFormatError("Predictions JSON exceeds the 100 MB safety limit")
    try:
        payload = json.loads(
            prediction_path.read_text(encoding="utf-8"),
            parse_constant=lambda value: _reject_json_constant(value),
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise DatasetFormatError(f"Could not parse predictions JSON: {exc}") from exc

    bbox_format = "xywh"
    normalized = False
    if isinstance(payload, dict):
        bbox_format = payload.get("bbox_format", "xywh")
        normalized = payload.get("normalized", False)
        entries = payload.get("images")
    else:
        entries = payload
    if bbox_format not in {"xywh", "xyxy"}:
        raise DatasetFormatError("bbox_format must be 'xywh' or 'xyxy'")
    if not isinstance(normalized, bool):
        raise DatasetFormatError("normalized must be true or false")
    if not isinstance(entries, list):
        raise DatasetFormatError(
            "Predictions must be a list or an object containing an 'images' list"
        )

    image_map = {image.relative_path: image for image in dataset.images}
    by_basename: dict[str, list[str]] = defaultdict(list)
    for key in image_map:
        by_basename[PurePosixPath(key).name].append(key)
    result: dict[str, list[Prediction]] = defaultdict(list)
    for entry_index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise DatasetFormatError(f"Prediction entry {entry_index} must be an object")
        if "predictions" in entry:
            image_value = entry.get("image")
            raw_predictions = entry["predictions"]
        else:
            image_value = entry.get("image")
            raw_predictions = [entry]
        image_key = _resolve_image_key(dataset, image_map, by_basename, image_value)
        if not isinstance(raw_predictions, list):
            raise DatasetFormatError(f"Predictions for {image_key} must be a list")
        image = image_map[image_key]
        for raw in raw_predictions:
            result[image_key].append(_parse_prediction(raw, image, bbox_format, normalized))
    for values in result.values():
        values.sort(key=lambda prediction: prediction.confidence, reverse=True)
    return dict(result)


def _resolve_image_key(
    dataset: Dataset,
    image_map: dict[str, ImageRecord],
    by_basename: dict[str, list[str]],
    value: Any,
) -> str:
    if not isinstance(value, str) or not value:
        raise DatasetFormatError("Every prediction entry needs a non-empty 'image' string")
    pure = PurePosixPath(value.replace("\\", "/"))
    if pure.is_absolute() or ".." in pure.parts:
        raise DatasetFormatError(f"Unsafe image path in predictions: {value}")
    normalized_key = pure.as_posix().lstrip("./")
    if normalized_key in image_map:
        return normalized_key
    matches = by_basename.get(pure.name, [])
    if len(matches) == 1:
        return matches[0]
    # This additionally checks symlink traversal before producing the error.
    resolve_within(dataset.root, normalized_key)
    raise DatasetFormatError(f"Prediction references an unknown or ambiguous image: {value}")


def _parse_prediction(
    raw: Any,
    image: ImageRecord,
    bbox_format: str,
    normalized: bool,
) -> Prediction:
    if not isinstance(raw, dict):
        raise DatasetFormatError(f"A prediction for {image.relative_path} is not an object")
    class_value = raw.get("class_id", raw.get("category_id"))
    confidence = raw.get("confidence", raw.get("score"))
    box = raw.get("bbox")
    if not isinstance(class_value, int) or isinstance(class_value, bool) or class_value < 0:
        raise DatasetFormatError("Prediction class_id/category_id must be a non-negative integer")
    if not isinstance(confidence, (int, float)) or isinstance(confidence, bool):
        raise DatasetFormatError("Prediction confidence/score must be numeric")
    confidence = float(confidence)
    if not math.isfinite(confidence) or not 0 <= confidence <= 1:
        raise DatasetFormatError("Prediction confidence must be finite and within [0, 1]")
    if (
        not isinstance(box, list)
        or len(box) != 4
        or not all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in box)
    ):
        raise DatasetFormatError("Prediction bbox must be a four-number list")
    values = [float(value) for value in box]
    if not all(math.isfinite(value) for value in values):
        raise DatasetFormatError("Prediction bbox contains NaN or infinity")
    if normalized:
        values[0] *= image.width
        values[2] *= image.width
        values[1] *= image.height
        values[3] *= image.height
    parsed_box = xywh_to_xyxy(values) if bbox_format == "xywh" else tuple(values)
    if parsed_box[2] <= parsed_box[0] or parsed_box[3] <= parsed_box[1]:
        raise DatasetFormatError("Prediction bbox must have positive width and height")
    return Prediction(class_value, parsed_box, confidence)


def _greedy_matches(
    ground_truth: list[Any],
    gt_boxes: list[Box],
    predictions: list[Prediction],
    threshold: float,
) -> list[tuple[int, int]]:
    candidates: list[tuple[float, float, int, int]] = []
    for pred_index, prediction in enumerate(predictions):
        for gt_index, annotation in enumerate(ground_truth):
            overlap = iou(prediction.box, gt_boxes[gt_index])
            if prediction.class_id == annotation.class_id and overlap >= threshold:
                candidates.append((prediction.confidence, overlap, gt_index, pred_index))
    candidates.sort(reverse=True)
    used_gt: set[int] = set()
    used_predictions: set[int] = set()
    matches = []
    for _, _, gt_index, pred_index in candidates:
        if gt_index not in used_gt and pred_index not in used_predictions:
            used_gt.add(gt_index)
            used_predictions.add(pred_index)
            matches.append((gt_index, pred_index))
    return matches


def _failure_type(
    prediction: Prediction,
    ground_truth: list[Any],
    gt_boxes: list[Box],
    matched_gt: set[int],
    match_iou: float,
    localization_iou: float,
) -> tuple[str, int | None, float]:
    overlaps = sorted(
        ((iou(prediction.box, box), index) for index, box in enumerate(gt_boxes)), reverse=True
    )
    same_class = [
        (overlap, index)
        for overlap, index in overlaps
        if ground_truth[index].class_id == prediction.class_id
    ]
    duplicate = [item for item in same_class if item[0] >= match_iou and item[1] in matched_gt]
    if duplicate:
        overlap, index = duplicate[0]
        return "duplicate_detection", index, overlap
    classification = [
        (overlap, index)
        for overlap, index in overlaps
        if overlap >= match_iou and ground_truth[index].class_id != prediction.class_id
    ]
    if classification:
        overlap, index = classification[0]
        return "classification_error", index, overlap
    localization = [
        item for item in same_class if localization_iou <= item[0] < match_iou
    ]
    if localization:
        overlap, index = localization[0]
        return "localization_error", index, overlap
    overlap, index = overlaps[0] if overlaps else (0.0, None)
    return "background_false_positive", index, overlap


def _confidence_bucket(confidence: float) -> str:
    if confidence < 0.25:
        return "[0.00,0.25)"
    if confidence < 0.5:
        return "[0.25,0.50)"
    if confidence < 0.75:
        return "[0.50,0.75)"
    return "[0.75,1.00]"


def _round_box(box: Box) -> list[float]:
    return [round(value, 4) for value in box]


def _reject_json_constant(value: str) -> None:
    raise DatasetFormatError(f"Non-standard JSON numeric constant is not allowed: {value}")
