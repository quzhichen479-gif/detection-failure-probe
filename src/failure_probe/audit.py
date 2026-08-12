"""YOLO dataset audit metrics and issue detection."""

from __future__ import annotations

import statistics
from collections import Counter
from pathlib import Path
from typing import Any

from failure_probe.dataset import load_dataset
from failure_probe.geometry import iou, scale_bucket
from failure_probe.models import Annotation, Dataset, ImageRecord


def audit_dataset(
    dataset: str | Path | Dataset,
    *,
    near_duplicate_iou: float = 0.95,
) -> dict[str, Any]:
    """Audit a YOLO dataset and return JSON-serializable metrics and findings."""
    loaded = load_dataset(dataset) if not isinstance(dataset, Dataset) else dataset
    if not 0 < near_duplicate_iou <= 1:
        raise ValueError("near_duplicate_iou must be in (0, 1]")

    issues: list[dict[str, Any]] = []
    class_counts: Counter[int] = Counter()
    class_images: Counter[int] = Counter()
    widths: list[float] = []
    heights: list[float] = []
    areas: list[float] = []
    aspect_ratios: list[float] = []
    scale_counts: Counter[str] = Counter()
    duplicates: list[dict[str, Any]] = []
    image_payloads: list[dict[str, Any]] = []
    valid_count = 0
    invalid_count = 0

    def add_issue(issue_type: str, severity: str, image: str | None, **details: Any) -> str:
        issue_id = f"issue_{len(issues) + 1:06d}"
        issues.append(
            {"id": issue_id, "type": issue_type, "severity": severity, "image": image, **details}
        )
        return issue_id

    for record in loaded.images:
        image_issue_types: list[str] = []
        if record.image_error:
            add_issue("unreadable_image", "error", record.relative_path, message=record.image_error)
            image_issue_types.append("unreadable_image")
        if not record.label_exists:
            add_issue(
                "missing_label",
                "warning",
                record.relative_path,
                label=record.label_relative_path,
                message="image has no corresponding label file",
            )
            image_issue_types.append("missing_label")
        elif record.label_empty:
            add_issue(
                "empty_label",
                "info",
                record.relative_path,
                label=record.label_relative_path,
                message="label file contains no annotations",
            )
            image_issue_types.append("empty_label")

        serialized_annotations: list[dict[str, Any]] = []
        seen_classes: set[int] = set()
        for annotation in record.annotations:
            payload = _annotation_payload(annotation, record)
            annotation_issues: list[str] = []
            if not annotation.valid or record.image_error:
                invalid_count += 1
                issue_id = add_issue(
                    "invalid_box",
                    "error",
                    record.relative_path,
                    label=record.label_relative_path,
                    line=annotation.line,
                    message=annotation.error or "box belongs to an unreadable image",
                )
                annotation_issues.append(issue_id)
            else:
                valid_count += 1
                seen_classes.add(annotation.class_id)
                class_counts[annotation.class_id] += 1
                pixel_width = annotation.width * record.width
                pixel_height = annotation.height * record.height
                pixel_area = pixel_width * pixel_height
                widths.append(pixel_width)
                heights.append(pixel_height)
                areas.append(pixel_area)
                aspect = max(pixel_width / pixel_height, pixel_height / pixel_width)
                aspect_ratios.append(aspect)
                scale_counts[scale_bucket(pixel_area)] += 1
                if pixel_width < 2 or pixel_height < 2:
                    issue_id = add_issue(
                        "tiny_box",
                        "warning",
                        record.relative_path,
                        label=record.label_relative_path,
                        line=annotation.line,
                        message="box is narrower or shorter than 2 pixels",
                    )
                    annotation_issues.append(issue_id)
                    image_issue_types.append("tiny_box")
                if aspect >= 10:
                    issue_id = add_issue(
                        "extreme_aspect_ratio",
                        "warning",
                        record.relative_path,
                        label=record.label_relative_path,
                        line=annotation.line,
                        ratio=round(aspect, 4),
                        message="box aspect ratio is at least 10:1",
                    )
                    annotation_issues.append(issue_id)
                    image_issue_types.append("extreme_aspect_ratio")
            payload["issue_ids"] = annotation_issues
            serialized_annotations.append(payload)
        for class_id in seen_classes:
            class_images[class_id] += 1

        for first_index, first in enumerate(record.annotations):
            if not first.valid or record.image_error:
                continue
            for second_index in range(first_index + 1, len(record.annotations)):
                second = record.annotations[second_index]
                if not second.valid or first.class_id != second.class_id:
                    continue
                exact = _normalized_box(first) == _normalized_box(second)
                overlap = iou(
                    first.pixel_box(record.width, record.height),
                    second.pixel_box(record.width, record.height),
                )
                if exact or overlap >= near_duplicate_iou:
                    duplicate_type = (
                        "duplicate_annotation" if exact else "near_duplicate_annotation"
                    )
                    issue_id = add_issue(
                        duplicate_type,
                        "warning",
                        record.relative_path,
                        label=record.label_relative_path,
                        lines=[first.line, second.line],
                        iou=round(overlap, 6),
                        message="same-class annotations are identical or almost identical",
                    )
                    duplicates.append(
                        {
                            "issue_id": issue_id,
                            "image": record.relative_path,
                            "class_id": first.class_id,
                            "lines": [first.line, second.line],
                            "iou": round(overlap, 6),
                            "exact": exact,
                        }
                    )
                    image_issue_types.append(duplicate_type)

        image_payloads.append(
            {
                "image": record.relative_path,
                "label": record.label_relative_path,
                "width": record.width,
                "height": record.height,
                "label_exists": record.label_exists,
                "label_empty": record.label_empty,
                "issue_types": sorted(set(image_issue_types)),
                "annotations": serialized_annotations,
            }
        )

    for label in loaded.orphan_labels:
        add_issue(
            "missing_image",
            "error",
            None,
            label=label,
            message="label file has no corresponding configured image",
        )

    class_distribution = [
        {
            "class_id": class_id,
            "name": name,
            "annotations": class_counts[class_id],
            "images": class_images[class_id],
        }
        for class_id, name in loaded.names.items()
    ]
    issue_counts = Counter(issue["type"] for issue in issues)
    small_count = scale_counts["small"]
    return {
        "schema_version": 1,
        "dataset": {
            "yaml": loaded.yaml_path.name,
            "root": str(loaded.root),
            "splits": loaded.split_names,
            "classes": loaded.names,
        },
        "summary": {
            "images": len(loaded.images),
            "valid_annotations": valid_count,
            "invalid_annotations": invalid_count,
            "missing_labels": issue_counts["missing_label"],
            "missing_images": issue_counts["missing_image"],
            "empty_label_images": issue_counts["empty_label"],
            "duplicate_annotations": issue_counts["duplicate_annotation"],
            "near_duplicate_annotations": issue_counts["near_duplicate_annotation"],
            "suspicious_annotations": sum(
                issue_counts[key] for key in ("tiny_box", "extreme_aspect_ratio")
            ),
            "small_object_ratio": round(small_count / valid_count, 6) if valid_count else 0.0,
        },
        "class_distribution": class_distribution,
        "box_statistics": {
            "pixel_width": _describe(widths),
            "pixel_height": _describe(heights),
            "pixel_area": _describe(areas),
            "aspect_ratio": _describe(aspect_ratios),
            "scale_buckets": dict(sorted(scale_counts.items())),
            "small_definition": "pixel area < 32^2 at native image resolution",
        },
        "duplicates": duplicates,
        "issues": issues,
        "images": image_payloads,
    }


def _annotation_payload(annotation: Annotation, image: ImageRecord) -> dict[str, Any]:
    box = annotation.pixel_box(image.width, image.height) if image.width and image.height else None
    return {
        "class_id": annotation.class_id,
        "line": annotation.line,
        "valid": annotation.valid,
        "error": annotation.error,
        "yolo": [
            annotation.x_center,
            annotation.y_center,
            annotation.width,
            annotation.height,
        ],
        "bbox_xyxy": [round(value, 4) for value in box] if box else None,
    }


def _normalized_box(annotation: Annotation) -> tuple[float, float, float, float]:
    return (
        annotation.x_center,
        annotation.y_center,
        annotation.width,
        annotation.height,
    )


def _describe(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "min": None, "median": None, "mean": None, "p95": None, "max": None}
    ordered = sorted(values)
    p95_index = min(len(ordered) - 1, max(0, int(0.95 * len(ordered) + 0.999999) - 1))
    return {
        "count": len(values),
        "min": round(ordered[0], 4),
        "median": round(statistics.median(ordered), 4),
        "mean": round(statistics.fmean(ordered), 4),
        "p95": round(ordered[p95_index], 4),
        "max": round(ordered[-1], 4),
    }
