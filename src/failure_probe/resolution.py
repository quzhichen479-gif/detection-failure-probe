"""Resolution survival diagnostics for ground-truth boxes."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from failure_probe.dataset import load_dataset
from failure_probe.models import Dataset


def resolution_survival(
    dataset: str | Path | Dataset,
    resolutions: Iterable[int] = (320, 640, 1280),
) -> dict[str, Any]:
    """Measure box pixel geometry after aspect-preserving resize.

    These are geometric diagnostics only. They do not predict accuracy, recall,
    or any other model-performance metric.
    """
    loaded = load_dataset(dataset) if not isinstance(dataset, Dataset) else dataset
    requested = sorted(set(resolutions))
    if not requested or any(not isinstance(value, int) or value <= 0 for value in requested):
        raise ValueError("resolutions must contain positive integers")
    per_resolution: list[dict[str, Any]] = []
    details: list[dict[str, Any]] = []
    total_valid = sum(
        annotation.valid and not image.image_error
        for image in loaded.images
        for annotation in image.annotations
    )
    for resolution in requested:
        thresholds: Counter[str] = Counter()
        buckets: Counter[str] = Counter()
        widths: list[float] = []
        heights: list[float] = []
        areas: list[float] = []
        for image in loaded.images:
            if image.image_error:
                continue
            resize_scale = min(resolution / image.width, resolution / image.height)
            for annotation in image.annotations:
                if not annotation.valid:
                    continue
                width = annotation.width * image.width * resize_scale
                height = annotation.height * image.height * resize_scale
                pixel_area = width * height
                widths.append(width)
                heights.append(height)
                areas.append(pixel_area)
                minimum_side = min(width, height)
                for threshold in (1, 2, 4, 8, 16):
                    if minimum_side >= threshold:
                        thresholds[f"min_side_ge_{threshold}px"] += 1
                bucket = _survival_bucket(minimum_side)
                buckets[bucket] += 1
                details.append(
                    {
                        "image": image.relative_path,
                        "line": annotation.line,
                        "class_id": annotation.class_id,
                        "resolution": resolution,
                        "pixel_width": round(width, 4),
                        "pixel_height": round(height, 4),
                        "pixel_area": round(pixel_area, 4),
                        "scale_bucket": bucket,
                    }
                )
        per_resolution.append(
            {
                "resolution": resolution,
                "valid_objects": total_valid,
                "mean_pixel_width": _mean(widths),
                "mean_pixel_height": _mean(heights),
                "mean_pixel_area": _mean(areas),
                "scale_buckets": dict(sorted(buckets.items())),
                "survival": {
                    key: {
                        "count": thresholds[key],
                        "ratio": round(thresholds[key] / total_valid, 6) if total_valid else 0.0,
                    }
                    for key in (f"min_side_ge_{value}px" for value in (1, 2, 4, 8, 16))
                },
            }
        )
    return {
        "schema_version": 1,
        "method": "aspect-preserving resize into a square input; padding does not change box size",
        "disclaimer": (
            "Resolution survival is a geometric diagnostic and must not be interpreted as a "
            "prediction of model performance."
        ),
        "resolutions": per_resolution,
        "objects": details,
    }


def _survival_bucket(minimum_side: float) -> str:
    if minimum_side < 1:
        return "subpixel"
    if minimum_side < 4:
        return "1-4px"
    if minimum_side < 8:
        return "4-8px"
    if minimum_side < 16:
        return "8-16px"
    if minimum_side < 32:
        return "16-32px"
    return "32px+"


def _mean(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 4) if values else None
