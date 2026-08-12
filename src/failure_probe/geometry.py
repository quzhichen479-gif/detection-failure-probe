"""Small dependency-free geometry helpers."""

from __future__ import annotations

from typing import TypeAlias

Box: TypeAlias = tuple[float, float, float, float]


def yolo_to_xyxy(
    x_center: float,
    y_center: float,
    width: float,
    height: float,
    image_width: int,
    image_height: int,
) -> Box:
    """Convert normalized YOLO center-width-height coordinates to pixel xyxy."""
    x1 = (x_center - width / 2) * image_width
    y1 = (y_center - height / 2) * image_height
    x2 = (x_center + width / 2) * image_width
    y2 = (y_center + height / 2) * image_height
    return x1, y1, x2, y2


def xywh_to_xyxy(box: list[float] | tuple[float, ...]) -> Box:
    """Convert top-left pixel xywh coordinates to xyxy."""
    x, y, width, height = box
    return x, y, x + width, y + height


def area(box: Box) -> float:
    """Return non-negative box area."""
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def iou(first: Box, second: Box) -> float:
    """Return intersection over union for two xyxy boxes."""
    intersection = area(
        (
            max(first[0], second[0]),
            max(first[1], second[1]),
            min(first[2], second[2]),
            min(first[3], second[3]),
        )
    )
    union = area(first) + area(second) - intersection
    return intersection / union if union > 0 else 0.0


def scale_bucket(pixel_area: float) -> str:
    """Use common COCO-style object-area buckets."""
    if pixel_area < 32**2:
        return "small"
    if pixel_area < 96**2:
        return "medium"
    return "large"
