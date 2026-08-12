"""Internal dataset models shared by audit and analysis."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from failure_probe.geometry import Box, yolo_to_xyxy


@dataclass(frozen=True)
class Annotation:
    """One parsed YOLO annotation."""

    class_id: int
    x_center: float
    y_center: float
    width: float
    height: float
    line: int
    valid: bool = True
    error: str | None = None

    def pixel_box(self, image_width: int, image_height: int) -> Box:
        return yolo_to_xyxy(
            self.x_center,
            self.y_center,
            self.width,
            self.height,
            image_width,
            image_height,
        )


@dataclass
class ImageRecord:
    """An image and its corresponding YOLO label information."""

    path: Path
    relative_path: str
    label_path: Path
    label_relative_path: str
    width: int
    height: int
    annotations: list[Annotation] = field(default_factory=list)
    label_exists: bool = True
    label_empty: bool = False
    image_error: str | None = None


@dataclass
class Dataset:
    """Loaded dataset with paths constrained to ``root``."""

    yaml_path: Path
    root: Path
    names: dict[int, str]
    images: list[ImageRecord]
    orphan_labels: list[str]
    split_names: list[str]
