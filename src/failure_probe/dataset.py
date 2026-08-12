"""Safe, read-only loading of small and medium YOLO detection datasets."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import yaml
from PIL import Image

from failure_probe.errors import DatasetFormatError
from failure_probe.models import Annotation, Dataset, ImageRecord
from failure_probe.paths import relative_posix, resolve_within

IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
MAX_CONFIG_BYTES = 1_000_000
MAX_LABEL_BYTES = 10_000_000


def load_dataset(dataset_yaml: str | Path) -> Dataset:
    """Load a YOLO dataset without executing config or annotation content.

    By default, every path referenced by the YAML must remain beneath the YAML's
    directory. This makes path traversal and accidental reads of unrelated files
    fail closed.
    """
    yaml_path = Path(dataset_yaml).resolve(strict=True)
    if not yaml_path.is_file() or yaml_path.suffix.lower() not in {".yaml", ".yml"}:
        raise DatasetFormatError(f"Expected an existing .yaml or .yml file: {yaml_path}")
    if yaml_path.stat().st_size > MAX_CONFIG_BYTES:
        raise DatasetFormatError("Dataset YAML is larger than the 1 MB safety limit")
    try:
        config = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise DatasetFormatError(f"Could not safely parse dataset YAML: {exc}") from exc
    if not isinstance(config, dict):
        raise DatasetFormatError("Dataset YAML must contain a mapping")

    config_dir = yaml_path.parent.resolve()
    root_value = config.get("path", ".")
    if not isinstance(root_value, str):
        raise DatasetFormatError("Dataset 'path' must be a string")
    root = resolve_within(config_dir, root_value, must_exist=True)
    if not root.is_dir():
        raise DatasetFormatError(f"Dataset root is not a directory: {root}")

    names = _parse_names(config.get("names"))
    sources, split_names = _image_sources(config)
    image_paths: dict[Path, Path] = {}
    for source in sources:
        for image_path, source_root in _expand_source(root, source):
            image_paths[image_path] = source_root
    if not image_paths:
        raise DatasetFormatError("No supported images found in configured dataset splits")

    labels_config = config.get("labels")
    custom_label_root = None
    if labels_config is not None:
        if not isinstance(labels_config, str):
            raise DatasetFormatError("Optional 'labels' must be a directory string")
        custom_label_root = resolve_within(root, labels_config, must_exist=True)
        if not custom_label_root.is_dir():
            raise DatasetFormatError("Optional 'labels' path is not a directory")

    records: list[ImageRecord] = []
    expected_labels: set[Path] = set()
    label_roots: set[Path] = set()
    for image_path, source_root in sorted(image_paths.items(), key=lambda item: str(item[0])):
        label_path, label_root = _label_for_image(
            image_path, source_root, root, custom_label_root
        )
        expected_labels.add(label_path)
        label_roots.add(label_root)
        width, height, image_error = _image_size(image_path)
        annotations, empty = _read_label(label_path, set(names))
        records.append(
            ImageRecord(
                path=image_path,
                relative_path=relative_posix(image_path, root),
                label_path=label_path,
                label_relative_path=relative_posix(label_path, root),
                width=width,
                height=height,
                annotations=annotations,
                label_exists=label_path.is_file(),
                label_empty=empty,
                image_error=image_error,
            )
        )

    orphan_labels: list[str] = []
    for label_root in label_roots:
        if not label_root.is_dir():
            continue
        for label_path in label_root.rglob("*.txt"):
            resolved = label_path.resolve()
            if resolved not in expected_labels:
                orphan_labels.append(relative_posix(resolved, root))

    return Dataset(
        yaml_path=yaml_path,
        root=root,
        names=names,
        images=records,
        orphan_labels=sorted(set(orphan_labels)),
        split_names=split_names,
    )


def _parse_names(value: Any) -> dict[int, str]:
    if isinstance(value, list) and all(isinstance(name, str) for name in value):
        if not value:
            raise DatasetFormatError("Dataset 'names' may not be empty")
        return dict(enumerate(value))
    if isinstance(value, dict):
        result: dict[int, str] = {}
        for key, name in value.items():
            if not isinstance(name, str):
                raise DatasetFormatError("Every class name must be a string")
            try:
                class_id = int(key)
            except (TypeError, ValueError) as exc:
                raise DatasetFormatError(f"Invalid class id in names: {key}") from exc
            if class_id < 0:
                raise DatasetFormatError("Class ids must be non-negative")
            result[class_id] = name
        if not result:
            raise DatasetFormatError("Dataset 'names' may not be empty")
        return dict(sorted(result.items()))
    raise DatasetFormatError("Dataset YAML requires 'names' as a list or mapping")


def _image_sources(config: dict[str, Any]) -> tuple[list[str], list[str]]:
    keys = (
        ["images"]
        if "images" in config
        else [key for key in ("train", "val", "test") if key in config]
    )
    if not keys:
        raise DatasetFormatError("Dataset YAML needs 'images' or train/val/test image paths")
    sources: list[str] = []
    for key in keys:
        value = config[key]
        values = value if isinstance(value, list) else [value]
        if not values or not all(isinstance(item, str) for item in values):
            raise DatasetFormatError(f"Dataset split '{key}' must contain path strings")
        sources.extend(values)
    return sources, keys


def _expand_source(root: Path, source: str) -> list[tuple[Path, Path]]:
    source_path = resolve_within(root, source, must_exist=True)
    if source_path.is_dir():
        images: list[tuple[Path, Path]] = []
        for path in source_path.rglob("*"):
            if path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            image_path = resolve_within(root, path, must_exist=True)
            if image_path.is_file():
                images.append((image_path, source_path))
        return images
    if source_path.is_file() and source_path.suffix.lower() == ".txt":
        if source_path.stat().st_size > MAX_CONFIG_BYTES:
            raise DatasetFormatError(f"Image list is larger than 1 MB: {source_path}")
        items: list[tuple[Path, Path]] = []
        for raw_line in source_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            image_path = resolve_within(root, line, must_exist=True)
            if image_path.suffix.lower() not in IMAGE_EXTENSIONS or not image_path.is_file():
                raise DatasetFormatError(f"Unsupported image in list: {line}")
            items.append((image_path, root))
        return items
    if source_path.is_file() and source_path.suffix.lower() in IMAGE_EXTENSIONS:
        return [(source_path, source_path.parent)]
    raise DatasetFormatError(f"Unsupported dataset image source: {source_path}")


def _label_for_image(
    image_path: Path,
    source_root: Path,
    dataset_root: Path,
    custom_label_root: Path | None,
) -> tuple[Path, Path]:
    if custom_label_root is not None:
        relative = image_path.relative_to(source_root).with_suffix(".txt")
        return resolve_within(custom_label_root, relative), custom_label_root
    relative = image_path.relative_to(dataset_root)
    parts = list(relative.parts)
    if "images" in parts:
        index = parts.index("images")
        parts[index] = "labels"
        label_relative = Path(*parts).with_suffix(".txt")
        label_root = dataset_root.joinpath(*parts[: index + 1]).resolve()
    else:
        label_root = (dataset_root / "labels").resolve()
        relative_to_source = image_path.relative_to(source_root).with_suffix(".txt")
        label_relative = Path("labels") / relative_to_source
    return resolve_within(dataset_root, label_relative), label_root


def _image_size(path: Path) -> tuple[int, int, str | None]:
    try:
        with Image.open(path) as image:
            width, height = image.size
            if width <= 0 or height <= 0:
                return 0, 0, "image has non-positive dimensions"
            return width, height, None
    except (OSError, Image.DecompressionBombError) as exc:
        return 0, 0, f"could not read image metadata: {exc}"


def _read_label(path: Path, valid_class_ids: set[int]) -> tuple[list[Annotation], bool]:
    if not path.is_file():
        return [], False
    if path.stat().st_size > MAX_LABEL_BYTES:
        return [_invalid_annotation(0, "label file exceeds 10 MB safety limit")], False
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        return [_invalid_annotation(0, f"could not read label: {exc}")], False
    annotations: list[Annotation] = []
    nonempty_lines = [
        (number, line.strip()) for number, line in enumerate(lines, 1) if line.strip()
    ]
    for line_number, line in nonempty_lines:
        parts = line.split()
        if len(parts) != 5:
            annotations.append(_invalid_annotation(line_number, "expected exactly 5 fields"))
            continue
        try:
            raw_class, x_center, y_center, width, height = (float(value) for value in parts)
        except ValueError:
            annotations.append(_invalid_annotation(line_number, "contains a non-numeric field"))
            continue
        values = (raw_class, x_center, y_center, width, height)
        if not all(math.isfinite(value) for value in values):
            annotations.append(_invalid_annotation(line_number, "contains NaN or infinity"))
            continue
        class_id = int(raw_class)
        error = None
        if raw_class != class_id or class_id not in valid_class_ids:
            error = "class id is not a valid configured integer"
        elif width <= 0 or height <= 0:
            error = "box width and height must be positive"
        elif not all(0 <= value <= 1 for value in (x_center, y_center, width, height)):
            error = "normalized coordinates must be within [0, 1]"
        elif (
            x_center - width / 2 < 0
            or y_center - height / 2 < 0
            or x_center + width / 2 > 1
            or y_center + height / 2 > 1
        ):
            error = "box extends outside the image"
        annotations.append(
            Annotation(
                class_id=class_id,
                x_center=x_center,
                y_center=y_center,
                width=width,
                height=height,
                line=line_number,
                valid=error is None,
                error=error,
            )
        )
    return annotations, len(nonempty_lines) == 0


def _invalid_annotation(line: int, error: str) -> Annotation:
    return Annotation(0, 0.0, 0.0, 0.0, 0.0, line, valid=False, error=error)
