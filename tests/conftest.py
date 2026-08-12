from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image


@pytest.fixture
def sample_dataset(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "dataset"
    images = root / "images"
    labels = root / "labels"
    images.mkdir(parents=True)
    labels.mkdir()
    for name in ("a", "b", "c", "d"):
        Image.new("RGB", (100, 100), (30, 40, 50)).save(images / f"{name}.png")
    (labels / "a.txt").write_text(
        "0 0.5 0.5 0.2 0.2\n0 0.5 0.5 0.2 0.2\n1 0.9 0.9 0.3 0.3\n",
        encoding="utf-8",
    )
    (labels / "c.txt").write_text("1 0.2 0.2 0.2 0.2\n", encoding="utf-8")
    (labels / "d.txt").write_text("", encoding="utf-8")
    (labels / "orphan.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    dataset_yaml = root / "dataset.yaml"
    dataset_yaml.write_text(
        "path: .\ntrain: images\nnames:\n  0: alpha\n  1: beta\n",
        encoding="utf-8",
    )
    predictions = tmp_path / "predictions.json"
    predictions.write_text(
        json.dumps(
            {
                "bbox_format": "xywh",
                "images": [
                    {
                        "image": "images/a.png",
                        "predictions": [
                            {"class_id": 0, "bbox": [40, 40, 20, 20], "confidence": 0.9}
                        ],
                    },
                    {
                        "image": "images/c.png",
                        "predictions": [
                            {"class_id": 1, "bbox": [10, 10, 20, 20], "confidence": 0.95},
                            {"class_id": 1, "bbox": [10, 10, 20, 20], "confidence": 0.8},
                            {"class_id": 0, "bbox": [10, 10, 20, 20], "confidence": 0.7},
                            {"class_id": 1, "bbox": [20, 10, 20, 20], "confidence": 0.6},
                            {"class_id": 0, "bbox": [70, 70, 10, 10], "confidence": 0.2},
                        ],
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    return dataset_yaml, predictions
