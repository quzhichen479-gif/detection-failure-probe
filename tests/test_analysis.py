from __future__ import annotations

import pytest

from failure_probe import analyze_predictions
from failure_probe.errors import DatasetFormatError


def test_prediction_failure_taxonomy(sample_dataset) -> None:
    dataset_yaml, predictions = sample_dataset
    result = analyze_predictions(dataset_yaml, predictions)

    assert result["summary"]["tp"] == 2
    assert result["summary"]["fp"] == 4
    assert result["summary"]["fn"] == 1
    assert result["summary"]["failure_types"] == {
        "background_false_positive": 1,
        "classification_error": 1,
        "duplicate_detection": 1,
        "localization_error": 1,
    }
    assert result["per_class"][1]["recall"] == 1.0


def test_flat_prediction_list_and_basename_matching(sample_dataset, tmp_path) -> None:
    dataset_yaml, _ = sample_dataset
    prediction_file = tmp_path / "flat.json"
    prediction_file.write_text(
        '[{"image":"c.png","class_id":1,"bbox":[10,10,20,20],"score":0.9}]',
        encoding="utf-8",
    )

    result = analyze_predictions(dataset_yaml, prediction_file)

    assert result["summary"]["tp"] == 1


@pytest.mark.parametrize(
    "payload",
    [
        '[{"image":"../secret.png","class_id":1,"bbox":[10,10,20,20],"score":0.9}]',
        '[{"image":"c.png","class_id":1,"bbox":[10,10,20,20],"score":NaN}]',
    ],
)
def test_unsafe_or_nonstandard_predictions_are_rejected(
    sample_dataset,
    tmp_path,
    payload: str,
) -> None:
    dataset_yaml, _ = sample_dataset
    prediction_file = tmp_path / "unsafe.json"
    prediction_file.write_text(payload, encoding="utf-8")

    with pytest.raises(DatasetFormatError):
        analyze_predictions(dataset_yaml, prediction_file)
