from __future__ import annotations

from failure_probe import audit_dataset, load_dataset, resolution_survival


def test_audit_detects_integrity_and_annotation_issues(sample_dataset) -> None:
    dataset_yaml, _ = sample_dataset
    dataset = load_dataset(dataset_yaml)
    result = audit_dataset(dataset)

    assert result["summary"]["images"] == 4
    assert result["summary"]["valid_annotations"] == 3
    assert result["summary"]["invalid_annotations"] == 1
    assert result["summary"]["missing_labels"] == 1
    assert result["summary"]["missing_images"] == 1
    assert result["summary"]["empty_label_images"] == 1
    assert result["summary"]["duplicate_annotations"] == 1
    assert result["class_distribution"][0]["annotations"] == 2


def test_resolution_survival_is_geometric_and_multiscale(sample_dataset) -> None:
    dataset_yaml, _ = sample_dataset
    result = resolution_survival(dataset_yaml, [50, 100])

    assert [item["resolution"] for item in result["resolutions"]] == [50, 100]
    assert result["resolutions"][1]["mean_pixel_width"] == 20.0
    assert "must not be interpreted" in result["disclaimer"]


def test_noncontiguous_class_ids_are_supported(sample_dataset) -> None:
    dataset_yaml, _ = sample_dataset
    text = dataset_yaml.read_text(encoding="utf-8")
    dataset_yaml.write_text(text.replace("1: beta", "2: beta"), encoding="utf-8")
    label = dataset_yaml.parent / "labels" / "c.txt"
    label.write_text("2 0.2 0.2 0.2 0.2\n", encoding="utf-8")

    result = audit_dataset(dataset_yaml)

    assert result["summary"]["invalid_annotations"] == 1
    assert result["class_distribution"][1]["class_id"] == 2
    assert result["class_distribution"][1]["annotations"] == 1
