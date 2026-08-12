from __future__ import annotations

import json
from pathlib import Path

import pytest

from failure_probe.errors import RunFormatError, UnsafePathError
from failure_probe.paths import create_run_dir, validate_run_dir
from failure_probe.workflow import run_analysis, run_audit


def test_dataset_path_traversal_is_rejected(tmp_path: Path) -> None:
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    yaml_path = config_dir / "dataset.yaml"
    yaml_path.write_text("path: ..\ntrain: images\nnames: [thing]\n", encoding="utf-8")

    from failure_probe import load_dataset

    with pytest.raises(UnsafePathError):
        load_dataset(yaml_path)


def test_run_names_never_overwrite(tmp_path: Path) -> None:
    first = create_run_dir(tmp_path / "runs", "audit", "fixed")
    second = create_run_dir(tmp_path / "runs", "audit", "fixed")

    assert first.name == "fixed"
    assert second.name == "fixed_01"
    assert first.is_dir() and second.is_dir()


def test_complete_workflows_and_static_report(sample_dataset, tmp_path: Path) -> None:
    dataset_yaml, predictions = sample_dataset
    runs = tmp_path / "runs"
    audit_run = run_audit(dataset_yaml, runs_dir=runs, run_name="audit_test", resolutions=[64])
    analysis_run = run_analysis(
        dataset_yaml,
        predictions,
        runs_dir=runs,
        run_name="analysis_test",
        resolutions=[64],
    )

    assert (audit_run / "report.html").is_file()
    assert (analysis_run / "analysis.json").is_file()
    assert "Resolution survival" in (analysis_run / "report.html").read_text(encoding="utf-8")
    notes = json.loads((analysis_run / "reviewer_notes.json").read_text(encoding="utf-8"))
    assert notes == {"schema_version": 1, "notes": {}}


def test_unsafe_run_name_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(UnsafePathError):
        create_run_dir(tmp_path / "runs", "audit", "../escape")


def test_run_directory_symlink_is_rejected(tmp_path: Path) -> None:
    target = create_run_dir(tmp_path / "runs", "audit", "real")
    link = tmp_path / "linked_run"
    try:
        link.symlink_to(target, target_is_directory=True)
    except OSError:
        pytest.skip("directory symlinks are not available on this platform")

    with pytest.raises(RunFormatError):
        validate_run_dir(link)
