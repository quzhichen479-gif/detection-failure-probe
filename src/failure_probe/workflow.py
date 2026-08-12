"""Run orchestration kept separate from reusable analysis APIs."""

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path

from failure_probe.analysis import analyze_predictions
from failure_probe.audit import audit_dataset
from failure_probe.dataset import load_dataset
from failure_probe.paths import atomic_write_json, create_run_dir
from failure_probe.report import generate_report
from failure_probe.resolution import resolution_survival


def run_audit(
    dataset_yaml: str | Path,
    *,
    runs_dir: str | Path = "runs",
    run_name: str | None = None,
    resolutions: Iterable[int] = (320, 640, 1280),
) -> Path:
    """Execute a complete dataset audit and return the new run directory."""
    dataset = load_dataset(dataset_yaml)
    audit = audit_dataset(dataset)
    survival = resolution_survival(dataset, resolutions)
    run_dir = create_run_dir(runs_dir, "audit", run_name)
    _write_manifest(run_dir, "audit", dataset.yaml_path, dataset.root)
    atomic_write_json(run_dir / "audit.json", audit)
    atomic_write_json(run_dir / "resolution.json", survival)
    atomic_write_json(run_dir / "reviewer_notes.json", {"schema_version": 1, "notes": {}})
    generate_report(run_dir)
    return run_dir


def run_analysis(
    dataset_yaml: str | Path,
    predictions: str | Path,
    *,
    runs_dir: str | Path = "runs",
    run_name: str | None = None,
    resolutions: Iterable[int] = (320, 640, 1280),
    match_iou: float = 0.5,
    localization_iou: float = 0.1,
) -> Path:
    """Execute dataset audit, prediction analysis, and resolution diagnostics."""
    dataset = load_dataset(dataset_yaml)
    audit = audit_dataset(dataset)
    analysis = analyze_predictions(
        dataset,
        predictions,
        match_iou=match_iou,
        localization_iou=localization_iou,
    )
    survival = resolution_survival(dataset, resolutions)
    run_dir = create_run_dir(runs_dir, "analysis", run_name)
    _write_manifest(run_dir, "analysis", dataset.yaml_path, dataset.root)
    atomic_write_json(run_dir / "audit.json", audit)
    atomic_write_json(run_dir / "analysis.json", analysis)
    atomic_write_json(run_dir / "resolution.json", survival)
    atomic_write_json(run_dir / "reviewer_notes.json", {"schema_version": 1, "notes": {}})
    generate_report(run_dir)
    return run_dir


def _write_manifest(run_dir: Path, run_type: str, yaml_path: Path, dataset_root: Path) -> None:
    atomic_write_json(
        run_dir / "manifest.json",
        {
            "schema_version": 1,
            "tool": "detection-failure-probe",
            "run_type": run_type,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "dataset_yaml": str(yaml_path),
            "dataset_root": str(dataset_root),
        },
    )
