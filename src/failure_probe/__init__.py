"""Detection Failure Probe public API."""

from failure_probe.analysis import analyze_predictions
from failure_probe.audit import audit_dataset
from failure_probe.dataset import load_dataset
from failure_probe.report import generate_report
from failure_probe.resolution import resolution_survival

__all__ = [
    "analyze_predictions",
    "audit_dataset",
    "generate_report",
    "load_dataset",
    "resolution_survival",
]

__version__ = "0.1.0a0"
