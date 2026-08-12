"""Filesystem boundary and atomic-write helpers."""

from __future__ import annotations

import json
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from failure_probe.errors import RunFormatError, UnsafePathError

RUN_MARKER = ".failure-probe-run"
RUN_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,79}$")


def resolve_within(root: Path, candidate: str | Path, *, must_exist: bool = False) -> Path:
    """Resolve a path and reject traversal, symlink escapes, and foreign absolute paths."""
    root = root.resolve(strict=True)
    value = Path(candidate)
    resolved = (value if value.is_absolute() else root / value).resolve(strict=must_exist)
    if resolved != root and root not in resolved.parents:
        raise UnsafePathError(f"Path escapes allowed root {root}: {candidate}")
    return resolved


def relative_posix(path: Path, root: Path) -> str:
    """Return a stable relative path for JSON artifacts."""
    return path.resolve().relative_to(root.resolve()).as_posix()


def create_run_dir(runs_dir: str | Path, prefix: str, run_name: str | None = None) -> Path:
    """Create a fresh run directory without overwriting an existing path."""
    requested_root = Path(runs_dir).absolute()
    if requested_root.exists() and requested_root.is_symlink():
        raise UnsafePathError(f"Runs directory may not be a symlink: {requested_root}")
    root = requested_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    stem = run_name or f"{prefix}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    if not RUN_NAME.fullmatch(stem):
        raise UnsafePathError(
            "Run name must contain only letters, digits, dot, underscore, or dash"
        )
    for suffix in range(1000):
        name = stem if suffix == 0 else f"{stem}_{suffix:02d}"
        path = resolve_within(root, name)
        try:
            path.mkdir()
        except FileExistsError:
            continue
        (path / RUN_MARKER).write_text("detection-failure-probe\n", encoding="utf-8")
        return path
    raise UnsafePathError(f"Could not allocate a unique run directory under {root}")


def validate_run_dir(run_dir: str | Path) -> Path:
    """Require a real run directory with our marker and manifest."""
    requested = Path(run_dir).absolute()
    if requested.is_symlink():
        raise RunFormatError(f"Run directory may not be a symlink: {requested}")
    path = requested.resolve(strict=True)
    if not path.is_dir():
        raise RunFormatError(f"Not a regular run directory: {path}")
    if not (path / RUN_MARKER).is_file() or not (path / "manifest.json").is_file():
        raise RunFormatError(f"Missing Failure Probe run metadata in: {path}")
    return path


def atomic_write_json(path: Path, payload: Any) -> None:
    """Write JSON atomically without following an existing symlink."""
    _atomic_write(path, json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def atomic_write_text(path: Path, text: str) -> None:
    """Write UTF-8 text atomically without following an existing symlink."""
    _atomic_write(path, text)


def _atomic_write(path: Path, text: str) -> None:
    parent = path.parent.resolve(strict=True)
    if path.exists() and path.is_symlink():
        raise UnsafePathError(f"Refusing to overwrite symlink: {path}")
    if path.resolve(strict=False).parent != parent:
        raise UnsafePathError(f"Output escapes its parent: {path}")
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=parent, text=True)
    temporary_path = Path(temporary)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)
