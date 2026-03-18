"""Shared path helpers for the AEOSBench repository."""

from __future__ import annotations

import os
from pathlib import Path


def project_root() -> Path:
    """Return the repository root."""
    return Path(__file__).resolve().parent.parent


def _resolve_path(path_str: str) -> Path:
    path = Path(path_str).expanduser()
    if not path.is_absolute():
        path = project_root() / path
    return path.resolve()


def data_root() -> Path:
    """Return the configured data root."""
    override = os.environ.get("AEOS_DATA_ROOT")
    if override:
        return _resolve_path(override)
    return (project_root() / "data").resolve()


def benchmark_data_root() -> Path:
    """Return the released benchmark data directory."""
    root = data_root()
    nested_root = root / "data"
    if nested_root.is_dir():
        return nested_root
    return root


def project_relative_path(path: Path) -> Path:
    """Return a path relative to the project root when possible."""
    resolved = Path(path).expanduser().resolve()
    try:
        return resolved.relative_to(project_root())
    except ValueError:
        return resolved

