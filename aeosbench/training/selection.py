"""Selection manifests for iterative retraining inputs."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class SelectionEntry:
    split: str
    id_: int
    source_kind: str
    epoch: int | None
    trajectory_path: Path
    metrics_path: Path | None


@dataclass(frozen=True)
class SelectionManifest:
    path: Path
    entries: list[SelectionEntry]


def _coerce_path(value: Any, *, base_dir: Path, name: str) -> Path:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    path = Path(value)
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return path


def load_selection_manifest(path: str | Path) -> SelectionManifest:
    resolved = Path(path).resolve()
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("selection manifest must be a mapping")
    entries_payload = payload.get("entries")
    if not isinstance(entries_payload, list):
        raise TypeError("selection manifest must define an 'entries' list")

    entries: list[SelectionEntry] = []
    for index, entry_payload in enumerate(entries_payload):
        if not isinstance(entry_payload, dict):
            raise TypeError(f"selection entry {index} must be a mapping")
        split = entry_payload.get("split")
        if not isinstance(split, str):
            raise TypeError(f"selection entry {index}.split must be a string")
        id_value = entry_payload.get("id")
        if not isinstance(id_value, int):
            raise TypeError(f"selection entry {index}.id must be an int")
        source_payload = entry_payload.get("source")
        if not isinstance(source_payload, dict):
            raise TypeError(f"selection entry {index}.source must be a mapping")
        source_kind = source_payload.get("kind")
        if source_kind not in {"released", "rollout"}:
            raise ValueError(
                f"selection entry {index}.source.kind must be 'released' or 'rollout'"
            )
        epoch = source_payload.get("epoch")
        if epoch is not None and not isinstance(epoch, int):
            raise TypeError(f"selection entry {index}.source.epoch must be an int or null")
        trajectory_path = _coerce_path(
            source_payload.get("trajectory_path"),
            base_dir=resolved.parent,
            name=f"selection entry {index}.source.trajectory_path",
        )
        metrics_path_value = source_payload.get("metrics_path")
        metrics_path = None
        if metrics_path_value is not None:
            metrics_path = _coerce_path(
                metrics_path_value,
                base_dir=resolved.parent,
                name=f"selection entry {index}.source.metrics_path",
            )
        entries.append(
            SelectionEntry(
                split=split,
                id_=id_value,
                source_kind=source_kind,
                epoch=epoch,
                trajectory_path=trajectory_path,
                metrics_path=metrics_path,
            )
        )
    return SelectionManifest(path=resolved, entries=entries)


def save_selection_manifest(path: str | Path, entries: list[SelectionEntry]) -> Path:
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "entries": [
            {
                "split": entry.split,
                "id": entry.id_,
                "source": {
                    "kind": entry.source_kind,
                    "epoch": entry.epoch,
                    "trajectory_path": str(entry.trajectory_path),
                    "metrics_path": None if entry.metrics_path is None else str(entry.metrics_path),
                },
            }
            for entry in entries
        ]
    }
    resolved.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return resolved
