"""Released dataset layout helpers."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from aeosbench.paths import benchmark_data_root, data_root


@dataclass(frozen=True)
class AnnotationSelection:
    ids: list[int]
    epochs: list[int] | None = None

    def epoch_at(self, index: int, *, default: int | None = None) -> int:
        if self.epochs is None:
            if default is None:
                raise ValueError("annotation payload does not include epochs")
            return default
        return self.epochs[index]


@dataclass(frozen=True)
class ScenarioRef:
    split: str
    id_: int
    epoch: int


def _normalize_int_list(values: Any, *, name: str) -> list[int]:
    if not isinstance(values, list):
        raise TypeError(f"{name} must be a list")
    return [int(value) for value in values]


def load_annotations(path: str | Path) -> AnnotationSelection:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return AnnotationSelection(ids=_normalize_int_list(payload, name="ids"))
    if not isinstance(payload, dict):
        raise TypeError(f"unsupported annotation payload: {type(payload)!r}")
    ids = _normalize_int_list(payload["ids"], name="ids")
    epochs_value = payload.get("epochs")
    epochs = None
    if epochs_value is not None:
        epochs = _normalize_int_list(epochs_value, name="epochs")
        if len(ids) != len(epochs):
            raise ValueError("annotation ids and epochs must have the same length")
    return AnnotationSelection(ids=ids, epochs=epochs)


def annotation_path(split: str) -> Path:
    return benchmark_data_root() / "annotations" / f"{split}.json"


def scenario_refs(split: str, *, limit: int | None = None) -> list[ScenarioRef]:
    selection = load_annotations(annotation_path(split))
    ids = selection.ids[:limit]
    return [
        ScenarioRef(split=split, id_=id_, epoch=selection.epoch_at(index, default=1))
        for index, id_ in enumerate(ids)
    ]


def constellation_path(split: str, id_: int) -> Path:
    return benchmark_data_root() / "constellations" / split / f"{id_ // 1000:02d}" / f"{id_:05d}.json"


def taskset_path(split: str, id_: int) -> Path:
    return benchmark_data_root() / "tasksets" / split / f"{id_ // 1000:02d}" / f"{id_:05d}.json"


def trajectory_root_for_epoch(epoch: int) -> Path:
    return data_root() / f"trajectories.{int(epoch)}"


def trajectory_metrics_path(split: str, id_: int, *, epoch: int) -> Path:
    return trajectory_root_for_epoch(epoch) / split / f"{id_ // 1000:02d}" / f"{id_:05d}.json"


def trajectory_payload_path(split: str, id_: int, *, epoch: int) -> Path:
    return trajectory_root_for_epoch(epoch) / split / f"{id_ // 1000:02d}" / f"{id_:05d}.pth"
