"""Released dataset layout helpers."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import random
from typing import Any

from aeosbench.paths import benchmark_data_root, data_root


def _resolved_dataset_root(dataset_root: Path | None) -> Path:
    return data_root() if dataset_root is None else Path(dataset_root)


def _resolved_benchmark_root(dataset_root: Path | None) -> Path:
    if dataset_root is None:
        return benchmark_data_root()
    return Path(dataset_root) / "data"


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


def _annotation_split_name(split: str) -> str:
    if split == "test_official64":
        return "test"
    return split


def _data_split_name(split: str) -> str:
    if split in {"test_official64", "test_random64", "test_all"}:
        return "test"
    return split


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


def annotation_path(split: str, *, dataset_root: Path | None = None) -> Path:
    return _resolved_benchmark_root(dataset_root) / "annotations" / f"{_annotation_split_name(split)}.json"


def raw_scenario_ids(split: str, *, dataset_root: Path | None = None) -> list[int]:
    root = _resolved_benchmark_root(dataset_root) / "constellations" / _data_split_name(split)
    return sorted(int(path.stem) for path in root.rglob("*.json"))


def _scenario_refs_from_ids(
    split: str,
    ids: list[int],
    *,
    epoch: int,
) -> list[ScenarioRef]:
    data_split = _data_split_name(split)
    return [ScenarioRef(split=data_split, id_=id_, epoch=epoch) for id_ in ids]


def scenario_refs(
    split: str,
    *,
    limit: int | None = None,
    dataset_root: Path | None = None,
) -> list[ScenarioRef]:
    if split == "test_random64":
        ids = sorted(random.Random(42).sample(raw_scenario_ids("test", dataset_root=dataset_root), 64))
        return _scenario_refs_from_ids(split, ids[:limit], epoch=1)
    if split == "test_all":
        ids = raw_scenario_ids("test", dataset_root=dataset_root)
        return _scenario_refs_from_ids(split, ids[:limit], epoch=1)

    selection = load_annotations(annotation_path(split, dataset_root=dataset_root))
    ids = selection.ids[:limit]
    return [
        ScenarioRef(
            split=_data_split_name(split),
            id_=id_,
            epoch=selection.epoch_at(index, default=1),
        )
        for index, id_ in enumerate(ids)
    ]


def constellation_path(split: str, id_: int, *, dataset_root: Path | None = None) -> Path:
    return _resolved_benchmark_root(dataset_root) / "constellations" / split / f"{id_ // 1000:02d}" / f"{id_:05d}.json"


def taskset_path(split: str, id_: int, *, dataset_root: Path | None = None) -> Path:
    return _resolved_benchmark_root(dataset_root) / "tasksets" / split / f"{id_ // 1000:02d}" / f"{id_:05d}.json"


def trajectory_root_for_epoch(epoch: int, *, dataset_root: Path | None = None) -> Path:
    return _resolved_dataset_root(dataset_root) / f"trajectories.{int(epoch)}"


def trajectory_metrics_path(split: str, id_: int, *, epoch: int, dataset_root: Path | None = None) -> Path:
    return trajectory_root_for_epoch(epoch, dataset_root=dataset_root) / split / f"{id_ // 1000:02d}" / f"{id_:05d}.json"


def trajectory_payload_path(split: str, id_: int, *, epoch: int, dataset_root: Path | None = None) -> Path:
    return trajectory_root_for_epoch(epoch, dataset_root=dataset_root) / split / f"{id_ // 1000:02d}" / f"{id_:05d}.pth"
