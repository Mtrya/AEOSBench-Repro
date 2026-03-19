"""YAML config loading for dataset generation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


def _required_mapping(payload: dict[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise TypeError(f"{key} must be a mapping")
    return value


def _required_int(payload: dict[str, Any], key: str) -> int:
    if key not in payload:
        raise KeyError(f"missing required key: {key}")
    return int(payload[key])


def _optional_int(payload: dict[str, Any], key: str, default: int) -> int:
    if key not in payload:
        return default
    return int(payload[key])


def _optional_float(payload: dict[str, Any], key: str, default: float) -> float:
    if key not in payload:
        return default
    return float(payload[key])


def _optional_bool(payload: dict[str, Any], key: str, default: bool) -> bool:
    if key not in payload:
        return default
    return bool(payload[key])


@dataclass(frozen=True)
class SplitCounts:
    train: int
    val_seen: int
    val_unseen: int
    test: int


@dataclass(frozen=True)
class AssetsConfig:
    train_count: int
    val_unseen_count: int
    test_count: int
    retry_factor: int
    screening_task_count: int
    screening_horizon: int
    screening_acceptance_threshold: float


@dataclass(frozen=True)
class ScenariosConfig:
    counts: SplitCounts
    min_satellites: int
    max_satellites: int


@dataclass(frozen=True)
class TasksConfig:
    min_tasks: int
    max_tasks: int
    min_duration: int
    max_duration: int
    max_time_step: int


@dataclass(frozen=True)
class RolloutsConfig:
    epoch: int
    max_time_step: int
    enabled: bool


@dataclass(frozen=True)
class CurationConfig:
    min_completion_rate: float


@dataclass(frozen=True)
class StatisticsConfig:
    enabled: bool


@dataclass(frozen=True)
class DatasetGenerationConfig:
    seed: int
    assets: AssetsConfig
    scenarios: ScenariosConfig
    tasks: TasksConfig
    rollouts: RolloutsConfig
    curation: CurationConfig
    statistics: StatisticsConfig


def load_generation_config(path: str | Path) -> DatasetGenerationConfig:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("generation config must be a mapping")

    assets_payload = _required_mapping(payload, "assets")
    scenarios_payload = _required_mapping(payload, "scenarios")
    counts_payload = _required_mapping(scenarios_payload, "counts")
    tasks_payload = _required_mapping(payload, "tasks")
    rollouts_payload = _required_mapping(payload, "rollouts")
    curation_payload = _required_mapping(payload, "curation")
    statistics_payload = _required_mapping(payload, "statistics")

    return DatasetGenerationConfig(
        seed=int(payload.get("seed", 42)),
        assets=AssetsConfig(
            train_count=_required_int(assets_payload, "train_count"),
            val_unseen_count=_required_int(assets_payload, "val_unseen_count"),
            test_count=_required_int(assets_payload, "test_count"),
            retry_factor=_optional_int(assets_payload, "retry_factor", 8),
            screening_task_count=_optional_int(assets_payload, "screening_task_count", 36),
            screening_horizon=_optional_int(assets_payload, "screening_horizon", 7200),
            screening_acceptance_threshold=_optional_float(
                assets_payload,
                "screening_acceptance_threshold",
                0.99,
            ),
        ),
        scenarios=ScenariosConfig(
            counts=SplitCounts(
                train=_required_int(counts_payload, "train"),
                val_seen=_required_int(counts_payload, "val_seen"),
                val_unseen=_required_int(counts_payload, "val_unseen"),
                test=_required_int(counts_payload, "test"),
            ),
            min_satellites=_optional_int(scenarios_payload, "min_satellites", 1),
            max_satellites=_optional_int(scenarios_payload, "max_satellites", 50),
        ),
        tasks=TasksConfig(
            min_tasks=_optional_int(tasks_payload, "min_tasks", 50),
            max_tasks=_optional_int(tasks_payload, "max_tasks", 300),
            min_duration=_optional_int(tasks_payload, "min_duration", 15),
            max_duration=_optional_int(tasks_payload, "max_duration", 60),
            max_time_step=_optional_int(tasks_payload, "max_time_step", 3600),
        ),
        rollouts=RolloutsConfig(
            epoch=_optional_int(rollouts_payload, "epoch", 1),
            max_time_step=_optional_int(rollouts_payload, "max_time_step", 3600),
            enabled=_optional_bool(rollouts_payload, "enabled", True),
        ),
        curation=CurationConfig(
            min_completion_rate=_optional_float(curation_payload, "min_completion_rate", 0.0),
        ),
        statistics=StatisticsConfig(
            enabled=_optional_bool(statistics_payload, "enabled", True),
        ),
    )
