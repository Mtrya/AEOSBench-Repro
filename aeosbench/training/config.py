"""Supervised-training config loading."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any

import yaml

from aeosbench.evaluation.model_config import AEOSFormerConfig, parse_aeosformer_config


@dataclass(frozen=True)
class StatisticsConfig:
    mode: str
    path: Path | None


@dataclass(frozen=True)
class DataConfig:
    split: str
    annotation_file: str | None
    timesteps_per_scenario: int | None
    limit: int | None
    statistics: StatisticsConfig


@dataclass(frozen=True)
class ConstraintLabelConfig:
    min_positive_run_length: int
    max_time_horizon: int


@dataclass(frozen=True)
class LossWeightsConfig:
    feasibility: float
    timing: float
    assignment: float


@dataclass(frozen=True)
class OptimizerConfig:
    type: str
    lr: float
    betas: tuple[float, float]
    weight_decay: float
    eps: float


@dataclass(frozen=True)
class SchedulerConfig:
    type: str
    warmup_iters: int
    warmup_start_factor: float
    cosine_eta_min: float


@dataclass(frozen=True)
class TrainingRuntimeConfig:
    iterations: int
    gradient_accumulation_steps: int
    num_workers: int
    pin_memory: bool
    autocast: bool
    log_interval: int
    checkpoint_interval: int
    validation_interval: int


@dataclass(frozen=True)
class ValidationConfig:
    split: str
    annotation_file: str | None
    timesteps_per_scenario: int | None
    max_scenarios: int | None
    enabled: bool


@dataclass(frozen=True)
class LoadedTrainingConfig:
    path: Path
    hash_: str
    model: AEOSFormerConfig
    data: DataConfig
    constraint_labels: ConstraintLabelConfig
    loss_weights: LossWeightsConfig
    optimizer: OptimizerConfig
    scheduler: SchedulerConfig
    training: TrainingRuntimeConfig
    validation: ValidationConfig


def _require_mapping(mapping: dict[str, Any], key: str) -> dict[str, Any]:
    value = mapping.get(key)
    if not isinstance(value, dict):
        raise TypeError(f"{key} must be a mapping")
    return value


def _require_int(mapping: dict[str, Any], key: str) -> int:
    value = mapping.get(key)
    if not isinstance(value, int):
        raise TypeError(f"{key} must be an int")
    return value


def _optional_int(mapping: dict[str, Any], key: str, default: int | None = None) -> int | None:
    value = mapping.get(key, default)
    if value is None:
        return None
    if not isinstance(value, int):
        raise TypeError(f"{key} must be an int or null")
    return value


def _optional_int_alias(
    mapping: dict[str, Any],
    primary_key: str,
    alias_keys: tuple[str, ...],
    default: int | None = None,
) -> int | None:
    present = [key for key in (primary_key, *alias_keys) if key in mapping]
    if len(present) > 1:
        raise ValueError(f"only one of {', '.join(repr(key) for key in (primary_key, *alias_keys))} may be set")
    if primary_key in mapping:
        return _optional_int(mapping, primary_key, default)
    for alias_key in alias_keys:
        if alias_key in mapping:
            return _optional_int(mapping, alias_key, default)
    return default


def _require_bool(mapping: dict[str, Any], key: str) -> bool:
    value = mapping.get(key)
    if not isinstance(value, bool):
        raise TypeError(f"{key} must be a bool")
    return value


def _require_float(mapping: dict[str, Any], key: str) -> float:
    value = mapping.get(key)
    if not isinstance(value, (int, float)):
        raise TypeError(f"{key} must be a float")
    return float(value)


def _optional_path(value: Any, *, base_dir: Path) -> Path | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError("statistics.path must be a string or null")
    path = Path(value)
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return path


def load_training_config(path: str | Path) -> LoadedTrainingConfig:
    resolved = Path(path).resolve()
    content = resolved.read_text(encoding="utf-8")
    payload = yaml.safe_load(content)
    if not isinstance(payload, dict):
        raise TypeError("training config must be a mapping")

    model_payload = _require_mapping(payload, "model")
    data_payload = _require_mapping(payload, "data")
    statistics_payload = _require_mapping(data_payload, "statistics")
    constraint_payload = _require_mapping(payload, "constraint_labels")
    loss_weights_payload = payload.get("loss_weights", {})
    if not isinstance(loss_weights_payload, dict):
        raise TypeError("loss_weights must be a mapping")
    optimizer_payload = _require_mapping(payload, "optimizer")
    scheduler_payload = _require_mapping(payload, "scheduler")
    training_payload = _require_mapping(payload, "training")
    validation_payload = _require_mapping(payload, "validation")

    statistics_mode = statistics_payload.get("mode")
    if statistics_mode not in {"load_or_compute", "load_only", "compute_only"}:
        raise ValueError(f"unsupported statistics.mode: {statistics_mode!r}")
    optimizer_type = optimizer_payload.get("type")
    if optimizer_type != "adamw":
        raise ValueError(f"unsupported optimizer.type: {optimizer_type!r}")
    scheduler_type = scheduler_payload.get("type")
    if scheduler_type != "warmup_cosine":
        raise ValueError(f"unsupported scheduler.type: {scheduler_type!r}")
    gradient_accumulation_steps = _optional_int(
        training_payload,
        "gradient_accumulation_steps",
        1,
    )
    if gradient_accumulation_steps is None or gradient_accumulation_steps < 1:
        raise ValueError("training.gradient_accumulation_steps must be >= 1")

    betas = optimizer_payload.get("betas")
    if not (
        isinstance(betas, list)
        and len(betas) == 2
        and all(isinstance(beta, (int, float)) for beta in betas)
    ):
        raise TypeError("optimizer.betas must be a [beta1, beta2] list")

    config_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()[:12]
    return LoadedTrainingConfig(
        path=resolved,
        hash_=config_hash,
        model=parse_aeosformer_config(model_payload),
        data=DataConfig(
            split=str(data_payload.get("split", "train")),
            annotation_file=(
                None
                if data_payload.get("annotation_file") is None
                else str(data_payload["annotation_file"])
            ),
            timesteps_per_scenario=_optional_int_alias(
                data_payload,
                "timesteps_per_scenario",
                ("timesteps_per_sample",),
                48,
            ),
            limit=_optional_int(data_payload, "limit", None),
            statistics=StatisticsConfig(
                mode=statistics_mode,
                path=_optional_path(
                    statistics_payload.get("path"),
                    base_dir=resolved.parent,
                ),
            ),
        ),
        constraint_labels=ConstraintLabelConfig(
            min_positive_run_length=_require_int(
                constraint_payload,
                "min_positive_run_length",
            ),
            max_time_horizon=_require_int(
                constraint_payload,
                "max_time_horizon",
            ),
        ),
        loss_weights=LossWeightsConfig(
            feasibility=_require_float(loss_weights_payload, "feasibility")
            if "feasibility" in loss_weights_payload
            else 1.0,
            timing=_require_float(loss_weights_payload, "timing")
            if "timing" in loss_weights_payload
            else 1.0,
            assignment=_require_float(loss_weights_payload, "assignment")
            if "assignment" in loss_weights_payload
            else 1.0,
        ),
        optimizer=OptimizerConfig(
            type=optimizer_type,
            lr=_require_float(optimizer_payload, "lr"),
            betas=(float(betas[0]), float(betas[1])),
            weight_decay=_require_float(optimizer_payload, "weight_decay"),
            eps=_require_float(optimizer_payload, "eps"),
        ),
        scheduler=SchedulerConfig(
            type=scheduler_type,
            warmup_iters=_require_int(scheduler_payload, "warmup_iters"),
            warmup_start_factor=_require_float(
                scheduler_payload,
                "warmup_start_factor",
            ),
            cosine_eta_min=_require_float(scheduler_payload, "cosine_eta_min"),
        ),
        training=TrainingRuntimeConfig(
            iterations=_require_int(training_payload, "iterations"),
            gradient_accumulation_steps=gradient_accumulation_steps,
            num_workers=_require_int(training_payload, "num_workers"),
            pin_memory=_require_bool(training_payload, "pin_memory"),
            autocast=_require_bool(training_payload, "autocast"),
            log_interval=_require_int(training_payload, "log_interval"),
            checkpoint_interval=_require_int(training_payload, "checkpoint_interval"),
            validation_interval=_require_int(training_payload, "validation_interval"),
        ),
        validation=ValidationConfig(
            split=str(validation_payload.get("split", "val_seen")),
            annotation_file=(
                None
                if validation_payload.get("annotation_file") is None
                else str(validation_payload["annotation_file"])
            ),
            timesteps_per_scenario=_optional_int_alias(
                validation_payload,
                "timesteps_per_scenario",
                ("timesteps_per_sample",),
                48,
            ),
            max_scenarios=_optional_int(validation_payload, "max_scenarios", None),
            enabled=_require_bool(validation_payload, "enabled"),
        ),
    )
