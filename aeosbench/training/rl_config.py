"""Reinforcement-learning config loading."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any

import yaml

from aeosbench.evaluation.model_config import AEOSFormerConfig, parse_aeosformer_config
from aeosbench.paths import project_root


@dataclass(frozen=True)
class RLInitializationConfig:
    actor_checkpoint: Path


@dataclass(frozen=True)
class RLEnvironmentConfig:
    split: str
    annotation_file: str | None
    limit: int | None
    num_envs: int


@dataclass(frozen=True)
class RLRewardConfig:
    completion_bonus: float
    visible_satellite_bonus: float
    idle_satellite_penalty: float
    scale: float


@dataclass(frozen=True)
class PPOConfig:
    total_timesteps: int
    n_steps: int
    batch_size: int
    n_epochs: int
    learning_rate: float
    gamma: float
    gae_lambda: float
    clip_range: float
    ent_coef: float
    vf_coef: float
    save_freq: int


@dataclass(frozen=True)
class IterativeConfig:
    outer_iterations: int
    rollout_limit: int | None
    min_cr_improvement: float


@dataclass(frozen=True)
class RetrainingConfig:
    supervised_config: Path


@dataclass(frozen=True)
class ValidationEvalConfig:
    enabled: bool
    split: str
    limit: int | None


@dataclass(frozen=True)
class LoadedRLConfig:
    path: Path
    hash_: str
    model: AEOSFormerConfig
    initialization: RLInitializationConfig
    environment: RLEnvironmentConfig
    reward: RLRewardConfig
    ppo: PPOConfig
    iterative: IterativeConfig
    supervised: RetrainingConfig
    validation_eval: ValidationEvalConfig


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


def _optional_int(mapping: dict[str, Any], key: str, default: int | None = None) -> int | None:
    value = mapping.get(key, default)
    if value is None:
        return None
    if not isinstance(value, int):
        raise TypeError(f"{key} must be an int or null")
    return value


def _resolve_path(value: Any, *, base_dir: Path, name: str) -> Path:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    path = Path(value)
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return path


def _resolve_project_path(value: Any, *, name: str) -> Path:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    path = Path(value)
    if not path.is_absolute():
        path = (project_root() / path).resolve()
    return path


def load_rl_config(path: str | Path) -> LoadedRLConfig:
    resolved = Path(path).resolve()
    content = resolved.read_text(encoding="utf-8")
    payload = yaml.safe_load(content)
    if not isinstance(payload, dict):
        raise TypeError("rl config must be a mapping")

    model_payload = _require_mapping(payload, "model")
    initialization_payload = _require_mapping(payload, "initialization")
    environment_payload = _require_mapping(payload, "environment")
    reward_payload = _require_mapping(payload, "reward")
    ppo_payload = _require_mapping(payload, "ppo")
    iterative_payload = _require_mapping(payload, "iterative")
    supervised_payload = _require_mapping(payload, "supervised")
    validation_payload = payload.get("validation_eval", {"enabled": False})
    if not isinstance(validation_payload, dict):
        raise TypeError("validation_eval must be a mapping")

    config_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()[:12]
    num_envs = _require_int(environment_payload, "num_envs")
    if num_envs < 1:
        raise ValueError("environment.num_envs must be >= 1")

    outer_iterations = _require_int(iterative_payload, "outer_iterations")
    if outer_iterations < 1:
        raise ValueError("iterative.outer_iterations must be >= 1")

    return LoadedRLConfig(
        path=resolved,
        hash_=config_hash,
        model=parse_aeosformer_config(model_payload),
        initialization=RLInitializationConfig(
            actor_checkpoint=_resolve_project_path(
                initialization_payload.get("actor_checkpoint"),
                name="initialization.actor_checkpoint",
            ),
        ),
        environment=RLEnvironmentConfig(
            split=str(environment_payload.get("split", "train")),
            annotation_file=(
                None
                if environment_payload.get("annotation_file") is None
                else str(environment_payload["annotation_file"])
            ),
            limit=_optional_int(environment_payload, "limit", None),
            num_envs=num_envs,
        ),
        reward=RLRewardConfig(
            completion_bonus=_require_float(reward_payload, "completion_bonus"),
            visible_satellite_bonus=_require_float(reward_payload, "visible_satellite_bonus"),
            idle_satellite_penalty=_require_float(reward_payload, "idle_satellite_penalty"),
            scale=_require_float(reward_payload, "scale"),
        ),
        ppo=PPOConfig(
            total_timesteps=_require_int(ppo_payload, "total_timesteps"),
            n_steps=_require_int(ppo_payload, "n_steps"),
            batch_size=_require_int(ppo_payload, "batch_size"),
            n_epochs=_require_int(ppo_payload, "n_epochs"),
            learning_rate=_require_float(ppo_payload, "learning_rate"),
            gamma=_require_float(ppo_payload, "gamma"),
            gae_lambda=_require_float(ppo_payload, "gae_lambda"),
            clip_range=_require_float(ppo_payload, "clip_range"),
            ent_coef=_require_float(ppo_payload, "ent_coef"),
            vf_coef=_require_float(ppo_payload, "vf_coef"),
            save_freq=_require_int(ppo_payload, "save_freq"),
        ),
        iterative=IterativeConfig(
            outer_iterations=outer_iterations,
            rollout_limit=_optional_int(iterative_payload, "rollout_limit", None),
            min_cr_improvement=_require_float(iterative_payload, "min_cr_improvement"),
        ),
        supervised=RetrainingConfig(
            supervised_config=_resolve_project_path(
                supervised_payload.get("config"),
                name="supervised.config",
            ),
        ),
        validation_eval=ValidationEvalConfig(
            enabled=bool(validation_payload.get("enabled", False)),
            split=str(validation_payload.get("split", "val_seen")),
            limit=_optional_int(validation_payload, "limit", None),
        ),
    )
