"""Evaluation model-config parsing."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import hashlib
from typing import Any

import yaml


@dataclass(frozen=True)
class AEOSFormerConfig:
    time_embedding_dim: int
    sensor_type_embedding_dim: int
    tasks_data_embedding_dim: int
    encoder_width: int
    encoder_depth: int
    encoder_num_heads: int
    sensor_enabled_embedding_dim: int
    constellation_data_embedding_dim: int
    decoder_width: int
    decoder_depth: int
    decoder_num_heads: int
    time_model_hidden_dim: int


@dataclass(frozen=True)
class LoadedModelConfig:
    path: Path
    hash_: str
    model: AEOSFormerConfig


def _require_int(mapping: dict[str, Any], key: str) -> int:
    value = mapping.get(key)
    if not isinstance(value, int):
        raise TypeError(f"model.{key} must be an int")
    return value


def load_model_config(path: str | Path) -> LoadedModelConfig:
    resolved = Path(path)
    content = resolved.read_text(encoding="utf-8")
    payload = yaml.safe_load(content)
    if not isinstance(payload, dict):
        raise TypeError("model config must be a mapping")
    model_payload = payload.get("model")
    if not isinstance(model_payload, dict):
        raise TypeError("model config must define a top-level 'model' mapping")
    model_type = model_payload.get("type")
    if model_type != "aeosformer":
        raise ValueError(f"unsupported model.type: {model_type!r}")
    model = AEOSFormerConfig(
        time_embedding_dim=_require_int(model_payload, "time_embedding_dim"),
        sensor_type_embedding_dim=_require_int(model_payload, "sensor_type_embedding_dim"),
        tasks_data_embedding_dim=_require_int(model_payload, "tasks_data_embedding_dim"),
        encoder_width=_require_int(model_payload, "encoder_width"),
        encoder_depth=_require_int(model_payload, "encoder_depth"),
        encoder_num_heads=_require_int(model_payload, "encoder_num_heads"),
        sensor_enabled_embedding_dim=_require_int(
            model_payload,
            "sensor_enabled_embedding_dim",
        ),
        constellation_data_embedding_dim=_require_int(
            model_payload,
            "constellation_data_embedding_dim",
        ),
        decoder_width=_require_int(model_payload, "decoder_width"),
        decoder_depth=_require_int(model_payload, "decoder_depth"),
        decoder_num_heads=_require_int(model_payload, "decoder_num_heads"),
        time_model_hidden_dim=_require_int(model_payload, "time_model_hidden_dim"),
    )
    config_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()[:12]
    return LoadedModelConfig(path=resolved, hash_=config_hash, model=model)
