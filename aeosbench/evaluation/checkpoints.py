"""Checkpoint loading helpers."""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Any

import torch

from aeosbench.models.aeosformer import AEOSFormerActor

from .model_config import AEOSFormerConfig

_IGNORED_TRAINING_KEYS = {
    "_transformer._time_model._mse_loss._weight._steps",
    "_transformer._time_model._bce_loss._weight._steps",
    "_ce_loss._weight._steps",
}


def _unwrap_state_dict(payload: Any) -> OrderedDict[str, torch.Tensor]:
    if isinstance(payload, OrderedDict):
        return payload
    if isinstance(payload, dict):
        state_dict = payload.get("state_dict")
        if isinstance(state_dict, OrderedDict):
            return state_dict
    raise TypeError(f"unsupported checkpoint payload: {type(payload)!r}")


def normalized_state_dict(path: str | Path) -> OrderedDict[str, torch.Tensor]:
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    state_dict = _unwrap_state_dict(payload)
    return OrderedDict(
        (key, value)
        for key, value in state_dict.items()
        if key not in _IGNORED_TRAINING_KEYS
    )


def build_actor(config: AEOSFormerConfig) -> AEOSFormerActor:
    return AEOSFormerActor(
        time_embedding_dim=config.time_embedding_dim,
        sensor_type_embedding_dim=config.sensor_type_embedding_dim,
        tasks_data_embedding_dim=config.tasks_data_embedding_dim,
        encoder_width=config.encoder_width,
        encoder_depth=config.encoder_depth,
        encoder_num_heads=config.encoder_num_heads,
        sensor_enabled_embedding_dim=config.sensor_enabled_embedding_dim,
        constellation_data_embedding_dim=config.constellation_data_embedding_dim,
        decoder_width=config.decoder_width,
        decoder_depth=config.decoder_depth,
        decoder_num_heads=config.decoder_num_heads,
        time_model_hidden_dim=config.time_model_hidden_dim,
    )


def load_actor_checkpoint(config: AEOSFormerConfig, path: str | Path) -> AEOSFormerActor:
    model = build_actor(config)
    state_dict = normalized_state_dict(path)
    missing, unexpected = model.load_state_dict(state_dict, strict=True)
    if missing or unexpected:
        raise RuntimeError(
            f"checkpoint mismatch for {path}: missing={missing}, unexpected={unexpected}"
        )
    model.eval()
    return model
