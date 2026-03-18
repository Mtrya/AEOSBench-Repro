"""Statistics loading and normalization helpers."""

from __future__ import annotations

from collections import namedtuple
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
import sys
import types
from typing import Any, Iterator

import torch

from aeosbench.paths import data_root


@dataclass(frozen=True)
class Statistics:
    constellation_mean: torch.Tensor
    constellation_std: torch.Tensor
    taskset_mean: torch.Tensor
    taskset_std: torch.Tensor

    def normalize_constellation(self, values: torch.Tensor) -> torch.Tensor:
        return (values - self.constellation_mean) / (self.constellation_std + 1e-6)

    def normalize_taskset(self, values: torch.Tensor) -> torch.Tensor:
        return (values - self.taskset_mean) / (self.taskset_std + 1e-6)


def statistics_path() -> Path:
    return data_root() / "statistics_new.pth"


def _coerce_statistics(payload: Any) -> Statistics:
    if isinstance(payload, Statistics):
        return payload
    if isinstance(payload, dict):
        keys = {
            "constellation_mean",
            "constellation_std",
            "taskset_mean",
            "taskset_std",
        }
        if keys.issubset(payload):
            return Statistics(
                constellation_mean=torch.as_tensor(payload["constellation_mean"]),
                constellation_std=torch.as_tensor(payload["constellation_std"]),
                taskset_mean=torch.as_tensor(payload["taskset_mean"]),
                taskset_std=torch.as_tensor(payload["taskset_std"]),
            )
    fields = getattr(payload, "_fields", ())
    if tuple(fields) == (
        "constellation_mean",
        "constellation_std",
        "taskset_mean",
        "taskset_std",
    ):
        return Statistics(
            constellation_mean=torch.as_tensor(payload.constellation_mean),
            constellation_std=torch.as_tensor(payload.constellation_std),
            taskset_mean=torch.as_tensor(payload.taskset_mean),
            taskset_std=torch.as_tensor(payload.taskset_std),
        )
    raise TypeError(f"unsupported statistics payload: {type(payload)!r}")


@contextmanager
def _legacy_statistics_alias() -> Iterator[None]:
    saved = {
        "constellation": sys.modules.get("constellation"),
        "constellation.new_transformers": sys.modules.get("constellation.new_transformers"),
        "constellation.new_transformers.types": sys.modules.get(
            "constellation.new_transformers.types"
        ),
    }

    try:
        pkg = types.ModuleType("constellation")
        subpkg = types.ModuleType("constellation.new_transformers")
        types_module = types.ModuleType("constellation.new_transformers.types")
        legacy_statistics = namedtuple(
            "Statistics",
            ["constellation_mean", "constellation_std", "taskset_mean", "taskset_std"],
        )
        legacy_statistics.__module__ = "constellation.new_transformers.types"
        types_module.Statistics = legacy_statistics
        sys.modules["constellation"] = pkg
        sys.modules["constellation.new_transformers"] = subpkg
        sys.modules["constellation.new_transformers.types"] = types_module
        yield
    finally:
        for name, module in saved.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


def load_statistics(path: str | Path | None = None) -> Statistics:
    resolved = statistics_path() if path is None else Path(path)
    try:
        payload = torch.load(resolved, map_location="cpu", weights_only=False)
    except ModuleNotFoundError as exc:
        if exc.name != "constellation":
            raise
        with _legacy_statistics_alias():
            payload = torch.load(resolved, map_location="cpu", weights_only=False)
    return _coerce_statistics(payload)
