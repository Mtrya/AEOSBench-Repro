from collections import namedtuple
import sys
import types

import torch

from aeosbench.evaluation.statistics import Statistics, load_statistics


def test_load_statistics_supports_native_mapping(tmp_path):
    path = tmp_path / "statistics.pth"
    payload = {
        "constellation_mean": torch.ones(56),
        "constellation_std": torch.full((56,), 2.0),
        "taskset_mean": torch.ones(6),
        "taskset_std": torch.full((6,), 3.0),
    }
    torch.save(payload, path)

    statistics = load_statistics(path)

    assert isinstance(statistics, Statistics)
    assert statistics.constellation_mean.shape == (56,)
    assert statistics.taskset_std.shape == (6,)


def test_load_statistics_supports_legacy_namedtuple(tmp_path):
    path = tmp_path / "statistics.pth"
    statistics_tuple = namedtuple(
        "Statistics",
        ["constellation_mean", "constellation_std", "taskset_mean", "taskset_std"],
    )
    statistics_tuple.__module__ = "constellation.new_transformers.types"

    constellation_pkg = types.ModuleType("constellation")
    new_transformers_pkg = types.ModuleType("constellation.new_transformers")
    types_module = types.ModuleType("constellation.new_transformers.types")
    types_module.Statistics = statistics_tuple
    saved_modules = {
        "constellation": sys.modules.get("constellation"),
        "constellation.new_transformers": sys.modules.get("constellation.new_transformers"),
        "constellation.new_transformers.types": sys.modules.get(
            "constellation.new_transformers.types"
        ),
    }
    sys.modules["constellation"] = constellation_pkg
    sys.modules["constellation.new_transformers"] = new_transformers_pkg
    sys.modules["constellation.new_transformers.types"] = types_module
    try:
        torch.save(
            statistics_tuple(
                torch.zeros(56),
                torch.ones(56),
                torch.zeros(6),
                torch.ones(6),
            ),
            path,
        )
    finally:
        for name, module in saved_modules.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module

    statistics = load_statistics(path)

    assert isinstance(statistics, Statistics)
    assert statistics.constellation_std.shape == (56,)
    assert statistics.taskset_mean.shape == (6,)
