from aeosbench.evaluation.statistics import Statistics
from aeosbench.training.rl_config import RLEnvironmentConfig, RLRewardConfig
from aeosbench.training.rl_env import RLEnvironment


def _dummy_statistics() -> Statistics:
    import torch

    return Statistics(
        constellation_mean=torch.zeros(56),
        constellation_std=torch.ones(56),
        taskset_mean=torch.zeros(6),
        taskset_std=torch.ones(6),
    )


def test_rl_env_build_uses_spawn_for_subprocess_envs(monkeypatch):
    seen: dict[str, object] = {}

    def fake_subproc(factories, start_method=None):
        seen["num_factories"] = len(factories)
        seen["start_method"] = start_method
        return object()

    monkeypatch.setattr("aeosbench.training.rl_env.SubprocVecEnv", fake_subproc)

    RLEnvironment.build(
        RLEnvironmentConfig(
            split="train",
            annotation_file=None,
            limit=1,
            num_envs=2,
        ),
        RLRewardConfig(
            completion_bonus=100.0,
            visible_satellite_bonus=2.0,
            satellite_existence_cost=1.0,
            scale=10.0,
        ),
        statistics=_dummy_statistics(),
        seed=3407,
    )

    assert seen["num_factories"] == 2
    assert seen["start_method"] == "spawn"


def test_rl_env_decode_action_clamps_out_of_range_indices_to_idle():
    env = object.__new__(RLEnvironment)
    env._runtime = type(
        "Runtime",
        (),
        {
            "environment": type("Environment", (), {"num_satellites": 2})(),
            "task_manager": type("TaskManager", (), {"ongoing_tasks": [object(), object()]})(),
        },
    )()

    decoded = env._decode_action(__import__("numpy").array([0, 5, 9], dtype="int32"))

    assert decoded == [-1, -1]
