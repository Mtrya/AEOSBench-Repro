"""Gymnasium environment for AEOSBench PPO training."""

from __future__ import annotations

from functools import partial
import random
from pathlib import Path
from typing import Any, TypedDict, cast

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import numpy.typing as npt
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
import torch

from aeosbench.constants import MAX_TIME_STEP, SATELLITE_DIM, TASK_DIM
from aeosbench.data import Constellation, SensorType, TaskSet
from aeosbench.evaluation.basilisk_env import BasiliskEnvironment
from aeosbench.evaluation.layout import annotation_path, load_annotations
from aeosbench.evaluation.statistics import Statistics, load_statistics
from aeosbench.paths import benchmark_data_root
from aeosbench.sim import ScenarioRuntime, TaskManager, build_actor_observation
from aeosbench.training.rl_config import RLEnvironmentConfig, RLRewardConfig

MAX_NUM_SATELLITES = 51
MAX_NUM_TASKS = 302


class Observation(TypedDict):
    num_satellites: int
    num_tasks: int
    time_step: int
    constellation_sensor_type: npt.NDArray[np.uint8]
    constellation_sensor_enabled: npt.NDArray[np.uint8]
    constellation_data: npt.NDArray[np.float32]
    tasks_sensor_type: npt.NDArray[np.uint8]
    tasks_data: npt.NDArray[np.float32]


null_observation = Observation(
    num_satellites=1,
    num_tasks=1,
    time_step=1,
    constellation_sensor_type=np.zeros(MAX_NUM_SATELLITES, np.uint8),
    constellation_sensor_enabled=np.zeros(MAX_NUM_SATELLITES, np.uint8),
    constellation_data=np.zeros((MAX_NUM_SATELLITES, SATELLITE_DIM), np.float32),
    tasks_sensor_type=np.zeros(MAX_NUM_TASKS, np.uint8),
    tasks_data=np.zeros((MAX_NUM_TASKS, TASK_DIM), np.float32),
)


def _resolve_annotation_file(split: str, annotation_file: str | None) -> str:
    if annotation_file is None:
        return str(annotation_path(split))
    candidate = Path(annotation_file)
    if candidate.is_absolute():
        return str(candidate)
    return str(annotation_path(split).parent / candidate)


def _load_refs(config: RLEnvironmentConfig) -> list[int]:
    selection = load_annotations(_resolve_annotation_file(config.split, config.annotation_file))
    return selection.ids[: config.limit]


def _constellation_path(split: str, id_: int) -> str:
    return str(benchmark_data_root() / "constellations" / split / f"{id_ // 1000:02d}" / f"{id_:05d}.json")


def _taskset_path(split: str, id_: int) -> str:
    return str(benchmark_data_root() / "tasksets" / split / f"{id_ // 1000:02d}" / f"{id_:05d}.json")


class Padding:
    def __call__(self, observation: Observation) -> Observation:
        padded = Observation(
            num_satellites=observation["num_satellites"],
            num_tasks=observation["num_tasks"],
            time_step=observation["time_step"],
            constellation_sensor_type=null_observation["constellation_sensor_type"].copy(),
            constellation_sensor_enabled=null_observation["constellation_sensor_enabled"].copy(),
            constellation_data=null_observation["constellation_data"].copy(),
            tasks_sensor_type=null_observation["tasks_sensor_type"].copy(),
            tasks_data=null_observation["tasks_data"].copy(),
        )
        ns = observation["constellation_data"].shape[0]
        nt = observation["tasks_data"].shape[0]
        padded["constellation_sensor_type"][:ns] = observation["constellation_sensor_type"]
        padded["constellation_sensor_enabled"][:ns] = observation["constellation_sensor_enabled"]
        padded["constellation_data"][:ns] = observation["constellation_data"]
        padded["tasks_sensor_type"][:nt] = observation["tasks_sensor_type"]
        padded["tasks_data"][:nt] = observation["tasks_data"]
        return padded


class RLEnvironment(gym.Env[Observation, npt.NDArray[np.int32]]):
    @classmethod
    def build(
        cls,
        environment_config: RLEnvironmentConfig,
        reward_config: RLRewardConfig,
        *,
        statistics: Statistics,
        seed: int,
    ) -> gym.Env[Any, Any]:
        num_envs = environment_config.num_envs
        factory = partial(
            cls,
            environment_config=environment_config,
            reward_config=reward_config,
            statistics=statistics,
            seed=seed,
        )
        if num_envs == 1:
            return DummyVecEnv([factory])
        return SubprocVecEnv([factory for _ in range(num_envs)], start_method="spawn")

    def __init__(
        self,
        *,
        environment_config: RLEnvironmentConfig,
        reward_config: RLRewardConfig,
        statistics: Statistics,
        seed: int,
    ) -> None:
        super().__init__()
        self._split = environment_config.split
        self._refs = _load_refs(environment_config)
        if not self._refs:
            raise RuntimeError(f"no RL scenarios available for split {self._split!r}")
        self._reward = reward_config
        self._statistics = statistics
        self._rng = random.Random(seed)
        self._padding = Padding()
        self._current_id: int | None = None
        self._runtime: ScenarioRuntime | None = None

        self.observation_space = spaces.Dict(  # type: ignore[assignment]
            dict(
                num_satellites=spaces.Discrete(MAX_NUM_SATELLITES),
                num_tasks=spaces.Discrete(MAX_NUM_TASKS),
                time_step=spaces.Discrete(MAX_TIME_STEP),
                constellation_sensor_type=spaces.MultiDiscrete([len(SensorType)] * MAX_NUM_SATELLITES),
                constellation_sensor_enabled=spaces.MultiBinary(MAX_NUM_SATELLITES),
                constellation_data=spaces.Box(
                    low=-1e5,
                    high=1e5,
                    shape=(MAX_NUM_SATELLITES, SATELLITE_DIM),
                    dtype=np.float32,
                ),
                tasks_sensor_type=spaces.MultiDiscrete([len(SensorType)] * MAX_NUM_TASKS),
                tasks_data=spaces.Box(
                    low=-1e3,
                    high=1e3,
                    shape=(MAX_NUM_TASKS, TASK_DIM),
                    dtype=np.float32,
                ),
            )
        )
        self.action_space = spaces.MultiDiscrete([MAX_NUM_TASKS] * MAX_NUM_SATELLITES)  # type: ignore[assignment]

    def _require_runtime(self) -> ScenarioRuntime:
        if self._runtime is None:
            raise RuntimeError("environment is not initialized")
        return self._runtime

    def _select_id(self) -> int:
        return self._rng.choice(self._refs)

    def _current_observation(self) -> Observation:
        runtime = self._require_runtime()
        raw = build_actor_observation(
            runtime.environment,
            runtime.task_manager,
            self._statistics,
            device=torch.device("cpu"),
        )
        observation = Observation(
            num_satellites=int(raw["constellation_data"].shape[1]),
            num_tasks=int(raw["tasks_data"].shape[1]),
            time_step=int(raw["time_steps"][0].item()),
            constellation_sensor_type=cast(
                npt.NDArray[np.uint8],
                raw["constellation_sensor_type"].squeeze(0).numpy().astype(np.uint8),
            ),
            constellation_sensor_enabled=cast(
                npt.NDArray[np.uint8],
                raw["constellation_sensor_enabled"].squeeze(0).numpy().astype(np.uint8),
            ),
            constellation_data=cast(
                npt.NDArray[np.float32],
                raw["constellation_data"].squeeze(0).numpy().astype(np.float32),
            ),
            tasks_sensor_type=cast(
                npt.NDArray[np.uint8],
                raw["tasks_sensor_type"].squeeze(0).numpy().astype(np.uint8),
            ),
            tasks_data=cast(
                npt.NDArray[np.float32],
                raw["tasks_data"].squeeze(0).numpy().astype(np.float32),
            ),
        )
        return self._padding(observation)

    def _decode_action(self, action: npt.NDArray[np.int32]) -> list[int]:
        runtime = self._require_runtime()
        active = action[: runtime.environment.num_satellites]
        num_tasks = len(runtime.task_manager.ongoing_tasks)
        decoded: list[int] = []
        for value in active.tolist():
            task_index = int(value) - 1
            if task_index < -1 or task_index >= num_tasks:
                task_index = -1
            decoded.append(task_index)
        return decoded

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[Observation, dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self._rng.seed(seed)

        id_ = self._select_id()
        self._current_id = id_
        constellation = Constellation.load(_constellation_path(self._split, id_))
        taskset = TaskSet.load(_taskset_path(self._split, id_))
        environment = BasiliskEnvironment(constellation=constellation, taskset=taskset)
        self._runtime = ScenarioRuntime(
            environment=environment,
            task_manager=TaskManager(taskset, environment.timer),
        )
        self._runtime.skip_idle()
        if self._runtime.done:
            return null_observation, dict(id=id_)
        return self._current_observation(), dict(id=id_)

    def step(
        self,
        action: npt.NDArray[np.int32],
    ) -> tuple[Observation, float, bool, bool, dict[str, Any]]:
        runtime = self._require_runtime()
        assignment = self._decode_action(action)
        summary = runtime.step(assignment)
        reward = (
            (self._reward.completion_bonus * summary.newly_completed_tasks)
            + (self._reward.visible_satellite_bonus * summary.num_visible_satellites)
            - (self._reward.satellite_existence_cost * runtime.environment.num_satellites)
        ) / self._reward.scale
        terminated = runtime.task_manager.all_closed
        truncated = runtime.environment.timer.time >= MAX_TIME_STEP
        if terminated or truncated:
            return null_observation, reward, terminated, truncated, {"id": self._current_id}
        return self._current_observation(), reward, terminated, truncated, {"id": self._current_id}

    def close(self) -> None:
        self._runtime = None
        super().close()


def load_rl_statistics() -> Statistics:
    return load_statistics()
