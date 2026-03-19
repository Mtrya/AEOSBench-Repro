from types import SimpleNamespace

import torch

from aeosbench.data import SensorType, Task, TaskSet
from aeosbench.sim.runtime import ScenarioRuntime, TaskManager


class _FakeEnvironment:
    def __init__(self, visibility: torch.Tensor) -> None:
        self.timer = SimpleNamespace(time=0)
        self.num_satellites = visibility.shape[0]
        self._visibility = visibility

    def is_visible(self, taskset: TaskSet) -> torch.Tensor:
        return self._visibility.clone()

    def apply_assignment(self, assignment: list[int], taskset: TaskSet) -> None:
        return None

    def step(self) -> None:
        self.timer.time += 1


def test_runtime_visible_satellite_count_masks_inactive_tasks():
    taskset = TaskSet(
        [
            Task(
                id_=0,
                release_time=0,
                due_time=5,
                duration=10,
                coordinate=(0.0, 0.0),
                sensor_type=SensorType.VISIBLE,
            ),
            Task(
                id_=1,
                release_time=10,
                due_time=20,
                duration=10,
                coordinate=(0.0, 0.0),
                sensor_type=SensorType.VISIBLE,
            ),
        ]
    )
    visibility = torch.tensor(
        [
            [False, True],
            [False, False],
        ],
        dtype=torch.bool,
    )
    environment = _FakeEnvironment(visibility)
    runtime = ScenarioRuntime(
        environment=environment,
        task_manager=TaskManager(taskset, environment.timer),
    )

    summary = runtime.step([-1, -1])

    assert summary.num_visible_satellites == 0
