"""Shared Basilisk rollout runtime for evaluation and RL."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from aeosbench.constants import MAX_TIME_STEP
from aeosbench.data import TaskSet
from aeosbench.evaluation.metrics import RawMetrics
from aeosbench.evaluation.statistics import Statistics
from aeosbench.evaluation.basilisk_env import BasiliskEnvironment


class TaskManager:
    def __init__(self, taskset: TaskSet, timer: Any) -> None:
        self.taskset = taskset
        self.timer = timer
        self.progress = torch.zeros(len(taskset), dtype=torch.float32)
        self.succeeded_flags = torch.zeros(len(taskset), dtype=torch.bool)

    @property
    def ongoing_flags(self) -> torch.Tensor:
        current_time = self.timer.time
        return ~self.succeeded_flags & torch.tensor(
            [task.release_time <= current_time <= task.due_time for task in self.taskset],
            dtype=torch.bool,
        )

    @property
    def ongoing_tasks(self) -> TaskSet:
        return TaskSet(task for task, flag in zip(self.taskset, self.ongoing_flags) if bool(flag))

    @property
    def num_succeeded_tasks(self) -> int:
        return int(self.succeeded_flags.sum().item())

    @property
    def all_closed(self) -> bool:
        current_time = self.timer.time
        closed_flags = self.succeeded_flags | torch.tensor(
            [task.due_time < current_time for task in self.taskset],
            dtype=torch.bool,
        )
        return bool(closed_flags.all().item())

    @property
    def is_idle(self) -> bool:
        return len(self.ongoing_tasks) == 0

    def record(self, visibility: torch.Tensor) -> int:
        durations = torch.tensor([task.duration for task in self.taskset], dtype=torch.float32)
        masked_visibility = visibility.clone()
        masked_visibility[:, ~self.ongoing_flags] = False
        any_visible = masked_visibility.any(dim=0)
        completed_before = self.num_succeeded_tasks
        self.progress = (self.progress + 1.0) * any_visible.to(dtype=torch.float32)
        self.succeeded_flags |= self.progress >= durations
        return self.num_succeeded_tasks - completed_before


@dataclass
class MetricsAccumulator:
    durations: torch.Tensor
    release_times: torch.Tensor
    max_progress: torch.Tensor
    completion_time: torch.Tensor
    working_time_steps: torch.Tensor

    @classmethod
    def create(cls, taskset: TaskSet, *, num_satellites: int) -> "MetricsAccumulator":
        durations = torch.tensor([task.duration for task in taskset], dtype=torch.float32)
        return cls(
            durations=durations,
            release_times=torch.tensor([task.release_time for task in taskset], dtype=torch.float32),
            max_progress=torch.zeros(len(taskset), dtype=torch.float32),
            completion_time=torch.full((len(taskset),), float("inf")),
            working_time_steps=torch.zeros(num_satellites, dtype=torch.float32),
        )

    def update(
        self,
        *,
        task_manager: TaskManager,
        environment: BasiliskEnvironment,
        assignment: list[int],
    ) -> None:
        self.max_progress = torch.maximum(self.max_progress, task_manager.progress)
        if task_manager.succeeded_flags.any():
            self.completion_time[task_manager.succeeded_flags] = torch.minimum(
                self.completion_time[task_manager.succeeded_flags],
                torch.full_like(
                    self.completion_time[task_manager.succeeded_flags],
                    float(environment.timer.time),
                ),
            )
        self.working_time_steps += (torch.tensor(assignment) != -1).to(dtype=torch.float32)

    def finalize(self, *, task_manager: TaskManager, environment: BasiliskEnvironment) -> RawMetrics:
        succeeded_flags = task_manager.succeeded_flags
        cr = float(task_manager.num_succeeded_tasks / len(task_manager.taskset))
        wcr = float(self.durations[succeeded_flags].sum().item() / self.durations.sum().item())
        pcr = float((self.max_progress / self.durations).mean().item())
        if succeeded_flags.any():
            tat_seconds = float(
                (self.completion_time[succeeded_flags] - self.release_times[succeeded_flags])
                .mean()
                .item()
            )
        else:
            tat_seconds = float("inf")
        current_constellation = environment.get_constellation().sort()
        sensor_power = torch.tensor(
            [satellite.sensor.power for satellite in current_constellation],
            dtype=torch.float32,
        )
        pc_watt_seconds = float((self.working_time_steps * sensor_power).sum().item())
        return RawMetrics(
            cr=cr,
            pcr=pcr,
            wcr=wcr,
            tat_seconds=tat_seconds,
            pc_watt_seconds=pc_watt_seconds,
        )


@dataclass(frozen=True)
class StepSummary:
    newly_completed_tasks: int
    num_visible_satellites: int


@dataclass
class TrajectoryRecorder:
    constellation_data: list[torch.Tensor] = field(default_factory=list)
    constellation_sensor_enabled: list[torch.Tensor] = field(default_factory=list)
    task_progress: list[torch.Tensor] = field(default_factory=list)
    actions_task_id: list[torch.Tensor] = field(default_factory=list)
    is_visible: list[torch.Tensor] = field(default_factory=list)

    def record(
        self,
        *,
        environment: BasiliskEnvironment,
        task_manager: TaskManager,
        assignment: list[int],
        visibility: torch.Tensor,
    ) -> None:
        constellation = environment.get_constellation()
        sensor_enabled, dynamic_data = constellation.dynamic_to_tensor()
        self.constellation_sensor_enabled.append(sensor_enabled.clone())
        self.constellation_data.append(dynamic_data.clone())
        self.task_progress.append(task_manager.progress.clone())
        self.actions_task_id.append(torch.tensor(assignment, dtype=torch.long))
        self.is_visible.append(visibility.clone())

    def to_payload(self) -> dict[str, object]:
        return {
            "constellation": {
                "sensor_enabled": torch.stack(self.constellation_sensor_enabled),
                "data": torch.stack(self.constellation_data),
            },
            "taskset": {
                "progress": torch.stack(self.task_progress),
            },
            "actions": {
                "task_id": torch.stack(self.actions_task_id),
            },
            "is_visible": torch.stack(self.is_visible),
        }


def build_actor_observation(
    environment: BasiliskEnvironment,
    task_manager: TaskManager,
    statistics: Statistics,
    *,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    constellation = environment.get_constellation()
    constellation_sensor_type, constellation_static = constellation.static_to_tensor()
    constellation_sensor_enabled, constellation_dynamic = constellation.dynamic_to_tensor()
    constellation_data = torch.cat((constellation_static, constellation_dynamic), dim=-1)
    constellation_data = statistics.normalize_constellation(constellation_data).unsqueeze(0)

    tasks = task_manager.ongoing_tasks
    tasks_sensor_type, tasks_static = tasks.to_tensor()
    tasks_static = tasks_static.clone()
    tasks_static[:, 0] -= float(environment.timer.time)
    tasks_static[:, 1] -= float(environment.timer.time)
    progress = task_manager.progress[task_manager.ongoing_flags].unsqueeze(-1)
    tasks_data = torch.cat((tasks_static, progress), dim=-1)
    tasks_data = statistics.normalize_taskset(tasks_data).unsqueeze(0)

    return {
        "time_steps": torch.tensor([environment.timer.time], dtype=torch.long, device=device),
        "constellation_sensor_type": (constellation_sensor_type - 1).unsqueeze(0).to(device),
        "constellation_sensor_enabled": constellation_sensor_enabled.unsqueeze(0).to(device),
        "constellation_data": constellation_data.to(device),
        "constellation_mask": torch.ones((1, constellation_data.shape[1]), dtype=torch.bool, device=device),
        "tasks_sensor_type": (tasks_sensor_type - 1).unsqueeze(0).to(device),
        "tasks_data": tasks_data.to(device),
        "tasks_mask": torch.ones((1, tasks_data.shape[1]), dtype=torch.bool, device=device),
    }


class ScenarioRuntime:
    def __init__(
        self,
        *,
        environment: BasiliskEnvironment,
        task_manager: TaskManager,
        recorder: TrajectoryRecorder | None = None,
    ) -> None:
        self.environment = environment
        self.task_manager = task_manager
        self.metrics = MetricsAccumulator.create(
            task_manager.taskset,
            num_satellites=environment.num_satellites,
        )
        self.recorder = recorder

    @property
    def done(self) -> bool:
        return self.task_manager.all_closed or self.environment.timer.time >= MAX_TIME_STEP

    def _step_once(self, assignment: list[int]) -> StepSummary:
        visibility = self.environment.is_visible(self.task_manager.taskset)
        active_visibility = visibility.clone()
        active_visibility[:, ~self.task_manager.ongoing_flags] = False
        if self.recorder is not None:
            self.recorder.record(
                environment=self.environment,
                task_manager=self.task_manager,
                assignment=assignment,
                visibility=visibility,
            )
        newly_completed_tasks = self.task_manager.record(visibility)
        self.metrics.update(
            task_manager=self.task_manager,
            environment=self.environment,
            assignment=assignment,
        )
        self.environment.apply_assignment(assignment, self.task_manager.ongoing_tasks)
        self.environment.step()
        return StepSummary(
            newly_completed_tasks=newly_completed_tasks,
            num_visible_satellites=int(active_visibility.any(dim=1).sum().item()),
        )

    def skip_idle(self) -> None:
        while self.task_manager.is_idle and not self.done:
            self._step_once([-1] * self.environment.num_satellites)

    def step(self, assignment: list[int]) -> StepSummary:
        summary = self._step_once(assignment)
        if not self.done:
            self.skip_idle()
        return summary

    def finalize_metrics(self) -> RawMetrics:
        return self.metrics.finalize(task_manager=self.task_manager, environment=self.environment)
