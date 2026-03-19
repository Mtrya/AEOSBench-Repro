"""Clean-room expert scheduler and rollout helpers for dataset generation."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path

import torch

from aeosbench.constants import MAX_OFF_NADIR_ANGLE, RADIUS_EARTH
from aeosbench.data import Constellation, TaskSet
from aeosbench.evaluation.basilisk_env import BasiliskEnvironment
from aeosbench.sim import ScenarioRuntime, TaskManager, TrajectoryRecorder


OMEGA_EARTH = (2.0 * math.pi) / (23 * 3600 + 56 * 60 + 4.09053)


@dataclass(frozen=True)
class TrajectoryMetrics:
    cr: float
    wcr: float
    pcr: float
    wpcr: float
    tat_seconds: float
    pc_watt_seconds: float

    def to_json_payload(self) -> dict[str, float]:
        return {
            "CR": self.cr,
            "WCR": self.wcr,
            "PCR": self.pcr,
            "WPCR": self.wpcr,
            "TAT": self.tat_seconds,
            "PC": self.pc_watt_seconds,
        }


def _task_positions_eci(taskset: TaskSet, *, time_step: int) -> torch.Tensor:
    theta = OMEGA_EARTH * time_step
    cosine = math.cos(theta)
    sine = math.sin(theta)
    ecef = torch.tensor([task.coordinate_ecef for task in taskset], dtype=torch.float32)
    x = ecef[:, 0] * cosine - ecef[:, 1] * sine
    y = ecef[:, 0] * sine + ecef[:, 1] * cosine
    z = ecef[:, 2]
    return torch.stack((x, y, z), dim=-1)


def _check_constraints(distance: torch.Tensor, orbital_radius: torch.Tensor) -> torch.Tensor:
    mask_distance = distance < RADIUS_EARTH
    cosine = ((distance.square() + orbital_radius.square() - RADIUS_EARTH**2) / (2 * distance * orbital_radius))
    return mask_distance & (cosine > math.cos(MAX_OFF_NADIR_ANGLE))


class GreedyExpertScheduler:
    def __init__(self) -> None:
        self._previous_assignment: torch.Tensor | None = None

    def reset(self, *, num_satellites: int) -> None:
        self._previous_assignment = torch.full((num_satellites,), -1, dtype=torch.int64)

    def step(self, *, environment: BasiliskEnvironment, task_manager: TaskManager) -> list[int]:
        satellites = environment.get_constellation().sort()
        if self._previous_assignment is None:
            self.reset(num_satellites=len(satellites))
        assert self._previous_assignment is not None

        ongoing_tasks = task_manager.ongoing_tasks
        if len(ongoing_tasks) == 0:
            self._previous_assignment = torch.full_like(self._previous_assignment, -1)
            return [-1] * len(satellites)

        task_positions = _task_positions_eci(ongoing_tasks, time_step=environment.timer.time)
        satellite_positions = torch.tensor(
            [satellite.rv[0].tolist() for satellite in satellites],
            dtype=torch.float32,
        )
        distance = torch.norm(
            satellite_positions.unsqueeze(1) - task_positions.unsqueeze(0),
            dim=-1,
        )
        orbital_radius = torch.norm(satellite_positions, dim=-1, keepdim=True)
        feasible = _check_constraints(distance, orbital_radius)

        task_ids = torch.tensor([task.id_ for task in ongoing_tasks], dtype=torch.int64)
        masked_distance = distance.masked_fill(~feasible, float("inf"))
        greedy_indices = masked_distance.argmin(dim=1)
        greedy_valid = torch.isfinite(masked_distance[torch.arange(len(satellites)), greedy_indices])

        chosen_indices = torch.full((len(satellites),), -1, dtype=torch.int64)
        chosen_task_ids = torch.full((len(satellites),), -1, dtype=torch.int64)
        chosen_indices[greedy_valid] = greedy_indices[greedy_valid]
        chosen_task_ids[greedy_valid] = task_ids[greedy_indices[greedy_valid]]

        previous = self._previous_assignment.clone()
        task_index_by_id = {int(task_id): index for index, task_id in enumerate(task_ids.tolist())}
        for satellite_index, previous_task_id in enumerate(previous.tolist()):
            if previous_task_id == -1:
                continue
            task_index = task_index_by_id.get(previous_task_id)
            if task_index is None or not bool(feasible[satellite_index, task_index]):
                continue
            chosen_indices[satellite_index] = task_index
            chosen_task_ids[satellite_index] = previous_task_id

        self._previous_assignment = chosen_task_ids
        return [int(value) for value in chosen_indices.tolist()]


def rollout_with_expert(
    *,
    constellation: Constellation,
    taskset: TaskSet,
    max_time_step: int,
) -> tuple[dict[str, object], TrajectoryMetrics]:
    environment = BasiliskEnvironment(constellation=constellation, taskset=taskset)
    task_manager = TaskManager(taskset, environment.timer)
    recorder = TrajectoryRecorder()
    runtime = ScenarioRuntime(
        environment=environment,
        task_manager=task_manager,
        recorder=recorder,
    )
    scheduler = GreedyExpertScheduler()
    scheduler.reset(num_satellites=environment.num_satellites)
    runtime.skip_idle()
    while not runtime.done and environment.timer.time < max_time_step:
        assignment = scheduler.step(environment=environment, task_manager=task_manager)
        runtime.step(assignment)
    raw = runtime.finalize_metrics()
    wpcr = float(runtime.metrics.max_progress.sum().item() / runtime.metrics.durations.sum().item())
    return recorder.to_payload(), TrajectoryMetrics(
        cr=raw.cr,
        wcr=raw.wcr,
        pcr=raw.pcr,
        wpcr=wpcr,
        tat_seconds=raw.tat_seconds,
        pc_watt_seconds=raw.pc_watt_seconds,
    )


def write_trajectory_outputs(
    *,
    payload_path: Path,
    metrics_path: Path,
    payload: dict[str, object],
    metrics: TrajectoryMetrics,
) -> None:
    payload_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, payload_path)
    metrics_path.write_text(
        json.dumps(metrics.to_json_payload(), indent=2),
        encoding="utf-8",
    )
