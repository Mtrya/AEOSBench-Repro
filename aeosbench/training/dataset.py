"""Supervised trajectory dataset and label generation."""

from __future__ import annotations

from dataclasses import dataclass
import random
from pathlib import Path

import torch

from aeosbench.data import Constellation, TaskSet
from aeosbench.evaluation.layout import (
    ScenarioRef,
    annotation_path,
    load_annotations,
    trajectory_payload_path,
)
from aeosbench.evaluation.statistics import Statistics
from aeosbench.paths import benchmark_data_root

from .config import ConstraintLabelConfig


@dataclass(frozen=True)
class SupervisedBatch:
    split: str
    scenario_id: int
    epoch: int
    time_steps: torch.Tensor
    constellation_sensor_type: torch.Tensor
    constellation_sensor_enabled: torch.Tensor
    constellation_data: torch.Tensor
    constellation_mask: torch.Tensor
    tasks_sensor_type: torch.Tensor
    tasks_data: torch.Tensor
    tasks_mask: torch.Tensor
    actions_task_id: torch.Tensor
    feasibility_target: torch.Tensor
    timing_target: torch.Tensor


def _resolve_annotation_path(split: str, annotation_file: str | None) -> Path:
    if annotation_file is None:
        return annotation_path(split)
    candidate = Path(annotation_file)
    if candidate.is_absolute():
        return candidate
    return benchmark_data_root() / "annotations" / candidate


def _scenario_refs(
    split: str,
    *,
    annotation_file: str | None,
    limit: int | None,
) -> list[ScenarioRef]:
    selection = load_annotations(_resolve_annotation_path(split, annotation_file))
    ids = selection.ids[:limit]
    return [
        ScenarioRef(split=split, id_=id_, epoch=selection.epoch_at(index, default=1))
        for index, id_ in enumerate(ids)
    ]


def _constellation_path(split: str, id_: int) -> Path:
    return benchmark_data_root() / "constellations" / split / f"{id_ // 1000:02d}" / f"{id_:05d}.json"


def _taskset_path(split: str, id_: int) -> Path:
    return benchmark_data_root() / "tasksets" / split / f"{id_ // 1000:02d}" / f"{id_:05d}.json"


def _build_constellation_tensors(
    ref: ScenarioRef,
    trajectory: dict[str, object],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    constellation = Constellation.load(_constellation_path(ref.split, ref.id_))
    sensor_type, static_data = constellation.static_to_tensor()
    dynamic_payload = trajectory["constellation"]
    if not isinstance(dynamic_payload, dict):
        raise TypeError("trajectory.constellation must be a mapping")
    sensor_enabled = torch.as_tensor(dynamic_payload["sensor_enabled"], dtype=torch.long)
    dynamic_data = torch.as_tensor(dynamic_payload["data"], dtype=torch.float32)

    repeated_sensor_type = sensor_type.unsqueeze(0).expand(dynamic_data.shape[0], -1)
    repeated_static_data = static_data.unsqueeze(0).expand(dynamic_data.shape[0], -1, -1)
    data = torch.cat((repeated_static_data, dynamic_data), dim=-1)
    mask = torch.ones_like(repeated_sensor_type, dtype=torch.bool)
    return repeated_sensor_type, sensor_enabled, data, mask


def _build_task_tensors(
    ref: ScenarioRef,
    trajectory: dict[str, object],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    taskset = TaskSet.load(_taskset_path(ref.split, ref.id_))
    sensor_type, static_data = taskset.to_tensor()

    taskset_payload = trajectory["taskset"]
    if not isinstance(taskset_payload, dict):
        raise TypeError("trajectory.taskset must be a mapping")
    progress = torch.as_tensor(taskset_payload["progress"], dtype=torch.float32)
    time_steps = torch.arange(progress.shape[0], dtype=torch.float32).unsqueeze(-1)

    repeated_sensor_type = sensor_type.unsqueeze(0).expand(progress.shape[0], -1)
    repeated_static_data = static_data.unsqueeze(0).expand(progress.shape[0], -1, -1).clone()
    repeated_static_data[..., 0] -= time_steps
    repeated_static_data[..., 1] -= time_steps
    data = torch.cat((repeated_static_data, progress.unsqueeze(-1)), dim=-1)

    durations = static_data[:, 2]
    finished_mask = progress >= durations
    finished_mask = finished_mask.cummax(dim=0).values
    mask = (repeated_static_data[..., 0] <= 0) & (repeated_static_data[..., 1] >= 0)
    mask[1:] &= ~finished_mask[:-1]
    return repeated_sensor_type, data, mask, durations


def _sample_time_indices(
    valid_indices: list[int],
    *,
    timesteps_per_sample: int | None,
    deterministic: bool,
    seed: int,
) -> list[int]:
    if not valid_indices:
        return []
    if timesteps_per_sample is None or len(valid_indices) <= timesteps_per_sample:
        return valid_indices
    if deterministic:
        rng = random.Random(seed)
        return sorted(rng.sample(valid_indices, timesteps_per_sample))
    return sorted(random.sample(valid_indices, timesteps_per_sample))


def _constraint_targets_for_times(
    actions_task_id: torch.Tensor,
    is_visible: torch.Tensor,
    progress: torch.Tensor,
    durations: torch.Tensor,
    time_indices: list[int],
    config: ConstraintLabelConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    total_steps, num_satellites = actions_task_id.shape
    num_tasks = progress.shape[1]
    completed_tasks = progress.amax(dim=0) >= durations
    time_axis = torch.arange(total_steps, dtype=torch.int32).unsqueeze(-1)
    task_ids = torch.arange(num_tasks, dtype=actions_task_id.dtype).unsqueeze(0)
    selected_axis = torch.as_tensor(time_indices, dtype=torch.int32).unsqueeze(-1)
    selected_feasible = torch.zeros(
        (len(time_indices), num_satellites, num_tasks),
        dtype=torch.bool,
    )
    selected_timing = torch.zeros(
        (len(time_indices), num_satellites, num_tasks),
        dtype=torch.float32,
    )

    for satellite_index in range(num_satellites):
        assigned_visible = (
            actions_task_id[:, satellite_index].unsqueeze(-1).eq(task_ids)
            & is_visible[:, satellite_index, :]
        )
        run_starts = assigned_visible.clone()
        for offset in range(1, config.min_positive_run_length):
            run_starts[:-offset] &= assigned_visible[offset:]
            run_starts[-offset:] = False

        next_start = torch.where(
            run_starts,
            time_axis.expand(-1, num_tasks),
            torch.full((total_steps, num_tasks), total_steps, dtype=torch.int32),
        )
        next_start = torch.flip(
            torch.cummin(torch.flip(next_start, dims=[0]), dim=0).values,
            dims=[0],
        )
        next_selected = next_start[time_indices]
        feasible = (next_selected < total_steps) & completed_tasks.unsqueeze(0)
        timing = (next_selected - selected_axis).clamp_min(0)
        timing = timing.clamp_max(config.max_time_horizon).to(dtype=torch.float32)

        selected_feasible[:, satellite_index, :] = feasible
        selected_timing[:, satellite_index, :] = timing

    return selected_feasible, selected_timing


class SupervisedTrajectoryDataset(torch.utils.data.Dataset[SupervisedBatch]):
    def __init__(
        self,
        *,
        split: str,
        annotation_file: str | None = None,
        timesteps_per_sample: int | None = 48,
        limit: int | None = None,
        constraint_labels: ConstraintLabelConfig,
        statistics: Statistics | None = None,
        deterministic_sampling: bool = False,
        seed: int = 3407,
    ) -> None:
        super().__init__()
        self._refs = _scenario_refs(
            split,
            annotation_file=annotation_file,
            limit=limit,
        )
        self._timesteps_per_sample = timesteps_per_sample
        self._constraint_labels = constraint_labels
        self._statistics = statistics
        self._deterministic_sampling = deterministic_sampling
        self._seed = seed

    def __len__(self) -> int:
        return len(self._refs)

    def __getitem__(self, index: int) -> SupervisedBatch:
        ref = self._refs[index]
        trajectory = torch.load(
            trajectory_payload_path(ref.split, ref.id_, epoch=ref.epoch),
            map_location="cpu",
            weights_only=False,
        )
        if not isinstance(trajectory, dict):
            raise TypeError(f"trajectory payload must be a mapping: {ref}")

        (
            constellation_sensor_type,
            constellation_sensor_enabled,
            constellation_data,
            constellation_mask,
        ) = _build_constellation_tensors(ref, trajectory)
        tasks_sensor_type, tasks_data, tasks_mask, durations = _build_task_tensors(
            ref,
            trajectory,
        )

        valid_indices = tasks_mask.any(dim=-1).nonzero(as_tuple=False).flatten().tolist()
        sampled_indices = _sample_time_indices(
            valid_indices,
            timesteps_per_sample=self._timesteps_per_sample,
            deterministic=self._deterministic_sampling,
            seed=self._seed + ref.id_ + ref.epoch * 100_000,
        )
        if not sampled_indices:
            raise RuntimeError(f"{ref.split}:{ref.id_} has no valid supervised timesteps")

        actions_payload = trajectory["actions"]
        if not isinstance(actions_payload, dict):
            raise TypeError("trajectory.actions must be a mapping")
        actions_task_id = torch.as_tensor(actions_payload["task_id"], dtype=torch.long)
        is_visible = torch.as_tensor(trajectory["is_visible"], dtype=torch.bool)
        progress = torch.as_tensor(trajectory["taskset"]["progress"], dtype=torch.float32)
        feasibility_target, timing_target = _constraint_targets_for_times(
            actions_task_id,
            is_visible,
            progress,
            durations,
            sampled_indices,
            self._constraint_labels,
        )

        sampled_steps = torch.as_tensor(sampled_indices, dtype=torch.long)
        constellation_sensor_type = constellation_sensor_type[sampled_indices]
        constellation_sensor_enabled = constellation_sensor_enabled[sampled_indices]
        constellation_data = constellation_data[sampled_indices]
        constellation_mask = constellation_mask[sampled_indices]
        tasks_sensor_type = tasks_sensor_type[sampled_indices]
        tasks_data = tasks_data[sampled_indices]
        tasks_mask = tasks_mask[sampled_indices]
        actions_task_id = actions_task_id[sampled_indices]

        task_is_valid = tasks_mask.any(dim=0)
        if not bool(task_is_valid.all()):
            tasks_sensor_type = tasks_sensor_type[:, task_is_valid]
            tasks_data = tasks_data[:, task_is_valid]
            tasks_mask = tasks_mask[:, task_is_valid]
            feasibility_target = feasibility_target[:, :, task_is_valid]
            timing_target = timing_target[:, :, task_is_valid]

            task_id_mapper = task_is_valid.cumsum(dim=0, dtype=torch.long) - 1
            actions_task_id = torch.where(
                actions_task_id == -1,
                actions_task_id,
                task_id_mapper[actions_task_id],
            )

        augmented_tasks_mask = torch.cat(
            (
                torch.ones((len(sampled_indices), 1), dtype=torch.bool),
                tasks_mask,
            ),
            dim=-1,
        )
        if not augmented_tasks_mask.gather(dim=-1, index=actions_task_id + 1).all():
            raise RuntimeError(f"invalid actions after task pruning for {ref.split}:{ref.id_}")

        if self._statistics is not None:
            constellation_data = self._statistics.normalize_constellation(constellation_data)
            tasks_data = self._statistics.normalize_taskset(tasks_data)

        return SupervisedBatch(
            split=ref.split,
            scenario_id=ref.id_,
            epoch=ref.epoch,
            time_steps=sampled_steps,
            constellation_sensor_type=constellation_sensor_type - 1,
            constellation_sensor_enabled=constellation_sensor_enabled,
            constellation_data=constellation_data,
            constellation_mask=constellation_mask,
            tasks_sensor_type=tasks_sensor_type - 1,
            tasks_data=tasks_data,
            tasks_mask=tasks_mask,
            actions_task_id=actions_task_id,
            feasibility_target=feasibility_target,
            timing_target=timing_target,
        )
