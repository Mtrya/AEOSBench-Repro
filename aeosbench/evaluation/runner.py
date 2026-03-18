"""End-to-end evaluation orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import Any

from Basilisk.utilities.supportDataTools import dataFetcher
from Basilisk.utilities.supportDataTools.dataFetcher import DataFile
import torch
from tqdm.auto import tqdm

from aeosbench.constants import MAX_TIME_STEP
from aeosbench.data import Constellation, TaskSet
from aeosbench.paths import project_relative_path

from .basilisk_env import BasiliskEnvironment
from .checkpoints import load_actor_checkpoint
from .layout import ScenarioRef, constellation_path, scenario_refs, taskset_path
from .metrics import DisplayMetrics, RawMetrics, mean_raw_metrics, provisional_cs
from .model_config import LoadedModelConfig
from .statistics import Statistics, load_statistics

DEFAULT_SPICE_KERNELS = (
    DataFile.EphemerisData.naif0012,
    DataFile.EphemerisData.pck00010,
    DataFile.EphemerisData.de_403_masses,
    DataFile.EphemerisData.de430,
)


@dataclass(frozen=True)
class EvaluationRequest:
    model_config: LoadedModelConfig
    checkpoints: list[Path]
    splits: list[str]
    limit: int | None
    device: str
    show_progress: bool = True


@dataclass(frozen=True)
class ScenarioResult:
    id_: int
    epoch: int
    raw_metrics: RawMetrics


@dataclass(frozen=True)
class EvaluationRow:
    split: str
    checkpoint_path: Path
    timestamp: str
    config_hash: str
    limit: int | None
    scenario_results: list[ScenarioResult]
    aggregate_raw_metrics: RawMetrics
    aggregate_display_metrics: DisplayMetrics
    cs_provisional: float


@dataclass(frozen=True)
class EvaluationResult:
    model_config_path: Path
    rows: list[EvaluationRow]


def _cached_support_path(rel_path: str) -> Path:
    return Path(dataFetcher.POOCH.path) / rel_path


def missing_support_data_files() -> list[Path]:
    missing: list[Path] = []
    for kernel in DEFAULT_SPICE_KERNELS:
        rel_path = dataFetcher.relpath(kernel)
        local = dataFetcher.local_support_path(rel_path)
        if local is not None and local.exists():
            continue
        cached = _cached_support_path(rel_path)
        if not cached.exists():
            missing.append(cached)
    return missing


def check_eval_prereqs(request: EvaluationRequest) -> None:
    missing_paths: list[Path] = []
    for split in request.splits:
        refs = scenario_refs(split, limit=request.limit)
        if not refs:
            missing_paths.append(constellation_path(split, 0).parent)
            continue
        first_ref = refs[0]
        for candidate in (
            constellation_path(first_ref.split, first_ref.id_),
            taskset_path(first_ref.split, first_ref.id_),
        ):
            if not candidate.exists():
                missing_paths.append(candidate)
    for checkpoint in request.checkpoints:
        if not checkpoint.exists():
            missing_paths.append(checkpoint)
    missing_paths.extend(missing_support_data_files())
    if missing_paths:
        formatted = "\n".join(f"- {path}" for path in missing_paths)
        raise RuntimeError(f"Evaluation prerequisites are missing:\n{formatted}")


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

    def record(self, visibility: torch.Tensor) -> None:
        durations = torch.tensor([task.duration for task in self.taskset], dtype=torch.float32)
        masked_visibility = visibility.clone()
        masked_visibility[:, ~self.ongoing_flags] = False
        any_visible = masked_visibility.any(dim=0)
        self.progress = (self.progress + 1.0) * any_visible.to(dtype=torch.float32)
        self.succeeded_flags |= self.progress >= durations


def _build_observation(
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


def _idle_assignment(environment: BasiliskEnvironment) -> list[int]:
    return [-1] * environment.num_satellites


def _skip_idle(
    environment: BasiliskEnvironment,
    task_manager: TaskManager,
    *,
    max_progress: torch.Tensor,
    completion_time: torch.Tensor,
    working_time_steps: torch.Tensor,
) -> None:
    while task_manager.is_idle and environment.timer.time < MAX_TIME_STEP and not task_manager.all_closed:
        assignment = _idle_assignment(environment)
        visibility = environment.is_visible(task_manager.taskset)
        task_manager.record(visibility)
        max_progress.copy_(torch.maximum(max_progress, task_manager.progress))
        if task_manager.succeeded_flags.any():
            completion_time[task_manager.succeeded_flags] = torch.minimum(
                completion_time[task_manager.succeeded_flags],
                torch.full_like(
                    completion_time[task_manager.succeeded_flags],
                    float(environment.timer.time),
                ),
            )
        environment.apply_assignment(assignment, task_manager.ongoing_tasks)
        working_time_steps += torch.tensor(
            [0] * environment.num_satellites,
            dtype=working_time_steps.dtype,
        )
        environment.step()


def evaluate_scenario(
    actor: Any,
    statistics: Statistics,
    ref: ScenarioRef,
    *,
    device: torch.device,
) -> ScenarioResult:
    constellation = Constellation.load(constellation_path(ref.split, ref.id_))
    taskset = TaskSet.load(taskset_path(ref.split, ref.id_))
    environment = BasiliskEnvironment(constellation=constellation, taskset=taskset)
    task_manager = TaskManager(taskset, environment.timer)
    durations = torch.tensor([task.duration for task in taskset], dtype=torch.float32)
    release_times = torch.tensor([task.release_time for task in taskset], dtype=torch.float32)
    max_progress = task_manager.progress.clone()
    completion_time = torch.full((len(taskset),), float("inf"))
    working_time_steps = torch.zeros(environment.num_satellites, dtype=torch.float32)
    _skip_idle(
        environment,
        task_manager,
        max_progress=max_progress,
        completion_time=completion_time,
        working_time_steps=working_time_steps,
    )

    while environment.timer.time < MAX_TIME_STEP and not task_manager.all_closed:
        if task_manager.is_idle:
            _skip_idle(
                environment,
                task_manager,
                max_progress=max_progress,
                completion_time=completion_time,
                working_time_steps=working_time_steps,
            )
            continue
        observation = _build_observation(
            environment,
            task_manager,
            statistics,
            device=device,
        )
        logits = actor.predict(**observation)
        action = logits.argmax(dim=-1).squeeze(0) - 1
        assignment = action.to(dtype=torch.int64).cpu().tolist()

        visibility = environment.is_visible(task_manager.taskset)
        task_manager.record(visibility)
        max_progress = torch.maximum(max_progress, task_manager.progress)
        if task_manager.succeeded_flags.any():
            completion_time[task_manager.succeeded_flags] = torch.minimum(
                completion_time[task_manager.succeeded_flags],
                torch.full_like(
                    completion_time[task_manager.succeeded_flags],
                    float(environment.timer.time),
                ),
            )
        working_time_steps += (torch.tensor(assignment) != -1).to(dtype=torch.float32)
        environment.apply_assignment(assignment, task_manager.ongoing_tasks)
        environment.step()

    succeeded_flags = task_manager.succeeded_flags
    cr = float(task_manager.num_succeeded_tasks / len(taskset))
    wcr = float(durations[succeeded_flags].sum().item() / durations.sum().item())
    pcr = float((max_progress / durations).mean().item())
    tat_seconds = float((completion_time[succeeded_flags] - release_times[succeeded_flags]).mean().item())
    current_constellation = environment.get_constellation().sort()
    sensor_power = torch.tensor([satellite.sensor.power for satellite in current_constellation], dtype=torch.float32)
    pc_watt_seconds = float((working_time_steps * sensor_power).sum().item())
    return ScenarioResult(
        id_=ref.id_,
        epoch=ref.epoch,
        raw_metrics=RawMetrics(
            cr=cr,
            pcr=pcr,
            wcr=wcr,
            tat_seconds=tat_seconds,
            pc_watt_seconds=pc_watt_seconds,
        ),
    )


def _resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def run_evaluation(request: EvaluationRequest) -> EvaluationResult:
    check_eval_prereqs(request)
    device = _resolve_device(request.device)
    timestamp = datetime.now(timezone.utc).isoformat()
    statistics = load_statistics()
    rows: list[EvaluationRow] = []
    evaluation_plan = [
        (checkpoint, split, scenario_refs(split, limit=request.limit))
        for checkpoint in request.checkpoints
        for split in request.splits
    ]
    total_scenarios = sum(len(refs) for _, _, refs in evaluation_plan)
    progress_enabled = request.show_progress and sys.stderr.isatty()
    with tqdm(
        total=total_scenarios,
        desc="Evaluating",
        unit="scenario",
        disable=not progress_enabled,
    ) as progress:
        for checkpoint, split, refs in evaluation_plan:
            progress.set_description(f"Evaluating {split}")
            actor = load_actor_checkpoint(request.model_config.model, checkpoint).to(device)
            scenario_results = []
            for ref in refs:
                scenario_results.append(
                    evaluate_scenario(actor, statistics, ref, device=device)
                )
                progress.update(1)
            aggregate_raw = mean_raw_metrics([result.raw_metrics for result in scenario_results])
            rows.append(
                EvaluationRow(
                    split=split,
                    checkpoint_path=project_relative_path(checkpoint),
                    timestamp=timestamp,
                    config_hash=request.model_config.hash_,
                    limit=request.limit,
                    scenario_results=scenario_results,
                    aggregate_raw_metrics=aggregate_raw,
                    aggregate_display_metrics=aggregate_raw.display,
                    cs_provisional=provisional_cs(aggregate_raw),
                )
            )
    return EvaluationResult(
        model_config_path=project_relative_path(request.model_config.path),
        rows=rows,
    )


def result_to_dict(result: EvaluationResult) -> dict[str, Any]:
    return {
        "model_config_path": str(result.model_config_path),
        "rows": [
            {
                "split": row.split,
                "checkpoint_path": str(row.checkpoint_path),
                "timestamp": row.timestamp,
                "config_hash": row.config_hash,
                "limit": row.limit,
                "raw_metrics": {
                    "CR": row.aggregate_raw_metrics.cr,
                    "PCR": row.aggregate_raw_metrics.pcr,
                    "WCR": row.aggregate_raw_metrics.wcr,
                    "TAT_seconds": row.aggregate_raw_metrics.tat_seconds,
                    "PC_watt_seconds": row.aggregate_raw_metrics.pc_watt_seconds,
                    "CS_provisional": row.cs_provisional,
                },
                "display_metrics": {
                    "CR_percent": row.aggregate_display_metrics.cr_percent,
                    "PCR_percent": row.aggregate_display_metrics.pcr_percent,
                    "WCR_percent": row.aggregate_display_metrics.wcr_percent,
                    "TAT_hours": row.aggregate_display_metrics.tat_hours,
                    "PC_Wh": row.aggregate_display_metrics.pc_wh,
                    "CS_provisional": row.cs_provisional,
                },
                "scenario_results": [
                    {
                        "id": scenario.id_,
                        "epoch": scenario.epoch,
                        "raw_metrics": {
                            "CR": scenario.raw_metrics.cr,
                            "PCR": scenario.raw_metrics.pcr,
                            "WCR": scenario.raw_metrics.wcr,
                            "TAT_seconds": scenario.raw_metrics.tat_seconds,
                            "PC_watt_seconds": scenario.raw_metrics.pc_watt_seconds,
                            "CS_provisional": provisional_cs(scenario.raw_metrics),
                        },
                    }
                    for scenario in row.scenario_results
                ],
            }
            for row in result.rows
        ],
    }
