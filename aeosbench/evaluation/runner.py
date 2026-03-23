"""End-to-end evaluation orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import Any

import torch
from tqdm.auto import tqdm

from aeosbench.data import Constellation, TaskSet
from aeosbench.paths import project_relative_path
from aeosbench.sim import ScenarioRuntime, TaskManager, build_actor_observation

from .basilisk_env import BasiliskEnvironment
from .checkpoints import load_actor_checkpoint
from .layout import ScenarioRef, constellation_path, scenario_refs, taskset_path
from .metrics import DisplayMetrics, RawMetrics, mean_raw_metrics, provisional_cs
from .model_config import LoadedModelConfig
from .statistics import Statistics, load_statistics

DEFAULT_SPICE_KERNELS = (
    "EphemerisData/naif0012.tls",
    "EphemerisData/pck00010.tpc",
    "EphemerisData/de-403-masses.tpc",
    "EphemerisData/de430.bsp",
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


def _cached_support_paths(rel_path: str) -> list[Path]:
    try:
        from Basilisk.utilities.supportDataTools import dataFetcher
    except ModuleNotFoundError:
        return []
    root = Path(dataFetcher.POOCH.path)
    return [
        root / rel_path,
        root / "supportData" / rel_path,
    ]


def _packaged_support_path(rel_path: str) -> Path | None:
    try:
        import Basilisk
    except ModuleNotFoundError:
        return None
    return Path(Basilisk.__file__).resolve().parent / "supportData" / rel_path


def support_data_path_candidates(rel_path: str) -> list[Path]:
    candidates: list[Path] = []
    candidates.extend(_cached_support_paths(rel_path))
    packaged = _packaged_support_path(rel_path)
    if packaged is not None:
        candidates.append(packaged)
    return candidates


def missing_support_data_files() -> list[Path]:
    missing: list[Path] = []
    for rel_path in DEFAULT_SPICE_KERNELS:
        candidates = support_data_path_candidates(rel_path)
        if any(candidate.exists() for candidate in candidates):
            continue
        if candidates:
            missing.append(candidates[0])
        else:
            missing.append(Path(rel_path))
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
    runtime = ScenarioRuntime(
        environment=environment,
        task_manager=TaskManager(taskset, environment.timer),
    )
    runtime.skip_idle()

    while not runtime.done:
        if runtime.task_manager.is_idle:
            runtime.skip_idle()
            continue
        observation = build_actor_observation(
            runtime.environment,
            runtime.task_manager,
            statistics,
            device=device,
        )
        logits = actor.predict(**observation)
        action = logits.argmax(dim=-1).squeeze(0) - 1
        assignment = action.to(dtype=torch.int64).cpu().tolist()
        runtime.step(assignment)

    return ScenarioResult(
        id_=ref.id_,
        epoch=ref.epoch,
        raw_metrics=runtime.finalize_metrics(),
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
