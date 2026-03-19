"""Orchestration for staged dataset generation."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import random
import shutil
from typing import Literal

from tqdm.auto import tqdm

from aeosbench.data import Constellation, TaskSet
from aeosbench.training.statistics import compute_statistics

from .config import DatasetGenerationConfig
from .expert import rollout_with_expert, write_trajectory_outputs
from .layout import GenerationLayout
from .sampling import (
    instantiate_satellite_from_asset,
    sample_screening_satellite,
    sample_screening_taskset,
    sample_taskset,
    write_constellation,
    write_json,
    write_orbit,
    write_taskset,
)

StageName = Literal["assets", "scenarios", "rollouts", "annotations", "statistics", "all"]
ORBITS_PER_SCENARIO = 100


@dataclass(frozen=True)
class GenerationRequest:
    config: DatasetGenerationConfig
    config_path: Path
    output_root: Path
    seed: int
    overwrite: bool
    show_progress: bool
    device: str


def _copy_config(request: GenerationRequest, layout: GenerationLayout) -> None:
    layout.generation_root.mkdir(parents=True, exist_ok=True)
    shutil.copy2(request.config_path, layout.generation_root / "config.yaml")
    manifest_path = layout.generation_root / "run_manifest.json"
    manifest = {
        "seed": request.seed,
        "output_root": str(request.output_root),
        "device": request.device,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def _write_rejection(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload) + "\n")


def _split_asset_counts(config: DatasetGenerationConfig) -> dict[str, int]:
    return {
        "train": config.assets.train_count,
        "val_unseen": config.assets.val_unseen_count,
        "test": config.assets.test_count,
    }


def _split_scenario_counts(config: DatasetGenerationConfig) -> dict[str, int]:
    counts = config.scenarios.counts
    return {
        "train": counts.train,
        "val_seen": counts.val_seen,
        "val_unseen": counts.val_unseen,
        "test": counts.test,
    }


def _load_assets_for_split(layout: GenerationLayout, split: str) -> list[Constellation]:
    root = layout.satellites_root / split
    files = sorted(root.glob("*.json"), key=lambda item: int(item.stem))
    return [Constellation.load(path) for path in files]


def _load_or_create_screening_taskset(
    request: GenerationRequest,
    layout: GenerationLayout,
) -> TaskSet:
    path = layout.screening_taskset_path
    if path.exists() and not request.overwrite:
        return TaskSet.load(path)
    taskset = sample_screening_taskset(
        random.Random(request.seed),
        size=request.config.assets.screening_task_count,
        horizon=request.config.assets.screening_horizon,
    )
    write_taskset(path, taskset)
    return taskset


def _orbit_id_for_scenario_satellite(scenario_id: int, satellite_id: int) -> int:
    if satellite_id >= ORBITS_PER_SCENARIO:
        raise ValueError(
            f"satellite_id={satellite_id} exceeds the orbit-id packing limit "
            f"of {ORBITS_PER_SCENARIO} per scenario"
        )
    return scenario_id * ORBITS_PER_SCENARIO + satellite_id


def generate_assets(request: GenerationRequest, layout: GenerationLayout) -> None:
    rng = random.Random(request.seed)
    rejection_log = layout.rejection_log_path("asset_rejections")
    if request.overwrite and rejection_log.exists():
        rejection_log.unlink()
    screening_taskset = _load_or_create_screening_taskset(request, layout)
    for split, target_count in _split_asset_counts(request.config).items():
        split_root = layout.satellites_root / split
        if request.overwrite and split_root.exists():
            shutil.rmtree(split_root)
        split_root.mkdir(parents=True, exist_ok=True)
        existing = sorted(split_root.glob("*.json"), key=lambda item: int(item.stem))
        if len(existing) >= target_count:
            continue

        max_attempts = target_count * request.config.assets.retry_factor
        next_id = len(existing)
        iterator = range(max_attempts)
        if request.show_progress:
            iterator = tqdm(iterator, desc=f"Generating assets:{split}", unit="attempt")
        for _ in iterator:
            if next_id >= target_count:
                break
            candidate = sample_screening_satellite(rng, orbit_id=next_id)
            constellation = Constellation({candidate.id_: candidate})
            _, metrics = rollout_with_expert(
                constellation=constellation,
                taskset=screening_taskset,
                max_time_step=request.config.assets.screening_horizon,
            )
            if metrics.cr <= request.config.assets.screening_acceptance_threshold:
                _write_rejection(
                    rejection_log,
                    {
                        "split": split,
                        "candidate_id": next_id,
                        "reason": "screening_threshold",
                        "cr": metrics.cr,
                    },
                )
                continue
            asset_path = layout.asset_path(split, next_id)
            write_constellation(asset_path, constellation)
            write_orbit(layout.orbit_path(next_id), candidate.orbit)
            next_id += 1
        if next_id < target_count:
            raise RuntimeError(f"could not generate enough accepted assets for split={split!r}")


def generate_scenarios(request: GenerationRequest, layout: GenerationLayout) -> None:
    rng = random.Random(request.seed + 1)
    asset_pool = {
        "train": _load_assets_for_split(layout, "train"),
        "val_unseen": _load_assets_for_split(layout, "val_unseen"),
        "test": _load_assets_for_split(layout, "test"),
    }
    if not asset_pool["train"] or not asset_pool["val_unseen"] or not asset_pool["test"]:
        raise RuntimeError("asset pools must be generated before scenarios")

    for split, target_count in _split_scenario_counts(request.config).items():
        if split == "val_seen":
            source_pool = asset_pool["train"]
        else:
            source_pool = asset_pool[split]
        iterator = range(target_count)
        if request.show_progress:
            iterator = tqdm(iterator, desc=f"Generating scenarios:{split}", unit="scenario")
        for scenario_id in iterator:
            constellation_path = layout.constellation_path(split, scenario_id)
            taskset_path = layout.taskset_path(split, scenario_id)
            if constellation_path.exists() and taskset_path.exists() and not request.overwrite:
                continue
            num_satellites = rng.randint(
                request.config.scenarios.min_satellites,
                request.config.scenarios.max_satellites,
            )
            num_satellites = min(num_satellites, len(source_pool))
            selected_assets = rng.sample(source_pool, num_satellites)
            satellites = {}
            for satellite_id, asset_constellation in enumerate(selected_assets):
                asset = asset_constellation.sort()[0]
                instantiated = instantiate_satellite_from_asset(
                    rng,
                    satellite_id=satellite_id,
                    orbit_id=_orbit_id_for_scenario_satellite(scenario_id, satellite_id),
                    asset=asset,
                )
                satellites[satellite_id] = instantiated
                write_orbit(layout.orbit_path(instantiated.orbit_id), instantiated.orbit)
            constellation = Constellation(satellites)
            num_tasks = rng.randint(request.config.tasks.min_tasks, request.config.tasks.max_tasks)
            taskset = sample_taskset(
                rng,
                num_tasks=num_tasks,
                min_duration=request.config.tasks.min_duration,
                max_duration=request.config.tasks.max_duration,
                max_time_step=request.config.tasks.max_time_step,
            )
            write_constellation(constellation_path, constellation)
            write_taskset(taskset_path, taskset)


def generate_rollouts(request: GenerationRequest, layout: GenerationLayout) -> None:
    if not request.config.rollouts.enabled:
        return
    rejection_log = layout.rejection_log_path("rollout_rejections")
    if request.overwrite and rejection_log.exists():
        rejection_log.unlink()
    epoch = request.config.rollouts.epoch
    for split, target_count in _split_scenario_counts(request.config).items():
        iterator = range(target_count)
        if request.show_progress:
            iterator = tqdm(iterator, desc=f"Generating rollouts:{split}", unit="scenario")
        for scenario_id in iterator:
            payload_path = layout.trajectory_payload_path(split, scenario_id, epoch=epoch)
            metrics_path = layout.trajectory_metrics_path(split, scenario_id, epoch=epoch)
            if payload_path.exists() and metrics_path.exists() and not request.overwrite:
                continue
            constellation = Constellation.load(layout.constellation_path(split, scenario_id))
            taskset = TaskSet.load(layout.taskset_path(split, scenario_id))
            try:
                payload, metrics = rollout_with_expert(
                    constellation=constellation,
                    taskset=taskset,
                    max_time_step=request.config.rollouts.max_time_step,
                )
            except Exception as error:
                _write_rejection(
                    rejection_log,
                    {
                        "split": split,
                        "scenario_id": scenario_id,
                        "reason": "simulation_error",
                        "error": repr(error),
                    },
                )
                continue
            if metrics.cr < request.config.curation.min_completion_rate:
                _write_rejection(
                    rejection_log,
                    {
                        "split": split,
                        "scenario_id": scenario_id,
                        "reason": "completion_rate",
                        "cr": metrics.cr,
                    },
                )
                continue
            write_trajectory_outputs(
                payload_path=payload_path,
                metrics_path=metrics_path,
                payload=payload,
                metrics=metrics,
            )


def generate_annotations(request: GenerationRequest, layout: GenerationLayout) -> None:
    epoch = request.config.rollouts.epoch
    for split, target_count in _split_scenario_counts(request.config).items():
        ids: list[int] = []
        epochs: list[int] = []
        for scenario_id in range(target_count):
            payload_path = layout.trajectory_payload_path(split, scenario_id, epoch=epoch)
            metrics_path = layout.trajectory_metrics_path(split, scenario_id, epoch=epoch)
            if payload_path.exists() and metrics_path.exists():
                ids.append(scenario_id)
                epochs.append(epoch)
        write_json(layout.annotation_path(split), {"ids": ids, "epochs": epochs})


def generate_statistics(request: GenerationRequest, layout: GenerationLayout) -> None:
    if not request.config.statistics.enabled:
        return
    if layout.statistics_path.exists() and not request.overwrite:
        return
    compute_statistics(
        split="train",
        annotation_file=None,
        output_path=layout.statistics_path,
        show_progress=request.show_progress,
        dataset_root=layout.output_root,
    )


def run_generation_stage(stage: StageName, request: GenerationRequest) -> None:
    layout = GenerationLayout(request.output_root)
    if request.overwrite and request.output_root.exists() and stage == "all":
        shutil.rmtree(request.output_root)
    layout.ensure_roots()
    _copy_config(request, layout)

    if stage in {"assets", "all"}:
        generate_assets(request, layout)
    if stage in {"scenarios", "all"}:
        generate_scenarios(request, layout)
    if stage in {"rollouts", "all"}:
        generate_rollouts(request, layout)
    if stage in {"annotations", "all"} and (stage != "all" or request.config.rollouts.enabled):
        generate_annotations(request, layout)
    if stage in {"statistics", "all"} and (stage != "all" or request.config.rollouts.enabled):
        generate_statistics(request, layout)
