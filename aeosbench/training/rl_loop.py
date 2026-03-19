"""Reinforcement-learning and iterative retraining orchestration."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime
import json
import os
from pathlib import Path
import random
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
import torch
from tqdm.auto import tqdm

from aeosbench.data import Constellation, TaskSet
from aeosbench.evaluation.layout import (
    annotation_path,
    load_annotations,
    trajectory_metrics_path,
    trajectory_payload_path,
)
from aeosbench.evaluation.metrics import RawMetrics
from aeosbench.evaluation.runner import EvaluationRequest, run_evaluation
from aeosbench.evaluation.statistics import Statistics, load_statistics
from aeosbench.paths import project_relative_path
from aeosbench.sim import ScenarioRuntime, TaskManager, TrajectoryRecorder, build_actor_observation

from .checkpoints import checkpoint_dir
from .config import load_training_config
from .rl_config import LoadedRLConfig
from .rl_env import RLEnvironment
from .rl_policy import AEOSFormerPPOPolicy
from .selection import SelectionEntry, SelectionManifest, load_selection_manifest, save_selection_manifest
from .trainer import TrainingRequest, run_training


@dataclass(frozen=True)
class RLTrainingRequest:
    config: LoadedRLConfig
    work_dir: Path
    device: str
    seed: int
    resume: str | Path | None = None
    show_progress: bool = True


@dataclass(frozen=True)
class RLState:
    completed_outer_iterations: int
    current_actor_checkpoint: Path
    current_selection_manifest: Path


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    return resolved


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _metrics_path(work_dir: Path) -> Path:
    return work_dir / "metrics.jsonl"


def _append_metrics(work_dir: Path, payload: dict[str, object]) -> None:
    path = _metrics_path(work_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def _state_path(work_dir: Path) -> Path:
    return work_dir / "state.json"


def _save_state(work_dir: Path, state: RLState) -> None:
    _state_path(work_dir).write_text(
        json.dumps(
            {
                "completed_outer_iterations": state.completed_outer_iterations,
                "current_actor_checkpoint": str(state.current_actor_checkpoint),
                "current_selection_manifest": str(state.current_selection_manifest),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def _load_state(work_dir: Path) -> RLState:
    payload = json.loads(_state_path(work_dir).read_text(encoding="utf-8"))
    return RLState(
        completed_outer_iterations=int(payload["completed_outer_iterations"]),
        current_actor_checkpoint=Path(payload["current_actor_checkpoint"]),
        current_selection_manifest=Path(payload["current_selection_manifest"]),
    )


def _selection_entry_for_released(split: str, id_: int, epoch: int) -> SelectionEntry:
    return SelectionEntry(
        split=split,
        id_=id_,
        source_kind="released",
        epoch=epoch,
        trajectory_path=trajectory_payload_path(split, id_, epoch=epoch),
        metrics_path=trajectory_metrics_path(split, id_, epoch=epoch),
    )


def _selection_annotation_path(config: LoadedRLConfig) -> Path:
    if config.environment.annotation_file is None:
        return annotation_path(config.environment.split)
    candidate = Path(config.environment.annotation_file)
    if candidate.is_absolute():
        return candidate
    return annotation_path(config.environment.split).parent / candidate


def _initialize_selection_manifest(config: LoadedRLConfig, work_dir: Path) -> Path:
    selection = load_annotations(_selection_annotation_path(config))
    entries = [
        _selection_entry_for_released(
            config.environment.split,
            id_,
            selection.epoch_at(index, default=1),
        )
        for index, id_ in enumerate(selection.ids[: config.environment.limit])
    ]
    return save_selection_manifest(work_dir / "selection" / "initial_train_manifest.json", entries)


def _load_raw_metrics(path: Path) -> RawMetrics:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return RawMetrics(
        cr=float(payload["CR"]),
        pcr=float(payload["PCR"]),
        wcr=float(payload["WCR"]),
        tat_seconds=float(payload["TAT"]),
        pc_watt_seconds=float(payload["PC"]),
    )


def _save_raw_metrics(path: Path, metrics: RawMetrics) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "CR": metrics.cr,
                "PCR": metrics.pcr,
                "WCR": metrics.wcr,
                "TAT": metrics.tat_seconds,
                "PC": metrics.pc_watt_seconds,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def _beats_baseline(candidate: RawMetrics, baseline: RawMetrics, *, min_cr_improvement: float) -> bool:
    if candidate.cr > baseline.cr + min_cr_improvement:
        return True
    if candidate.cr < baseline.cr + min_cr_improvement:
        return False
    candidate_tuple = (candidate.pcr, candidate.wcr, -candidate.tat_seconds, -candidate.pc_watt_seconds)
    baseline_tuple = (baseline.pcr, baseline.wcr, -baseline.tat_seconds, -baseline.pc_watt_seconds)
    return candidate_tuple > baseline_tuple


def _export_actor_checkpoint(algorithm: PPO, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    actor = algorithm.policy.mlp_extractor.actor  # type: ignore[attr-defined]
    torch.save(actor.state_dict(), output_path)
    return output_path


def _outer_dir(work_dir: Path, iteration: int) -> Path:
    return work_dir / f"outer_iter_{iteration}"


def _rollout_entry_path(outer_dir: Path, split: str, id_: int) -> tuple[Path, Path]:
    shard = outer_dir / "rollouts" / split / f"{id_ // 1000:02d}"
    return shard / f"{id_:05d}.pth", shard / f"{id_:05d}.json"


def _constellation_path(split: str, id_: int) -> Path:
    return annotation_path(split).parent.parent / "constellations" / split / f"{id_ // 1000:02d}" / f"{id_:05d}.json"


def _taskset_path(split: str, id_: int) -> Path:
    return annotation_path(split).parent.parent / "tasksets" / split / f"{id_ // 1000:02d}" / f"{id_:05d}.json"


def _rollout_entry(
    actor: torch.nn.Module,
    statistics: Statistics,
    entry: SelectionEntry,
    *,
    device: torch.device,
    trajectory_path: Path,
    metrics_path: Path,
) -> tuple[SelectionEntry, RawMetrics]:
    constellation = Constellation.load(_constellation_path(entry.split, entry.id_))
    taskset = TaskSet.load(_taskset_path(entry.split, entry.id_))
    from aeosbench.evaluation.basilisk_env import BasiliskEnvironment

    environment = BasiliskEnvironment(constellation=constellation, taskset=taskset)
    recorder = TrajectoryRecorder()
    runtime = ScenarioRuntime(
        environment=environment,
        task_manager=TaskManager(taskset, environment.timer),
        recorder=recorder,
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
        assignment = (logits.argmax(dim=-1).squeeze(0) - 1).to(dtype=torch.int64).cpu().tolist()
        runtime.step(assignment)
    raw_metrics = runtime.finalize_metrics()
    trajectory_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(recorder.to_payload(), trajectory_path)
    _save_raw_metrics(metrics_path, raw_metrics)
    return (
        SelectionEntry(
            split=entry.split,
            id_=entry.id_,
            source_kind="rollout",
            epoch=None,
            trajectory_path=trajectory_path,
            metrics_path=metrics_path,
        ),
        raw_metrics,
    )


def _run_rollout_collection(
    *,
    actor_checkpoint: Path,
    config: LoadedRLConfig,
    current_selection: SelectionManifest,
    outer_dir: Path,
    device: torch.device,
    statistics: Statistics,
    show_progress: bool,
    work_dir: Path,
) -> Path:
    from aeosbench.evaluation.checkpoints import load_actor_checkpoint

    actor = load_actor_checkpoint(config.model, actor_checkpoint).to(device)
    next_entries: list[SelectionEntry] = []
    candidates = current_selection.entries[: config.iterative.rollout_limit]
    if config.iterative.rollout_limit is None:
        candidates = current_selection.entries
    progress_enabled = show_progress and sys.stderr.isatty()
    iterator = candidates
    if progress_enabled:
        iterator = tqdm(candidates, desc="Collecting rollouts", unit="scenario")
    baseline_by_id = {entry.id_: entry for entry in current_selection.entries}
    for entry in iterator:
        trajectory_path, metrics_path = _rollout_entry_path(outer_dir, entry.split, entry.id_)
        candidate_entry, candidate_metrics = _rollout_entry(
            actor,
            statistics,
            entry,
            device=device,
            trajectory_path=trajectory_path,
            metrics_path=metrics_path,
        )
        baseline_entry = baseline_by_id[entry.id_]
        assert baseline_entry.metrics_path is not None
        baseline_metrics = _load_raw_metrics(baseline_entry.metrics_path)
        if _beats_baseline(
            candidate_metrics,
            baseline_metrics,
            min_cr_improvement=config.iterative.min_cr_improvement,
        ):
            next_entries.append(candidate_entry)
        else:
            next_entries.append(baseline_entry)
    if len(next_entries) != len(current_selection.entries):
        retained = {entry.id_ for entry in next_entries}
        for entry in current_selection.entries:
            if entry.id_ not in retained:
                next_entries.append(entry)
    next_entries = sorted(next_entries, key=lambda entry: entry.id_)
    accepted_count = sum(entry.source_kind == "rollout" for entry in next_entries)
    _append_metrics(
        work_dir,
        {
            "event": "rollout_selection",
            "accepted_rollouts": accepted_count,
            "total_scenarios": len(next_entries),
            "outer_iteration": int(outer_dir.name.split("_")[-1]),
        },
    )
    return save_selection_manifest(outer_dir / "selection" / "train_manifest.json", next_entries)


def _run_validation_eval(
    *,
    actor_checkpoint: Path,
    config: LoadedRLConfig,
    work_dir: Path,
) -> None:
    if not config.validation_eval.enabled:
        return
    from aeosbench.evaluation.model_config import LoadedModelConfig

    result = run_evaluation(
        EvaluationRequest(
            model_config=LoadedModelConfig(
                path=config.path,
                hash_=config.hash_,
                model=config.model,
            ),
            checkpoints=[actor_checkpoint],
            splits=[config.validation_eval.split],
            limit=config.validation_eval.limit,
            device="auto",
            show_progress=False,
        )
    )
    row = result.rows[0]
    _append_metrics(
        work_dir,
        {
            "event": "validation_eval",
            "split": row.split,
            "checkpoint_path": str(project_relative_path(actor_checkpoint)),
            "cr": row.aggregate_raw_metrics.cr,
            "pcr": row.aggregate_raw_metrics.pcr,
            "wcr": row.aggregate_raw_metrics.wcr,
            "tat_seconds": row.aggregate_raw_metrics.tat_seconds,
            "pc_watt_seconds": row.aggregate_raw_metrics.pc_watt_seconds,
        },
    )


def run_rl_training(request: RLTrainingRequest) -> Path:
    _seed_everything(request.seed)
    device = _resolve_device(request.device)
    if request.resume is not None:
        resume_dir = Path(request.resume).resolve()
        if request.work_dir.resolve() != resume_dir:
            raise ValueError("--resume and --work-dir must point to the same RL work directory")
    request.work_dir.mkdir(parents=True, exist_ok=True)
    request.work_dir.joinpath("config.yaml").write_text(
        request.config.path.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    _append_metrics(
        request.work_dir,
        {
            "event": "start",
            "timestamp": datetime.now().isoformat(),
            "device": str(device),
            "seed": request.seed,
            "config_hash": request.config.hash_,
        },
    )

    if request.resume is None:
        current_selection_manifest = _initialize_selection_manifest(request.config, request.work_dir)
        state = RLState(
            completed_outer_iterations=0,
            current_actor_checkpoint=request.config.initialization.actor_checkpoint,
            current_selection_manifest=current_selection_manifest,
        )
        _save_state(request.work_dir, state)
    else:
        state = _load_state(Path(request.resume))

    # RL keeps one fixed normalization snapshot for the whole outer loop.
    # Supervised retraining may recompute its own stats from the current selection,
    # but PPO rollout collection and exploration continue to use this initial set.
    statistics = load_statistics()
    progress_enabled = request.show_progress and sys.stderr.isatty()
    for outer_iteration in range(
        state.completed_outer_iterations + 1,
        request.config.iterative.outer_iterations + 1,
    ):
        outer_dir = _outer_dir(request.work_dir, outer_iteration)
        outer_dir.mkdir(parents=True, exist_ok=True)
        env = RLEnvironment.build(
            request.config.environment,
            request.config.reward,
            statistics=statistics,
            seed=request.seed + outer_iteration,
        )
        callback = CheckpointCallback(
            save_freq=request.config.ppo.save_freq,
            save_path=str(outer_dir / "ppo"),
            name_prefix="ppo",
        )
        algorithm = PPO(
            AEOSFormerPPOPolicy,
            env,
            policy_kwargs={
                "actor_config": request.config.model,
                "load_model_from": [str(state.current_actor_checkpoint)],
            },
            seed=request.seed + outer_iteration,
            device=str(device),
            n_steps=request.config.ppo.n_steps,
            batch_size=request.config.ppo.batch_size,
            n_epochs=request.config.ppo.n_epochs,
            learning_rate=request.config.ppo.learning_rate,
            gamma=request.config.ppo.gamma,
            gae_lambda=request.config.ppo.gae_lambda,
            clip_range=request.config.ppo.clip_range,
            ent_coef=request.config.ppo.ent_coef,
            vf_coef=request.config.ppo.vf_coef,
            verbose=1,
        )
        try:
            algorithm.learn(
                total_timesteps=request.config.ppo.total_timesteps,
                callback=callback,
                progress_bar=progress_enabled,
            )
            final_ppo_path = outer_dir / "ppo" / "ppo_final"
            algorithm.save(str(final_ppo_path))
            exported_actor_path = _export_actor_checkpoint(algorithm, outer_dir / "export" / "model.pth")
        finally:
            env.close()
        current_selection = load_selection_manifest(state.current_selection_manifest)
        next_manifest_path = _run_rollout_collection(
            actor_checkpoint=exported_actor_path,
            config=request.config,
            current_selection=current_selection,
            outer_dir=outer_dir,
            device=device,
            statistics=statistics,
            show_progress=request.show_progress,
            work_dir=request.work_dir,
        )
        supervised_config = load_training_config(request.config.supervised.supervised_config)
        supervised_config = replace(
            supervised_config,
            initialization=replace(
                supervised_config.initialization,
                checkpoint=exported_actor_path,
            ),
            data=replace(
                supervised_config.data,
                selection_manifest=next_manifest_path,
            ),
        )
        sl_work_dir = outer_dir / "sl"
        run_training(
            TrainingRequest(
                config=supervised_config,
                work_dir=sl_work_dir,
                device=request.device,
                seed=request.seed + outer_iteration,
                show_progress=request.show_progress,
            )
        )
        next_actor_checkpoint = checkpoint_dir(
            sl_work_dir,
            supervised_config.training.iterations,
        ) / "model.pth"
        _run_validation_eval(
            actor_checkpoint=next_actor_checkpoint,
            config=request.config,
            work_dir=request.work_dir,
        )
        state = RLState(
            completed_outer_iterations=outer_iteration,
            current_actor_checkpoint=next_actor_checkpoint,
            current_selection_manifest=next_manifest_path,
        )
        _save_state(request.work_dir, state)
        _append_metrics(
            request.work_dir,
            {
                "event": "outer_iteration_complete",
                "outer_iteration": outer_iteration,
                "ppo_path": str(project_relative_path(final_ppo_path.with_suffix(".zip"))),
                "actor_checkpoint": str(project_relative_path(next_actor_checkpoint)),
                "selection_manifest": str(project_relative_path(next_manifest_path)),
            },
        )
    return request.work_dir
