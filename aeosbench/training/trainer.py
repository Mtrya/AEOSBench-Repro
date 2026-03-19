"""Supervised AEOS-Former training loop."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
import random
import sys
from typing import Iterable

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from aeosbench.evaluation.checkpoints import normalized_state_dict
from aeosbench.evaluation.checkpoints import build_actor as build_eval_actor

from .checkpoints import load_checkpoint, resolve_resume_path, save_checkpoint
from .config import LoadedTrainingConfig, LossWeightsConfig
from .dataset import SupervisedBatch, SupervisedTrajectoryDataset
from .losses import LossSummary, compute_supervised_losses
from .statistics import ensure_statistics


@dataclass(frozen=True)
class TrainingRequest:
    config: LoadedTrainingConfig
    work_dir: Path
    device: str
    seed: int
    resume: str | Path | None = None
    load_model_from: tuple[Path, ...] = ()
    show_progress: bool = True


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


def _worker_init_fn(base_seed: int):
    def _init(worker_id: int) -> None:
        random.seed(base_seed + worker_id)
        torch.manual_seed(base_seed + worker_id)

    return _init


def _move_batch(batch: SupervisedBatch, device: torch.device) -> SupervisedBatch:
    return SupervisedBatch(
        split=batch.split,
        scenario_id=batch.scenario_id,
        epoch=batch.epoch,
        time_steps=batch.time_steps.to(device),
        constellation_sensor_type=batch.constellation_sensor_type.to(device),
        constellation_sensor_enabled=batch.constellation_sensor_enabled.to(device),
        constellation_data=batch.constellation_data.to(device),
        constellation_mask=batch.constellation_mask.to(device),
        tasks_sensor_type=batch.tasks_sensor_type.to(device),
        tasks_data=batch.tasks_data.to(device),
        tasks_mask=batch.tasks_mask.to(device),
        actions_task_id=batch.actions_task_id.to(device),
        feasibility_target=batch.feasibility_target.to(device),
        timing_target=batch.timing_target.to(device),
    )


def _build_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    iterations: int,
    warmup_iters: int,
    warmup_start_factor: float,
    cosine_eta_min: float,
) -> torch.optim.lr_scheduler.LRScheduler:
    linear = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=warmup_start_factor,
        total_iters=max(warmup_iters - 1, 1),
    )
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(iterations - warmup_iters, 1),
        eta_min=cosine_eta_min,
    )
    return torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[linear, cosine],
        milestones=[warmup_iters],
    )


def _overlay_model_weights(model: torch.nn.Module, paths: Iterable[Path]) -> None:
    for path in paths:
        state_dict = normalized_state_dict(path)
        model.load_state_dict(state_dict, strict=False)


def _metrics_path(work_dir: Path) -> Path:
    return work_dir / "metrics.jsonl"


def _append_metrics(work_dir: Path, payload: dict[str, object]) -> None:
    path = _metrics_path(work_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def _build_dataloader(
    dataset: SupervisedTrajectoryDataset,
    *,
    num_workers: int,
    pin_memory: bool,
    shuffle: bool,
    seed: int,
) -> DataLoader[SupervisedBatch]:
    return DataLoader(
        dataset,
        batch_size=None,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=_worker_init_fn(seed),
    )


def _evaluate_validation(
    model: torch.nn.Module,
    loader: DataLoader[SupervisedBatch],
    *,
    device: torch.device,
    autocast_enabled: bool,
    loss_weights: LossWeightsConfig,
) -> dict[str, float]:
    totals = {
        "total": 0.0,
        "feasibility": 0.0,
        "timing": 0.0,
        "assignment": 0.0,
        "assignment_accuracy": 0.0,
    }
    count = 0
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = _move_batch(batch, device)
            with torch.autocast(device_type="cuda", enabled=autocast_enabled):
                outputs = model.forward_supervised(
                    batch.time_steps,
                    batch.constellation_sensor_type,
                    batch.constellation_sensor_enabled,
                    batch.constellation_data,
                    batch.constellation_mask,
                    batch.tasks_sensor_type,
                    batch.tasks_data,
                    batch.tasks_mask,
                )
                losses = compute_supervised_losses(
                    outputs,
                    batch,
                    weights=loss_weights,
                )
            totals["total"] += float(losses.total.item())
            totals["feasibility"] += float(losses.feasibility.item())
            totals["timing"] += float(losses.timing.item())
            totals["assignment"] += float(losses.assignment.item())
            totals["assignment_accuracy"] += float(losses.assignment_accuracy.item())
            count += 1
    if count == 0:
        raise RuntimeError("validation loader produced no batches")
    return {key: value / count for key, value in totals.items()}


def _print_step(
    prefix: str,
    iteration: int,
    total_iterations: int,
    metrics: dict[str, float],
    *,
    progress: tqdm | None = None,
) -> None:
    joined = " ".join(f"{key}={value:.4f}" for key, value in metrics.items())
    message = f"[{prefix}] iter={iteration}/{total_iterations} {joined}"
    if progress is not None:
        progress.write(message)
        return
    print(message)


def run_training(request: TrainingRequest) -> Path:
    _seed_everything(request.seed)
    device = _resolve_device(request.device)
    work_dir = request.work_dir
    work_dir.mkdir(parents=True, exist_ok=True)
    (work_dir / "config.yaml").write_text(
        request.config.path.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    progress_enabled = request.show_progress and sys.stderr.isatty()

    statistics = ensure_statistics(
        mode=request.config.data.statistics.mode,
        path=request.config.data.statistics.path,
        split=request.config.data.split,
        annotation_file=request.config.data.annotation_file,
        show_progress=progress_enabled,
    )

    train_dataset = SupervisedTrajectoryDataset(
        split=request.config.data.split,
        annotation_file=request.config.data.annotation_file,
        timesteps_per_scenario=request.config.data.timesteps_per_scenario,
        limit=request.config.data.limit,
        constraint_labels=request.config.constraint_labels,
        statistics=statistics,
        deterministic_sampling=False,
        seed=request.seed,
    )
    train_loader = _build_dataloader(
        train_dataset,
        num_workers=request.config.training.num_workers,
        pin_memory=request.config.training.pin_memory,
        shuffle=True,
        seed=request.seed,
    )

    validation_loader: DataLoader[SupervisedBatch] | None = None
    if request.config.validation.enabled:
        validation_dataset = SupervisedTrajectoryDataset(
            split=request.config.validation.split,
            annotation_file=request.config.validation.annotation_file,
            timesteps_per_scenario=request.config.validation.timesteps_per_scenario,
            limit=request.config.validation.max_scenarios,
            constraint_labels=request.config.constraint_labels,
            statistics=statistics,
            deterministic_sampling=True,
            seed=request.seed,
        )
        validation_loader = _build_dataloader(
            validation_dataset,
            num_workers=0,
            pin_memory=False,
            shuffle=False,
            seed=request.seed,
        )

    model = build_eval_actor(request.config.model).to(device)
    if request.load_model_from:
        _overlay_model_weights(model, request.load_model_from)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=request.config.optimizer.lr,
        betas=request.config.optimizer.betas,
        weight_decay=request.config.optimizer.weight_decay,
        eps=request.config.optimizer.eps,
    )
    lr_scheduler = _build_scheduler(
        optimizer,
        iterations=request.config.training.iterations,
        warmup_iters=request.config.scheduler.warmup_iters,
        warmup_start_factor=request.config.scheduler.warmup_start_factor,
        cosine_eta_min=request.config.scheduler.cosine_eta_min,
    )
    autocast_enabled = request.config.training.autocast and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=autocast_enabled)

    start_iteration = 0
    resume_path = resolve_resume_path(work_dir, request.resume)
    if resume_path is not None:
        meta = load_checkpoint(
            resume_path,
            model=model,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            scaler=scaler,
        )
        start_iteration = int(meta.get("iter", 0))

    _append_metrics(
        work_dir,
        {
            "event": "start",
            "timestamp": datetime.now().isoformat(),
            "device": str(device),
            "seed": request.seed,
            "config_hash": request.config.hash_,
            "gradient_accumulation_steps": request.config.training.gradient_accumulation_steps,
        },
    )

    data_iter = iter(train_loader)
    with tqdm(
        total=request.config.training.iterations,
        initial=start_iteration,
        desc="Training",
        disable=not progress_enabled,
    ) as progress:
        for iteration in range(start_iteration + 1, request.config.training.iterations + 1):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            metric_sums = {
                "loss": 0.0,
                "feasibility_loss": 0.0,
                "timing_loss": 0.0,
                "assignment_loss": 0.0,
                "assignment_accuracy": 0.0,
            }

            for _ in range(request.config.training.gradient_accumulation_steps):
                try:
                    batch = next(data_iter)
                except StopIteration:
                    data_iter = iter(train_loader)
                    batch = next(data_iter)

                batch = _move_batch(batch, device)
                with torch.autocast(device_type="cuda", enabled=autocast_enabled):
                    outputs = model.forward_supervised(
                        batch.time_steps,
                        batch.constellation_sensor_type,
                        batch.constellation_sensor_enabled,
                        batch.constellation_data,
                        batch.constellation_mask,
                        batch.tasks_sensor_type,
                        batch.tasks_data,
                        batch.tasks_mask,
                    )
                    losses = compute_supervised_losses(
                        outputs,
                        batch,
                        weights=request.config.loss_weights,
                    )
                    scaled_loss = losses.total / request.config.training.gradient_accumulation_steps

                if autocast_enabled:
                    scaler.scale(scaled_loss).backward()
                else:
                    scaled_loss.backward()

                metric_sums["loss"] += float(losses.total.item())
                metric_sums["feasibility_loss"] += float(losses.feasibility.item())
                metric_sums["timing_loss"] += float(losses.timing.item())
                metric_sums["assignment_loss"] += float(losses.assignment.item())
                metric_sums["assignment_accuracy"] += float(losses.assignment_accuracy.item())

            if autocast_enabled:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            lr_scheduler.step()

            train_metrics = {
                key: value / request.config.training.gradient_accumulation_steps
                for key, value in metric_sums.items()
            }
            train_metrics["lr"] = float(optimizer.param_groups[0]["lr"])
            progress.update(1)
            progress.set_postfix(
                loss=f"{train_metrics['loss']:.4f}",
                acc=f"{train_metrics['assignment_accuracy']:.4f}",
            )
            if (
                iteration == 1
                or iteration == request.config.training.iterations
                or iteration % request.config.training.log_interval == 0
            ):
                _print_step(
                    "train",
                    iteration,
                    request.config.training.iterations,
                    train_metrics,
                    progress=progress if progress_enabled else None,
                )
                _append_metrics(
                    work_dir,
                    {"event": "train", "iter": iteration, **train_metrics},
                )

            if (
                validation_loader is not None
                and (
                    iteration == request.config.training.iterations
                    or iteration % request.config.training.validation_interval == 0
                )
            ):
                validation_metrics = _evaluate_validation(
                    model,
                    validation_loader,
                    device=device,
                    autocast_enabled=autocast_enabled,
                    loss_weights=request.config.loss_weights,
                )
                _print_step(
                    "val",
                    iteration,
                    request.config.training.iterations,
                    validation_metrics,
                    progress=progress if progress_enabled else None,
                )
                _append_metrics(
                    work_dir,
                    {"event": "validation", "iter": iteration, **validation_metrics},
                )

            if (
                iteration == request.config.training.iterations
                or iteration % request.config.training.checkpoint_interval == 0
            ):
                save_checkpoint(
                    work_dir=work_dir,
                    iteration=iteration,
                    model=model,
                    optimizer=optimizer,
                    lr_scheduler=lr_scheduler,
                    scaler=scaler,
                    seed=request.seed,
                    autocast=autocast_enabled,
                    config_path=request.config.path,
                    config_hash=request.config.hash_,
                )

    return work_dir
