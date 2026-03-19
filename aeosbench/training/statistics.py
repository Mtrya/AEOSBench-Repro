"""Statistics loading and bootstrap helpers for supervised training."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from tqdm.auto import tqdm

from aeosbench.evaluation.statistics import Statistics, load_statistics, statistics_path

from .dataset import _build_constellation_tensors, _build_task_tensors, _scenario_refs


@dataclass
class RunningMoments:
    count: int = 0
    sum_: torch.Tensor | None = None
    sum_squares: torch.Tensor | None = None

    def update(self, values: torch.Tensor) -> None:
        flattened = values.reshape(-1, values.shape[-1]).to(dtype=torch.float64)
        batch_count = flattened.shape[0]
        batch_sum = flattened.sum(dim=0)
        batch_sum_squares = flattened.square().sum(dim=0)

        if self.sum_ is None:
            self.sum_ = batch_sum
            self.sum_squares = batch_sum_squares
        else:
            self.sum_ += batch_sum
            assert self.sum_squares is not None
            self.sum_squares += batch_sum_squares
        self.count += batch_count

    def finalize(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.count == 0 or self.sum_ is None or self.sum_squares is None:
            raise RuntimeError("No samples were collected while computing statistics")
        mean = self.sum_ / self.count
        variance = (self.sum_squares / self.count) - mean.square()
        return mean.float(), variance.clamp_min(0).sqrt().float()


def compute_statistics(
    *,
    split: str = "train",
    annotation_file: str | None = None,
    selection_manifest: str | Path | None = None,
    output_path: Path | None = None,
    show_progress: bool = True,
    dataset_root: Path | None = None,
) -> Path:
    refs = _scenario_refs(
        split,
        annotation_file=annotation_file,
        selection_manifest=selection_manifest,
        epoch=None,
        limit=None,
        dataset_root=dataset_root,
    )
    constellation_stats = RunningMoments()
    taskset_stats = RunningMoments()
    iterator = refs
    if show_progress:
        iterator = tqdm(refs, desc="Computing training statistics")
    for ref in iterator:
        trajectory = torch.load(
            ref.trajectory_path,
            map_location="cpu",
            weights_only=False,
        )
        if not isinstance(trajectory, dict):
            raise TypeError(f"trajectory payload must be a mapping: {ref}")
        _, _, constellation_data, _ = _build_constellation_tensors(ref, trajectory, dataset_root=dataset_root)
        _, task_data, task_mask, _ = _build_task_tensors(ref, trajectory, dataset_root=dataset_root)
        valid_indices = task_mask.any(dim=-1)
        task_is_valid = task_mask[valid_indices].any(dim=0)
        constellation_stats.update(constellation_data[valid_indices])
        taskset_stats.update(task_data[valid_indices][:, task_is_valid])

    constellation_mean, constellation_std = constellation_stats.finalize()
    taskset_mean, taskset_std = taskset_stats.finalize()
    path = statistics_path() if output_path is None else output_path
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        Statistics(
            constellation_mean=constellation_mean,
            constellation_std=constellation_std,
            taskset_mean=taskset_mean,
            taskset_std=taskset_std,
        ),
        path,
    )
    return path


def ensure_statistics(
    *,
    mode: str,
    path: Path | None = None,
    split: str = "train",
    annotation_file: str | None = None,
    selection_manifest: str | Path | None = None,
    show_progress: bool = True,
) -> Statistics:
    stats_path = statistics_path() if path is None else path
    if mode == "load_only":
        return load_statistics(stats_path)
    if mode == "compute_only":
        compute_statistics(
            split=split,
            annotation_file=annotation_file,
            selection_manifest=selection_manifest,
            output_path=stats_path,
            show_progress=show_progress,
        )
        return load_statistics(stats_path)
    if mode != "load_or_compute":
        raise ValueError(f"unsupported statistics mode: {mode!r}")
    if stats_path.exists():
        return load_statistics(stats_path)
    compute_statistics(
        split=split,
        annotation_file=annotation_file,
        selection_manifest=selection_manifest,
        output_path=stats_path,
        show_progress=show_progress,
    )
    return load_statistics(stats_path)
