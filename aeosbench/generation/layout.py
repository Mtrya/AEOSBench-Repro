"""Filesystem helpers for generated dataset trees."""

from __future__ import annotations

from pathlib import Path


class GenerationLayout:
    def __init__(self, output_root: Path) -> None:
        self.output_root = output_root

    @property
    def benchmark_root(self) -> Path:
        return self.output_root / "data"

    @property
    def annotations_root(self) -> Path:
        return self.benchmark_root / "annotations"

    @property
    def constellations_root(self) -> Path:
        return self.benchmark_root / "constellations"

    @property
    def tasksets_root(self) -> Path:
        return self.benchmark_root / "tasksets"

    @property
    def satellites_root(self) -> Path:
        return self.benchmark_root / "satellites"

    @property
    def orbits_root(self) -> Path:
        return self.benchmark_root / "orbits"

    @property
    def generation_root(self) -> Path:
        return self.output_root / "generation"

    @property
    def statistics_path(self) -> Path:
        return self.output_root / "statistics_new.pth"

    @property
    def screening_taskset_path(self) -> Path:
        return self.tasksets_root / "mrp.json"

    def trajectory_root(self, epoch: int) -> Path:
        return self.output_root / f"trajectories.{int(epoch)}"

    def ensure_roots(self) -> None:
        for path in (
            self.annotations_root,
            self.constellations_root,
            self.tasksets_root,
            self.satellites_root,
            self.orbits_root,
            self.generation_root,
        ):
            path.mkdir(parents=True, exist_ok=True)

    def asset_path(self, split: str, asset_id: int) -> Path:
        return self.satellites_root / split / f"{asset_id}.json"

    def orbit_path(self, orbit_id: int) -> Path:
        return self.orbits_root / f"{orbit_id}.json"

    def constellation_path(self, split: str, scenario_id: int) -> Path:
        return self.constellations_root / split / f"{scenario_id // 1000:02d}" / f"{scenario_id:05d}.json"

    def taskset_path(self, split: str, scenario_id: int) -> Path:
        return self.tasksets_root / split / f"{scenario_id // 1000:02d}" / f"{scenario_id:05d}.json"

    def trajectory_payload_path(self, split: str, scenario_id: int, *, epoch: int) -> Path:
        return self.trajectory_root(epoch) / split / f"{scenario_id // 1000:02d}" / f"{scenario_id:05d}.pth"

    def trajectory_metrics_path(self, split: str, scenario_id: int, *, epoch: int) -> Path:
        return self.trajectory_root(epoch) / split / f"{scenario_id // 1000:02d}" / f"{scenario_id:05d}.json"

    def annotation_path(self, split: str) -> Path:
        return self.annotations_root / f"{split}.json"

    def rejection_log_path(self, name: str) -> Path:
        return self.generation_root / f"{name}.jsonl"
