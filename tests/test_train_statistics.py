from pathlib import Path

import torch

from aeosbench.training.statistics import compute_statistics


def test_compute_statistics_ignores_training_subset_limit(monkeypatch, tmp_path):
    seen: dict[str, int | None] = {}

    def fake_scenario_refs(
        split: str,
        *,
        annotation_file: str | None,
        selection_manifest,
        epoch: int | None,
        limit: int | None,
        dataset_root=None,
    ):
        seen["limit"] = limit
        seen["epoch"] = epoch
        seen["dataset_root"] = dataset_root
        return [type("Ref", (), {"split": split, "id_": 1, "epoch": 1, "trajectory_path": Path("/tmp/train-1-1.pth")})()]

    monkeypatch.setattr("aeosbench.training.statistics._scenario_refs", fake_scenario_refs)
    monkeypatch.setattr(
        "aeosbench.training.statistics.torch.load",
        lambda *args, **kwargs: {"constellation": {}, "taskset": {}},
    )
    monkeypatch.setattr(
        "aeosbench.training.statistics._build_constellation_tensors",
        lambda ref, trajectory, dataset_root=None: (
            torch.empty(0),
            torch.empty(0),
            torch.tensor([[[1.0] * 56]], dtype=torch.float32),
            torch.empty(0),
        ),
    )
    monkeypatch.setattr(
        "aeosbench.training.statistics._build_task_tensors",
        lambda ref, trajectory, dataset_root=None: (
            torch.empty(0),
            torch.tensor([[[2.0] * 6]], dtype=torch.float32),
            torch.tensor([[True]], dtype=torch.bool),
            torch.empty(0),
        ),
    )

    output_path = tmp_path / "statistics.pth"
    compute_statistics(
        split="train",
        annotation_file="train.json",
        output_path=output_path,
        show_progress=False,
    )

    assert seen["limit"] is None
    assert seen["epoch"] is None
    assert output_path.exists()
