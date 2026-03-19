import json
from pathlib import Path

from aeosbench.training.config import ConstraintLabelConfig
from aeosbench.training.dataset import SupervisedTrajectoryDataset
from aeosbench.training.selection import load_selection_manifest


def test_selection_manifest_loads_and_dataset_consumes_real_rollout_source(tmp_path):
    manifest_path = tmp_path / "manifest.json"
    trajectory_path = Path("data/trajectories.3/train/00/00000.pth").resolve()
    metrics_path = Path("data/trajectories.3/train/00/00000.json").resolve()
    manifest_path.write_text(
        json.dumps(
            {
                "entries": [
                    {
                        "split": "train",
                        "id": 0,
                        "source": {
                            "kind": "released",
                            "epoch": 3,
                            "trajectory_path": str(trajectory_path),
                            "metrics_path": str(metrics_path),
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    manifest = load_selection_manifest(manifest_path)
    assert manifest.entries[0].id_ == 0
    assert manifest.entries[0].epoch == 3

    dataset = SupervisedTrajectoryDataset(
        split="train",
        selection_manifest=manifest_path,
        timesteps_per_scenario=4,
        limit=1,
        constraint_labels=ConstraintLabelConfig(
            min_positive_run_length=3,
            max_time_horizon=100,
        ),
        statistics=None,
        deterministic_sampling=True,
        seed=3407,
    )
    batch = dataset[0]

    assert batch.scenario_id == 0
    assert batch.epoch == 3
    assert batch.time_steps.shape == (4,)
    assert batch.constellation_data.shape[-1] == 56


def test_selection_manifest_with_null_epoch_is_accepted_for_rollout_sources(tmp_path):
    manifest_path = tmp_path / "manifest.json"
    trajectory_path = Path("data/trajectories.3/train/00/00000.pth").resolve()
    metrics_path = Path("data/trajectories.3/train/00/00000.json").resolve()
    manifest_path.write_text(
        json.dumps(
            {
                "entries": [
                    {
                        "split": "train",
                        "id": 0,
                        "source": {
                            "kind": "rollout",
                            "epoch": None,
                            "trajectory_path": str(trajectory_path),
                            "metrics_path": str(metrics_path),
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    dataset = SupervisedTrajectoryDataset(
        split="train",
        selection_manifest=manifest_path,
        timesteps_per_scenario=4,
        limit=1,
        constraint_labels=ConstraintLabelConfig(
            min_positive_run_length=3,
            max_time_horizon=100,
        ),
        statistics=None,
        deterministic_sampling=True,
        seed=3407,
    )
    batch = dataset[0]

    assert batch.scenario_id == 0
    assert batch.epoch is None
    assert batch.time_steps.shape == (4,)
