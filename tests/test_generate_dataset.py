import json
import subprocess
import sys

from aeosbench.evaluation.layout import scenario_refs
from aeosbench.paths import project_root
from aeosbench.training.config import ConstraintLabelConfig
from aeosbench.training.dataset import SupervisedTrajectoryDataset


def test_tiny_generation_round_trip(tmp_path, monkeypatch):
    output_root = tmp_path / "generated"
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/generate_dataset.py",
            "all",
            "configs/dataset/tiny.yaml",
            "--output-root",
            str(output_root),
            "--device",
            "cpu",
            "--no-progress",
        ],
        cwd=project_root(),
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr

    assert (output_root / "data/annotations/train.json").exists()
    assert (output_root / "data/constellations/train/00/00000.json").exists()
    assert (output_root / "data/tasksets/train/00/00000.json").exists()
    assert (output_root / "data/satellites/train/0.json").exists()
    assert (output_root / "trajectories.1/train/00/00000.pth").exists()
    assert (output_root / "statistics_new.pth").exists()

    metrics = json.loads((output_root / "trajectories.1/train/00/00000.json").read_text())
    assert set(metrics) == {"CR", "WCR", "PCR", "WPCR", "TAT", "PC"}

    monkeypatch.setenv("AEOS_DATA_ROOT", str(output_root))
    refs = scenario_refs("train")
    assert len(refs) == 1
    assert refs[0].split == "train"
    assert refs[0].id_ == 0
    assert refs[0].epoch == 1

    dataset = SupervisedTrajectoryDataset(
        split="train",
        annotation_file="train.json",
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
    assert batch.constellation_data.shape[-1] == 56
    assert batch.tasks_data.shape[-1] == 6
