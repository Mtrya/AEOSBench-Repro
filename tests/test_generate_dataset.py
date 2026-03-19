from dataclasses import replace
import json
import os
import random
import subprocess
import sys

from aeosbench.generation import load_generation_config, run_generation_stage
from aeosbench.generation.layout import GenerationLayout
from aeosbench.generation.pipeline import GenerationRequest
from aeosbench.generation.sampling import sample_sensor
from aeosbench.evaluation.layout import scenario_refs
from aeosbench.paths import project_root
from aeosbench.training.config import ConstraintLabelConfig
from aeosbench.training.dataset import SupervisedTrajectoryDataset


def _tiny_config():
    return load_generation_config(project_root() / "configs/dataset/tiny.yaml")


def _request(tmp_path, config, *, overwrite=False):
    return GenerationRequest(
        config=config,
        config_path=project_root() / "configs/dataset/tiny.yaml",
        output_root=tmp_path / "generated",
        seed=config.seed,
        overwrite=overwrite,
        show_progress=False,
        device="cpu",
    )


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
    assert (output_root / "data/tasksets/mrp.json").exists()
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


def test_assets_overwrite_rebuilds_existing_files(tmp_path):
    config = _tiny_config()
    request = _request(tmp_path, config)
    run_generation_stage("assets", request)

    layout = GenerationLayout(request.output_root)
    asset_path = layout.asset_path("train", 0)
    original = asset_path.read_text(encoding="utf-8")
    asset_path.write_text('{"corrupted": true}', encoding="utf-8")

    run_generation_stage("assets", request)
    assert asset_path.read_text(encoding="utf-8") == '{"corrupted": true}'

    run_generation_stage("assets", _request(tmp_path, config, overwrite=True))
    assert asset_path.read_text(encoding="utf-8") == original


def test_scenarios_clamp_satellite_count_to_available_asset_pool(tmp_path):
    base = _tiny_config()
    config = replace(
        base,
        scenarios=replace(
            base.scenarios,
            counts=replace(base.scenarios.counts, train=0, val_seen=0, val_unseen=1, test=1),
            min_satellites=2,
            max_satellites=2,
        ),
    )
    request = _request(tmp_path, config)
    run_generation_stage("assets", request)
    run_generation_stage("scenarios", request)

    layout = GenerationLayout(request.output_root)
    val_unseen = json.loads(layout.constellation_path("val_unseen", 0).read_text(encoding="utf-8"))
    test = json.loads(layout.constellation_path("test", 0).read_text(encoding="utf-8"))
    assert len(val_unseen["satellites"]) == 1
    assert len(test["satellites"]) == 1


def test_all_stage_skips_annotations_and_statistics_when_rollouts_disabled(tmp_path):
    base = _tiny_config()
    config = replace(
        base,
        rollouts=replace(base.rollouts, enabled=False),
        statistics=replace(base.statistics, enabled=True),
    )
    request = _request(tmp_path, config)
    run_generation_stage("all", request)

    layout = GenerationLayout(request.output_root)
    assert not layout.annotation_path("train").exists()
    assert not layout.statistics_path.exists()


def test_annotations_allow_rollout_shortfalls(tmp_path, monkeypatch):
    from aeosbench.generation.expert import TrajectoryMetrics
    from aeosbench.generation import pipeline as pipeline_module

    base = _tiny_config()
    config = replace(
        base,
        scenarios=replace(
            base.scenarios,
            counts=replace(base.scenarios.counts, train=2, val_seen=0, val_unseen=0, test=0),
        ),
    )
    request = _request(tmp_path, config)
    run_generation_stage("assets", request)
    run_generation_stage("scenarios", request)

    call_count = {"value": 0}

    def fake_rollout_with_expert(*, constellation, taskset, max_time_step):
        del constellation, taskset, max_time_step
        if call_count["value"] == 1:
            raise RuntimeError("sim failure")
        call_count["value"] += 1
        return {}, TrajectoryMetrics(
            cr=1.0,
            wcr=1.0,
            pcr=1.0,
            wpcr=1.0,
            tat_seconds=0.0,
            pc_watt_seconds=0.0,
        )

    monkeypatch.setattr(pipeline_module, "rollout_with_expert", fake_rollout_with_expert)
    run_generation_stage("rollouts", request)
    run_generation_stage("annotations", request)

    layout = GenerationLayout(request.output_root)
    annotations = json.loads(layout.annotation_path("train").read_text(encoding="utf-8"))
    assert annotations == {"ids": [0], "epochs": [1]}
    assert layout.rejection_log_path("rollout_rejections").exists()


def test_generation_statistics_do_not_mutate_aeos_data_root(tmp_path, monkeypatch):
    config = _tiny_config()
    request = _request(tmp_path, config)
    monkeypatch.setenv("AEOS_DATA_ROOT", "/tmp/original-root")
    run_generation_stage("all", request)
    assert os.environ["AEOS_DATA_ROOT"] == "/tmp/original-root"


def test_screening_sensor_is_fixed_and_generous():
    sensor = sample_sensor(random.Random(0), screening=True)
    assert sensor.half_field_of_view == 5.0
    assert sensor.power == 5.0
