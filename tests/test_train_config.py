from pathlib import Path

from aeosbench.training.config import load_training_config


def test_load_training_config_parses_tiny_yaml():
    config = load_training_config("configs/train_sl/tiny.yaml")

    assert config.model.encoder_width == 64
    assert config.initialization.checkpoint is None
    assert config.data.timesteps_per_scenario == 8
    assert config.constraint_labels.min_positive_run_length == 3
    assert config.loss_weights.feasibility == 1.0
    assert config.loss_weights.timing == 1.0
    assert config.loss_weights.assignment == 1.0
    assert config.training.gradient_accumulation_steps == 2
    assert config.training.iterations == 1


def test_load_training_config_accepts_legacy_timesteps_per_sample_alias(tmp_path):
    config_path = tmp_path / "legacy.yaml"
    config_path.write_text(
        Path("configs/train_sl/tiny.yaml").read_text(encoding="utf-8").replace(
            "timesteps_per_scenario",
            "timesteps_per_sample",
        ),
        encoding="utf-8",
    )

    config = load_training_config(config_path)

    assert config.data.timesteps_per_scenario == 8
    assert config.validation.timesteps_per_scenario == 8


def test_load_training_config_resolves_initialization_checkpoint_from_project_root(tmp_path):
    config_path = tmp_path / "init.yaml"
    config_path.write_text(
        Path("configs/train_sl/tiny.yaml").read_text(encoding="utf-8").replace(
            "checkpoint: null",
            "checkpoint: data/model/model.pth",
        ),
        encoding="utf-8",
    )

    config = load_training_config(config_path)

    assert config.initialization.checkpoint is not None
    assert config.initialization.checkpoint.name == "model.pth"
    assert config.initialization.checkpoint.exists()
