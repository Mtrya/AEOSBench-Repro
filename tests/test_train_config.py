from aeosbench.training.config import load_training_config


def test_load_training_config_parses_tiny_yaml():
    config = load_training_config("configs/train_sl/tiny.yaml")

    assert config.model.encoder_width == 64
    assert config.data.timesteps_per_sample == 8
    assert config.constraint_labels.min_positive_run_length == 3
    assert config.loss_weights.feasibility == 1.0
    assert config.loss_weights.timing == 1.0
    assert config.loss_weights.assignment == 1.0
    assert config.training.iterations == 1
