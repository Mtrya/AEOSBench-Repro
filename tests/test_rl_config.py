from aeosbench.training.rl_config import load_rl_config


def test_load_rl_config_parses_tiny_yaml():
    config = load_rl_config("configs/train_rl/tiny.yaml")

    assert config.model.encoder_width == 512
    assert config.environment.split == "train"
    assert config.environment.limit == 1
    assert config.environment.num_envs == 1
    assert config.iterative.outer_iterations == 1
    assert config.ppo.total_timesteps == 20
    assert config.reward.completion_bonus == 100.0
    assert config.supervised.supervised_config.name == "tiny.yaml"
    assert config.validation_eval.enabled is False
