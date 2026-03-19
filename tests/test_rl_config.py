import pytest

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
    assert config.reward.satellite_existence_cost == 1.0
    assert config.supervised.supervised_config.name == "tiny.yaml"
    assert config.validation_eval.enabled is False


def test_load_rl_config_rejects_removed_idle_satellite_penalty_key(tmp_path):
    path = tmp_path / "old.yaml"
    path.write_text(
        """
model:
  type: aeosformer
  time_embedding_dim: 64
  sensor_type_embedding_dim: 128
  tasks_data_embedding_dim: 128
  encoder_width: 512
  encoder_depth: 12
  encoder_num_heads: 16
  sensor_enabled_embedding_dim: 128
  constellation_data_embedding_dim: 128
  decoder_width: 512
  decoder_depth: 12
  decoder_num_heads: 16
  time_model_hidden_dim: 1024
initialization:
  actor_checkpoint: data/model/model.pth
environment:
  split: train
  annotation_file: null
  limit: 1
  num_envs: 1
reward:
  completion_bonus: 100.0
  visible_satellite_bonus: 2.0
  idle_satellite_penalty: 1.0
  scale: 10.0
ppo:
  total_timesteps: 20
  n_steps: 4
  batch_size: 4
  n_epochs: 1
  learning_rate: 0.0003
  gamma: 0.99
  gae_lambda: 0.95
  clip_range: 0.2
  ent_coef: 0.0
  vf_coef: 0.5
  save_freq: 10
iterative:
  outer_iterations: 1
  rollout_limit: 1
  min_cr_improvement: 0.0
supervised:
  config: configs/train_sl/tiny.yaml
validation_eval:
  enabled: false
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(TypeError, match="satellite_existence_cost"):
        load_rl_config(path)
