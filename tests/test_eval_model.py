import torch

from aeosbench.evaluation.checkpoints import load_actor_checkpoint, normalized_state_dict
from aeosbench.evaluation.model_config import load_model_config


def test_load_model_config_parses_official_yaml():
    config = load_model_config("configs/eval/official_aeosformer.yaml")

    assert config.model.encoder_depth == 12
    assert config.model.decoder_num_heads == 16
    assert len(config.hash_) == 12


def test_normalized_state_dict_drops_training_only_keys():
    state_dict = normalized_state_dict("data/model/model.pth")

    assert "_ce_loss._weight._steps" not in state_dict
    assert "_transformer._time_model._mse_loss._weight._steps" not in state_dict
    assert "_transformer._time_model._bce_loss._weight._steps" not in state_dict
    assert "_transformer._encoder._blocks.0._norm1.weight" in state_dict


def test_official_checkpoint_loads_and_runs_forward():
    config = load_model_config("configs/eval/official_aeosformer.yaml")
    model = load_actor_checkpoint(config.model, "data/model/model.pth")

    logits = model.predict(
        time_steps=torch.tensor([0]),
        constellation_sensor_type=torch.zeros((1, 2), dtype=torch.long),
        constellation_sensor_enabled=torch.zeros((1, 2), dtype=torch.long),
        constellation_data=torch.zeros((1, 2, 56), dtype=torch.float32),
        constellation_mask=torch.ones((1, 2), dtype=torch.bool),
        tasks_sensor_type=torch.zeros((1, 3), dtype=torch.long),
        tasks_data=torch.zeros((1, 3, 6), dtype=torch.float32),
        tasks_mask=torch.ones((1, 3), dtype=torch.bool),
    )

    assert logits.shape == (1, 2, 4)
