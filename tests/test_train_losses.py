import torch

from aeosbench.models.aeosformer import SupervisedOutputs
from aeosbench.training.dataset import SupervisedBatch
from aeosbench.training.losses import compute_supervised_losses


def test_compute_supervised_losses_returns_scalar_metrics():
    batch = SupervisedBatch(
        split="train",
        scenario_id=0,
        epoch=1,
        time_steps=torch.tensor([0]),
        constellation_sensor_type=torch.zeros((1, 1), dtype=torch.long),
        constellation_sensor_enabled=torch.zeros((1, 1), dtype=torch.long),
        constellation_data=torch.zeros((1, 1, 56), dtype=torch.float32),
        constellation_mask=torch.ones((1, 1), dtype=torch.bool),
        tasks_sensor_type=torch.zeros((1, 2), dtype=torch.long),
        tasks_data=torch.zeros((1, 2, 6), dtype=torch.float32),
        tasks_mask=torch.tensor([[True, True]], dtype=torch.bool),
        actions_task_id=torch.tensor([[1]], dtype=torch.long),
        feasibility_target=torch.tensor([[[False, True]]], dtype=torch.bool),
        timing_target=torch.tensor([[[0.0, 3.0]]], dtype=torch.float32),
    )
    outputs = SupervisedOutputs(
        feasibility_logits=torch.tensor([[[0.0, 4.0]]], dtype=torch.float32),
        timing_predictions=torch.tensor([[[1.0, 2.5]]], dtype=torch.float32),
        assignment_logits=torch.tensor([[[0.0, -1.0, 3.0]]], dtype=torch.float32),
    )

    summary = compute_supervised_losses(outputs, batch)

    assert summary.total.ndim == 0
    assert summary.feasibility.ndim == 0
    assert summary.timing.ndim == 0
    assert summary.assignment.ndim == 0
    assert 0.0 <= float(summary.assignment_accuracy.item()) <= 1.0
