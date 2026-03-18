"""Supervised AEOS-Former losses."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from aeosbench.models.aeosformer import SupervisedOutputs

from .dataset import SupervisedBatch


@dataclass(frozen=True)
class LossSummary:
    total: torch.Tensor
    feasibility: torch.Tensor
    timing: torch.Tensor
    assignment: torch.Tensor
    assignment_accuracy: torch.Tensor


def compute_supervised_losses(outputs: SupervisedOutputs, batch: SupervisedBatch) -> LossSummary:
    pair_mask = batch.constellation_mask.unsqueeze(-1) & batch.tasks_mask.unsqueeze(-2)
    feasibility_targets = batch.feasibility_target.to(dtype=torch.float32)

    if pair_mask.any():
        feasibility = F.binary_cross_entropy_with_logits(
            outputs.feasibility_logits[pair_mask],
            feasibility_targets[pair_mask],
        )
    else:
        feasibility = outputs.feasibility_logits.new_zeros(())

    timing_mask = pair_mask & batch.feasibility_target
    if timing_mask.any():
        timing = F.mse_loss(
            outputs.timing_predictions[timing_mask],
            batch.timing_target[timing_mask],
        )
    else:
        timing = outputs.timing_predictions.new_zeros(())

    assignment_targets = batch.actions_task_id + 1
    assignment_logits = outputs.assignment_logits.reshape(-1, outputs.assignment_logits.shape[-1])
    assignment = F.cross_entropy(
        assignment_logits,
        assignment_targets.reshape(-1),
    )
    predictions = outputs.assignment_logits.argmax(dim=-1) - 1
    assignment_accuracy = (predictions == batch.actions_task_id).to(dtype=torch.float32).mean()

    total = feasibility + timing + assignment
    return LossSummary(
        total=total,
        feasibility=feasibility,
        timing=timing,
        assignment=assignment,
        assignment_accuracy=assignment_accuracy,
    )
