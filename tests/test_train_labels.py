import torch

from aeosbench.training.config import ConstraintLabelConfig
from aeosbench.training.dataset import _constraint_targets_for_times


def test_constraint_targets_follow_clean_future_run_rule():
    actions = torch.tensor(
        [
            [1],
            [1],
            [0],
            [0],
            [0],
            [-1],
        ],
        dtype=torch.long,
    )
    is_visible = torch.tensor(
        [
            [[False, False]],
            [[False, False]],
            [[True, False]],
            [[True, False]],
            [[True, False]],
            [[False, False]],
        ],
        dtype=torch.bool,
    )
    progress = torch.tensor(
        [
            [0, 0],
            [0, 0],
            [1, 0],
            [2, 0],
            [3, 0],
            [3, 0],
        ],
        dtype=torch.float32,
    )
    durations = torch.tensor([3, 1], dtype=torch.float32)

    feasible, timing = _constraint_targets_for_times(
        actions,
        is_visible,
        progress,
        durations,
        [0, 2, 4],
        ConstraintLabelConfig(min_positive_run_length=3, max_time_horizon=100),
    )

    assert feasible[:, 0, 0].tolist() == [True, True, False]
    assert feasible[:, 0, 1].tolist() == [False, False, False]
    assert timing[0, 0, 0].item() == 2.0
    assert timing[1, 0, 0].item() == 0.0
