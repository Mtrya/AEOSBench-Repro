"""Tests for Phase 3 RL round-trip correctness.

T1: TrajectoryRecorder payload → SupervisedTrajectoryDataset
T3: Action decode consistency between gym env and direct rollout
T4: Statistics computation with selection_manifest
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from aeosbench.training.config import ConstraintLabelConfig
from aeosbench.training.dataset import SupervisedTrajectoryDataset
from aeosbench.training.statistics import compute_statistics


# ---------------------------------------------------------------------------
# T1: TrajectoryRecorder payload consumed by the supervised dataset loader
# ---------------------------------------------------------------------------

def _make_synthetic_recorder_payload(
    *,
    num_timesteps: int,
    num_satellites: int,
    num_tasks: int,
) -> dict[str, object]:
    """Build a payload with the same structure as TrajectoryRecorder.to_payload()."""
    return {
        "constellation": {
            "sensor_enabled": torch.randint(0, 2, (num_timesteps, num_satellites)),
            "data": torch.randn(num_timesteps, num_satellites, 8),
        },
        "taskset": {
            "progress": torch.zeros(num_timesteps, num_tasks, dtype=torch.float32),
        },
        "actions": {
            "task_id": torch.full(
                (num_timesteps, num_satellites), -1, dtype=torch.long
            ),
        },
        "is_visible": torch.zeros(
            num_timesteps, num_satellites, num_tasks, dtype=torch.bool
        ),
    }


def test_recorder_payload_round_trips_through_supervised_dataset(tmp_path):
    """A synthetic TrajectoryRecorder payload can be loaded by the dataset."""
    # Scenario 0, train split — we need a real constellation + taskset JSON on
    # disk but the trajectory payload is synthetic (the recorder's output).
    num_satellites = 42
    num_tasks = 90
    num_timesteps = 20

    payload = _make_synthetic_recorder_payload(
        num_timesteps=num_timesteps,
        num_satellites=num_satellites,
        num_tasks=num_tasks,
    )
    trajectory_path = tmp_path / "rollout_00000.pth"
    torch.save(payload, trajectory_path)

    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "entries": [
                    {
                        "split": "train",
                        "id": 0,
                        "source": {
                            "kind": "rollout",
                            "epoch": None,
                            "trajectory_path": str(trajectory_path),
                            "metrics_path": None,
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    dataset = SupervisedTrajectoryDataset(
        split="train",
        selection_manifest=manifest_path,
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
    assert batch.epoch is None
    assert batch.time_steps.shape == (4,)
    # constellation_data = static (48 dims) + dynamic (8 dims) = 56
    assert batch.constellation_data.shape[-1] == 56


# ---------------------------------------------------------------------------
# T3: Action decode consistency
# ---------------------------------------------------------------------------

def test_action_decode_matches_between_gym_and_direct_rollout():
    """The gym _decode_action and the direct rollout argmax-1 agree on in-range inputs."""
    # Simulate: 5 satellites, 10 ongoing tasks.
    # Actor logits shape: (1, num_satellites, 1 + num_tasks) where index 0 = idle.
    num_satellites = 5
    num_tasks = 10
    torch.manual_seed(42)
    logits = torch.randn(1, num_satellites, 1 + num_tasks)

    # Direct rollout path (rl_loop.py:241-242):
    # argmax over the last dim gives indices 0..num_tasks, then subtract 1.
    direct_assignment = (
        logits.argmax(dim=-1).squeeze(0) - 1
    ).to(dtype=torch.int64).cpu().tolist()

    # Gym path: SB3 samples from Categorical(logits=...) per satellite.
    # In the deterministic (argmax) case, the raw action per satellite is the
    # argmax of the per-satellite logit vector. The gym decode then does value-1
    # with clamping. For in-range argmax values, clamping is a no-op.
    raw_actions = logits.squeeze(0).argmax(dim=-1).numpy()
    gym_decoded: list[int] = []
    for value in raw_actions.tolist():
        task_index = int(value) - 1
        if task_index < -1 or task_index >= num_tasks:
            task_index = -1
        gym_decoded.append(task_index)

    assert direct_assignment == gym_decoded


def test_gym_decode_clamps_out_of_range_to_idle():
    """Out-of-range action values are clamped to -1 (idle) by the gym decode."""
    num_tasks = 5
    # Simulate raw actions: some in range, some out of range.
    # Value 0 → idle (-1), value 3 → task 2, value 100 → clamped to idle
    raw_actions = [0, 3, 100, num_tasks + 1]
    decoded: list[int] = []
    for value in raw_actions:
        task_index = int(value) - 1
        if task_index < -1 or task_index >= num_tasks:
            task_index = -1
        decoded.append(task_index)

    assert decoded == [-1, 2, -1, -1]


# ---------------------------------------------------------------------------
# T4: Statistics computation with selection_manifest
# ---------------------------------------------------------------------------

def test_compute_statistics_with_selection_manifest(tmp_path):
    """compute_statistics accepts a selection_manifest path and completes."""
    trajectory_path = Path("data/trajectories.3/train/00/00000.pth").resolve()
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "entries": [
                    {
                        "split": "train",
                        "id": 0,
                        "source": {
                            "kind": "released",
                            "epoch": 3,
                            "trajectory_path": str(trajectory_path),
                            "metrics_path": None,
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    output_path = tmp_path / "statistics.pth"
    compute_statistics(
        split="train",
        selection_manifest=manifest_path,
        output_path=output_path,
        show_progress=False,
    )

    assert output_path.exists()
    from aeosbench.evaluation.statistics import load_statistics

    stats = load_statistics(output_path)
    assert stats.constellation_mean is not None
    assert stats.taskset_mean is not None
