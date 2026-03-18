from aeosbench.evaluation.statistics import load_statistics
from aeosbench.training.config import ConstraintLabelConfig
from aeosbench.training.dataset import SupervisedTrajectoryDataset


def test_supervised_dataset_loads_real_train_sample():
    dataset = SupervisedTrajectoryDataset(
        split="train",
        annotation_file="train.json",
        timesteps_per_sample=8,
        limit=1,
        constraint_labels=ConstraintLabelConfig(
            min_positive_run_length=3,
            max_time_horizon=100,
        ),
        statistics=load_statistics(),
        deterministic_sampling=True,
        seed=3407,
    )

    batch = dataset[0]

    assert batch.split == "train"
    assert batch.scenario_id == 0
    assert batch.time_steps.shape == (8,)
    assert batch.constellation_data.shape[0] == 8
    assert batch.constellation_data.shape[-1] == 56
    assert batch.tasks_data.shape[0] == 8
    assert batch.tasks_data.shape[-1] == 6
    assert batch.feasibility_target.shape[:2] == batch.actions_task_id.shape
    assert batch.timing_target.shape == batch.feasibility_target.shape
    assert batch.tasks_mask.any()
