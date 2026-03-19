from aeosbench.generation.config import load_generation_config
from aeosbench.paths import project_root


def test_generation_config_loads_tiny():
    config = load_generation_config(project_root() / "configs/dataset/tiny.yaml")

    assert config.seed == 7
    assert config.assets.train_count == 2
    assert config.scenarios.counts.train == 1
    assert config.rollouts.epoch == 1
    assert config.statistics.enabled is True
