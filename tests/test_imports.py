from aeosbench import __version__
from aeosbench.paths import benchmark_data_root, data_root, project_root


def test_package_imports():
    assert __version__
    assert project_root().name == "AEOSBench-Repro"
    assert data_root().name == "data"
    assert benchmark_data_root().name == "data"

