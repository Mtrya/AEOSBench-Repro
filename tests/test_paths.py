from pathlib import Path

from aeosbench import paths


def test_default_data_root_resolves_to_project_data(monkeypatch):
    monkeypatch.delenv("AEOS_DATA_ROOT", raising=False)

    assert paths.data_root() == paths.project_root() / "data"


def test_aeos_data_root_override(monkeypatch, tmp_path):
    override_root = tmp_path / "custom-data"
    override_root.mkdir()
    monkeypatch.setenv("AEOS_DATA_ROOT", str(override_root))

    assert paths.data_root() == override_root.resolve()


def test_relative_aeos_data_root_override_is_project_relative(monkeypatch, tmp_path):
    relative_root = Path("tmp-data")
    target_root = paths.project_root() / relative_root
    target_root.mkdir(exist_ok=True)
    monkeypatch.setenv("AEOS_DATA_ROOT", str(relative_root))

    assert paths.data_root() == target_root.resolve()


def test_benchmark_data_root_prefers_nested_layout(monkeypatch, tmp_path):
    outer_root = tmp_path / "aeos-data"
    nested_root = outer_root / "data"
    nested_root.mkdir(parents=True)
    monkeypatch.setenv("AEOS_DATA_ROOT", str(outer_root))

    assert paths.benchmark_data_root() == nested_root.resolve()


def test_benchmark_data_root_falls_back_to_flat_layout(monkeypatch, tmp_path):
    flat_root = tmp_path / "flat-data"
    flat_root.mkdir()
    monkeypatch.setenv("AEOS_DATA_ROOT", str(flat_root))

    assert paths.benchmark_data_root() == flat_root.resolve()


def test_project_relative_path_inside_repo():
    path = paths.project_root() / "data" / "model" / "model.pth"

    assert paths.project_relative_path(path) == Path("data/model/model.pth")


def test_project_relative_path_outside_repo(tmp_path):
    path = tmp_path / "artifact.txt"
    path.write_text("artifact", encoding="utf-8")

    assert paths.project_relative_path(path) == path.resolve()

