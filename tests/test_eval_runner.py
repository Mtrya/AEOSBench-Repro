from pathlib import Path

from aeosbench.evaluation import runner


def test_missing_support_data_accepts_packaged_basilisk_layout(tmp_path, monkeypatch):
    support_root = tmp_path / "Basilisk" / "supportData"
    ephemeris_root = support_root / "EphemerisData"
    ephemeris_root.mkdir(parents=True)
    for rel_path in runner.DEFAULT_SPICE_KERNELS:
        path = support_root / rel_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")

    monkeypatch.setattr(runner, "_cached_support_path", lambda rel_path: None)
    monkeypatch.setattr(
        runner,
        "_packaged_support_path",
        lambda rel_path: support_root / rel_path,
    )

    assert runner.missing_support_data_files() == []
