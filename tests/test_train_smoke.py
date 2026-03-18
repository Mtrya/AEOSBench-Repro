import subprocess
import sys

from aeosbench.paths import project_root


def test_train_sl_tiny_smoke(tmp_path):
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/train_sl.py",
            "configs/train_sl/tiny.yaml",
            "--work-dir",
            str(tmp_path / "train_smoke"),
            "--device",
            "cpu",
            "--seed",
            "123",
        ],
        cwd=project_root(),
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert (tmp_path / "train_smoke" / "metrics.jsonl").exists()
    assert (tmp_path / "train_smoke" / "checkpoints" / "iter_1" / "model.pth").exists()
