import subprocess
import sys

from aeosbench.paths import project_root


SCRIPT_PATHS = [
    "scripts/eval.py",
    "scripts/train_sl.py",
    "scripts/train_rl.py",
    "scripts/run_baseline.py",
    "scripts/generate_dataset.py",
]


def test_scripts_help():
    root = project_root()

    for script in SCRIPT_PATHS:
        completed = subprocess.run(
            [sys.executable, script, "--help"],
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
        )
        assert completed.returncode == 0, completed.stderr
        assert "usage:" in completed.stdout
