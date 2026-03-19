from pathlib import Path

from aeosbench.cli.train_rl import main


def test_train_rl_resume_defaults_work_dir_to_resume_path(monkeypatch, tmp_path):
    captured: dict[str, object] = {}
    resume_dir = tmp_path / "existing-run"

    monkeypatch.setattr("aeosbench.cli.train_rl.load_rl_config", lambda path: object())

    def fake_run_training(request):
        captured["work_dir"] = request.work_dir
        captured["resume"] = request.resume
        return request.work_dir

    monkeypatch.setattr("aeosbench.cli.train_rl.run_rl_training", fake_run_training)

    exit_code = main(["configs/train_rl/tiny.yaml", "--resume", str(resume_dir)])

    assert exit_code == 0
    assert captured["resume"] == resume_dir
    assert captured["work_dir"] == resume_dir
