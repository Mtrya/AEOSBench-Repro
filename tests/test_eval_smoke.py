import pytest

from aeosbench.evaluation.model_config import load_model_config
from aeosbench.evaluation.layout import ScenarioRef
from aeosbench.evaluation.runner import EvaluationRequest, run_evaluation


@pytest.mark.slow
def test_real_eval_smoke_runs_one_case(monkeypatch):
    monkeypatch.setattr(
        "aeosbench.evaluation.runner.scenario_refs",
        lambda split, limit=None: [ScenarioRef(split="val_unseen", id_=238, epoch=1)],
    )
    config = load_model_config("configs/eval/official_aeosformer.yaml")
    request = EvaluationRequest(
        model_config=config,
        checkpoints=[__import__("pathlib").Path("data/model/model.pth")],
        splits=["val_unseen"],
        limit=1,
        device="cpu",
    )

    result = run_evaluation(request)

    assert len(result.rows) == 1
    assert result.rows[0].split == "val_unseen"
    assert len(result.rows[0].scenario_results) == 1
