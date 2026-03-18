from aeosbench.evaluation.metrics import RawMetrics, provisional_cs
from aeosbench.evaluation.reports import render_markdown, render_terminal_table
from aeosbench.evaluation.runner import EvaluationResult, EvaluationRow, ScenarioResult


def test_raw_metrics_display_converts_time_and_power_units():
    metrics = RawMetrics(
        cr=0.25,
        pcr=0.50,
        wcr=0.75,
        tat_seconds=7200.0,
        pc_watt_seconds=36000.0,
    )

    display = metrics.display

    assert display.cr_percent == 25.0
    assert display.pcr_percent == 50.0
    assert display.wcr_percent == 75.0
    assert display.tat_hours == 2.0
    assert display.pc_wh == 10.0


def test_provisional_cs_uses_documented_formula():
    metrics = RawMetrics(
        cr=0.3,
        pcr=0.4,
        wcr=0.5,
        tat_seconds=3600.0,
        pc_watt_seconds=36000.0,
    )

    score = provisional_cs(metrics)

    assert score > 0
    assert round(score, 4) == round((0.6 * 0.3 + 0.2 * 0.4 + 0.2 * 0.5) ** -1 + 1.0 + 0.1, 4)


def test_render_terminal_table_includes_provisional_note():
    raw = RawMetrics(
        cr=0.2,
        pcr=0.3,
        wcr=0.4,
        tat_seconds=1800.0,
        pc_watt_seconds=7200.0,
    )
    row = EvaluationRow(
        split="val_seen",
        checkpoint_path=__import__("pathlib").Path("data/model/model.pth"),
        timestamp="2026-03-18T00:00:00+00:00",
        config_hash="abc123def456",
        limit=1,
        scenario_results=[ScenarioResult(id_=238, epoch=1, raw_metrics=raw)],
        aggregate_raw_metrics=raw,
        aggregate_display_metrics=raw.display,
        cs_provisional=provisional_cs(raw),
    )
    result = EvaluationResult(
        model_config_path=__import__("pathlib").Path("configs/eval/official_aeosformer.yaml"),
        rows=[row],
    )

    rendered = render_terminal_table(result)

    assert "Split" in rendered
    assert "val_seen" in rendered
    assert "CS is provisional" in rendered


def test_render_markdown_includes_row_metadata():
    raw = RawMetrics(
        cr=0.2,
        pcr=0.3,
        wcr=0.4,
        tat_seconds=1800.0,
        pc_watt_seconds=7200.0,
    )
    row = EvaluationRow(
        split="val_seen",
        checkpoint_path=__import__("pathlib").Path("data/model/model.pth"),
        timestamp="2026-03-18T00:00:00+00:00",
        config_hash="abc123def456",
        limit=1,
        scenario_results=[ScenarioResult(id_=238, epoch=1, raw_metrics=raw)],
        aggregate_raw_metrics=raw,
        aggregate_display_metrics=raw.display,
        cs_provisional=provisional_cs(raw),
    )
    result = EvaluationResult(
        model_config_path=__import__("pathlib").Path("configs/eval/official_aeosformer.yaml"),
        rows=[row],
    )

    rendered = render_markdown(result)

    assert "Timestamp" in rendered
    assert "Config Hash" in rendered
    assert "2026-03-18T00:00:00+00:00" in rendered
    assert "abc123def456" in rendered
