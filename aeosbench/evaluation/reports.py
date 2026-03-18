"""Evaluation report rendering."""

from __future__ import annotations

import json
from pathlib import Path

from .runner import EvaluationResult, result_to_dict


def write_json_report(result: EvaluationResult, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "summary.json"
    path.write_text(json.dumps(result_to_dict(result), indent=2), encoding="utf-8")
    return path


def render_markdown(result: EvaluationResult) -> str:
    lines = [
        "# AEOSBench Evaluation Summary",
        "",
        f"- Model config: `{result.model_config_path}`",
        "",
        "| Split | Checkpoint | Timestamp | Config Hash | Limit | CR % | PCR % | WCR % | TAT h | PC Wh | CS* |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in result.rows:
        metrics = row.aggregate_display_metrics
        lines.append(
            "| "
            + " | ".join(
                [
                    row.split,
                    str(row.checkpoint_path),
                    row.timestamp,
                    row.config_hash,
                    "-" if row.limit is None else str(row.limit),
                    f"{metrics.cr_percent:.2f}",
                    f"{metrics.pcr_percent:.2f}",
                    f"{metrics.wcr_percent:.2f}",
                    f"{metrics.tat_hours:.4f}",
                    f"{metrics.pc_wh:.2f}",
                    f"{row.cs_provisional:.4f}",
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "* `CS` is provisional in Phase 1 and follows the formula transcribed in `WANG2025TOWARDS.md`.",
        ]
    )
    return "\n".join(lines)


def write_markdown_report(result: EvaluationResult, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "summary.md"
    path.write_text(render_markdown(result), encoding="utf-8")
    return path


def render_terminal_table(result: EvaluationResult) -> str:
    headers = ["Split", "Checkpoint", "CR %", "PCR %", "WCR %", "TAT h", "PC Wh", "CS*"]
    rows = []
    for row in result.rows:
        metrics = row.aggregate_display_metrics
        rows.append(
            [
                row.split,
                str(row.checkpoint_path),
                f"{metrics.cr_percent:.2f}",
                f"{metrics.pcr_percent:.2f}",
                f"{metrics.wcr_percent:.2f}",
                f"{metrics.tat_hours:.4f}",
                f"{metrics.pc_wh:.2f}",
                f"{row.cs_provisional:.4f}",
            ]
        )
    widths = [
        max(len(header), *(len(str(row[index])) for row in rows))
        for index, header in enumerate(headers)
    ]
    parts = [
        " | ".join(header.ljust(widths[index]) for index, header in enumerate(headers)),
        "-+-".join("-" * width for width in widths),
    ]
    for row in rows:
        parts.append(" | ".join(str(value).ljust(widths[index]) for index, value in enumerate(row)))
    parts.append("")
    parts.append("* CS is provisional in Phase 1.")
    return "\n".join(parts)
