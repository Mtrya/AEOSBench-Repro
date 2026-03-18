"""Public evaluation script entrypoint."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

from aeosbench.evaluation.model_config import load_model_config
from aeosbench.evaluation.reports import (
    render_terminal_table,
    write_json_report,
    write_markdown_report,
)
from aeosbench.evaluation.runner import EvaluationRequest, run_evaluation


def build_parser():
    parser = argparse.ArgumentParser(
        prog="python scripts/eval.py",
        description="Evaluate AEOSBench checkpoints.",
    )
    parser.add_argument("model_config", type=Path)
    parser.add_argument("checkpoints", nargs="+", type=Path)
    parser.add_argument(
        "--split",
        dest="splits",
        action="append",
        choices=["val_seen", "val_unseen", "test"],
        required=True,
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
    )
    parser.add_argument(
        "--format",
        dest="formats",
        action="append",
        choices=["terminal", "json", "md"],
        default=None,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable the scenario-level progress bar.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    model_config = load_model_config(args.model_config)
    request = EvaluationRequest(
        model_config=model_config,
        checkpoints=args.checkpoints,
        splits=args.splits,
        limit=args.limit,
        device=args.device,
        show_progress=not args.no_progress,
    )
    result = run_evaluation(request)
    formats = ["terminal", "json", "md"] if not args.formats else args.formats
    output_dir = args.output_dir
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path("outputs") / "eval" / timestamp
    if "terminal" in formats:
        print(render_terminal_table(result))
    if "json" in formats:
        write_json_report(result, output_dir)
    if "md" in formats:
        write_markdown_report(result, output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
