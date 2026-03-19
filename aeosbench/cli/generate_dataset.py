"""Public dataset generation script entrypoint."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from aeosbench.generation import load_generation_config, run_generation_stage
from aeosbench.generation.pipeline import GenerationRequest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python scripts/generate_dataset.py",
        description="Generate AEOSBench dataset artifacts.",
    )
    parser.add_argument(
        "stage",
        choices=["assets", "scenarios", "rollouts", "annotations", "statistics", "all"],
        help="Which generation stage to run.",
    )
    parser.add_argument(
        "config",
        type=Path,
        help="Path to the dataset generation YAML config.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="Root directory for the generated dataset tree.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional override for the config seed.",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Execution device hint for rollout stages.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite any existing generated artifacts at the output root.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm progress bars.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    config_path = args.config.resolve()
    output_root = args.output_root.resolve()
    config = load_generation_config(config_path)
    seed = config.seed if args.seed is None else int(args.seed)
    run_generation_stage(
        args.stage,
        GenerationRequest(
            config=config,
            config_path=config_path,
            output_root=output_root,
            seed=seed,
            overwrite=bool(args.overwrite),
            show_progress=not bool(args.no_progress),
            device=str(args.device),
        ),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
