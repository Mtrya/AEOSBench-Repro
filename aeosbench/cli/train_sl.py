"""Public supervised training script entrypoint."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

from aeosbench.training.config import load_training_config
from aeosbench.training.trainer import TrainingRequest, run_training


def build_parser():
    parser = argparse.ArgumentParser(
        prog="python scripts/train_sl.py",
        description="Train AEOS-Former with supervised learning.",
    )
    parser.add_argument("config", type=Path)
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=None,
        help="Override the default timestamped work directory under outputs/train_sl/",
    )
    parser.add_argument(
        "--resume",
        default=None,
        help="Resume from an explicit checkpoint directory or use 'latest' in the work dir.",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
    )
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--load-model-from", nargs="*", type=Path, default=[])
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    config = load_training_config(args.config)
    work_dir = args.work_dir
    if work_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        work_dir = Path("outputs") / "train_sl" / f"{args.config.stem}_{timestamp}"
    run_training(
        TrainingRequest(
            config=config,
            work_dir=work_dir,
            device=args.device,
            seed=args.seed,
            resume=args.resume,
            load_model_from=tuple(args.load_model_from),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
