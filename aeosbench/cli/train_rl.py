"""Public reinforcement-learning script entrypoint."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

from aeosbench.training.rl_config import load_rl_config
from aeosbench.training.rl_loop import RLTrainingRequest, run_rl_training


def build_parser():
    parser = argparse.ArgumentParser(
        prog="python scripts/train_rl.py",
        description="Train AEOSBench policies with reinforcement learning.",
    )
    parser.add_argument("config", type=Path)
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=None,
        help="Override the default timestamped work directory under outputs/train_rl/",
    )
    parser.add_argument(
        "--resume",
        default=None,
        help="Resume from an existing RL work directory state.",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm/SB3 progress bars during rollout collection and PPO learning.",
    )
    parser.add_argument("--seed", type=int, default=3407)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    config = load_rl_config(args.config)
    work_dir = args.work_dir
    if work_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        work_dir = Path("outputs") / "train_rl" / f"{args.config.stem}_{timestamp}"
    run_rl_training(
        RLTrainingRequest(
            config=config,
            work_dir=work_dir,
            device=args.device,
            seed=args.seed,
            resume=args.resume,
            show_progress=not args.no_progress,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
