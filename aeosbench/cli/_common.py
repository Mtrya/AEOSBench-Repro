"""Shared helpers for thin Phase 0 CLI stubs."""

from __future__ import annotations

import argparse
from collections.abc import Sequence


def build_placeholder_parser(
    prog: str,
    description: str,
) -> argparse.ArgumentParser:
    return argparse.ArgumentParser(
        prog=prog,
        description=description,
    )


def run_placeholder(
    parser: argparse.ArgumentParser,
    argv: Sequence[str] | None = None,
) -> int:
    parser.parse_args(argv)
    parser.exit(
        status=1,
        message=(
            f"{parser.prog} is a Phase 0 placeholder. "
            "Implementation lands in a later phase.\n"
        ),
    )

