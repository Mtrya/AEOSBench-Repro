"""Public dataset generation script entrypoint."""

from __future__ import annotations

from collections.abc import Sequence

from aeosbench.cli._common import build_placeholder_parser, run_placeholder


def build_parser():
    return build_placeholder_parser(
        prog="python scripts/generate_dataset.py",
        description="Generate AEOSBench dataset artifacts.",
    )


def main(argv: Sequence[str] | None = None) -> int:
    return run_placeholder(build_parser(), argv)


if __name__ == "__main__":
    raise SystemExit(main())

