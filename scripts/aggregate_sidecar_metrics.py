#!/usr/bin/env python3
"""Aggregate released trajectory sidecar metrics for a split."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from aeosbench.evaluation.layout import annotation_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate released trajectory sidecar metrics for a split.",
    )
    parser.add_argument(
        "--split",
        choices=("train", "val_seen", "val_unseen", "test"),
        required=True,
    )
    parser.add_argument(
        "--epoch",
        type=int,
        default=1,
        help="Trajectory epoch root to read from (default: 1).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    annotations = json.loads(annotation_path(args.split).read_text(encoding="utf-8"))
    ids = annotations["ids"]
    rows = []
    for id_ in ids:
        metric_path = Path(f"data/trajectories.{args.epoch}/{args.split}/{id_ // 1000:02d}/{id_:05d}.json")
        rows.append(json.loads(metric_path.read_text(encoding="utf-8")))

    means = {key: statistics.mean(row[key] for row in rows) for key in ("CR", "PCR", "WCR", "TAT", "PC")}
    weighted_score = 0.6 * means["CR"] + 0.2 * means["PCR"] + 0.2 * means["WCR"]
    cs_provisional = (weighted_score ** -1) + (means["TAT"] / 3600.0) + ((means["PC"] / 3600.0) / 100.0)
    payload = {
        "split": args.split,
        "epoch": args.epoch,
        "count": len(rows),
        "raw_metrics": {
            "CR": means["CR"],
            "PCR": means["PCR"],
            "WCR": means["WCR"],
            "TAT_seconds": means["TAT"],
            "PC_watt_seconds": means["PC"],
            "CS_provisional": cs_provisional,
        },
        "display_metrics": {
            "CR_percent": means["CR"] * 100.0,
            "PCR_percent": means["PCR"] * 100.0,
            "WCR_percent": means["WCR"] * 100.0,
            "TAT_hours": means["TAT"] / 3600.0,
            "PC_Wh": means["PC"] / 3600.0,
            "CS_provisional": cs_provisional,
        },
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
