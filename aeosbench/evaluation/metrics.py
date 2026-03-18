"""Evaluation metrics and report-friendly conversions."""

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class RawMetrics:
    cr: float
    pcr: float
    wcr: float
    tat_seconds: float
    pc_watt_seconds: float

    @property
    def display(self) -> "DisplayMetrics":
        return DisplayMetrics(
            cr_percent=self.cr * 100.0,
            pcr_percent=self.pcr * 100.0,
            wcr_percent=self.wcr * 100.0,
            tat_hours=self.tat_seconds / 3600.0,
            pc_wh=self.pc_watt_seconds / 3600.0,
        )


@dataclass(frozen=True)
class DisplayMetrics:
    cr_percent: float
    pcr_percent: float
    wcr_percent: float
    tat_hours: float
    pc_wh: float


def provisional_cs(raw: RawMetrics) -> float:
    weighted_score = 0.6 * raw.cr + 0.2 * raw.pcr + 0.2 * raw.wcr
    if weighted_score <= 0:
        return math.inf
    return (weighted_score ** -1) + (raw.tat_seconds / 3600.0) + ((raw.pc_watt_seconds / 3600.0) / 100.0)


def mean_raw_metrics(metrics: list[RawMetrics]) -> RawMetrics:
    if not metrics:
        raise ValueError("cannot aggregate an empty metrics list")
    count = float(len(metrics))
    return RawMetrics(
        cr=sum(metric.cr for metric in metrics) / count,
        pcr=sum(metric.pcr for metric in metrics) / count,
        wcr=sum(metric.wcr for metric in metrics) / count,
        tat_seconds=sum(metric.tat_seconds for metric in metrics) / count,
        pc_watt_seconds=sum(metric.pc_watt_seconds for metric in metrics) / count,
    )
