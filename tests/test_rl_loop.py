from aeosbench.evaluation.metrics import RawMetrics
from aeosbench.training.rl_loop import _beats_baseline


def test_rl_acceptance_is_cr_first_then_tie_breakers():
    baseline = RawMetrics(
        cr=0.4,
        pcr=0.5,
        wcr=0.45,
        tat_seconds=500.0,
        pc_watt_seconds=1000.0,
    )
    better_cr = RawMetrics(
        cr=0.5,
        pcr=0.1,
        wcr=0.1,
        tat_seconds=900.0,
        pc_watt_seconds=2000.0,
    )
    better_tie_break = RawMetrics(
        cr=0.4,
        pcr=0.6,
        wcr=0.45,
        tat_seconds=500.0,
        pc_watt_seconds=1000.0,
    )
    worse_same_cr = RawMetrics(
        cr=0.4,
        pcr=0.4,
        wcr=0.45,
        tat_seconds=500.0,
        pc_watt_seconds=1000.0,
    )

    assert _beats_baseline(better_cr, baseline, min_cr_improvement=0.0) is True
    assert _beats_baseline(better_tie_break, baseline, min_cr_improvement=0.0) is True
    assert _beats_baseline(worse_same_cr, baseline, min_cr_improvement=0.0) is False
