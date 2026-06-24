"""Tests for training readiness snapshot logic."""

from dashboard.components.training_readiness import compute_readiness_snapshot


def test_closest_horizon_picks_fewest_remaining():
    status = {
        "labeled_by_horizon": {"6h": 90, "12h": 50, "24h": 10},
        "calibration_min_rows": 100,
        "retrain_min_rows": 400,
    }
    snap = compute_readiness_snapshot(status)
    assert snap["calibration"]["horizon"] == "6h"
    assert snap["calibration"]["remaining"] == 10
    assert snap["calibration"]["ready"] is False
    assert snap["calibration"]["progress_pct"] == 90.0


def test_all_ready_when_every_horizon_meets_threshold():
    status = {
        "labeled_by_horizon": {tf: 500 for tf in ["6h", "12h", "18h"]},
        "calibration_min_rows": 100,
        "retrain_min_rows": 400,
    }
    # Patch TIMEFRAMES is global — use counts that exceed min for all in module
    from src.horizons import TIMEFRAMES

    status["labeled_by_horizon"] = {tf: 500 for tf in TIMEFRAMES}
    snap = compute_readiness_snapshot(status)
    assert snap["calibration"]["ready"] is True
    assert snap["calibration"]["horizon"] == "all"
    assert snap["retrain"]["ready"] is True


def test_retrain_uses_higher_threshold():
    status = {
        "labeled_by_horizon": {tf: 150 for tf in ["6h"]},
        "calibration_min_rows": 100,
        "retrain_min_rows": 400,
    }
    from src.horizons import TIMEFRAMES

    labeled = {tf: 150 for tf in TIMEFRAMES}
    status["labeled_by_horizon"] = labeled
    snap = compute_readiness_snapshot(status)
    assert snap["calibration"]["ready"] is True
    assert snap["retrain"]["ready"] is False
    assert snap["retrain"]["remaining"] == 250
