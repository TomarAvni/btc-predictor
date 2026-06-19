"""Unit tests for the closed-loop calibrator refit + promotion guard.

The model-retraining path is intentionally NOT exercised here: it requires the
full historical price parquet + heavy XGBoost training that belongs in the
cloud. These tests cover the parts that decide whether we *learn from mistakes*
safely: the promotion predicate and the live calibrator refit's promote / keep /
skip / dry-run behavior.
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.calibration import ProbabilityCalibrator  # noqa: E402
from src.training import closed_loop  # noqa: E402


class TestPromotionPredicate(unittest.TestCase):
    def test_lower_is_better_brier(self) -> None:
        # Candidate must be <= current + tol.
        self.assertTrue(closed_loop.is_not_worse(0.20, 0.25, 0.0, higher_is_better=False))
        self.assertFalse(closed_loop.is_not_worse(0.26, 0.25, 0.0, higher_is_better=False))
        self.assertTrue(closed_loop.is_not_worse(0.26, 0.25, 0.02, higher_is_better=False))

    def test_higher_is_better_accuracy(self) -> None:
        self.assertTrue(closed_loop.is_not_worse(0.55, 0.50, 0.0, higher_is_better=True))
        self.assertFalse(closed_loop.is_not_worse(0.45, 0.50, 0.0, higher_is_better=True))
        self.assertTrue(closed_loop.is_not_worse(0.49, 0.50, 0.01, higher_is_better=True))

    def test_nan_never_passes(self) -> None:
        self.assertFalse(closed_loop.is_not_worse(float("nan"), 0.5, 0.0, higher_is_better=True))
        self.assertFalse(closed_loop.is_not_worse(0.5, float("nan"), 0.0, higher_is_better=False))


def _miscalibrated_rows(n_per_bucket: int = 60) -> list[dict]:
    """Build live rows where raw probs are systematically miscalibrated.

    Two prob buckets (0.3 and 0.7) each have a TRUE up-rate of 0.5. The identity
    map (current empty calibrator) is therefore over/under-confident, while an
    isotonic fit collapses both onto ~0.5 and lowers Brier -> a clean promote.
    Rows are interleaved so any chronological split stays balanced.
    """
    rows: list[dict] = []
    stamp = 0
    for i in range(n_per_bucket):
        for prob in (0.3, 0.7):
            for label in (1, 0):  # balanced -> true up-rate 0.5 in each bucket
                stamp += 1
                rows.append({
                    "run_number": stamp,
                    "timeframe": "24h",
                    "prediction_timestamp": f"2026-01-01T{stamp:06d}",
                    "direction_prob": prob,
                    "label_up": label,
                })
    return rows


class TestLiveCalibratorRefit(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.model_dir = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_skips_when_insufficient_data(self) -> None:
        rows = [
            {"run_number": i, "timeframe": "24h",
             "prediction_timestamp": f"2026-01-01T{i:04d}",
             "direction_prob": 0.6, "label_up": i % 2}
            for i in range(10)
        ]
        report = closed_loop.refit_live_calibrator(
            rows, serving_model_dir=self.model_dir, min_rows=100,
        )
        self.assertIn("24h", report["skipped_horizons"])
        self.assertFalse((self.model_dir / "probability_calibrator.pkl").exists())

    def test_promotes_and_writes_when_calibration_improves(self) -> None:
        rows = _miscalibrated_rows()
        report = closed_loop.refit_live_calibrator(
            rows, serving_model_dir=self.model_dir,
            min_rows=100, holdout_frac=0.3, tolerance=0.0,
        )
        self.assertIn("24h", report["promoted_horizons"])
        h = report["horizons"]["24h"]
        self.assertLessEqual(h["candidate_brier"], h["current_brier"])
        # Promoted -> calibrator persisted to the serving dir with this horizon.
        cal_path = self.model_dir / "probability_calibrator.pkl"
        self.assertTrue(cal_path.exists())
        reloaded = ProbabilityCalibrator(model_dir=self.model_dir).load()
        self.assertTrue(reloaded.has_horizon("24h"))

    def test_dry_run_decides_but_writes_nothing(self) -> None:
        rows = _miscalibrated_rows()
        report = closed_loop.refit_live_calibrator(
            rows, serving_model_dir=self.model_dir,
            min_rows=100, holdout_frac=0.3, dry_run=True,
        )
        self.assertIn("24h", report["promoted_horizons"])
        self.assertFalse(report["written"])
        self.assertFalse((self.model_dir / "probability_calibrator.pkl").exists())

    def test_no_data_is_clean_noop(self) -> None:
        report = closed_loop.refit_live_calibrator([], serving_model_dir=self.model_dir)
        self.assertEqual(report["promoted_horizons"], [])
        self.assertFalse((self.model_dir / "probability_calibrator.pkl").exists())


class TestModelRetrainGracefulSkip(unittest.TestCase):
    """The heavy retrain path must degrade to a clean skip, never crash."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.serving = Path(self._tmp.name) / "serving"
        self.staging = Path(self._tmp.name) / "staging"

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_skips_when_no_live_data(self) -> None:
        out = closed_loop.retrain_models_with_live_blend(
            [], serving_model_dir=self.serving, staging_model_dir=self.staging,
        )
        self.assertEqual(out["status"], "skipped")
        self.assertEqual(out["reason"], "no_live_data")

    def test_skips_when_insufficient_rows_per_horizon(self) -> None:
        rows = [
            {"run_number": i, "timeframe": "24h", "horizon_hours": 24,
             "prediction_timestamp": f"2026-01-01T{i:04d}", "label_up": i % 2,
             "actual_return_pct": 1.0, "direction_prob": 0.6,
             "features": {"rsi_14": 50.0}}
            for i in range(20)
        ]
        out = closed_loop.retrain_models_with_live_blend(
            rows, serving_model_dir=self.serving, staging_model_dir=self.staging,
            min_rows_per_horizon=400,
        )
        self.assertEqual(out["status"], "skipped")
        self.assertEqual(out["reason"], "insufficient_live_data_per_horizon")


if __name__ == "__main__":
    unittest.main(verbosity=2)
