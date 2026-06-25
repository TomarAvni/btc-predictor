"""Unit tests for prediction pipeline freshness detection."""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.prediction_freshness import (  # noqa: E402
    check_prediction_freshness,
    latest_prediction_timestamp,
)


class TestPredictionFreshness(unittest.TestCase):
    def test_missing_log_is_stale(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            freshness = check_prediction_freshness(
                Path(tmp) / "missing.log",
                max_age=timedelta(hours=3),
                now=datetime(2026, 6, 16, 12, 0, tzinfo=timezone.utc),
            )

        self.assertTrue(freshness.is_stale)
        self.assertIsNone(freshness.latest_at)
        self.assertIn("no prediction", freshness.reason)

    def test_latest_timestamp_uses_last_prediction_header(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "predictions.log"
            path.write_text(
                "\n".join([
                    "[2026-06-16 06:00 UTC] -- Prediction Run #41",
                    "noise",
                    "[2026-06-16 09:30 UTC] -- Prediction Run #42",
                ]),
                encoding="utf-8",
            )

            latest = latest_prediction_timestamp(path)

        self.assertEqual(
            latest,
            datetime(2026, 6, 16, 9, 30, tzinfo=timezone.utc),
        )

    def test_fresh_when_latest_prediction_is_under_threshold(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "predictions.log"
            path.write_text(
                "[2026-06-16 09:30 UTC] -- Prediction Run #42\n",
                encoding="utf-8",
            )
            freshness = check_prediction_freshness(
                path,
                max_age=timedelta(hours=3),
                now=datetime(2026, 6, 16, 12, 0, tzinfo=timezone.utc),
            )

        self.assertFalse(freshness.is_stale)
        self.assertEqual(freshness.age, timedelta(hours=2, minutes=30))

    def test_stale_when_latest_prediction_reaches_threshold(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "predictions.log"
            path.write_text(
                "[2026-06-16 09:00 UTC] -- Prediction Run #42\n",
                encoding="utf-8",
            )
            freshness = check_prediction_freshness(
                path,
                max_age=timedelta(hours=3),
                now=datetime(2026, 6, 16, 12, 0, tzinfo=timezone.utc),
            )

        self.assertTrue(freshness.is_stale)
        self.assertEqual(freshness.age, timedelta(hours=3))


if __name__ == "__main__":
    unittest.main(verbosity=2)
