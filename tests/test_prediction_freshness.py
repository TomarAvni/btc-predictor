"""Unit tests for prediction log freshness checks."""

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
    def test_latest_prediction_timestamp_uses_most_recent_header(self) -> None:
        text = """
[2026-06-18 12:43 UTC] -- Prediction Run #41
body
[2026-06-18 16:40 UTC] -- Prediction Run #42
body
"""

        self.assertEqual(
            latest_prediction_timestamp(text),
            datetime(2026, 6, 18, 16, 40, tzinfo=timezone.utc),
        )

    def test_fresh_log_returns_success(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "predictions.log"
            path.write_text("[2026-06-18 16:40 UTC] -- Prediction Run #42\n", encoding="utf-8")

            result = check_prediction_freshness(
                path,
                max_age=timedelta(hours=3),
                now=datetime(2026, 6, 18, 18, 0, tzinfo=timezone.utc),
            )

        self.assertTrue(result.is_fresh)
        self.assertEqual(result.age, timedelta(hours=1, minutes=20))

    def test_stale_log_returns_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "predictions.log"
            path.write_text("[2026-06-18 12:43 UTC] -- Prediction Run #41\n", encoding="utf-8")

            result = check_prediction_freshness(
                path,
                max_age=timedelta(hours=3),
                now=datetime(2026, 6, 18, 18, 0, tzinfo=timezone.utc),
            )

        self.assertFalse(result.is_fresh)
        self.assertIn("stale", result.reason)

    def test_missing_log_returns_failure(self) -> None:
        result = check_prediction_freshness(
            "/tmp/does-not-exist-predictions.log",
            now=datetime(2026, 6, 18, 18, 0, tzinfo=timezone.utc),
        )

        self.assertFalse(result.is_fresh)
        self.assertIsNone(result.latest_timestamp)


if __name__ == "__main__":
    unittest.main(verbosity=2)
