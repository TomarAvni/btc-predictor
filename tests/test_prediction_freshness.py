"""Unit tests for prediction pipeline freshness checks."""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.prediction_freshness import (  # noqa: E402
    is_prediction_stale,
    latest_prediction_timestamp,
    parse_prediction_timestamp,
    prediction_age_hours,
)


class TestPredictionFreshness(unittest.TestCase):
    def test_parse_prediction_header(self) -> None:
        timestamp = parse_prediction_timestamp("[2026-06-17 00:26 UTC] -- Prediction Run #41")
        self.assertEqual(timestamp, datetime(2026, 6, 17, 0, 26, tzinfo=timezone.utc))

    def test_ignores_non_header_lines(self) -> None:
        self.assertIsNone(parse_prediction_timestamp("PREDICTIONS:"))

    def test_latest_prediction_timestamp_uses_newest_header(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "predictions.log"
            log_path.write_text(
                "\n".join(
                    [
                        "[2026-06-17 00:26 UTC] -- Prediction Run #41",
                        "PREDICTIONS:",
                        "[2026-06-17 02:07 UTC] -- Prediction Run #42",
                    ]
                ),
                encoding="utf-8",
            )

            self.assertEqual(
                latest_prediction_timestamp(log_path),
                datetime(2026, 6, 17, 2, 7, tzinfo=timezone.utc),
            )

    def test_missing_log_is_stale(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            self.assertTrue(is_prediction_stale(Path(tmp) / "missing.log"))

    def test_age_and_staleness_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "predictions.log"
            log_path.write_text("[2026-06-17 00:26 UTC] -- Prediction Run #41\n", encoding="utf-8")
            now = datetime(2026, 6, 17, 3, 26, tzinfo=timezone.utc)

            self.assertAlmostEqual(prediction_age_hours(log_path, now=now), 3.0)
            self.assertFalse(is_prediction_stale(log_path, max_age_hours=3.0, now=now))
            self.assertTrue(is_prediction_stale(log_path, max_age_hours=2.99, now=now))


if __name__ == "__main__":
    unittest.main(verbosity=2)
