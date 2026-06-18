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
    main,
    parse_prediction_timestamps,
)


class TestPredictionFreshness(unittest.TestCase):
    def test_parse_prediction_timestamps(self) -> None:
        text = """
================================================================================
[2026-06-18 01:07 UTC] -- Prediction Run #12
================================================================================
[2026-06-18 04:37 UTC] -- Prediction Run #13
"""
        timestamps = parse_prediction_timestamps(text)

        self.assertEqual(len(timestamps), 2)
        self.assertEqual(timestamps[0], datetime(2026, 6, 18, 1, 7, tzinfo=timezone.utc))
        self.assertEqual(timestamps[1], datetime(2026, 6, 18, 4, 37, tzinfo=timezone.utc))

    def test_latest_prediction_timestamp_normalizes_to_utc(self) -> None:
        latest = latest_prediction_timestamp(
            [
                datetime(2026, 6, 18, 4, 0),
                datetime(2026, 6, 18, 5, 0, tzinfo=timezone.utc),
            ]
        )

        self.assertEqual(latest, datetime(2026, 6, 18, 5, 0, tzinfo=timezone.utc))

    def test_fresh_log_is_fresh(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "predictions.log"
            log_path.write_text(
                "[2026-06-18 04:37 UTC] -- Prediction Run #13\n",
                encoding="utf-8",
            )

            result = check_prediction_freshness(
                log_path,
                max_age=timedelta(hours=3),
                now=datetime(2026, 6, 18, 6, 0, tzinfo=timezone.utc),
            )

        self.assertTrue(result.is_fresh)

    def test_stale_log_is_not_fresh(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "predictions.log"
            log_path.write_text(
                "[2026-06-18 01:00 UTC] -- Prediction Run #13\n",
                encoding="utf-8",
            )

            result = check_prediction_freshness(
                log_path,
                max_age=timedelta(hours=3),
                now=datetime(2026, 6, 18, 6, 0, tzinfo=timezone.utc),
            )

        self.assertFalse(result.is_fresh)
        self.assertIn("stale", result.reason)

    def test_missing_log_is_not_fresh(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            result = check_prediction_freshness(
                Path(tmpdir) / "missing.log",
                max_age=timedelta(hours=3),
                now=datetime(2026, 6, 18, 6, 0, tzinfo=timezone.utc),
            )

        self.assertFalse(result.is_fresh)
        self.assertIn("no prediction timestamps", result.reason)

    def test_cli_returns_failure_for_stale_log(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "predictions.log"
            log_path.write_text(
                "[2026-06-18 01:00 UTC] -- Prediction Run #13\n",
                encoding="utf-8",
            )

            exit_code = main(["--path", str(log_path), "--max-age-hours", "0.1"])

        self.assertEqual(exit_code, 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
