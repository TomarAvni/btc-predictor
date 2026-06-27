"""Tests for prediction pipeline freshness checks."""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.prediction_freshness import (  # noqa: E402
    build_parser,
    check_prediction_freshness,
    latest_prediction_run_at,
    parse_prediction_timestamp,
    _resolve_watchdog_max_age,
)


class TestPredictionFreshness(unittest.TestCase):
    def test_parse_prediction_timestamp_is_utc_aware(self) -> None:
        parsed = parse_prediction_timestamp("2026-06-17 05:26 UTC")

        self.assertEqual(parsed, datetime(2026, 6, 17, 5, 26, tzinfo=timezone.utc))

    def test_latest_prediction_run_at_uses_newest_header(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "predictions.log"
            log_path.write_text(
                """
================================================================================
[2026-06-17 00:26 UTC] -- Prediction Run #41
================================================================================
[2026-06-17 05:26 UTC] -- Prediction Run #42
================================================================================
""",
                encoding="utf-8",
            )

            self.assertEqual(
                latest_prediction_run_at(log_path),
                datetime(2026, 6, 17, 5, 26, tzinfo=timezone.utc),
            )

    def test_check_prediction_freshness_accepts_recent_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "predictions.log"
            log_path.write_text(
                "[2026-06-17 05:26 UTC] -- Prediction Run #42",
                encoding="utf-8",
            )

            result = check_prediction_freshness(
                log_path,
                max_age=timedelta(hours=3),
                now=datetime(2026, 6, 17, 7, 0, tzinfo=timezone.utc),
            )

            self.assertTrue(result.is_fresh)
            self.assertEqual(result.age, timedelta(hours=1, minutes=34))

    def test_check_prediction_freshness_rejects_stale_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "predictions.log"
            log_path.write_text(
                "[2026-06-17 00:26 UTC] -- Prediction Run #41",
                encoding="utf-8",
            )

            result = check_prediction_freshness(
                log_path,
                max_age=timedelta(hours=3),
                now=datetime(2026, 6, 17, 5, 0, tzinfo=timezone.utc),
            )

            self.assertFalse(result.is_fresh)
            self.assertEqual(result.age, timedelta(hours=4, minutes=34))

    def test_check_prediction_freshness_rejects_missing_or_unparseable_log(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            missing = Path(tmp) / "missing.log"
            corrupt = Path(tmp) / "predictions.log"
            corrupt.write_text("no prediction headers here", encoding="utf-8")

            missing_result = check_prediction_freshness(missing)
            corrupt_result = check_prediction_freshness(corrupt)

            self.assertFalse(missing_result.is_fresh)
            self.assertIsNone(missing_result.latest_run_at)
            self.assertFalse(corrupt_result.is_fresh)
            self.assertIsNone(corrupt_result.latest_run_at)

    def test_resolve_watchdog_max_age_prefers_minutes(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["--max-age-minutes", "60"])

        self.assertEqual(_resolve_watchdog_max_age(args), timedelta(minutes=60))


if __name__ == "__main__":
    unittest.main(verbosity=2)
