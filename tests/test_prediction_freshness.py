"""Tests for prediction log freshness checks."""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.prediction_freshness import (  # noqa: E402
    is_prediction_stale,
    latest_prediction_timestamp,
    latest_prediction_timestamp_from_file,
    main,
    parse_prediction_timestamp,
)


class TestPredictionFreshness(unittest.TestCase):
    def test_parse_prediction_header(self) -> None:
        parsed = parse_prediction_timestamp("[2026-06-17 19:37 UTC] -- Prediction Run #85")

        self.assertEqual(parsed, datetime(2026, 6, 17, 19, 37, tzinfo=timezone.utc))

    def test_ignores_non_prediction_lines(self) -> None:
        self.assertIsNone(parse_prediction_timestamp("SIGNAL SUMMARY:"))

    def test_latest_prediction_timestamp_uses_newest_header(self) -> None:
        latest = latest_prediction_timestamp(
            [
                "[2026-06-17 17:30 UTC] -- Prediction Run #84",
                "noise",
                "[2026-06-17 19:37 UTC] -- Prediction Run #85",
                "[2026-06-17 14:08 UTC] -- Prediction Run #83",
            ]
        )

        self.assertEqual(latest, datetime(2026, 6, 17, 19, 37, tzinfo=timezone.utc))

    def test_fresh_log_is_not_stale(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "predictions.log"
            path.write_text("[2026-06-17 19:37 UTC] -- Prediction Run #85\n", encoding="utf-8")

            self.assertFalse(
                is_prediction_stale(
                    path,
                    max_age=timedelta(hours=3),
                    now=datetime(2026, 6, 17, 21, 0, tzinfo=timezone.utc),
                )
            )

    def test_old_log_is_stale(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "predictions.log"
            path.write_text("[2026-06-17 17:30 UTC] -- Prediction Run #84\n", encoding="utf-8")

            self.assertTrue(
                is_prediction_stale(
                    path,
                    max_age=timedelta(hours=3),
                    now=datetime(2026, 6, 17, 21, 0, tzinfo=timezone.utc),
                )
            )

    def test_missing_or_unparseable_log_is_stale(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            missing = Path(tmp) / "missing.log"
            bad = Path(tmp) / "bad.log"
            bad.write_text("no prediction headers\n", encoding="utf-8")

            self.assertIsNone(latest_prediction_timestamp_from_file(missing))
            self.assertTrue(is_prediction_stale(missing, max_age=timedelta(hours=3)))
            self.assertTrue(is_prediction_stale(bad, max_age=timedelta(hours=3)))

    def test_cli_exit_codes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "predictions.log"
            path.write_text("[2026-06-17 19:37 UTC] -- Prediction Run #85\n", encoding="utf-8")

            self.assertEqual(main(["--path", str(path), "--max-age-hours", "999999"]), 0)
            self.assertEqual(main(["--path", str(path), "--max-age-hours", "0"]), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
