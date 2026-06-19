"""Tests for prediction freshness checks used by the Actions watchdog."""

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
    latest_prediction_run_at,
    main,
    parse_prediction_run_timestamp,
)


class TestPredictionFreshness(unittest.TestCase):
    def test_parse_prediction_run_header(self) -> None:
        line = "[2026-06-19 02:21 UTC] -- Prediction Run #71"
        self.assertEqual(
            parse_prediction_run_timestamp(line),
            datetime(2026, 6, 19, 2, 21, tzinfo=timezone.utc),
        )

    def test_ignores_non_header_lines(self) -> None:
        self.assertIsNone(parse_prediction_run_timestamp("PREDICTIONS:"))

    def test_latest_prediction_run_at_returns_newest_header(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "predictions.log"
            log_path.write_text(
                "\n".join(
                    [
                        "[2026-06-19 00:21 UTC] -- Prediction Run #69",
                        "bad line",
                        "[2026-06-19 02:21 UTC] -- Prediction Run #71",
                        "[2026-06-19 01:21 UTC] -- Prediction Run #70",
                    ]
                ),
                encoding="utf-8",
            )

            self.assertEqual(
                latest_prediction_run_at(log_path),
                datetime(2026, 6, 19, 2, 21, tzinfo=timezone.utc),
            )

    def test_missing_log_is_stale(self) -> None:
        result = check_prediction_freshness(
            Path("/tmp/definitely-missing-predictions.log"),
            now=datetime(2026, 6, 19, 6, 0, tzinfo=timezone.utc),
        )

        self.assertFalse(result.is_fresh)
        self.assertIsNone(result.latest_run_at)

    def test_recent_prediction_is_fresh(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "predictions.log"
            log_path.write_text("[2026-06-19 04:00 UTC] -- Prediction Run #72\n", encoding="utf-8")

            result = check_prediction_freshness(
                log_path,
                max_age=timedelta(hours=3),
                now=datetime(2026, 6, 19, 6, 0, tzinfo=timezone.utc),
            )

            self.assertTrue(result.is_fresh)

    def test_old_prediction_is_stale(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "predictions.log"
            log_path.write_text("[2026-06-19 02:21 UTC] -- Prediction Run #71\n", encoding="utf-8")

            result = check_prediction_freshness(
                log_path,
                max_age=timedelta(hours=3),
                now=datetime(2026, 6, 19, 6, 0, tzinfo=timezone.utc),
            )

            self.assertFalse(result.is_fresh)

    def test_cli_exit_code_reflects_freshness(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "predictions.log"
            log_path.write_text("[2099-06-19 02:21 UTC] -- Prediction Run #1\n", encoding="utf-8")

            self.assertEqual(main(["--path", str(log_path), "--max-age-hours", "3"]), 0)

            log_path.write_text("[2000-06-19 02:21 UTC] -- Prediction Run #1\n", encoding="utf-8")
            self.assertEqual(main(["--log-path", str(log_path), "--max-age-hours", "3"]), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
