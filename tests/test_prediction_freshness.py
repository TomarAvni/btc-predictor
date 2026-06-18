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
    check_prediction_freshness,
    latest_prediction_timestamp,
    main,
    parse_prediction_timestamps,
)


class TestPredictionFreshness(unittest.TestCase):
    def test_parse_prediction_timestamps_ignores_non_headers(self) -> None:
        timestamps = parse_prediction_timestamps(
            [
                "================================================================================",
                "[2026-06-18 00:00 UTC] -- Prediction Run #101",
                "BTC Price: 104000",
                "[2026-06-18 00:30 UTC] -- Prediction Run #102",
            ]
        )

        self.assertEqual(
            timestamps,
            [
                datetime(2026, 6, 18, 0, 0, tzinfo=timezone.utc),
                datetime(2026, 6, 18, 0, 30, tzinfo=timezone.utc),
            ],
        )

    def test_latest_prediction_timestamp_uses_newest_header(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "predictions.log"
            path.write_text(
                "\n".join(
                    [
                        "[2026-06-18 01:00 UTC] -- Prediction Run #103",
                        "[2026-06-18 00:30 UTC] -- Prediction Run #102",
                    ]
                ),
                encoding="utf-8",
            )

            self.assertEqual(
                latest_prediction_timestamp(path),
                datetime(2026, 6, 18, 1, 0, tzinfo=timezone.utc),
            )

    def test_fresh_when_latest_prediction_within_max_age(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "predictions.log"
            path.write_text(
                "[2026-06-18 00:30 UTC] -- Prediction Run #102\n",
                encoding="utf-8",
            )

            status = check_prediction_freshness(
                path,
                max_age=timedelta(hours=3),
                now=datetime(2026, 6, 18, 3, 29, tzinfo=timezone.utc),
            )

            self.assertTrue(status.is_fresh)

    def test_stale_when_latest_prediction_exceeds_max_age(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "predictions.log"
            path.write_text(
                "[2026-06-18 00:30 UTC] -- Prediction Run #102\n",
                encoding="utf-8",
            )

            status = check_prediction_freshness(
                path,
                max_age=timedelta(hours=3),
                now=datetime(2026, 6, 18, 3, 31, tzinfo=timezone.utc),
            )

            self.assertFalse(status.is_fresh)

    def test_missing_or_malformed_log_is_stale(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            missing = Path(tmpdir) / "missing.log"
            malformed = Path(tmpdir) / "predictions.log"
            malformed.write_text("no run headers here\n", encoding="utf-8")

            self.assertFalse(check_prediction_freshness(missing).is_fresh)
            self.assertFalse(check_prediction_freshness(malformed).is_fresh)

    def test_cli_exit_code_reflects_freshness(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "predictions.log"
            path.write_text(
                "[2000-01-01 00:30 UTC] -- Prediction Run #102\n",
                encoding="utf-8",
            )

            self.assertEqual(main(["--path", str(path), "--max-age-hours", "9999999"]), 0)
            self.assertEqual(main(["--path", str(path), "--max-age-hours", "0"]), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
