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
    check_freshness,
    latest_prediction_timestamp,
    main,
    parse_prediction_timestamp,
)


class TestPredictionFreshness(unittest.TestCase):
    def test_parse_prediction_header(self) -> None:
        parsed = parse_prediction_timestamp(
            "[2026-06-18 23:22 UTC] -- Prediction Run #69"
        )
        self.assertEqual(parsed, datetime(2026, 6, 18, 23, 22, tzinfo=timezone.utc))

    def test_parse_ignores_non_header_lines(self) -> None:
        self.assertIsNone(parse_prediction_timestamp("PREDICTIONS:"))

    def test_latest_prediction_timestamp_uses_newest_entry(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "predictions.log"
            path.write_text(
                "\n".join(
                    [
                        "[2026-06-18 10:00 UTC] -- Prediction Run #67",
                        "PREDICTIONS:",
                        "[2026-06-18 12:30 UTC] -- Prediction Run #68",
                        "[2026-06-18 11:00 UTC] -- Prediction Run #66",
                    ]
                ),
                encoding="utf-8",
            )

            self.assertEqual(
                latest_prediction_timestamp(path),
                datetime(2026, 6, 18, 12, 30, tzinfo=timezone.utc),
            )

    def test_fresh_when_latest_prediction_is_within_limit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "predictions.log"
            path.write_text(
                "[2026-06-18 12:30 UTC] -- Prediction Run #68\n",
                encoding="utf-8",
            )

            status = check_freshness(
                path,
                max_age=timedelta(hours=3),
                now=datetime(2026, 6, 18, 14, 0, tzinfo=timezone.utc),
            )

            self.assertTrue(status.is_fresh)
            self.assertEqual(status.age, timedelta(hours=1, minutes=30))

    def test_stale_when_latest_prediction_exceeds_limit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "predictions.log"
            path.write_text(
                "[2026-06-18 10:00 UTC] -- Prediction Run #67\n",
                encoding="utf-8",
            )

            status = check_freshness(
                path,
                max_age=timedelta(hours=3),
                now=datetime(2026, 6, 18, 13, 1, tzinfo=timezone.utc),
            )

            self.assertFalse(status.is_fresh)

    def test_missing_log_is_stale(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            status = check_freshness(Path(tmp) / "missing.log")

            self.assertFalse(status.is_fresh)
            self.assertIsNone(status.latest_prediction_at)

    def test_cli_exit_codes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "predictions.log"
            path.write_text(
                "[2026-06-18 10:00 UTC] -- Prediction Run #67\n",
                encoding="utf-8",
            )

            self.assertEqual(main(["--path", str(path), "--max-age-hours", "99999"]), 0)
            self.assertEqual(main(["--path", str(path), "--max-age-hours", "0.001"]), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
