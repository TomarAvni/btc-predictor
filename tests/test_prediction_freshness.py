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
    latest_prediction_time,
    parse_prediction_timestamps,
)


def _write_log(path: Path, timestamps: list[str]) -> None:
    blocks = [
        "================================================================================\n"
        f"[{timestamp} UTC] -- Prediction Run #{index}\n"
        "================================================================================\n"
        "\n"
        "PREDICTIONS:\n"
        "  6h    | UP    | +0.4%    | Confidence: 66%\n"
        for index, timestamp in enumerate(timestamps, start=1)
    ]
    path.write_text("\n".join(blocks), encoding="utf-8")


class TestPredictionFreshness(unittest.TestCase):
    def test_parse_prediction_timestamps(self) -> None:
        text = (
            "[2026-06-18 04:43 UTC] -- Prediction Run #57\n"
            "not a header\n"
            "[2026-06-18 05:43 UTC] -- Prediction Run #58\n"
        )

        parsed = list(parse_prediction_timestamps(text))

        self.assertEqual(
            parsed,
            [
                datetime(2026, 6, 18, 4, 43, tzinfo=timezone.utc),
                datetime(2026, 6, 18, 5, 43, tzinfo=timezone.utc),
            ],
        )

    def test_latest_prediction_time_uses_newest_timestamp(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "predictions.log"
            _write_log(path, ["2026-06-18 05:43", "2026-06-18 04:43"])

            self.assertEqual(
                latest_prediction_time(path),
                datetime(2026, 6, 18, 5, 43, tzinfo=timezone.utc),
            )

    def test_fresh_log_passes(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "predictions.log"
            _write_log(path, ["2026-06-18 06:30"])

            result = check_freshness(
                path,
                max_age=timedelta(hours=3),
                now=datetime(2026, 6, 18, 9, 0, tzinfo=timezone.utc),
            )

            self.assertTrue(result.is_fresh)

    def test_stale_log_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "predictions.log"
            _write_log(path, ["2026-06-18 04:43"])

            result = check_freshness(
                path,
                max_age=timedelta(hours=3),
                now=datetime(2026, 6, 18, 9, 1, tzinfo=timezone.utc),
            )

            self.assertFalse(result.is_fresh)
            self.assertIsNotNone(result.age)

    def test_missing_log_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            result = check_freshness(
                Path(tmpdir) / "predictions.log",
                max_age=timedelta(hours=3),
                now=datetime(2026, 6, 18, 9, 1, tzinfo=timezone.utc),
            )

            self.assertFalse(result.is_fresh)
            self.assertIn("No prediction runs", result.message())


if __name__ == "__main__":
    unittest.main(verbosity=2)
