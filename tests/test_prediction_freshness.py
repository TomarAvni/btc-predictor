"""Tests for prediction freshness checks."""

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
)


def _log_entry(timestamp: str, run_number: int = 1) -> str:
    return "\n".join(
        [
            "=" * 80,
            f"[{timestamp}] -- Prediction Run #{run_number}",
            "=" * 80,
            "",
            "PREDICTIONS:",
            "  6h    | UP    | +1.0%    | Confidence: 55%",
        ]
    )


class TestPredictionFreshness(unittest.TestCase):
    def test_latest_prediction_timestamp_uses_newest_header(self) -> None:
        text = "\n".join(
            [
                _log_entry("2026-06-18 08:00 UTC", 1),
                _log_entry("2026-06-18 11:00 UTC", 2),
                _log_entry("2026-06-18 09:00 UTC", 3),
            ]
        )

        latest = latest_prediction_timestamp(text)

        self.assertEqual(latest, datetime(2026, 6, 18, 11, 0, tzinfo=timezone.utc))

    def test_fresh_when_latest_prediction_within_threshold(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "predictions.log"
            path.write_text(_log_entry("2026-06-18 10:30 UTC"), encoding="utf-8")

            result = check_prediction_freshness(
                path,
                max_age=timedelta(hours=1),
                now=datetime(2026, 6, 18, 11, 0, tzinfo=timezone.utc),
            )

        self.assertTrue(result.is_fresh)
        self.assertEqual(result.age, timedelta(minutes=30))

    def test_stale_when_latest_prediction_exceeds_threshold(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "predictions.log"
            path.write_text(_log_entry("2026-06-18 08:59 UTC"), encoding="utf-8")

            result = check_prediction_freshness(
                path,
                max_age=timedelta(hours=1),
                now=datetime(2026, 6, 18, 10, 0, tzinfo=timezone.utc),
            )

        self.assertFalse(result.is_fresh)
        self.assertEqual(result.reason, "latest prediction is stale")

    def test_missing_log_is_stale(self) -> None:
        result = check_prediction_freshness(
            "/tmp/does-not-exist-predictions.log",
            now=datetime(2026, 6, 18, 12, 0, tzinfo=timezone.utc),
        )

        self.assertFalse(result.is_fresh)
        self.assertIn("does not exist", result.reason)

    def test_log_without_headers_is_stale(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "predictions.log"
            path.write_text("no prediction entries yet", encoding="utf-8")

            result = check_prediction_freshness(
                path,
                now=datetime(2026, 6, 18, 12, 0, tzinfo=timezone.utc),
            )

        self.assertFalse(result.is_fresh)
        self.assertIn("no prediction headers", result.reason)

    def test_cli_returns_success_for_fresh_log(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "predictions.log"
            now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
            path.write_text(_log_entry(now), encoding="utf-8")

            exit_code = main(["--path", str(path), "--max-age-hours", "1"])

        self.assertEqual(exit_code, 0)

    def test_cli_returns_failure_for_stale_log(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "predictions.log"
            path.write_text(_log_entry("2000-01-01 00:00 UTC"), encoding="utf-8")

            exit_code = main(["--path", str(path), "--max-age-hours", "1"])

        self.assertEqual(exit_code, 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
