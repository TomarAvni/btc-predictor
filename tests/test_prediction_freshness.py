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
    get_prediction_freshness,
    latest_prediction_run,
    main,
)


def _write_log(path: Path, timestamps: list[str]) -> None:
    blocks = [
        f"================================================================================\n"
        f"[{timestamp}] -- Prediction Run #{idx}\n"
        f"\n"
        f"PREDICTIONS:\n"
        f"  6h    | UP    | +0.4%   | Confidence: 66%\n"
        for idx, timestamp in enumerate(timestamps, start=1)
    ]
    path.write_text("\n".join(blocks), encoding="utf-8")


class TestPredictionFreshness(unittest.TestCase):
    def test_latest_prediction_run_returns_newest_header(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "predictions.log"
            _write_log(path, ["2026-06-17 10:00 UTC", "2026-06-17 12:30 UTC"])

            latest = latest_prediction_run(path)

        self.assertIsNotNone(latest)
        assert latest is not None
        self.assertEqual(latest[0], datetime(2026, 6, 17, 12, 30, tzinfo=timezone.utc))
        self.assertEqual(latest[1], 2)

    def test_fresh_prediction_is_not_stale(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "predictions.log"
            _write_log(path, ["2026-06-17 12:00 UTC"])

            freshness = get_prediction_freshness(
                path,
                now=datetime(2026, 6, 17, 14, 59, tzinfo=timezone.utc),
            )

        self.assertFalse(freshness.is_stale(timedelta(hours=3)))
        self.assertEqual(freshness.run_number, 1)

    def test_prediction_older_than_threshold_is_stale(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "predictions.log"
            _write_log(path, ["2026-06-17 12:00 UTC"])

            freshness = get_prediction_freshness(
                path,
                now=datetime(2026, 6, 17, 15, 1, tzinfo=timezone.utc),
            )

        self.assertTrue(freshness.is_stale(timedelta(hours=3)))

    def test_missing_or_malformed_log_is_stale(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            missing = Path(tmp) / "missing.log"
            malformed = Path(tmp) / "predictions.log"
            malformed.write_text("no prediction headers here\n", encoding="utf-8")

            missing_freshness = get_prediction_freshness(missing)
            malformed_freshness = get_prediction_freshness(malformed)

        self.assertTrue(missing_freshness.is_stale(timedelta(hours=3)))
        self.assertTrue(malformed_freshness.is_stale(timedelta(hours=3)))
        self.assertIsNone(missing_freshness.latest_timestamp)
        self.assertIsNone(malformed_freshness.latest_timestamp)

    def test_cli_exit_codes_reflect_freshness(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "predictions.log"
            _write_log(path, ["2026-06-17 12:00 UTC"])

            self.assertEqual(main(["--path", str(path), "--max-age-hours", "99999"]), 0)
            self.assertEqual(main(["--path", str(path), "--max-age-hours", "0"]), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
