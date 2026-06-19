"""Unit tests for prediction pipeline freshness checks."""

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
    iter_prediction_timestamps,
    latest_prediction_timestamp,
)


class TestPredictionFreshness(unittest.TestCase):
    def _write_log(self, content: str) -> Path:
        handle = tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False)
        self.addCleanup(lambda: Path(handle.name).unlink(missing_ok=True))
        with handle:
            handle.write(content)
        return Path(handle.name)

    def test_iter_prediction_timestamps_ignores_non_headers(self) -> None:
        lines = [
            "================================================================================\n",
            "[2026-06-19 00:21 UTC] -- Prediction Run #70\n",
            "not a header\n",
            "[bad timestamp] -- Prediction Run #71\n",
        ]

        timestamps = list(iter_prediction_timestamps(lines))

        self.assertEqual(timestamps, [datetime(2026, 6, 19, 0, 21, tzinfo=timezone.utc)])

    def test_latest_prediction_timestamp_returns_newest_header(self) -> None:
        log_path = self._write_log(
            "[2026-06-19 00:21 UTC] -- Prediction Run #70\n"
            "PREDICTIONS:\n"
            "[2026-06-19 02:21 UTC] -- Prediction Run #71\n"
        )

        latest = latest_prediction_timestamp(log_path)

        self.assertEqual(latest, datetime(2026, 6, 19, 2, 21, tzinfo=timezone.utc))

    def test_fresh_when_latest_prediction_within_max_age(self) -> None:
        log_path = self._write_log("[2026-06-19 02:21 UTC] -- Prediction Run #71\n")
        now = datetime(2026, 6, 19, 3, 0, tzinfo=timezone.utc)

        result = check_prediction_freshness(
            log_path=log_path,
            max_age=timedelta(hours=3),
            now=now,
        )

        self.assertTrue(result.is_fresh)
        self.assertEqual(result.age, timedelta(minutes=39))

    def test_stale_when_latest_prediction_exceeds_max_age(self) -> None:
        log_path = self._write_log("[2026-06-18 23:21 UTC] -- Prediction Run #70\n")
        now = datetime(2026, 6, 19, 3, 0, tzinfo=timezone.utc)

        result = check_prediction_freshness(
            log_path=log_path,
            max_age=timedelta(hours=3),
            now=now,
        )

        self.assertFalse(result.is_fresh)
        self.assertEqual(result.age, timedelta(hours=3, minutes=39))

    def test_missing_log_is_stale(self) -> None:
        result = check_prediction_freshness(
            log_path=Path(tempfile.gettempdir()) / "missing-predictions.log",
            max_age=timedelta(hours=3),
            now=datetime(2026, 6, 19, 3, 0, tzinfo=timezone.utc),
        )

        self.assertFalse(result.is_fresh)
        self.assertIsNone(result.latest_timestamp)


if __name__ == "__main__":
    unittest.main(verbosity=2)
