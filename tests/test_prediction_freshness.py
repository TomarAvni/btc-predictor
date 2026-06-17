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
    is_prediction_fresh,
    latest_prediction_timestamp,
    prediction_age,
)


class TestPredictionFreshness(unittest.TestCase):
    def write_log(self, content: str) -> Path:
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        path = Path(temp_dir.name) / "predictions.log"
        path.write_text(content, encoding="utf-8")
        return path

    def test_latest_prediction_timestamp_uses_newest_entry(self) -> None:
        path = self.write_log(
            """
================================================================================
[2026-06-17 05:28 UTC] -- Prediction Run #42
================================================================================
[2026-06-17 10:18 UTC] -- Prediction Run #43
================================================================================
"""
        )

        self.assertEqual(
            latest_prediction_timestamp(path),
            datetime(2026, 6, 17, 10, 18, tzinfo=timezone.utc),
        )

    def test_fresh_when_latest_prediction_within_limit(self) -> None:
        path = self.write_log("[2026-06-17 10:18 UTC] -- Prediction Run #43\n")
        now = datetime(2026, 6, 17, 12, 1, tzinfo=timezone.utc)

        self.assertTrue(
            is_prediction_fresh(path, max_age=timedelta(hours=3), now=now)
        )

    def test_stale_when_latest_prediction_exceeds_limit(self) -> None:
        path = self.write_log("[2026-06-17 05:28 UTC] -- Prediction Run #43\n")
        now = datetime(2026, 6, 17, 12, 1, tzinfo=timezone.utc)

        self.assertFalse(
            is_prediction_fresh(path, max_age=timedelta(hours=3), now=now)
        )

    def test_missing_log_has_no_age_and_is_not_fresh(self) -> None:
        path = Path(tempfile.gettempdir()) / "missing-predictions.log"

        self.assertIsNone(prediction_age(path))
        self.assertFalse(is_prediction_fresh(path, max_age=timedelta(hours=3)))

    def test_malformed_entries_are_ignored(self) -> None:
        path = self.write_log(
            """
[not a timestamp] -- Prediction Run #1
[2026-06-17 10:18 UTC] -- Prediction Run #2
"""
        )

        self.assertEqual(
            latest_prediction_timestamp(path),
            datetime(2026, 6, 17, 10, 18, tzinfo=timezone.utc),
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
