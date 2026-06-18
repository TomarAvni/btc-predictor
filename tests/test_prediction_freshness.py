"""Unit tests for prediction pipeline freshness checks."""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.prediction_freshness import check_prediction_freshness, main  # noqa: E402


class TestPredictionFreshness(unittest.TestCase):
    def _write_log(self, text: str) -> Path:
        tmp = tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False)
        self.addCleanup(lambda: Path(tmp.name).unlink(missing_ok=True))
        with tmp:
            tmp.write(text)
        return Path(tmp.name)

    def test_fresh_when_latest_prediction_within_max_age(self) -> None:
        path = self._write_log(
            "[2026-06-18 00:00 UTC] -- Prediction Run #1\n"
            "[2026-06-18 02:00 UTC] -- Prediction Run #2\n"
        )

        result = check_prediction_freshness(
            path,
            max_age=timedelta(hours=3),
            now=datetime(2026, 6, 18, 4, 30, tzinfo=timezone.utc),
        )

        self.assertTrue(result.is_fresh)
        self.assertEqual(result.latest_timestamp, datetime(2026, 6, 18, 2, 0, tzinfo=timezone.utc))

    def test_stale_when_latest_prediction_exceeds_max_age(self) -> None:
        path = self._write_log("[2026-06-17 23:46 UTC] -- Prediction Run #10\n")

        result = check_prediction_freshness(
            path,
            max_age=timedelta(hours=3),
            now=datetime(2026, 6, 18, 3, 3, tzinfo=timezone.utc),
        )

        self.assertFalse(result.is_fresh)
        self.assertGreater(result.age, timedelta(hours=3))

    def test_missing_log_is_stale(self) -> None:
        result = check_prediction_freshness(
            Path("/tmp/definitely-missing-predictions.log"),
            max_age=timedelta(hours=3),
            now=datetime(2026, 6, 18, 3, 3, tzinfo=timezone.utc),
        )

        self.assertFalse(result.is_fresh)
        self.assertIsNone(result.latest_timestamp)

    def test_cli_exit_codes_reflect_freshness(self) -> None:
        fresh = self._write_log("[2026-06-18 02:00 UTC] -- Prediction Run #2\n")
        stale = self._write_log("[2026-06-17 23:46 UTC] -- Prediction Run #10\n")

        self.assertEqual(main(["--path", str(fresh), "--max-age-hours", "99999"]), 0)
        self.assertEqual(main(["--path", str(stale), "--max-age-hours", "0.1"]), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
