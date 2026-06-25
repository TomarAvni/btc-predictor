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
    check_freshness,
    latest_prediction_run_at,
    main,
)


class TestPredictionFreshness(unittest.TestCase):
    def test_missing_log_is_stale(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "missing.log"
            status = check_freshness(
                path,
                max_age=timedelta(hours=3),
                now=datetime(2026, 6, 17, 12, 0, tzinfo=timezone.utc),
            )

        self.assertIsNone(status.latest_run_at)
        self.assertFalse(status.is_fresh)

    def test_latest_prediction_run_uses_newest_header(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "predictions.log"
            path.write_text(
                "\n".join(
                    [
                        "[2026-06-17 05:26 UTC] -- Prediction Run #41",
                        "PREDICTIONS:",
                        "[2026-06-17 10:15 UTC] -- Prediction Run #42",
                        "PREDICTIONS:",
                    ]
                ),
                encoding="utf-8",
            )

            latest = latest_prediction_run_at(path)

        self.assertEqual(latest, datetime(2026, 6, 17, 10, 15, tzinfo=timezone.utc))

    def test_recent_prediction_is_fresh(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "predictions.log"
            path.write_text(
                "[2026-06-17 10:15 UTC] -- Prediction Run #42\n",
                encoding="utf-8",
            )
            status = check_freshness(
                path,
                max_age=timedelta(hours=3),
                now=datetime(2026, 6, 17, 12, 0, tzinfo=timezone.utc),
            )

        self.assertTrue(status.is_fresh)

    def test_old_prediction_is_stale(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "predictions.log"
            path.write_text(
                "[2026-06-17 08:15 UTC] -- Prediction Run #42\n",
                encoding="utf-8",
            )
            status = check_freshness(
                path,
                max_age=timedelta(hours=3),
                now=datetime(2026, 6, 17, 12, 0, tzinfo=timezone.utc),
            )

        self.assertFalse(status.is_fresh)

    def test_cli_exit_codes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fresh = Path(tmp) / "fresh.log"
            stale = Path(tmp) / "stale.log"
            fresh.write_text(
                f"[{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}] -- Prediction Run #1\n",
                encoding="utf-8",
            )
            stale.write_text(
                "[2020-01-01 00:00 UTC] -- Prediction Run #1\n",
                encoding="utf-8",
            )

            self.assertEqual(main(["--log-path", str(fresh), "--max-age-hours", "3"]), 0)
            self.assertEqual(main(["--log-path", str(stale), "--max-age-hours", "3"]), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
