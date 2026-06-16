"""Unit tests for scheduled prediction freshness checks."""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.prediction_freshness import (  # noqa: E402
    latest_prediction_time,
    should_run_prediction,
)


class TestPredictionFreshness(unittest.TestCase):
    def test_latest_prediction_time_uses_newest_header(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "predictions.log"
            path.write_text(
                "\n".join(
                    [
                        "[2026-06-15 10:45 UTC] -- Prediction Run #10",
                        "PREDICTIONS:",
                        "[2026-06-15 11:15 UTC] -- Prediction Run #11",
                    ]
                ),
                encoding="utf-8",
            )

            self.assertEqual(
                latest_prediction_time(path),
                datetime(2026, 6, 15, 11, 15, tzinfo=timezone.utc),
            )

    def test_missing_log_is_stale_for_schedule(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "missing.log"

            self.assertTrue(
                should_run_prediction(
                    event_name="schedule",
                    predictions_path=path,
                    threshold=timedelta(minutes=25),
                    now=datetime(2026, 6, 15, 11, 30, tzinfo=timezone.utc),
                )
            )

    def test_schedule_skips_fresh_prediction(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "predictions.log"
            path.write_text(
                "[2026-06-15 11:15 UTC] -- Prediction Run #11\n",
                encoding="utf-8",
            )

            self.assertFalse(
                should_run_prediction(
                    event_name="schedule",
                    predictions_path=path,
                    threshold=timedelta(minutes=25),
                    now=datetime(2026, 6, 15, 11, 30, tzinfo=timezone.utc),
                )
            )

    def test_schedule_runs_stale_prediction(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "predictions.log"
            path.write_text(
                "[2026-06-15 11:00 UTC] -- Prediction Run #11\n",
                encoding="utf-8",
            )

            self.assertTrue(
                should_run_prediction(
                    event_name="schedule",
                    predictions_path=path,
                    threshold=timedelta(minutes=25),
                    now=datetime(2026, 6, 15, 11, 30, tzinfo=timezone.utc),
                )
            )

    def test_workflow_dispatch_always_runs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "predictions.log"
            path.write_text(
                "[2026-06-15 11:29 UTC] -- Prediction Run #11\n",
                encoding="utf-8",
            )

            self.assertTrue(
                should_run_prediction(
                    event_name="workflow_dispatch",
                    predictions_path=path,
                    threshold=timedelta(minutes=25),
                    now=datetime(2026, 6, 15, 11, 30, tzinfo=timezone.utc),
                )
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
