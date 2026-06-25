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
    latest_prediction_jsonl_time,
    latest_prediction_log_time,
    latest_prediction_time,
    should_run_prediction,
)


class TestPredictionFreshness(unittest.TestCase):
    def test_latest_prediction_jsonl_time_uses_newest_valid_record(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "predictions.jsonl"
            path.write_text(
                "\n".join(
                    [
                        '{"timestamp": "2026-06-15T10:45:00Z"}',
                        "not-json",
                        '{"timestamp": "2026-06-15T11:15:00+00:00"}',
                    ]
                ),
                encoding="utf-8",
            )

            self.assertEqual(
                latest_prediction_jsonl_time(path),
                datetime(2026, 6, 15, 11, 15, tzinfo=timezone.utc),
            )

    def test_latest_prediction_log_time_uses_newest_header(self) -> None:
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
                latest_prediction_log_time(path),
                datetime(2026, 6, 15, 11, 15, tzinfo=timezone.utc),
            )

    def test_latest_prediction_time_prefers_newest_source(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            jsonl = Path(tmp) / "predictions.jsonl"
            log = Path(tmp) / "predictions.log"
            jsonl.write_text(
                '{"timestamp": "2026-06-15T11:10:00Z"}\n',
                encoding="utf-8",
            )
            log.write_text(
                "[2026-06-15 11:15 UTC] -- Prediction Run #11\n",
                encoding="utf-8",
            )

            self.assertEqual(
                latest_prediction_time(jsonl, log),
                datetime(2026, 6, 15, 11, 15, tzinfo=timezone.utc),
            )

    def test_missing_logs_are_stale_for_schedule(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            self.assertTrue(
                should_run_prediction(
                    event_name="schedule",
                    jsonl_path=Path(tmp) / "missing.jsonl",
                    log_path=Path(tmp) / "missing.log",
                    threshold=timedelta(minutes=25),
                    now=datetime(2026, 6, 15, 11, 30, tzinfo=timezone.utc),
                )
            )

    def test_schedule_skips_fresh_prediction(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "predictions.jsonl"
            path.write_text(
                '{"timestamp": "2026-06-15T11:15:00Z"}\n',
                encoding="utf-8",
            )

            self.assertFalse(
                should_run_prediction(
                    event_name="schedule",
                    jsonl_path=path,
                    log_path=Path(tmp) / "missing.log",
                    threshold=timedelta(minutes=25),
                    now=datetime(2026, 6, 15, 11, 30, tzinfo=timezone.utc),
                )
            )

    def test_schedule_runs_stale_prediction(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "predictions.jsonl"
            path.write_text(
                '{"timestamp": "2026-06-15T11:00:00Z"}\n',
                encoding="utf-8",
            )

            self.assertTrue(
                should_run_prediction(
                    event_name="schedule",
                    jsonl_path=path,
                    log_path=Path(tmp) / "missing.log",
                    threshold=timedelta(minutes=25),
                    now=datetime(2026, 6, 15, 11, 30, tzinfo=timezone.utc),
                )
            )

    def test_workflow_dispatch_always_runs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "predictions.jsonl"
            path.write_text(
                '{"timestamp": "2026-06-15T11:29:00Z"}\n',
                encoding="utf-8",
            )

            self.assertTrue(
                should_run_prediction(
                    event_name="workflow_dispatch",
                    jsonl_path=path,
                    log_path=Path(tmp) / "missing.log",
                    threshold=timedelta(minutes=25),
                    now=datetime(2026, 6, 15, 11, 30, tzinfo=timezone.utc),
                )
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
