"""Unit tests for prediction workflow freshness checks."""

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
                        '{"timestamp": "2026-06-16T10:00:00+00:00"}',
                        "not json",
                        '{"timestamp": "2026-06-16T12:30:00Z"}',
                        '{"timestamp": ""}',
                    ]
                ),
                encoding="utf-8",
            )

            self.assertEqual(
                latest_prediction_jsonl_time(path),
                datetime(2026, 6, 16, 12, 30, tzinfo=timezone.utc),
            )

    def test_latest_prediction_log_time_uses_newest_header(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "predictions.log"
            path.write_text(
                "\n".join(
                    [
                        "[2026-06-16 09:30 UTC]",
                        "PREDICTIONS:",
                        "[2026-06-16 11:00 UTC]",
                    ]
                ),
                encoding="utf-8",
            )

            self.assertEqual(
                latest_prediction_log_time(path),
                datetime(2026, 6, 16, 11, 0, tzinfo=timezone.utc),
            )

    def test_latest_prediction_time_prefers_newest_source(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            jsonl_path = Path(tmp) / "predictions.jsonl"
            log_path = Path(tmp) / "predictions.log"
            jsonl_path.write_text(
                '{"timestamp": "2026-06-16T13:15:00+00:00"}\n',
                encoding="utf-8",
            )
            log_path.write_text("[2026-06-16 12:45 UTC]\n", encoding="utf-8")

            self.assertEqual(
                latest_prediction_time(jsonl_path, log_path),
                datetime(2026, 6, 16, 13, 15, tzinfo=timezone.utc),
            )

    def test_scheduled_run_skips_when_prediction_is_fresh(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            jsonl_path = Path(tmp) / "predictions.jsonl"
            log_path = Path(tmp) / "predictions.log"
            jsonl_path.write_text(
                '{"timestamp": "2026-06-16T13:00:00+00:00"}\n',
                encoding="utf-8",
            )

            self.assertFalse(
                should_run_prediction(
                    event_name="schedule",
                    jsonl_path=jsonl_path,
                    log_path=log_path,
                    threshold=timedelta(minutes=25),
                    now=datetime(2026, 6, 16, 13, 20, tzinfo=timezone.utc),
                )
            )

    def test_scheduled_run_proceeds_when_prediction_is_stale(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            jsonl_path = Path(tmp) / "predictions.jsonl"
            log_path = Path(tmp) / "predictions.log"
            log_path.write_text("[2026-06-16 13:00 UTC]\n", encoding="utf-8")

            self.assertTrue(
                should_run_prediction(
                    event_name="schedule",
                    jsonl_path=jsonl_path,
                    log_path=log_path,
                    threshold=timedelta(minutes=25),
                    now=datetime(2026, 6, 16, 13, 30, tzinfo=timezone.utc),
                )
            )

    def test_manual_run_always_proceeds(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            jsonl_path = Path(tmp) / "predictions.jsonl"
            log_path = Path(tmp) / "predictions.log"
            jsonl_path.write_text(
                '{"timestamp": "2026-06-16T13:00:00+00:00"}\n',
                encoding="utf-8",
            )

            self.assertTrue(
                should_run_prediction(
                    event_name="workflow_dispatch",
                    jsonl_path=jsonl_path,
                    log_path=log_path,
                    threshold=timedelta(hours=24),
                    now=datetime(2026, 6, 16, 13, 1, tzinfo=timezone.utc),
                )
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
