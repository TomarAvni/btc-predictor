"""Unit tests for prediction pipeline freshness detection."""

from __future__ import annotations

import json
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
    latest_prediction_timestamp_from_jsonl,
    latest_prediction_timestamp_from_log,
)


class TestPredictionFreshness(unittest.TestCase):
    def test_missing_log_is_stale(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            freshness = check_prediction_freshness(
                Path(tmp) / "missing.log",
                jsonl_path=Path(tmp) / "missing.jsonl",
                max_age=timedelta(hours=1),
                now=datetime(2026, 6, 16, 12, 0, tzinfo=timezone.utc),
            )

        self.assertTrue(freshness.is_stale)
        self.assertIsNone(freshness.latest_at)
        self.assertIn("no prediction", freshness.reason)

    def test_latest_timestamp_uses_last_prediction_header(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "predictions.log"
            path.write_text(
                "\n".join(
                    [
                        "[2026-06-16 06:00 UTC] -- Prediction Run #41",
                        "noise",
                        "[2026-06-16 09:30 UTC] -- Prediction Run #42",
                    ]
                ),
                encoding="utf-8",
            )

            latest = latest_prediction_timestamp_from_log(path)

        self.assertEqual(
            latest,
            datetime(2026, 6, 16, 9, 30, tzinfo=timezone.utc),
        )

    def test_latest_timestamp_from_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "predictions.jsonl"
            records = [
                {"run_number": 1, "timestamp": "2026-06-16T06:00:00Z"},
                {"run_number": 2, "timestamp": "2026-06-16T09:30:00Z"},
            ]
            path.write_text(
                "\n".join(json.dumps(record) for record in records),
                encoding="utf-8",
            )

            latest = latest_prediction_timestamp_from_jsonl(path)

        self.assertEqual(
            latest,
            datetime(2026, 6, 16, 9, 30, tzinfo=timezone.utc),
        )

    def test_latest_timestamp_prefers_newest_across_sources(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "predictions.log"
            jsonl_path = Path(tmp) / "predictions.jsonl"
            log_path.write_text(
                "[2026-06-16 08:00 UTC] -- Prediction Run #41\n",
                encoding="utf-8",
            )
            jsonl_path.write_text(
                json.dumps({"run_number": 42, "timestamp": "2026-06-16T10:00:00Z"}),
                encoding="utf-8",
            )

            latest, source = latest_prediction_timestamp(log_path, jsonl_path)

        self.assertEqual(
            latest,
            datetime(2026, 6, 16, 10, 0, tzinfo=timezone.utc),
        )
        self.assertEqual(source, "predictions.jsonl")

    def test_fresh_when_latest_prediction_is_under_threshold(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "predictions.log"
            path.write_text(
                "[2026-06-16 09:30 UTC] -- Prediction Run #42\n",
                encoding="utf-8",
            )
            freshness = check_prediction_freshness(
                path,
                jsonl_path=Path(tmp) / "missing.jsonl",
                max_age=timedelta(hours=1),
                now=datetime(2026, 6, 16, 10, 0, tzinfo=timezone.utc),
            )

        self.assertFalse(freshness.is_stale)
        self.assertEqual(freshness.age, timedelta(minutes=30))

    def test_stale_when_latest_prediction_reaches_threshold(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "predictions.log"
            path.write_text(
                "[2026-06-16 09:00 UTC] -- Prediction Run #42\n",
                encoding="utf-8",
            )
            freshness = check_prediction_freshness(
                path,
                jsonl_path=Path(tmp) / "missing.jsonl",
                max_age=timedelta(hours=1),
                now=datetime(2026, 6, 16, 12, 0, tzinfo=timezone.utc),
            )

        self.assertTrue(freshness.is_stale)
        self.assertEqual(freshness.age, timedelta(hours=3))


if __name__ == "__main__":
    unittest.main(verbosity=2)
