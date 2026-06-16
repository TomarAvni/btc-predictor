"""Unit tests for prediction freshness checks."""

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
)


class TestPredictionFreshness(unittest.TestCase):
    def test_jsonl_latest_timestamp_takes_precedence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            jsonl = root / "predictions.jsonl"
            text_log = root / "predictions.log"
            jsonl.write_text(
                "\n".join(
                    [
                        '{"timestamp": "2026-06-16T08:32:02Z"}',
                        "not json",
                        '{"timestamp": "2026-06-16T13:16:41Z"}',
                    ]
                ),
                encoding="utf-8",
            )
            text_log.write_text(
                "[2026-06-16 14:00 UTC] -- Prediction Run #99\n",
                encoding="utf-8",
            )

            latest, source = latest_prediction_timestamp(
                jsonl_path=jsonl,
                text_log_path=text_log,
            )

        self.assertEqual(
            latest,
            datetime(2026, 6, 16, 13, 16, 41, tzinfo=timezone.utc),
        )
        self.assertEqual(source, str(jsonl))

    def test_falls_back_to_text_log_headers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            jsonl = root / "missing.jsonl"
            text_log = root / "predictions.log"
            text_log.write_text(
                "\n".join(
                    [
                        "[2026-06-16 08:32 UTC] -- Prediction Run #30",
                        "noise",
                        "[2026-06-16 13:16 UTC] -- Prediction Run #33",
                    ]
                ),
                encoding="utf-8",
            )

            latest, source = latest_prediction_timestamp(
                jsonl_path=jsonl,
                text_log_path=text_log,
            )

        self.assertEqual(
            latest,
            datetime(2026, 6, 16, 13, 16, tzinfo=timezone.utc),
        )
        self.assertEqual(source, str(text_log))

    def test_fresh_when_latest_within_threshold(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            jsonl = Path(tmp) / "predictions.jsonl"
            jsonl.write_text(
                '{"timestamp": "2026-06-16T13:16:41Z"}\n',
                encoding="utf-8",
            )

            status = check_prediction_freshness(
                max_age=timedelta(hours=3),
                now=datetime(2026, 6, 16, 15, 0, tzinfo=timezone.utc),
                jsonl_path=jsonl,
                text_log_path=Path(tmp) / "predictions.log",
            )

        self.assertFalse(status.is_stale)
        self.assertEqual(status.age, timedelta(hours=1, minutes=43, seconds=19))

    def test_stale_when_latest_exceeds_threshold(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            jsonl = Path(tmp) / "predictions.jsonl"
            jsonl.write_text(
                '{"timestamp": "2026-06-16T13:16:41Z"}\n',
                encoding="utf-8",
            )

            status = check_prediction_freshness(
                max_age=timedelta(hours=3),
                now=datetime(2026, 6, 16, 18, 1, 18, tzinfo=timezone.utc),
                jsonl_path=jsonl,
                text_log_path=Path(tmp) / "predictions.log",
            )

        self.assertTrue(status.is_stale)
        self.assertEqual(status.age, timedelta(hours=4, minutes=44, seconds=37))

    def test_stale_when_no_prediction_timestamp_exists(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            status = check_prediction_freshness(
                max_age=timedelta(hours=3),
                now=datetime(2026, 6, 16, 18, 0, tzinfo=timezone.utc),
                jsonl_path=Path(tmp) / "missing.jsonl",
                text_log_path=Path(tmp) / "missing.log",
            )

        self.assertTrue(status.is_stale)
        self.assertIsNone(status.age)


if __name__ == "__main__":
    unittest.main(verbosity=2)
