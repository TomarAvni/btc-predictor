"""Tests for prediction freshness checks."""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.prediction_freshness import (  # noqa: E402
    check_prediction_freshness,
    latest_jsonl_timestamp,
    latest_text_log_timestamp,
    main,
)


class TestPredictionFreshness(unittest.TestCase):
    def test_jsonl_latest_timestamp_ignores_bad_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "predictions.jsonl"
            path.write_text(
                "\n".join(
                    [
                        '{"timestamp": "2026-06-17T05:28:36Z"}',
                        "not json",
                        '{"timestamp": "2026-06-17T06:00:00+00:00"}',
                        '{"timestamp": "bad"}',
                    ]
                ),
                encoding="utf-8",
            )

            self.assertEqual(
                latest_jsonl_timestamp(path),
                datetime(2026, 6, 17, 6, 0, tzinfo=timezone.utc),
            )

    def test_text_log_latest_timestamp(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "predictions.log"
            path.write_text(
                """
================================================================================
[2026-06-17 05:28 UTC] -- Prediction Run #42
================================================================================
[2026-06-17 08:01 UTC] -- Prediction Run #43
""",
                encoding="utf-8",
            )

            self.assertEqual(
                latest_text_log_timestamp(path),
                datetime(2026, 6, 17, 8, 1, tzinfo=timezone.utc),
            )

    def test_fresh_when_latest_prediction_within_max_age(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            jsonl = Path(tmp) / "predictions.jsonl"
            text_log = Path(tmp) / "predictions.log"
            jsonl.write_text(
                '{"timestamp": "2026-06-17T08:30:00Z"}\n',
                encoding="utf-8",
            )

            result = check_prediction_freshness(
                max_age_hours=3,
                now=datetime(2026, 6, 17, 10, 0, tzinfo=timezone.utc),
                jsonl_path=jsonl,
                text_log_path=text_log,
            )

            self.assertTrue(result.is_fresh)
            self.assertEqual(result.source, str(jsonl))

    def test_stale_when_latest_prediction_exceeds_max_age(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            jsonl = Path(tmp) / "predictions.jsonl"
            text_log = Path(tmp) / "predictions.log"
            jsonl.write_text(
                '{"timestamp": "2026-06-17T05:28:00Z"}\n',
                encoding="utf-8",
            )

            result = check_prediction_freshness(
                max_age_hours=3,
                now=datetime(2026, 6, 17, 9, 1, tzinfo=timezone.utc),
                jsonl_path=jsonl,
                text_log_path=text_log,
            )

            self.assertFalse(result.is_fresh)

    def test_missing_artifacts_are_stale(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = check_prediction_freshness(
                max_age_hours=3,
                now=datetime(2026, 6, 17, 9, 1, tzinfo=timezone.utc),
                jsonl_path=Path(tmp) / "missing.jsonl",
                text_log_path=Path(tmp) / "missing.log",
            )

            self.assertFalse(result.is_fresh)
            self.assertIsNone(result.latest_timestamp)

    def test_cli_exit_code_reflects_freshness(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            jsonl = Path(tmp) / "predictions.jsonl"
            text_log = Path(tmp) / "predictions.log"

            self.assertEqual(
                main(
                    [
                        "--max-age-hours",
                        "3",
                        "--jsonl-path",
                        str(jsonl),
                        "--text-log-path",
                        str(text_log),
                    ]
                ),
                1,
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
