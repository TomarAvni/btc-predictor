"""Unit tests for prediction log freshness checks."""

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
    iter_prediction_headers,
    main,
)


NOW = datetime(2026, 6, 19, 9, 0, tzinfo=timezone.utc)


def _write_log(path: Path, *headers: str) -> None:
    blocks = []
    for index, header in enumerate(headers, start=1):
        blocks.append(
            "\n".join([
                "=" * 80,
                f"[{header}] -- Prediction Run #{index}",
                "=" * 80,
                "",
                "PREDICTIONS:",
                "  6h    | UP    | +0.1%    | Confidence: 66%",
            ])
        )
    path.write_text("\n".join(blocks), encoding="utf-8")


class TestPredictionFreshness(unittest.TestCase):
    def test_fresh_when_latest_header_within_window(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "predictions.log"
            _write_log(path, "2026-06-19 07:30 UTC")

            status = check_prediction_freshness(path, now=NOW)

            self.assertTrue(status.is_fresh)
            self.assertEqual(status.latest_run_number, 1)
            self.assertEqual(status.age, timedelta(minutes=90))

    def test_stale_when_latest_header_exceeds_window(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "predictions.log"
            _write_log(path, "2026-06-19 05:59 UTC")

            status = check_prediction_freshness(path, now=NOW)

            self.assertFalse(status.is_fresh)
            self.assertIn("older", status.reason)

    def test_missing_log_is_stale(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            status = check_prediction_freshness(Path(tmp) / "missing.log", now=NOW)

            self.assertFalse(status.is_fresh)
            self.assertIsNone(status.latest_run_at)
            self.assertIn("does not exist", status.reason)

    def test_log_without_headers_is_stale(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "predictions.log"
            path.write_text("no prediction runs yet\n", encoding="utf-8")

            status = check_prediction_freshness(path, now=NOW)

            self.assertFalse(status.is_fresh)
            self.assertIn("no prediction run headers", status.reason)

    def test_uses_newest_timestamp_not_last_header(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "predictions.log"
            _write_log(path, "2026-06-19 08:00 UTC", "2026-06-19 04:00 UTC")

            status = check_prediction_freshness(path, now=NOW)

            self.assertTrue(status.is_fresh)
            self.assertEqual(status.latest_run_at, datetime(2026, 6, 19, 8, 0, tzinfo=timezone.utc))

    def test_uses_highest_run_number_when_latest_timestamp_ties(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "predictions.log"
            _write_log(path, "2026-06-19 08:00 UTC", "2026-06-19 08:00 UTC")

            status = check_prediction_freshness(path, now=NOW)

            self.assertTrue(status.is_fresh)
            self.assertEqual(status.latest_run_number, 2)

    def test_future_timestamp_is_not_fresh(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "predictions.log"
            _write_log(path, "2026-06-19 10:00 UTC")

            status = check_prediction_freshness(path, now=NOW)

            self.assertFalse(status.is_fresh)
            self.assertIn("future", status.reason)

    def test_iter_prediction_headers_ignores_malformed_headers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "predictions.log"
            path.write_text(
                "\n".join([
                    "[not a date] -- Prediction Run #1",
                    "[2026-06-19 08:00 UTC] -- Prediction Run #2",
                ]),
                encoding="utf-8",
            )

            self.assertEqual(
                iter_prediction_headers(path),
                [(datetime(2026, 6, 19, 8, 0, tzinfo=timezone.utc), 2)],
            )

    def test_cli_accepts_log_path_alias(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "empty.log"
            path.write_text("", encoding="utf-8")

            self.assertEqual(main(["--log-path", str(path), "--max-age-hours", "3"]), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
