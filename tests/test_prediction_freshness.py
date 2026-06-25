"""Tests for prediction log freshness checks."""

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
    latest_prediction_run,
    main,
)


def _write_log(path: Path, headers: list[str]) -> None:
    chunks = []
    for header in headers:
        chunks.append(
            "\n".join(
                [
                    "=" * 80,
                    f"[{header}] -- Prediction Run #{len(chunks) + 1}",
                    "=" * 80,
                    "",
                    "PREDICTIONS:",
                    "  6h    | UP    | +0.1%    | Confidence: 66%",
                ]
            )
        )
    path.write_text("\n".join(chunks), encoding="utf-8")


class TestPredictionFreshness(unittest.TestCase):
    def test_latest_prediction_run_uses_newest_header(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "predictions.log"
            _write_log(
                log_path,
                [
                    "2026-06-18 09:00 UTC",
                    "2026-06-18 12:30 UTC",
                ],
            )

            latest = latest_prediction_run(log_path)

        self.assertIsNotNone(latest)
        timestamp, run_number = latest
        self.assertEqual(timestamp, datetime(2026, 6, 18, 12, 30, tzinfo=timezone.utc))
        self.assertEqual(run_number, 2)

    def test_fresh_when_latest_run_is_within_limit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "predictions.log"
            _write_log(log_path, ["2026-06-18 12:30 UTC"])

            freshness = check_prediction_freshness(
                log_path,
                max_age_hours=3,
                now=datetime(2026, 6, 18, 15, 0, tzinfo=timezone.utc),
            )

        self.assertTrue(freshness.is_fresh)
        self.assertAlmostEqual(freshness.age_hours, 2.5)

    def test_stale_when_latest_run_exceeds_limit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "predictions.log"
            _write_log(log_path, ["2026-06-18 11:30 UTC"])

            freshness = check_prediction_freshness(
                log_path,
                max_age_hours=3,
                now=datetime(2026, 6, 18, 15, 0, tzinfo=timezone.utc),
            )

        self.assertFalse(freshness.is_fresh)
        self.assertAlmostEqual(freshness.age_hours, 3.5)

    def test_missing_log_is_stale(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            freshness = check_prediction_freshness(
                Path(tmp) / "missing.log",
                max_age_hours=3,
                now=datetime(2026, 6, 18, 15, 0, tzinfo=timezone.utc),
            )

        self.assertFalse(freshness.is_fresh)
        self.assertIsNone(freshness.age_hours)

    def test_cli_exit_codes_match_freshness(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "predictions.log"
            _write_log(log_path, ["2026-06-18 12:30 UTC"])

            self.assertEqual(main(["--path", str(log_path), "--max-age-hours", "99999"]), 0)
            self.assertEqual(main(["--path", str(log_path), "--max-age-hours", "0.1"]), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
