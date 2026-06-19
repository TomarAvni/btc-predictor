"""Unit tests for the append-only labeled training store."""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.training import labeled_store  # noqa: E402


def _score(run: int, tf: str, feats: dict | None, direction: str = "UP",
           actual: str = "UP", prob: float = 0.6) -> dict:
    return {
        "run_number": run,
        "timeframe": tf,
        "prediction_timestamp": f"2026-01-{run:02d}T00:00:00+00:00",
        "predicted_direction": direction,
        "direction_prob": prob,
        "predicted_magnitude": 1.5,
        "confidence": 60,
        "calibrated": False,
        "model_source": "ensemble",
        "used_ml": True,
        "actual_return_pct": 1.2,
        "actual_direction": actual,
        "direction_correct": direction == actual,
        "magnitude_error": 0.3,
        "start_price": 100.0,
        "end_price": 101.2,
        "features": feats if feats is not None else {},
    }


class TestLabeledStore(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.store = Path(self._tmp.name) / "labeled.jsonl"

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_append_skips_empty_features_and_unknown_horizons(self) -> None:
        scores = [
            _score(1, "24h", {"rsi_14": 55.0}),
            _score(2, "24h", {}),          # empty features -> skipped
            _score(3, "999h", {"rsi": 1}), # unknown horizon -> skipped
        ]
        added = labeled_store.append_labeled_rows(scores, path=self.store)
        self.assertEqual(added, 1)
        rows = labeled_store.load_labeled_rows(self.store)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["timeframe"], "24h")
        self.assertEqual(rows[0]["label_up"], 1)

    def test_label_up_reflects_realized_direction(self) -> None:
        scores = [_score(1, "24h", {"rsi_14": 55.0}, actual="DOWN")]
        labeled_store.append_labeled_rows(scores, path=self.store)
        rows = labeled_store.load_labeled_rows(self.store)
        self.assertEqual(rows[0]["label_up"], 0)

    def test_dedup_accumulates_not_overwrites(self) -> None:
        labeled_store.append_labeled_rows([_score(1, "24h", {"rsi_14": 1.0})], path=self.store)
        labeled_store.append_labeled_rows([_score(1, "24h", {"rsi_14": 1.0})], path=self.store)  # dup
        added = labeled_store.append_labeled_rows([_score(2, "24h", {"rsi_14": 2.0})], path=self.store)
        self.assertEqual(added, 1)
        rows = labeled_store.load_labeled_rows(self.store)
        self.assertEqual(len(rows), 2)  # accumulated, not overwritten

    def test_same_run_different_horizons_are_distinct(self) -> None:
        scores = [
            _score(1, "24h", {"rsi_14": 1.0}),
            _score(1, "168h", {"rsi_14": 1.0}),
        ]
        added = labeled_store.append_labeled_rows(scores, path=self.store)
        self.assertEqual(added, 2)

    def test_frame_expands_features(self) -> None:
        labeled_store.append_labeled_rows(
            [_score(1, "24h", {"rsi_14": 55.0, "macd": 1.2})], path=self.store
        )
        rows = labeled_store.load_labeled_rows(self.store)
        df = labeled_store.labeled_rows_to_frame(rows)
        self.assertIn("feat__rsi_14", df.columns)
        self.assertIn("feat__macd", df.columns)
        self.assertEqual(labeled_store.feature_columns(df), ["feat__rsi_14", "feat__macd"])

    def test_per_horizon_counts(self) -> None:
        labeled_store.append_labeled_rows([
            _score(1, "24h", {"rsi_14": 1.0}),
            _score(2, "24h", {"rsi_14": 1.0}),
            _score(3, "168h", {"rsi_14": 1.0}),
        ], path=self.store)
        counts = labeled_store.per_horizon_counts(self.store)
        self.assertEqual(counts, {"24h": 2, "168h": 1})

    def test_empty_store_returns_empty_frame(self) -> None:
        df = labeled_store.labeled_rows_to_frame(path=self.store)
        self.assertTrue(df.empty)


if __name__ == "__main__":
    unittest.main(verbosity=2)
