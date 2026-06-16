"""Scorer must score parallel model tracks independently and partition stats.

Verifies the ``model_id`` dimension: two tracks (``numbers`` and ``llm_direct``)
for the same run+horizon are scored separately (no collision) and surface in the
per-track ``by_model`` breakdown of rolling accuracy.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.engine import scorer  # noqa: E402


def _write_price_parquet(path: Path, start: str, hours: int, start_price: float = 100.0) -> None:
    idx = pd.date_range(start=start, periods=hours, freq="h", tz="UTC")
    close = [start_price * (1 + 0.001 * i) for i in range(hours)]
    df = pd.DataFrame(
        {"open": close, "high": close, "low": close, "close": close, "volume": [1.0] * hours},
        index=idx,
    )
    df.index.name = "timestamp"
    df.to_parquet(path)


class TestScorerMultiTrack(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.jsonl = self.tmp / "predictions.jsonl"
        self.scores = self.tmp / "scores.jsonl"
        self.price = self.tmp / "btc_hourly.parquet"
        _write_price_parquet(self.price, "2026-01-01", hours=24 * 40)
        self.now = pd.Timestamp("2026-03-01", tz="UTC")

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _pred(self, direction: str) -> list[dict]:
        return [{
            "timeframe": "24h", "direction": direction, "direction_prob": 0.6,
            "magnitude": 1.0, "confidence": 60,
        }]

    def test_two_tracks_same_run_scored_independently(self) -> None:
        # Same run_number + timeframe, two different model_source tracks.
        records = [
            {"run_number": 1, "timestamp": "2026-01-02T00:00:00Z", "used_ml": True,
             "model_source": "numbers", "predictions": self._pred("UP")},
            {"run_number": 1, "timestamp": "2026-01-02T00:00:00Z", "used_ml": False,
             "model_source": "llm_direct", "predictions": self._pred("DOWN")},
        ]
        with open(self.jsonl, "w", encoding="utf-8") as fh:
            for rec in records:
                fh.write(json.dumps(rec) + "\n")

        new = scorer.score_mature_predictions(
            price_path=self.price, scores_path=self.scores,
            now=self.now, predictions_jsonl_path=self.jsonl,
        )
        # Both tracks scored despite identical run_number:timeframe.
        self.assertEqual(len(new), 2)
        by_source = {r["model_source"]: r for r in new}
        self.assertIn("numbers", by_source)
        self.assertIn("llm_direct", by_source)
        # Upward drift -> numbers (UP) correct, llm_direct (DOWN) wrong.
        self.assertTrue(by_source["numbers"]["direction_correct"])
        self.assertFalse(by_source["llm_direct"]["direction_correct"])

        # Re-running does not double-score either track.
        again = scorer.score_mature_predictions(
            price_path=self.price, scores_path=self.scores,
            now=self.now, predictions_jsonl_path=self.jsonl,
        )
        self.assertEqual(len(again), 0)

    def test_rolling_accuracy_partitioned_by_model(self) -> None:
        records = [
            {"run_number": 1, "timestamp": "2026-01-02T00:00:00Z", "used_ml": True,
             "model_source": "numbers", "predictions": self._pred("UP")},
            {"run_number": 1, "timestamp": "2026-01-02T00:00:00Z", "used_ml": False,
             "model_source": "llm_direct", "predictions": self._pred("DOWN")},
        ]
        with open(self.jsonl, "w", encoding="utf-8") as fh:
            for rec in records:
                fh.write(json.dumps(rec) + "\n")
        new = scorer.score_mature_predictions(
            price_path=self.price, scores_path=self.scores,
            now=self.now, predictions_jsonl_path=self.jsonl,
        )

        rolling = scorer.compute_rolling_accuracy(scores=new, now=self.now)
        self.assertIn("by_model", rolling)
        self.assertIn("numbers", rolling["by_model"])
        self.assertIn("llm_direct", rolling["by_model"])
        num_acc = rolling["by_model"]["numbers"]["timeframes"]["24h"]["all_time"]
        llm_acc = rolling["by_model"]["llm_direct"]["timeframes"]["24h"]["all_time"]
        self.assertEqual(num_acc["direction_accuracy_pct"], 100.0)
        self.assertEqual(llm_acc["direction_accuracy_pct"], 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
