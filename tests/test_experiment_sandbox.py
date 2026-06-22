"""Tests for sandbox-only experiment helpers."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from experiments.export_sentiment_sandbox import build_seed


class TestExperimentSandbox(unittest.TestCase):
    def test_sentiment_seed_export_is_data_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            history = root / "history"
            performance = root / "performance"
            history.mkdir()
            performance.mkdir()
            (history / "twitter_market_state.json").write_text(
                json.dumps({"mood": 0.2}), encoding="utf-8"
            )
            (history / "twitter_notable_events.json").write_text(
                json.dumps([{"tweet_id": str(i)} for i in range(60)]), encoding="utf-8"
            )
            (performance / "rolling_accuracy.json").write_text(
                json.dumps({"24h": {"direction_accuracy": 0.55}}), encoding="utf-8"
            )

            seed = build_seed(history, performance)

        self.assertEqual(seed["source"], "btc-predictor")
        self.assertIn("AGPL", seed["license_boundary"])
        self.assertIn("held-out", seed["promotion_gate"])
        self.assertEqual(seed["market_state"], {"mood": 0.2})
        self.assertEqual(len(seed["notable_events"]), 50)
        self.assertEqual(seed["notable_events"][0]["tweet_id"], "10")
        self.assertEqual(seed["rolling_accuracy"]["24h"]["direction_accuracy"], 0.55)


if __name__ == "__main__":
    unittest.main(verbosity=2)
