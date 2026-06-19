from __future__ import annotations

import unittest

from src.models.blender import blend_predictions
from src.models.random_model import RandomBaselinePredictor


class TestRandomBaselinePredictor(unittest.TestCase):
    def test_random_baseline_is_reproducible(self) -> None:
        features = {"rsi": 51.0, "volume": 2.0}
        p1 = RandomBaselinePredictor(seed=7).predict(features, "24h")
        p2 = RandomBaselinePredictor(seed=7).predict(features, "24h")
        self.assertEqual(p1, p2)
        self.assertIn(p1["direction"], {"UP", "DOWN"})
        self.assertEqual(p1["model_source"], "random")


class TestBlendPredictions(unittest.TestCase):
    def test_blends_matching_timeframes(self) -> None:
        numeric = [{"timeframe": "24h", "direction_prob": 0.8, "magnitude": 2.0}]
        twitter = [{"timeframe": "24h", "direction_prob": 0.2, "magnitude": 4.0}]
        out = blend_predictions(numeric, twitter, numeric_weight=0.5)
        self.assertEqual(len(out), 1)
        self.assertAlmostEqual(out[0]["direction_prob"], 0.5)
        self.assertAlmostEqual(out[0]["magnitude"], 3.0)
        self.assertEqual(out[0]["direction"], "UP")
        self.assertEqual(out[0]["model_source"], "blended_50_50")

    def test_missing_twitter_returns_numeric_only(self) -> None:
        numeric = [{"timeframe": "24h", "direction": "UP", "confidence": 60, "magnitude": 2.0}]
        out = blend_predictions(numeric, [], numeric_weight=0.5)
        self.assertEqual(out[0]["model_source"], "blended_numeric_only")
        self.assertEqual(out[0]["blend_status"], "missing_twitter")


if __name__ == "__main__":
    unittest.main()
