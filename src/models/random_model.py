"""Seeded random prediction baseline.

This model is not intended for trading. It provides a reproducible coin-flip
baseline so dashboards and validation reports can show whether the real modules
beat chance.
"""

from __future__ import annotations

import random
from typing import Any

from src.horizons import TIMEFRAMES


class RandomBaselinePredictor:
    """Generate reproducible random UP/DOWN predictions."""

    def __init__(self, seed: int = 42) -> None:
        self.seed = seed

    def predict(
        self,
        features: dict[str, float] | None = None,
        timeframe: str = "24h",
    ) -> dict[str, Any]:
        rng = random.Random(f"{self.seed}:{timeframe}:{sorted((features or {}).items())[:8]}")
        direction_prob = rng.random()
        direction = "UP" if direction_prob >= 0.5 else "DOWN"
        confidence = int(round(50 + abs(direction_prob - 0.5) * 100))
        # Keep magnitude small and explicit; this is a baseline, not an edge model.
        magnitude = round(0.25 + rng.random() * 1.75, 2)
        return {
            "timeframe": timeframe,
            "direction": direction,
            "direction_prob": direction_prob,
            "magnitude": magnitude,
            "predicted_return": magnitude if direction == "UP" else -magnitude,
            "confidence": confidence,
            "model_source": "random",
            "calibrated": False,
        }

    def predict_all_horizons(
        self,
        features: dict[str, float] | None = None,
    ) -> list[dict[str, Any]]:
        return [self.predict(features, tf) for tf in TIMEFRAMES]
