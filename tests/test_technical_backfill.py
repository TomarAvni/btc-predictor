from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from scripts.backfill_technical import backfill_technical_history, build_technical_history
from src.training.feature_builder import TrainingFeatureBuilder


def _price_frame(hours: int = 240) -> pd.DataFrame:
    idx = pd.date_range("2026-01-01", periods=hours, freq="h", tz="UTC")
    close = pd.Series([100.0 + i * 0.5 for i in range(hours)], index=idx)
    df = pd.DataFrame(
        {
            "open": close - 0.2,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": [1000.0 + i for i in range(hours)],
        },
        index=idx,
    )
    df.index.name = "timestamp"
    return df


class TestTechnicalBackfill(unittest.TestCase):
    def test_build_technical_history_uses_live_indicator_columns(self) -> None:
        history = build_technical_history(_price_frame())

        self.assertIn("rsi_14", history.columns)
        self.assertIn("macd", history.columns)
        self.assertIn("ema_50_200_cross", history.columns)
        self.assertIn("vwap", history.columns)
        self.assertNotIn("close", history.columns)
        self.assertEqual(history.index.name, "timestamp")
        self.assertEqual(len(history), 240)

    def test_backfill_writes_technical_parquet(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            price_dir = root / "price"
            price_dir.mkdir()
            _price_frame().to_parquet(price_dir / "btc_hourly.parquet")
            output = root / "history" / "technical.parquet"

            history = backfill_technical_history(price_dir=price_dir, output_path=output)

            self.assertTrue(output.exists())
            loaded = pd.read_parquet(output)
            self.assertEqual(len(loaded), len(history))
            self.assertIn("rsi_14", loaded.columns)


class TestFeatureBuilderPlaceholders(unittest.TestCase):
    def test_absent_placeholder_columns_are_not_added(self) -> None:
        features = TrainingFeatureBuilder().build_features(_price_frame(24))

        self.assertNotIn("exchange_netflow", features.columns)
        self.assertNotIn("fear_greed_index", features.columns)

    def test_absent_tweet_columns_are_not_added_when_opted_in(self) -> None:
        features = TrainingFeatureBuilder().build_features(
            _price_frame(24),
            include_tweets=True,
        )

        self.assertNotIn("tw_tweet_volume", features.columns)

    def test_real_placeholder_columns_are_preserved(self) -> None:
        price_df = _price_frame(24)
        price_df["fear_greed_index"] = range(24)

        features = TrainingFeatureBuilder().build_features(price_df)

        self.assertIn("fear_greed_index", features.columns)
        self.assertEqual(float(features["fear_greed_index"].iloc[-1]), 23.0)

    def test_real_tweet_columns_are_preserved_when_opted_in(self) -> None:
        price_df = _price_frame(24)
        price_df["tw_tweet_volume"] = range(24)

        features = TrainingFeatureBuilder().build_features(
            price_df,
            include_tweets=True,
        )

        self.assertIn("tw_tweet_volume", features.columns)
        self.assertEqual(float(features["tw_tweet_volume"].iloc[-1]), 23.0)


if __name__ == "__main__":
    unittest.main()
