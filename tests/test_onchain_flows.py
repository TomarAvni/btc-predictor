from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.collectors.onchain_flows import OnChainFlowCollector
from src.training.feature_builder import TrainingFeatureBuilder


class TestOnChainFlowCollector(unittest.TestCase):
    def test_summarize_large_flow_windows_and_heuristics(self) -> None:
        collector = OnChainFlowCollector()
        now = pd.Timestamp("2026-01-08T00:00:00Z")
        records = [
            {
                "timestamp": now - pd.Timedelta(hours=1),
                "tx_hash": "a",
                "total_btc": 200.0,
                "category": "unknown",
                "heuristic": "cold_storage_like",
            },
            {
                "timestamp": now - pd.Timedelta(hours=2),
                "tx_hash": "b",
                "total_btc": 300.0,
                "category": "unknown",
                "heuristic": "distribution_like",
            },
            {
                "timestamp": now - pd.Timedelta(days=2),
                "tx_hash": "c",
                "total_btc": 400.0,
                "category": "exchange",
                "heuristic": "neutral",
            },
        ]

        summary = collector._summarize(records, now)
        self.assertEqual(summary["whale_btc_moved_6h"], 500.0)
        self.assertEqual(summary["whale_btc_moved_24h"], 500.0)
        self.assertEqual(summary["whale_btc_moved_7d"], 900.0)
        self.assertEqual(summary["largest_whale_tx_btc_24h"], 300.0)
        self.assertAlmostEqual(summary["flow_accumulation_score"], -0.2)

    def test_label_file_created_when_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "labels.json"
            collector = OnChainFlowCollector(labels_path=path)
            labels = collector._load_labels()
            self.assertEqual(labels, {})
            self.assertTrue(path.exists())


class TestOnChainFlowFeatures(unittest.TestCase):
    def test_feature_builder_copies_onchain_flow_columns(self) -> None:
        idx = pd.date_range("2026-01-01", periods=3, freq="h", tz="UTC")
        price_df = pd.DataFrame(
            {
                "open": [100.0, 101.0, 102.0],
                "high": [101.0, 102.0, 103.0],
                "low": [99.0, 100.0, 101.0],
                "close": [100.0, 101.0, 102.0],
                "volume": [10.0, 11.0, 12.0],
                "whale_btc_moved_24h": [1.0, 2.0, 3.0],
                "net_exchange_flow_btc_24h": [-1.0, 0.0, 1.0],
            },
            index=idx,
        )

        features = TrainingFeatureBuilder().build_features(price_df)
        self.assertIn("whale_btc_moved_24h", features.columns)
        self.assertIn("net_exchange_flow_btc_24h", features.columns)
        self.assertEqual(float(features["whale_btc_moved_24h"].iloc[-1]), 3.0)
        self.assertEqual(float(features["net_exchange_flow_btc_24h"].iloc[-1]), 1.0)


if __name__ == "__main__":
    unittest.main()
