"""Unit tests for the X/Twitter LLM sentiment module (mock mode, no keys).

Covers ingestion filtering + engagement weighting, grounded extraction with
citation enforcement, hourly causal aggregation, and the manager's llm_direct
forecast shape.
"""

from __future__ import annotations

import asyncio
import os
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.collectors.twitter_sentiment import TwitterSentimentCollector  # noqa: E402
from src.engine.sentiment_manager import SentimentManager  # noqa: E402
from src.engine.sentiment_memory import SentimentMemory  # noqa: E402
from src.features import tweet_aggregator  # noqa: E402
from src.features.tweet_llm_reader import GroundedTweetReader, TweetExtraction  # noqa: E402
from src.horizons import HORIZON_HOURS  # noqa: E402

TEST_CONFIG = {
    "collectors": {
        "twitter": {
            "whitelist_handles": [
                "saylor", "WatcherGuru", "DocumentingBTC",
                "cointelegraph", "BitcoinMagazine",
            ],
            "min_followers": 500,
            "languages": ["en"],
            "whitelist_weight": 3.0,
        },
        "llm_reader": {"min_relevance": 0.15},
    }
}


class TestSentimentModule(unittest.TestCase):
    def setUp(self) -> None:
        # Force mock mode regardless of the developer's environment.
        self._saved_env = {
            k: os.environ.pop(k) for k in ("TWITTERAPI_KEY", "LLM_API_KEY")
            if k in os.environ
        }

    def tearDown(self) -> None:
        os.environ.update(self._saved_env)

    def _collect(self) -> pd.DataFrame:
        collector = TwitterSentimentCollector(TEST_CONFIG)
        self.assertTrue(collector.is_mock)
        return asyncio.run(collector.collect())

    def test_ingestion_filters_spam_and_language(self) -> None:
        df = self._collect()
        ids = set(df["id"])
        # Kept: whitelist + the >=500-follower English trader.
        self.assertEqual(ids, {"1001", "1002", "1003", "1004", "1007", "1008"})
        self.assertNotIn("1005", ids)  # Spanish, not whitelisted
        self.assertNotIn("1006", ids)  # spam bot, 80 followers

    def test_engagement_weight_boosts_whitelist(self) -> None:
        df = self._collect()
        saylor = df[df["id"] == "1001"]["engagement_weight"].iloc[0]
        trader = df[df["id"] == "1003"]["engagement_weight"].iloc[0]
        self.assertGreater(saylor, trader)

    def test_reader_grounding_and_signs(self) -> None:
        df = self._collect()
        reader = GroundedTweetReader(TEST_CONFIG)
        self.assertTrue(reader.is_mock)
        ext = {e.tweet_id: e for e in reader.read(df)}
        # All surviving extractions must be grounded with a real cited id.
        for e in ext.values():
            self.assertTrue(e.is_grounded)
            self.assertIn(e.tweet_id, e.cited_tweet_ids)
        self.assertGreater(ext["1001"].sentiment, 0)   # bought/inflows/record/rally
        self.assertLess(ext["1002"].sentiment, 0)      # incident/paused/nervous
        self.assertEqual(ext["1001"].self_reported_intent, "buy")
        self.assertEqual(ext["1003"].self_reported_intent, "sell")
        self.assertIn("hack", ext["1002"].event_flags)

    def test_reader_rejects_ungrounded(self) -> None:
        df = self._collect()
        reader = GroundedTweetReader(TEST_CONFIG)
        # An extraction citing a tweet outside the batch must be dropped.
        bogus = TweetExtraction(tweet_id="x", sentiment=0.9, cited_tweet_ids=["999999"])
        reader._read_mock = lambda _df: [bogus]  # type: ignore[assignment]
        self.assertEqual(reader.read(df), [])

    def test_aggregation_is_hourly_and_complete(self) -> None:
        df = self._collect()
        reader = GroundedTweetReader(TEST_CONFIG)
        signal = tweet_aggregator.aggregate(df, reader.read(df))
        self.assertFalse(signal.empty)
        for col in tweet_aggregator.SIGNAL_COLUMNS:
            self.assertIn(col, signal.columns)
        # Tweets span the 09:00 and 10:00 UTC hours -> two causal buckets.
        self.assertEqual(len(signal), 2)
        self.assertTrue((signal["tw_tweet_volume"] > 0).all())

    def test_manager_forecast_shape(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            memory = SentimentMemory(
                signal_path=tmp_path / "sig.parquet",
                events_path=tmp_path / "events.json",
                state_path=tmp_path / "state.json",
            )
            mgr = SentimentManager(config=TEST_CONFIG, memory=memory)
            result = asyncio.run(mgr.run_tick())

            self.assertTrue(result["mock"])
            self.assertEqual(result["model_source"], "llm_direct")
            self.assertGreater(len(result["predictions"]), 0)
            for p in result["predictions"]:
                self.assertIn(p["timeframe"], HORIZON_HOURS)
                self.assertLessEqual(HORIZON_HOURS[p["timeframe"]], 720)
                self.assertIn(p["direction"], ("UP", "DOWN"))
                self.assertGreaterEqual(p["direction_prob"], 0.05)
                self.assertLessEqual(p["direction_prob"], 0.95)
                self.assertGreaterEqual(p["confidence"], 10)
                self.assertLessEqual(p["confidence"], 95)
            # State persisted and carries grounded psychology fields.
            state = memory.get_state()
            self.assertIn("fear_greed", state)
            self.assertIn("mood", state)


if __name__ == "__main__":
    unittest.main(verbosity=2)
