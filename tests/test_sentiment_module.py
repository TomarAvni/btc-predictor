"""Unit tests for the X/Twitter LLM sentiment module (mock mode, no keys).

Covers ingestion filtering + engagement weighting, grounded extraction with
citation enforcement, hourly causal aggregation, and the manager's llm_direct
forecast shape.
"""

from __future__ import annotations

import asyncio
import math
import os
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.collectors.twitter_sentiment import (  # noqa: E402
    TwitterSentimentCollector,
    engagement_weight,
)
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
        # Kept: whitelist + the >=500-follower English accounts (trader, the
        # low-follower duplicate cluster 1009-1011, and the small high-engagement
        # account 1012).
        self.assertEqual(
            ids,
            {"1001", "1002", "1003", "1004", "1007", "1008",
             "1009", "1010", "1011", "1012"},
        )
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

    def test_reader_clamps_live_schema_values(self) -> None:
        reader = GroundedTweetReader(TEST_CONFIG)
        body = {
            "choices": [{
                "message": {
                    "content": (
                        '{"tweets":[{"tweet_id":"1001","sentiment":9,'
                        '"fear_greed":-20,"event_flags":["ETF","ETF"],'
                        '"self_reported_intent":"moon","conviction":3,'
                        '"relevance":-1,"cited_tweet_ids":["1001"]}]}'
                    )
                }
            }]
        }
        [item] = reader._parse_llm_response(body)
        self.assertEqual(item.sentiment, 1.0)
        self.assertEqual(item.fear_greed, 0.0)
        self.assertEqual(item.event_flags, ["etf"])
        self.assertEqual(item.self_reported_intent, "none")
        self.assertEqual(item.conviction, 1.0)
        self.assertEqual(item.relevance, 0.0)

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
            # Impact-quality signals wired into the state doc.
            self.assertIn("burst_intensity", state)
            self.assertIn("bot_ratio", state)

    # -- impact-quality tests (new) --------------------------------------------

    @staticmethod
    def _frame(rows: list[tuple]) -> pd.DataFrame:
        """Build a minimal raw-tweet frame: rows of (id, created_at, text, followers, ew)."""
        df = pd.DataFrame([
            {
                "id": r[0],
                "created_at": r[1],
                "text": r[2],
                "followers": r[3],
                "engagement_weight": r[4],
            }
            for r in rows
        ])
        df["created_at"] = pd.to_datetime(df["created_at"], utc=True)
        df = df.set_index("created_at")
        df.index.name = "timestamp"
        return df

    @staticmethod
    def _exts(ids: list[str]) -> list[TweetExtraction]:
        return [
            TweetExtraction(tweet_id=str(i), sentiment=1.0, relevance=1.0, cited_tweet_ids=[str(i)])
            for i in ids
        ]

    def test_novelty_saturation_discounts_duplicates(self) -> None:
        # Same count + same engagement; high followers so coordination never
        # fires, isolating the novelty/dedup effect. Followers >= 1000.
        unique = self._frame([
            ("u1", "2026-06-15T09:01:00Z", "bitcoin etf inflows hit a record high", 5000, 10.0),
            ("u2", "2026-06-15T09:02:00Z", "miners are holding through the dip", 5000, 10.0),
            ("u3", "2026-06-15T09:03:00Z", "lightning network adoption keeps growing", 5000, 10.0),
        ])
        dup = self._frame([
            ("d1", "2026-06-15T09:01:00Z", "bitcoin to the moon buy now", 5000, 10.0),
            ("d2", "2026-06-15T09:02:00Z", "bitcoin to the moon buy now", 5000, 10.0),
            ("d3", "2026-06-15T09:03:00Z", "bitcoin to the moon buy now", 5000, 10.0),
        ])
        sig_u = tweet_aggregator.aggregate(unique, self._exts(["u1", "u2", "u3"]))
        sig_d = tweet_aggregator.aggregate(dup, self._exts(["d1", "d2", "d3"]))

        self.assertGreater(
            sig_u["tw_effective_volume"].iloc[0],
            sig_d["tw_effective_volume"].iloc[0],
        )
        self.assertEqual(sig_u["tw_novelty_ratio"].iloc[0], 1.0)
        self.assertLess(sig_d["tw_novelty_ratio"].iloc[0], 1.0)

    def test_coordination_raises_bot_ratio(self) -> None:
        text = "bitcoin to the moon buy now"
        coordinated = self._frame([
            ("c1", "2026-06-15T09:01:00Z", text, 600, 5.0),
            ("c2", "2026-06-15T09:01:10Z", text, 700, 5.0),
            ("c3", "2026-06-15T09:01:20Z", text, 800, 5.0),
        ])
        organic = self._frame([
            ("o1", "2026-06-15T09:01:00Z", text, 50000, 5.0),
            ("o2", "2026-06-15T09:01:10Z", text, 60000, 5.0),
            ("o3", "2026-06-15T09:01:20Z", text, 70000, 5.0),
        ])
        sig_c = tweet_aggregator.aggregate(coordinated, self._exts(["c1", "c2", "c3"]))
        sig_o = tweet_aggregator.aggregate(organic, self._exts(["o1", "o2", "o3"]))

        self.assertGreater(sig_c["tw_bot_ratio"].iloc[0], 0.0)
        self.assertEqual(sig_o["tw_bot_ratio"].iloc[0], 0.0)
        self.assertGreater(
            sig_c["tw_bot_ratio"].iloc[0],
            sig_o["tw_bot_ratio"].iloc[0],
        )

    def test_burst_intensity_detects_spike(self) -> None:
        # Four tweets crammed into one 5-min sub-bin vs spread across the hour.
        packed = self._frame([
            ("p1", "2026-06-15T09:01:00Z", "alpha unique one", 5000, 5.0),
            ("p2", "2026-06-15T09:01:10Z", "bravo unique two", 5000, 5.0),
            ("p3", "2026-06-15T09:01:20Z", "charlie unique three", 5000, 5.0),
            ("p4", "2026-06-15T09:01:30Z", "delta unique four", 5000, 5.0),
        ])
        spread = self._frame([
            ("s1", "2026-06-15T09:00:00Z", "alpha unique one", 5000, 5.0),
            ("s2", "2026-06-15T09:15:00Z", "bravo unique two", 5000, 5.0),
            ("s3", "2026-06-15T09:30:00Z", "charlie unique three", 5000, 5.0),
            ("s4", "2026-06-15T09:45:00Z", "delta unique four", 5000, 5.0),
        ])
        sig_p = tweet_aggregator.aggregate(packed, self._exts(["p1", "p2", "p3", "p4"]))
        sig_s = tweet_aggregator.aggregate(spread, self._exts(["s1", "s2", "s3", "s4"]))

        self.assertGreater(
            sig_p["tw_burst_intensity"].iloc[0],
            sig_s["tw_burst_intensity"].iloc[0],
        )

    def test_engagement_rate_boosts_small_high_engagement_account(self) -> None:
        # Small account, huge engagement relative to its reach.
        likes, rts, quotes, replies, followers = 9000, 3000, 500, 1000, 600
        weight = engagement_weight(likes, rts, quotes, replies, followers, False, 3.0)
        virality = likes + 2.0 * rts + quotes + 0.5 * replies
        base = (1.0 + math.log1p(virality)) * (1.0 + math.log1p(followers) / 10.0)
        # Rate factor saturates near its cap -> a meaningful (>=1.5x) boost.
        self.assertGreater(weight, base * 1.5)

        # A low-engagement account with the SAME reach is boosted far less.
        low = engagement_weight(5, 1, 0, 0, followers, False, 3.0)
        base_low = (1.0 + math.log1p(5 + 2.0)) * (1.0 + math.log1p(followers) / 10.0)
        self.assertGreater(weight / base, low / base_low)


if __name__ == "__main__":
    unittest.main(verbosity=2)
