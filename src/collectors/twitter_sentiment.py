"""X/Twitter ingestion collector for the sentiment prediction module.

Pulls recent Bitcoin tweets from TwitterAPI.io (a third-party reseller) for a
broad keyword stream plus a trusted-handle whitelist tier, applies spam/bot
filtering, and annotates each tweet with an engagement-based influence weight so
viral / high-reach tweets count more downstream.

MOCK MODE: when the ``TWITTERAPI_KEY`` environment variable is absent the
collector loads ``tests/fixtures/sample_tweets.json`` instead of calling the
network, so the entire pipeline runs offline with no keys and no spend.

The collector returns *raw tweets* (one row per tweet, indexed by ``created_at``);
turning them into hourly features is the job of :mod:`src.features.tweet_aggregator`.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

import pandas as pd

from src import PROJECT_ROOT
from src.collectors.base import BaseCollector
from src.utils.cache import get_cached, set_cached
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

API_KEY_ENV = "TWITTERAPI_KEY"
MOCK_FIXTURE_PATH = PROJECT_ROOT / "tests" / "fixtures" / "sample_tweets.json"

DEFAULTS: dict[str, Any] = {
    "api_base": "https://api.twitterapi.io",
    "cadence_minutes": 10,
    "max_tweets_per_run": 1000,
    "monthly_tweet_cap": 4_320_000,
    "keywords": ["$BTC", "#Bitcoin", "bitcoin"],
    "whitelist_handles": [],
    "min_followers": 500,
    "languages": ["en"],
    "whitelist_weight": 3.0,
}

RAW_COLUMNS = [
    "id", "text", "author", "followers",
    "like_count", "retweet_count", "quote_count", "reply_count",
    "lang", "is_whitelist", "engagement_weight",
]

# Engagement-rate factor constants. A tweet's engagement *rate*
# (virality / followers) credits small accounts that punch above their reach:
# a 1k-follower tweet drawing thousands of interactions is a stronger organic
# signal than its raw reach implies. The boost is bounded so it tops up -- never
# dominates -- the reach/virality term.
RATE_SCALE = 5.0   # multiplier on the raw engagement rate before capping
RATE_CAP = 1.0     # max additive boost (rate_factor in [1.0, 1.0 + RATE_CAP])


def engagement_weight(
    like_count: int,
    retweet_count: int,
    quote_count: int,
    reply_count: int,
    followers: int,
    is_whitelist: bool,
    whitelist_weight: float,
) -> float:
    """Influence weight for a tweet.

    Combines virality (likes/retweets/quotes, retweets weighted highest because
    they amplify reach) with author reach (followers), on a log scale so a few
    viral tweets don't completely swamp the aggregate, then applies an
    engagement-*rate* boost (so small accounts punching above their reach are not
    under-credited) and finally a credibility multiplier for whitelisted handles.
    A retweeted/viral tweet therefore counts more -- by design.

        virality      = likes + 2*retweets + quotes + 0.5*replies
        engagement_rate = virality / max(followers, 1)
        rate_factor   = 1 + min(engagement_rate * RATE_SCALE, RATE_CAP)
        credibility   = whitelist_weight if whitelisted else 1
        weight = (1+log1p(virality)) * (1+log1p(followers)/10) * rate_factor * credibility
    """
    virality = like_count + 2.0 * retweet_count + quote_count + 0.5 * reply_count
    reach = math.log1p(max(followers, 0))
    engagement_rate = max(virality, 0.0) / max(followers, 1)
    rate_factor = 1.0 + min(engagement_rate * RATE_SCALE, RATE_CAP)
    credibility = max(whitelist_weight, 1.0) if is_whitelist else 1.0
    weight = (1.0 + math.log1p(max(virality, 0.0))) * (1.0 + reach / 10.0)
    weight *= rate_factor * credibility
    return round(weight, 4)


class TwitterSentimentCollector(BaseCollector):
    """Collects raw Bitcoin tweets (live via TwitterAPI.io, or mock fixtures)."""

    def __init__(self, config: dict | None = None) -> None:
        cfg = (config or {}).get("collectors", {}).get("twitter", {})
        self._cfg = {**DEFAULTS, **cfg}
        self._api_key = os.environ.get(API_KEY_ENV)

    # -- BaseCollector contract -------------------------------------------------

    @property
    def name(self) -> str:
        return "twitter_sentiment"

    @property
    def update_interval_seconds(self) -> int:
        return int(self._cfg["cadence_minutes"]) * 60

    @property
    def is_mock(self) -> bool:
        """True when running without an API key (offline fixture mode)."""
        return not self._api_key

    async def _collect(self) -> pd.DataFrame:
        raw = self._load_mock() if self.is_mock else await self._fetch_live()
        return self._to_frame(raw)

    # -- ingestion sources ------------------------------------------------------

    def _load_mock(self) -> list[dict[str, Any]]:
        if not MOCK_FIXTURE_PATH.exists():
            logger.warning("Mock fixture missing: %s", MOCK_FIXTURE_PATH)
            return []
        logger.info("[twitter_sentiment] MOCK MODE -- loading %s", MOCK_FIXTURE_PATH.name)
        return json.loads(MOCK_FIXTURE_PATH.read_text(encoding="utf-8"))

    async def _fetch_live(self) -> list[dict[str, Any]]:
        """Pull recent tweets from TwitterAPI.io (broad keywords + whitelist).

        Network failures are swallowed by ``BaseCollector.collect`` which returns
        an empty DataFrame, so a dead feed never crashes the predict path.
        """
        if not self._budget_ok():
            logger.warning("[twitter_sentiment] monthly tweet cap reached -- skipping pull")
            return []

        import httpx  # local import keeps mock mode dependency-free

        query = " OR ".join(self._cfg["keywords"])
        for handle in self._cfg["whitelist_handles"]:
            query += f" OR from:{handle}"

        url = f"{self._cfg['api_base'].rstrip('/')}/twitter/tweet/advanced_search"
        params = {"query": query, "queryType": "Latest"}
        headers = {"X-API-Key": self._api_key}

        collected: list[dict[str, Any]] = []
        limit = int(self._cfg["max_tweets_per_run"])
        async with httpx.AsyncClient(timeout=20) as client:
            cursor: str | None = None
            while len(collected) < limit:
                if cursor:
                    params["cursor"] = cursor
                resp = await client.get(url, params=params, headers=headers)
                resp.raise_for_status()
                payload = resp.json()
                collected.extend(payload.get("tweets", []))
                cursor = payload.get("next_cursor")
                if not cursor or not payload.get("has_next_page"):
                    break

        self._budget_consume(len(collected))
        # LIVE-MODE TODO (engagement velocity / two-pass capture): a planned
        # enhancement is to re-poll a sample of these tweets after a short delay
        # to measure how fast engagement is *accelerating* (dlikes/dt), a leading
        # impact signal. It is intentionally not implemented here because it
        # cannot run offline/mock and would add dead complexity to the batch path.
        return collected[:limit]

    def fetch_historical_day(self, day_start: pd.Timestamp) -> pd.DataFrame:
        """Fetch one UTC day of historical tweets and normalize to the raw frame.

        TwitterAPI.io exposes historical search through the same advanced-search
        endpoint. The query is constrained with since/until date operators so the
        downstream reader/backfill path uses the exact live normalization logic.
        """
        if self.is_mock:
            return self._to_frame(self._load_mock())
        if not self._budget_ok():
            logger.warning("[twitter_sentiment] monthly tweet cap reached -- skipping historical pull")
            return pd.DataFrame(columns=RAW_COLUMNS)

        import httpx

        start = pd.Timestamp(day_start)
        start = start.tz_localize("UTC") if start.tzinfo is None else start.tz_convert("UTC")
        start = start.normalize()
        end = start + pd.Timedelta(days=1)
        query = " OR ".join(self._cfg["keywords"])
        for handle in self._cfg["whitelist_handles"]:
            query += f" OR from:{handle}"
        query = f"({query}) since:{start.date()} until:{end.date()}"

        url = f"{self._cfg['api_base'].rstrip('/')}/twitter/tweet/advanced_search"
        params = {"query": query, "queryType": "Latest"}
        headers = {"X-API-Key": self._api_key}

        collected: list[dict[str, Any]] = []
        limit = int(self._cfg["max_tweets_per_run"])
        with httpx.Client(timeout=20) as client:
            cursor: str | None = None
            while len(collected) < limit:
                if cursor:
                    params["cursor"] = cursor
                resp = client.get(url, params=params, headers=headers)
                resp.raise_for_status()
                payload = resp.json()
                collected.extend(payload.get("tweets", []))
                cursor = payload.get("next_cursor")
                if not cursor or not payload.get("has_next_page"):
                    break

        self._budget_consume(len(collected))
        return self._to_frame(collected[:limit])

    # -- normalization + filtering ---------------------------------------------

    def _to_frame(self, raw: list[dict[str, Any]]) -> pd.DataFrame:
        whitelist = {h.lower() for h in self._cfg["whitelist_handles"]}
        languages = {lang.lower() for lang in self._cfg["languages"]}
        min_followers = int(self._cfg["min_followers"])
        wl_weight = float(self._cfg["whitelist_weight"])

        rows: list[dict[str, Any]] = []
        seen: set[str] = set()
        for t in raw:
            norm = self._normalize_tweet(t)
            if norm is None:
                continue
            if norm["id"] in seen:
                continue

            is_wl = norm["author"].lower() in whitelist
            if norm["lang"].lower() not in languages and not is_wl:
                continue
            if norm["followers"] < min_followers and not is_wl:
                continue
            if not norm["text"].strip():
                continue

            norm["is_whitelist"] = is_wl
            norm["engagement_weight"] = engagement_weight(
                norm["like_count"], norm["retweet_count"], norm["quote_count"],
                norm["reply_count"], norm["followers"], is_wl, wl_weight,
            )
            seen.add(norm["id"])
            rows.append(norm)

        if not rows:
            return pd.DataFrame(columns=RAW_COLUMNS)

        df = pd.DataFrame(rows)
        df["created_at"] = pd.to_datetime(df["created_at"], utc=True)
        df = df.set_index("created_at").sort_index()
        df.index.name = "timestamp"
        return df[RAW_COLUMNS]

    @staticmethod
    def _normalize_tweet(t: dict[str, Any]) -> dict[str, Any] | None:
        """Map either the fixture schema or the TwitterAPI.io schema to ours."""
        try:
            tweet_id = str(t.get("id") or t.get("id_str") or "").strip()
            if not tweet_id:
                return None
            author_obj = t.get("author") or {}
            if isinstance(author_obj, dict):
                author = author_obj.get("userName") or author_obj.get("screen_name") or ""
                followers = int(author_obj.get("followers") or author_obj.get("followers_count") or 0)
            else:
                author = str(author_obj)
                followers = int(t.get("followers", 0) or 0)
            return {
                "id": tweet_id,
                "created_at": t.get("created_at") or t.get("createdAt"),
                "text": str(t.get("text") or t.get("full_text") or ""),
                "author": str(author),
                "followers": followers,
                "like_count": int(t.get("like_count") or t.get("likeCount") or 0),
                "retweet_count": int(t.get("retweet_count") or t.get("retweetCount") or 0),
                "quote_count": int(t.get("quote_count") or t.get("quoteCount") or 0),
                "reply_count": int(t.get("reply_count") or t.get("replyCount") or 0),
                "lang": str(t.get("lang") or "en"),
            }
        except (TypeError, ValueError):
            return None

    # -- monthly budget guard ---------------------------------------------------

    def _budget_state(self) -> dict[str, Any]:
        month = pd.Timestamp.now(tz="UTC").strftime("%Y-%m")
        cached = get_cached("twitter_budget", max_age_minutes=60 * 24 * 40) or {}
        if cached.get("month") != month:
            return {"month": month, "count": 0}
        return cached

    def _budget_ok(self) -> bool:
        state = self._budget_state()
        return state["count"] < int(self._cfg["monthly_tweet_cap"])

    def _budget_consume(self, n: int) -> None:
        state = self._budget_state()
        state["count"] += int(n)
        set_cached("twitter_budget", state)
