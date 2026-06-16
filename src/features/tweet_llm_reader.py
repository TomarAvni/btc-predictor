"""Grounded LLM reader for Bitcoin tweets.

Reads tweets and emits a *descriptive* structured judgment per tweet -- sentiment,
fear/greed, event flags, self-reported buy/sell intent, conviction and relevance.
The guiding principle is **grounded, not guessed**: every extraction must cite the
tweet id(s) that justify it, and any ungrounded output is rejected (zeroed/dropped)
rather than trusted. Extraction is strictly descriptive and never forecasts price,
which keeps the LLM's after-the-fact knowledge from leaking into training features.

MOCK MODE: without ``LLM_API_KEY`` the reader uses a deterministic lexical model so
the pipeline runs offline and reproducibly. The same component (and its ``VERSION``)
is used for both historical backfill and live serving to avoid train/serve skew.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from src.utils.logger import setup_logger

logger = setup_logger(__name__)

API_KEY_ENV = "LLM_API_KEY"

# Bump when the prompt / model / schema changes; invalidates cached backfills so
# historical features always match the live extractor (train/serve parity).
VERSION = "reader-v1"

INTENT_BUY = "buy"
INTENT_SELL = "sell"
INTENT_HOLD = "hold"
INTENT_NONE = "none"

_BULLISH = {"bullish", "moon", "rally", "pump", "breakout", "ath", "record", "surge", "buy", "long", "accumulate", "accumulating", "stacking", "inflows", "euphoria"}
_BEARISH = {"bearish", "dump", "crash", "weak", "sell", "short", "shorting", "fear", "nervous", "cautious", "delays", "uncertainty", "incident", "hack", "paused", "capitulation"}
_FEAR = {"fear", "nervous", "cautious", "uncertainty", "panic", "weak", "crash", "dump"}
_GREED = {"greed", "euphoria", "moon", "fomo", "ath", "record", "rally", "surge"}

_EVENT_KEYWORDS = {
    "etf": "etf",
    "sec": "regulation",
    "regulat": "regulation",
    "hack": "hack",
    "security incident": "hack",
    "exploit": "hack",
    "halving": "halving",
    "breaking": "breaking_news",
}

_INTENT_BUY_PHRASES = ["just bought", "i'm buying", "im buying", "buying", "accumulat", "stacking", "aped in", "aping", "long here", "going long", "buy the dip"]
_INTENT_SELL_PHRASES = ["just sold", "i'm selling", "im selling", "selling", "shorting", "short here", "going short", "took profit", "exited"]
_INTENT_HOLD_PHRASES = ["holding", "hodl", "diamond hands", "not selling"]


@dataclass
class TweetExtraction:
    """Descriptive, grounded judgment for one tweet."""

    tweet_id: str
    sentiment: float = 0.0          # [-1, 1]
    fear_greed: float = 50.0        # [0, 100] (0 = extreme fear, 100 = extreme greed)
    event_flags: list[str] = field(default_factory=list)
    self_reported_intent: str = INTENT_NONE
    conviction: float = 0.0         # [0, 1]
    relevance: float = 0.0          # [0, 1]
    cited_tweet_ids: list[str] = field(default_factory=list)

    @property
    def is_grounded(self) -> bool:
        return bool(self.cited_tweet_ids)

    def to_dict(self) -> dict[str, Any]:
        return {
            "tweet_id": self.tweet_id,
            "sentiment": self.sentiment,
            "fear_greed": self.fear_greed,
            "event_flags": list(self.event_flags),
            "self_reported_intent": self.self_reported_intent,
            "conviction": self.conviction,
            "relevance": self.relevance,
            "cited_tweet_ids": list(self.cited_tweet_ids),
        }


class GroundedTweetReader:
    """Reads tweets into grounded :class:`TweetExtraction` records."""

    def __init__(self, config: dict | None = None) -> None:
        cfg = (config or {}).get("collectors", {}).get("llm_reader", {})
        self._cfg = cfg
        self._api_key = os.environ.get(API_KEY_ENV)
        self._min_relevance = float(cfg.get("min_relevance", 0.15))
        self.version = VERSION

    @property
    def is_mock(self) -> bool:
        return not self._api_key

    def read(self, tweets: pd.DataFrame) -> list[TweetExtraction]:
        """Return grounded extractions for the given tweets.

        Ungrounded outputs (no cited tweet ids, or citing ids not in the batch)
        are dropped so only evidence-backed judgments survive.
        """
        if tweets is None or tweets.empty:
            return []

        valid_ids = {str(i) for i in tweets["id"].tolist()}
        records = (
            self._read_mock(tweets) if self.is_mock else self._read_llm(tweets)
        )

        grounded: list[TweetExtraction] = []
        for rec in records:
            cited = [c for c in rec.cited_tweet_ids if c in valid_ids]
            if not cited:
                logger.debug("Dropping ungrounded extraction for %s", rec.tweet_id)
                continue
            rec.cited_tweet_ids = cited
            grounded.append(rec)
        return grounded

    # -- deterministic offline reader ------------------------------------------

    def _read_mock(self, tweets: pd.DataFrame) -> list[TweetExtraction]:
        out: list[TweetExtraction] = []
        for tid, text in zip(tweets["id"].astype(str), tweets["text"].astype(str)):
            out.append(self._lexical_extract(tid, text))
        return out

    def _lexical_extract(self, tweet_id: str, text: str) -> TweetExtraction:
        low = text.lower()
        tokens = set(low.replace("#", " ").replace("$", " ").replace(".", " ").replace(",", " ").split())

        bull = len(tokens & _BULLISH)
        bear = len(tokens & _BEARISH)
        total = bull + bear
        sentiment = 0.0 if total == 0 else round((bull - bear) / total, 4)

        fear = len(tokens & _FEAR)
        greed = len(tokens & _GREED)
        fg_total = fear + greed
        fear_greed = 50.0 if fg_total == 0 else round(50.0 + 50.0 * (greed - fear) / fg_total, 1)

        events = sorted({tag for kw, tag in _EVENT_KEYWORDS.items() if kw in low})

        intent = INTENT_NONE
        if any(p in low for p in _INTENT_BUY_PHRASES):
            intent = INTENT_BUY
        elif any(p in low for p in _INTENT_SELL_PHRASES):
            intent = INTENT_SELL
        elif any(p in low for p in _INTENT_HOLD_PHRASES):
            intent = INTENT_HOLD

        conviction = round(min(1.0, total / 4.0), 4)
        relevant = ("btc" in low) or ("bitcoin" in low) or bool(events) or total > 0
        relevance = 1.0 if relevant else 0.0

        return TweetExtraction(
            tweet_id=tweet_id,
            sentiment=sentiment,
            fear_greed=fear_greed,
            event_flags=events,
            self_reported_intent=intent,
            conviction=conviction,
            relevance=relevance,
            cited_tweet_ids=[tweet_id],  # grounded by construction: it read this tweet
        )

    # -- live LLM reader --------------------------------------------------------

    def _read_llm(self, tweets: pd.DataFrame) -> list[TweetExtraction]:
        """Batch the tweets to an LLM with a strict, citation-required schema.

        On any failure this falls back to the deterministic reader so the predict
        path keeps producing features rather than going dark.
        """
        try:
            import httpx

            batch_size = int(self._cfg.get("batch_size", 40))
            model = self._cfg.get("model", "gpt-5-nano")
            api_base = self._cfg.get("api_base", "https://api.openai.com/v1")
            out: list[TweetExtraction] = []
            ids = tweets["id"].astype(str).tolist()
            texts = tweets["text"].astype(str).tolist()

            with httpx.Client(timeout=30) as client:
                for start in range(0, len(ids), batch_size):
                    chunk_ids = ids[start:start + batch_size]
                    chunk_txt = texts[start:start + batch_size]
                    payload = self._build_request(model, chunk_ids, chunk_txt)
                    resp = client.post(
                        f"{api_base.rstrip('/')}/chat/completions",
                        headers={"Authorization": f"Bearer {self._api_key}"},
                        json=payload,
                    )
                    resp.raise_for_status()
                    out.extend(self._parse_llm_response(resp.json()))
            return out
        except Exception as exc:  # pragma: no cover - network/parse defensive path
            logger.warning("LLM reader failed (%s); falling back to lexical extractor", exc)
            return self._read_mock(tweets)

    def _build_request(self, model: str, ids: list[str], texts: list[str]) -> dict[str, Any]:
        numbered = "\n".join(f"[{i}] {t}" for i, t in zip(ids, texts))
        system = (
            "You read Bitcoin tweets and output ONLY a JSON array. For each tweet emit "
            "{tweet_id, sentiment(-1..1), fear_greed(0..100), event_flags[], "
            "self_reported_intent(buy|sell|hold|none), conviction(0..1), relevance(0..1), "
            "cited_tweet_ids[]}. Be DESCRIPTIVE only -- never predict price. "
            "cited_tweet_ids MUST contain the id(s) of the tweet(s) that justify your "
            "judgment; if you cannot ground it, omit the tweet."
        )
        return {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": numbered},
            ],
            "response_format": {"type": "json_object"},
        }

    def _parse_llm_response(self, body: dict[str, Any]) -> list[TweetExtraction]:
        import json

        content = body["choices"][0]["message"]["content"]
        data = json.loads(content)
        items = data if isinstance(data, list) else data.get("tweets", data.get("results", []))
        out: list[TweetExtraction] = []
        for it in items:
            try:
                out.append(TweetExtraction(
                    tweet_id=str(it["tweet_id"]),
                    sentiment=float(it.get("sentiment", 0.0)),
                    fear_greed=float(it.get("fear_greed", 50.0)),
                    event_flags=list(it.get("event_flags", []) or []),
                    self_reported_intent=str(it.get("self_reported_intent", INTENT_NONE)),
                    conviction=float(it.get("conviction", 0.0)),
                    relevance=float(it.get("relevance", 0.0)),
                    cited_tweet_ids=[str(c) for c in (it.get("cited_tweet_ids") or [])],
                ))
            except (KeyError, TypeError, ValueError):
                continue
        return out
