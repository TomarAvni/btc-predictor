"""Manager / synthesizer for the X/Twitter sentiment module.

Orchestrates the stateless workers (collector -> grounded reader -> aggregator ->
external memory) and synthesizes:

* a **market-psychology state** (crowd mood, fear/greed, conviction, positioning),
* the **llm_direct** forecast -- a grounded forecast derived from what was read
  (6h-interval path to ~1 week plus 30d), with direction, magnitude, long/short
  and a confidence.

The llm_direct confidence is intentionally the reader's own (uncalibrated): the
project scores it head-to-head against the calibrated/blended models so the
"LLM-direct vs calibrated" question is settled with data, not opinion.
"""

from __future__ import annotations

import math
from typing import Any

import pandas as pd

from src.collectors.twitter_sentiment import TwitterSentimentCollector
from src.engine.sentiment_memory import SentimentMemory
from src.features import tweet_aggregator
from src.features.tweet_llm_reader import GroundedTweetReader
from src.horizons import HORIZON_HOURS, TIMEFRAMES
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

MODEL_ID = "llm_direct"
MAX_HORIZON_HOURS = 720  # cap at 30 days


class SentimentManager:
    """Runs the sentiment pipeline and emits the llm_direct forecast."""

    def __init__(
        self,
        config: dict | None = None,
        collector: TwitterSentimentCollector | None = None,
        reader: GroundedTweetReader | None = None,
        memory: SentimentMemory | None = None,
    ) -> None:
        self.config = config or {}
        self.collector = collector or TwitterSentimentCollector(self.config)
        self.reader = reader or GroundedTweetReader(self.config)
        self.memory = memory or SentimentMemory()
        self._min_relevance = float(
            self.config.get("collectors", {}).get("llm_reader", {}).get("min_relevance", 0.15)
        )

    # -- full ingest + synthesize tick -----------------------------------------

    async def run_tick(self, money_signals: dict[str, float] | None = None) -> dict[str, Any]:
        """Ingest, read, aggregate, persist, and synthesize one cycle."""
        tweets = await self.collector.collect()
        extractions = self.reader.read(tweets)
        signal = tweet_aggregator.aggregate(tweets, extractions, self._min_relevance)

        if not signal.empty:
            self.memory.add_signal(signal)
        self.memory.record_events(tweets, extractions)

        latest = self.memory.latest_signal()
        state = self.build_state(latest, money_signals)
        self.memory.update_state(state)

        predictions = self.forecast(state)
        return {
            "predictions": predictions,
            "state": state,
            "model_source": MODEL_ID,
            "n_tweets": int(len(tweets)),
            "n_extractions": int(len(extractions)),
            "mock": self.collector.is_mock or self.reader.is_mock,
        }

    # -- market-psychology state -----------------------------------------------

    def build_state(
        self,
        latest_signal: dict[str, Any] | None,
        money_signals: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        sig = latest_signal or {}
        evidence = self.memory.retrieve_notable(limit=5)
        fg = float(sig.get("tw_fear_greed", 50.0))
        return {
            "mood": float(sig.get("tw_influencer_sentiment", 0.0)),
            "sentiment_mean": float(sig.get("tw_sentiment_mean", 0.0)),
            "momentum": float(sig.get("tw_sentiment_momentum", 0.0)),
            "fear_greed": fg,
            "fear_greed_label": _fg_label(fg),
            "positioning": float(sig.get("tw_intent_balance", 0.0)),
            "conviction": float(sig.get("tw_conviction_mean", 0.0)),
            "event_presence": float(sig.get("tw_event_presence", 0.0)),
            "volume_zscore": float(sig.get("tw_volume_zscore_24h", 0.0)),
            "money_signals": money_signals or {},
            "evidence": evidence,
            "as_of": sig.get("timestamp"),
        }

    # -- llm_direct forecast ----------------------------------------------------

    def forecast(self, state: dict[str, Any]) -> list[dict[str, Any]]:
        """Derive a per-horizon forecast from the crowd-psychology state.

        Grounded in what was read: combines influencer mood, momentum, positioning,
        a fear/greed contrarian nudge and (when available) hard "money" signals into
        a single directional score, then weakens it for longer horizons (tweet
        sentiment decays fast).
        """
        score = self._signal_score(state)
        conviction = max(0.0, min(1.0, state.get("conviction", 0.0)))

        predictions: list[dict[str, Any]] = []
        for tf in TIMEFRAMES:
            hours = HORIZON_HOURS.get(tf)
            if hours is None or hours > MAX_HORIZON_HOURS:
                continue

            decay = 1.0 / (1.0 + hours / 168.0)          # weaken with horizon
            eff = score * decay
            prob_up = 0.5 + 0.5 * math.tanh(2.0 * eff)
            prob_up = round(min(0.95, max(0.05, prob_up)), 4)

            direction = "UP" if prob_up >= 0.5 else "DOWN"
            magnitude = round(max(0.1, abs(eff) * 4.0 * math.sqrt(hours / 24.0)), 2)
            confidence = int(round(min(95, max(10, 50 + 45 * abs(eff) * (0.5 + 0.5 * conviction)))))

            predictions.append({
                "timeframe": tf,
                "direction": direction,
                "direction_prob": prob_up,
                "magnitude": magnitude,
                "confidence": confidence,
                "calibrated": False,
            })
        return predictions

    @staticmethod
    def _signal_score(state: dict[str, Any]) -> float:
        """Blend the psychology + money signals into a score in roughly [-1, 1]."""
        mood = state.get("mood", 0.0)
        momentum = state.get("momentum", 0.0)
        positioning = state.get("positioning", 0.0)
        # Fear/greed as a mild contrarian nudge: extreme fear -> slight up bias.
        fg = state.get("fear_greed", 50.0)
        contrarian = (50.0 - fg) / 100.0

        money = state.get("money_signals") or {}
        money_term = 0.0
        if money:
            money_term = sum(float(v) for v in money.values()) / (len(money) * 1.0)
            money_term = max(-1.0, min(1.0, money_term))

        score = (
            0.45 * mood
            + 0.20 * momentum
            + 0.20 * positioning
            + 0.10 * contrarian
            + 0.05 * money_term
        )
        return max(-1.0, min(1.0, score))


def _fg_label(value: float) -> str:
    if value <= 20:
        return "extreme_fear"
    if value <= 40:
        return "fear"
    if value < 60:
        return "neutral"
    if value < 80:
        return "greed"
    return "extreme_greed"
