"""Aggregate per-tweet LLM extractions into an hourly causal signal series.

Each hourly row summarizes only the tweets that occurred *in that hour* (strictly
causal -- no future information), engagement-weighted so viral / high-reach tweets
dominate. Rolling features (volume z-score, sentiment momentum) use trailing
windows only. The result is persisted to ``data/history/twitter_llm_signal.parquet``
following the project's ``data/history/<signal>.parquet`` convention so it is
picked up by :meth:`HistoricalDataLoader.get_merged_dataset`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from src import DATA_DIR
from src.features.tweet_llm_reader import (
    INTENT_BUY,
    INTENT_HOLD,
    INTENT_SELL,
    TweetExtraction,
)
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

SIGNAL_PATH = DATA_DIR / "history" / "twitter_llm_signal.parquet"

# All feature columns this module produces (tw_ prefix avoids collisions with
# the existing Fear & Greed `sentiment` signal).
SIGNAL_COLUMNS = [
    "tw_tweet_volume",
    "tw_volume_zscore_24h",
    "tw_sentiment_mean",
    "tw_influencer_sentiment",
    "tw_sentiment_momentum",
    "tw_bull_bear_ratio",
    "tw_fear_greed",
    "tw_intent_balance",
    "tw_event_presence",
    "tw_relevance_mean",
    "tw_conviction_mean",
]


def _weighted_mean(values: np.ndarray, weights: np.ndarray, default: float = 0.0) -> float:
    wsum = float(weights.sum())
    if wsum <= 0:
        return default
    return float((values * weights).sum() / wsum)


def aggregate(
    tweets: pd.DataFrame,
    extractions: Iterable[TweetExtraction],
    min_relevance: float = 0.15,
) -> pd.DataFrame:
    """Build the hourly signal frame from raw tweets + their extractions.

    Args:
        tweets: raw tweets (DatetimeIndex, must include ``id`` and
            ``engagement_weight``).
        extractions: grounded :class:`TweetExtraction` records.
        min_relevance: drop off-topic tweets below this relevance.

    Returns:
        Hourly DataFrame indexed by UTC hour with :data:`SIGNAL_COLUMNS`.
    """
    ext_by_id = {e.tweet_id: e for e in extractions}
    if tweets is None or tweets.empty or not ext_by_id:
        return pd.DataFrame(columns=SIGNAL_COLUMNS)

    df = tweets.copy()
    df["id"] = df["id"].astype(str)
    df = df[df["id"].isin(ext_by_id)]
    if df.empty:
        return pd.DataFrame(columns=SIGNAL_COLUMNS)

    df["sentiment"] = df["id"].map(lambda i: ext_by_id[i].sentiment)
    df["fear_greed"] = df["id"].map(lambda i: ext_by_id[i].fear_greed)
    df["conviction"] = df["id"].map(lambda i: ext_by_id[i].conviction)
    df["relevance"] = df["id"].map(lambda i: ext_by_id[i].relevance)
    df["intent"] = df["id"].map(lambda i: ext_by_id[i].self_reported_intent)
    df["has_event"] = df["id"].map(lambda i: 1.0 if ext_by_id[i].event_flags else 0.0)

    df = df[df["relevance"] >= min_relevance]
    if df.empty:
        return pd.DataFrame(columns=SIGNAL_COLUMNS)

    df["hour"] = df.index.floor("h")
    rows: list[dict] = []
    for hour, grp in df.groupby("hour"):
        w = grp["engagement_weight"].to_numpy(dtype=float)
        sent = grp["sentiment"].to_numpy(dtype=float)

        n_buy = int((grp["intent"] == INTENT_BUY).sum())
        n_sell = int((grp["intent"] == INTENT_SELL).sum())
        n_hold = int((grp["intent"] == INTENT_HOLD).sum())
        intent_total = n_buy + n_sell + n_hold
        intent_balance = (n_buy - n_sell) / intent_total if intent_total else 0.0

        bull_w = float(w[sent > 0].sum())
        bear_w = float(w[sent < 0].sum())
        bull_bear = (bull_w - bear_w) / (bull_w + bear_w) if (bull_w + bear_w) > 0 else 0.0

        rows.append({
            "hour": hour,
            "tw_tweet_volume": float(len(grp)),
            "tw_sentiment_mean": round(float(sent.mean()), 4),
            "tw_influencer_sentiment": round(_weighted_mean(sent, w), 4),
            "tw_bull_bear_ratio": round(bull_bear, 4),
            "tw_fear_greed": round(_weighted_mean(grp["fear_greed"].to_numpy(float), w, 50.0), 2),
            "tw_intent_balance": round(intent_balance, 4),
            "tw_event_presence": round(float(grp["has_event"].mean()), 4),
            "tw_relevance_mean": round(float(grp["relevance"].mean()), 4),
            "tw_conviction_mean": round(_weighted_mean(grp["conviction"].to_numpy(float), w), 4),
        })

    out = pd.DataFrame(rows).set_index("hour").sort_index()
    out.index.name = "timestamp"

    # Trailing-window (causal) rolling features.
    vol = out["tw_tweet_volume"]
    roll = vol.rolling(window=24, min_periods=2)
    out["tw_volume_zscore_24h"] = ((vol - roll.mean()) / roll.std(ddof=0)).fillna(0.0).round(4)
    sent_series = out["tw_influencer_sentiment"]
    out["tw_sentiment_momentum"] = (
        sent_series - sent_series.rolling(window=6, min_periods=1).mean()
    ).fillna(0.0).round(4)

    return out[SIGNAL_COLUMNS]


def persist(frame: pd.DataFrame, path: Path | None = None) -> Path:
    """Merge ``frame`` into the on-disk signal parquet (upsert by hour)."""
    out_path = path or SIGNAL_PATH
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if frame is None or frame.empty:
        return out_path

    if out_path.exists():
        existing = pd.read_parquet(out_path)
        if not isinstance(existing.index, pd.DatetimeIndex):
            existing.index = pd.to_datetime(existing.index, utc=True)
        combined = pd.concat([existing, frame])
        combined = combined[~combined.index.duplicated(keep="last")].sort_index()
    else:
        combined = frame.sort_index()

    combined.to_parquet(out_path)
    logger.info("Persisted tweet signal: %d hourly rows -> %s", len(combined), out_path.name)
    return out_path
