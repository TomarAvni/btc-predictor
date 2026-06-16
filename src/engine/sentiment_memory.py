"""External memory for the sentiment module.

Knowledge lives in a store, never in agent context windows. Three persistent
artifacts:

* the hourly signal time-series (``twitter_llm_signal.parquet``),
* a notable-events index (high-impact tweets) for lightweight retrieval (RAG-lite),
* a rolling market-state document the manager updates each cycle.

Reader/manager components are stateless and query this store.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from src import DATA_DIR
from src.features import tweet_aggregator
from src.features.tweet_llm_reader import TweetExtraction
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

HISTORY_DIR = DATA_DIR / "history"
EVENTS_PATH = HISTORY_DIR / "twitter_notable_events.json"
STATE_PATH = HISTORY_DIR / "twitter_market_state.json"
MAX_NOTABLE_EVENTS = 200


class SentimentMemory:
    """Persistent, queryable store for the sentiment module."""

    def __init__(
        self,
        signal_path: Path | None = None,
        events_path: Path | None = None,
        state_path: Path | None = None,
    ) -> None:
        self.signal_path = signal_path or tweet_aggregator.SIGNAL_PATH
        self.events_path = events_path or EVENTS_PATH
        self.state_path = state_path or STATE_PATH

    # -- signal time-series -----------------------------------------------------

    def add_signal(self, frame: pd.DataFrame) -> None:
        tweet_aggregator.persist(frame, self.signal_path)

    def load_signal(self) -> pd.DataFrame:
        if not self.signal_path.exists():
            return pd.DataFrame(columns=tweet_aggregator.SIGNAL_COLUMNS)
        df = pd.read_parquet(self.signal_path)
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index, utc=True)
        return df.sort_index()

    def latest_signal(self) -> dict[str, Any]:
        df = self.load_signal()
        if df.empty:
            return {}
        row = df.iloc[-1].to_dict()
        row["timestamp"] = df.index[-1].isoformat()
        return row

    # -- notable-events index (RAG-lite) ---------------------------------------

    def record_events(
        self,
        tweets: pd.DataFrame,
        extractions: Iterable[TweetExtraction],
        top_n: int = 10,
    ) -> list[dict[str, Any]]:
        """Store the highest-impact event-bearing tweets for later retrieval."""
        if tweets is None or tweets.empty:
            return []
        ext_by_id = {e.tweet_id: e for e in extractions}
        candidates: list[dict[str, Any]] = []
        for ts, row in tweets.iterrows():
            ext = ext_by_id.get(str(row["id"]))
            if ext is None or not ext.event_flags:
                continue
            candidates.append({
                "tweet_id": str(row["id"]),
                "timestamp": ts.isoformat(),
                "author": row.get("author"),
                "text": row.get("text"),
                "event_flags": ext.event_flags,
                "sentiment": ext.sentiment,
                "engagement_weight": float(row.get("engagement_weight", 0.0)),
            })
        candidates.sort(key=lambda c: c["engagement_weight"], reverse=True)
        top = candidates[:top_n]
        if top:
            self._append_events(top)
        return top

    def retrieve_notable(self, limit: int = 10) -> list[dict[str, Any]]:
        """Return the most recent notable events (newest first)."""
        events = self._load_events()
        return sorted(events, key=lambda e: e["timestamp"], reverse=True)[:limit]

    def _load_events(self) -> list[dict[str, Any]]:
        if not self.events_path.exists():
            return []
        try:
            return json.loads(self.events_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return []

    def _append_events(self, new_events: list[dict[str, Any]]) -> None:
        events = self._load_events()
        seen = {e["tweet_id"] for e in events}
        events.extend(e for e in new_events if e["tweet_id"] not in seen)
        events = sorted(events, key=lambda e: e["timestamp"])[-MAX_NOTABLE_EVENTS:]
        self.events_path.parent.mkdir(parents=True, exist_ok=True)
        self.events_path.write_text(json.dumps(events, indent=2), encoding="utf-8")

    # -- rolling market-state doc ----------------------------------------------

    def update_state(self, state: dict[str, Any]) -> None:
        state = {**state, "updated_at": pd.Timestamp.now(tz="UTC").isoformat()}
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")

    def get_state(self) -> dict[str, Any]:
        if not self.state_path.exists():
            return {}
        try:
            return json.loads(self.state_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}
