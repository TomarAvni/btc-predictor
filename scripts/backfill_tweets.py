"""One-time historical tweet backfill for training the calibrated/blended models.

Walks a date range day-by-day, pulls historical tweets (live) or replays the
fixtures across the range (mock), runs the SAME grounded reader + aggregator used
live (train/serve parity via the reader VERSION), and writes the hourly signal to
``data/history/twitter_llm_signal.parquet``.

Only ``llm_calibrated`` / ``blended`` need this history; ``llm_direct`` does not
train and is validated forward-only (no hindsight backtest).

Usage:
    python scripts/backfill_tweets.py --days 180
    python scripts/backfill_tweets.py --start 2026-01-01 --end 2026-06-01
"""

from __future__ import annotations

import argparse
import sys
from datetime import timedelta
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from main import load_config  # noqa: E402
from src.collectors.twitter_sentiment import TwitterSentimentCollector  # noqa: E402
from src.engine.sentiment_memory import SentimentMemory  # noqa: E402
from src.features import tweet_aggregator  # noqa: E402
from src.features.tweet_llm_reader import GroundedTweetReader  # noqa: E402
from src.utils.logger import setup_logger  # noqa: E402

logger = setup_logger("backfill_tweets")


def _daterange(start: pd.Timestamp, end: pd.Timestamp):
    day = start
    while day < end:
        yield day, day + timedelta(days=1)
        day += timedelta(days=1)


def _mock_day(collector: TwitterSentimentCollector, day_start: pd.Timestamp) -> pd.DataFrame:
    """Replay fixtures onto a given day so a demo history can be produced offline."""
    import asyncio

    tweets = asyncio.run(collector.collect())
    if tweets.empty:
        return tweets
    shifted = tweets.copy()
    # Re-anchor the fixtures' time-of-day onto day_start (keeps the hourly spread).
    base = shifted.index.min().normalize()
    shifted.index = shifted.index + (day_start.normalize() - base)
    shifted.index.name = "timestamp"
    return shifted


def main() -> int:
    parser = argparse.ArgumentParser(description="Historical tweet backfill")
    parser.add_argument("--days", type=int, default=180)
    parser.add_argument("--start", type=str, default=None)
    parser.add_argument("--end", type=str, default=None)
    args = parser.parse_args()

    end = pd.Timestamp(args.end, tz="UTC") if args.end else pd.Timestamp.now(tz="UTC").normalize()
    start = (
        pd.Timestamp(args.start, tz="UTC") if args.start
        else end - pd.Timedelta(days=args.days)
    )

    cfg = load_config()
    collector = TwitterSentimentCollector(cfg)
    reader = GroundedTweetReader(cfg)
    memory = SentimentMemory()
    min_rel = float(cfg.get("collectors", {}).get("llm_reader", {}).get("min_relevance", 0.15))

    logger.info(
        "Backfill %s -> %s (%s mode, reader %s)",
        start.date(), end.date(),
        "MOCK" if collector.is_mock else "LIVE", reader.version,
    )

    total_hours = 0
    for day_start, _day_end in _daterange(start, end):
        if collector.is_mock:
            tweets = _mock_day(collector, day_start)
        else:  # pragma: no cover - live path needs network + key
            tweets = collector.fetch_historical_day(day_start)  # type: ignore[attr-defined]
        if tweets is None or tweets.empty:
            continue
        extractions = reader.read(tweets)
        signal = tweet_aggregator.aggregate(tweets, extractions, min_rel)
        if not signal.empty:
            memory.add_signal(signal)
            total_hours += len(signal)

    logger.info("Backfill complete: %d hourly rows in %s", total_hours, memory.signal_path)
    print(f"Backfilled {total_hours} hourly signal rows -> {memory.signal_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
