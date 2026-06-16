"""Run a single X/Twitter sentiment tick (ingest -> read -> aggregate -> forecast).

Standalone entry point for cron / manual use, separate from the main predict
cycle. Runs in mock mode (fixtures) when ``TWITTERAPI_KEY`` / ``LLM_API_KEY``
are unset, so it works offline with no keys.

Usage:
    python scripts/run_sentiment_tick.py
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from main import load_config  # noqa: E402
from src.engine.sentiment_manager import SentimentManager  # noqa: E402


def main() -> int:
    cfg = load_config()
    mgr = SentimentManager(config=cfg)
    result = asyncio.run(mgr.run_tick())

    state = result.get("state", {})
    print(f"Mode: {'MOCK' if result.get('mock') else 'LIVE'}")
    print(f"Tweets: {result.get('n_tweets')} | grounded extractions: {result.get('n_extractions')}")
    print(
        f"Crowd mood: {state.get('mood'):+.3f} | fear/greed: {state.get('fear_greed')} "
        f"({state.get('fear_greed_label')}) | positioning: {state.get('positioning'):+.3f}"
    )
    print(f"llm_direct forecast ({len(result.get('predictions', []))} horizons):")
    for p in result.get("predictions", [])[:8]:
        print(
            f"  {p['timeframe']:>4} | {p['direction']:<4} | {p['magnitude']:+.2f}% "
            f"| conf {p['confidence']}%"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
