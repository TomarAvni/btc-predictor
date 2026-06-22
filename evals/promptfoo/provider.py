"""PromptFoo provider for the repo's grounded tweet reader.

The provider intentionally calls :class:`GroundedTweetReader` instead of a
separate prompt copy, so evals track the runtime extraction path used by
prediction runs.
"""

from __future__ import annotations

import json
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features.tweet_llm_reader import GroundedTweetReader  # noqa: E402


@contextmanager
def _maybe_disable_live_reader(allow_live: bool) -> Iterator[None]:
    saved = os.environ.get("LLM_API_KEY")
    if not allow_live:
        os.environ.pop("LLM_API_KEY", None)
    try:
        yield
    finally:
        if saved is not None:
            os.environ["LLM_API_KEY"] = saved


def _load_tweets(context: dict[str, Any]) -> list[dict[str, Any]]:
    raw = context.get("vars", {}).get("tweets_json", "[]")
    tweets = json.loads(raw) if isinstance(raw, str) else raw
    if not isinstance(tweets, list):
        raise ValueError("tweets_json must decode to a list")
    return tweets


def _to_frame(tweets: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for item in tweets:
        rows.append({
            "id": str(item["id"]),
            "text": str(item.get("text", "")),
        })
    return pd.DataFrame(rows)


def call_api(prompt: str, options: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    del prompt  # PromptFoo still requires a prompt; this provider uses vars.
    config = (options or {}).get("config", {})
    allow_live = bool(config.get("allow_live", False))

    with _maybe_disable_live_reader(allow_live):
        reader = GroundedTweetReader({"collectors": {"llm_reader": {"min_relevance": 0.0}}})
        output = [item.to_dict() for item in reader.read(_to_frame(_load_tweets(context)))]

    return {
        "output": json.dumps(output, sort_keys=True),
    }
