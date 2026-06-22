"""PromptFoo assertions for grounded tweet-reader outputs."""

from __future__ import annotations

import json
from typing import Any

FORECAST_TERMS = (
    "will hit",
    "will reach",
    "price target",
    "tomorrow",
    "next week",
    "guaranteed",
    "must hit",
)


def _fail(reason: str, score: float = 0.0) -> dict[str, Any]:
    return {"pass": False, "score": score, "reason": reason}


def _sign(value: float) -> str:
    if value > 0.15:
        return "positive"
    if value < -0.15:
        return "negative"
    return "neutral"


def _load_output(output: str) -> list[dict[str, Any]]:
    data = json.loads(output)
    if isinstance(data, dict):
        data = data.get("tweets", data.get("results", []))
    if not isinstance(data, list):
        raise ValueError("output must be a JSON list or object containing tweets/results")
    return data


def validate_grounded_reader(output: str, context: dict[str, Any]) -> dict[str, Any]:
    try:
        records = _load_output(output)
    except Exception as exc:
        return _fail(f"invalid JSON output: {exc}")

    vars_ = context.get("vars", {})
    expected = vars_.get("expected", {})
    input_ids = {str(item["id"]) for item in json.loads(vars_.get("tweets_json", "[]"))}
    by_id = {str(item.get("tweet_id")): item for item in records}

    for item in records:
        tid = str(item.get("tweet_id", ""))
        if tid not in input_ids:
            return _fail(f"unexpected tweet_id {tid!r}")
        cited = [str(c) for c in item.get("cited_tweet_ids", [])]
        if not cited:
            return _fail(f"{tid} has no cited_tweet_ids")
        if not set(cited).issubset(input_ids):
            return _fail(f"{tid} cites ids outside the batch: {cited}")
        for field in ("sentiment", "fear_greed", "conviction", "relevance"):
            if not isinstance(item.get(field), (int, float)):
                return _fail(f"{tid} field {field} must be numeric")
        if str(item.get("self_reported_intent")) not in {"buy", "sell", "hold", "none"}:
            return _fail(f"{tid} has invalid self_reported_intent")

    if expected.get("forbid_forecast_language", True):
        low = output.lower()
        for term in FORECAST_TERMS:
            if term in low:
                return _fail(f"output contains forecast language: {term}")

    checks = 0
    passed = 0
    for spec in expected.get("present", []):
        checks += 1
        tid = str(spec["id"])
        rec = by_id.get(tid)
        if rec is None:
            return _fail(f"missing expected tweet {tid}", passed / max(checks, 1))

        if "sentiment" in spec:
            checks += 1
            actual = _sign(float(rec.get("sentiment", 0.0)))
            if actual != spec["sentiment"]:
                return _fail(
                    f"{tid} sentiment {actual} != {spec['sentiment']}",
                    passed / max(checks, 1),
                )
            passed += 1

        if "intent" in spec:
            checks += 1
            actual = str(rec.get("self_reported_intent", "none"))
            if actual != spec["intent"]:
                return _fail(f"{tid} intent {actual} != {spec['intent']}", passed / max(checks, 1))
            passed += 1

        for event in spec.get("events", []):
            checks += 1
            if event not in (rec.get("event_flags") or []):
                return _fail(f"{tid} missing event {event}", passed / max(checks, 1))
            passed += 1

        if "relevance_min" in spec:
            checks += 1
            if float(rec.get("relevance", 0.0)) < float(spec["relevance_min"]):
                return _fail(f"{tid} relevance below minimum", passed / max(checks, 1))
            passed += 1

        if "relevance_max" in spec:
            checks += 1
            if float(rec.get("relevance", 0.0)) > float(spec["relevance_max"]):
                return _fail(f"{tid} relevance above maximum", passed / max(checks, 1))
            passed += 1

        passed += 1

    for tid in expected.get("absent", []):
        checks += 1
        if str(tid) in by_id:
            return _fail(f"tweet {tid} should have been omitted", passed / max(checks, 1))
        passed += 1

    return {
        "pass": True,
        "score": 1.0,
        "reason": f"passed {passed}/{max(checks, 1)} checks",
    }
