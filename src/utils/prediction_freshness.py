"""Freshness checks for the scheduled prediction workflow."""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

JSONL_TIMESTAMP_KEY = "timestamp"
PREDICTION_HEADER_RE = re.compile(r"^\[(\d{4}-\d{2}-\d{2}\s\d{2}:\d{2}\sUTC)\]")


def _ensure_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _parse_jsonl_timestamp(value: object) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None

    text = value.strip().replace("Z", "+00:00")
    try:
        return _ensure_utc(datetime.fromisoformat(text))
    except ValueError:
        return None


def latest_prediction_jsonl_time(path: Path) -> datetime | None:
    """Return the newest UTC timestamp from prediction JSONL records."""
    if not path.exists():
        return None

    latest: datetime | None = None
    with path.open(encoding="utf-8") as f:
        for line in f:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            ts = _parse_jsonl_timestamp(record.get(JSONL_TIMESTAMP_KEY))
            if ts is not None and (latest is None or ts > latest):
                latest = ts
    return latest


def latest_prediction_log_time(path: Path) -> datetime | None:
    """Return the newest UTC timestamp from a predictions.log-style file."""
    if not path.exists():
        return None

    latest: datetime | None = None
    with path.open(encoding="utf-8") as f:
        for line in f:
            match = PREDICTION_HEADER_RE.match(line)
            if not match:
                continue
            ts = datetime.strptime(match.group(1), "%Y-%m-%d %H:%M UTC").replace(
                tzinfo=timezone.utc
            )
            if latest is None or ts > latest:
                latest = ts
    return latest


def latest_prediction_time(jsonl_path: Path, log_path: Path) -> datetime | None:
    """Return the newest prediction timestamp from machine or text logs."""
    candidates = [
        latest_prediction_jsonl_time(jsonl_path),
        latest_prediction_log_time(log_path),
    ]
    return max((ts for ts in candidates if ts is not None), default=None)


def should_run_prediction(
    *,
    event_name: str,
    jsonl_path: Path,
    log_path: Path,
    threshold: timedelta,
    now: datetime | None = None,
) -> bool:
    """Manual runs always proceed; scheduled runs proceed only when stale."""
    if event_name == "workflow_dispatch":
        return True

    current_time = _ensure_utc(now or datetime.now(timezone.utc))
    latest = latest_prediction_time(jsonl_path, log_path)
    return latest is None or current_time - latest >= threshold


def _write_output(path: Path, should_run: bool) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(f"should_run={'true' if should_run else 'false'}\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Decide whether the predict workflow should do a full run."
    )
    parser.add_argument("--event-name", default="schedule")
    parser.add_argument("--jsonl-path", default="data/predictions/predictions.jsonl")
    parser.add_argument("--log-path", default="predictions.log")
    parser.add_argument("--threshold-minutes", type=int, default=25)
    parser.add_argument("--output", help="GitHub Actions output file path")
    args = parser.parse_args()

    threshold = timedelta(minutes=args.threshold_minutes)
    should_run = should_run_prediction(
        event_name=args.event_name,
        jsonl_path=Path(args.jsonl_path),
        log_path=Path(args.log_path),
        threshold=threshold,
    )

    status = "stale or manual" if should_run else "fresh"
    print(f"Prediction workflow freshness: {status}; should_run={should_run}")
    if args.output:
        _write_output(Path(args.output), should_run)


if __name__ == "__main__":
    main()
