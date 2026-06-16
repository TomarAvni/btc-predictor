"""Freshness checks for the scheduled prediction workflow."""

from __future__ import annotations

import argparse
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

PREDICTION_HEADER_RE = re.compile(
    r"^\[(\d{4}-\d{2}-\d{2}\s\d{2}:\d{2}\sUTC)\]"
)


def latest_prediction_time(path: Path) -> datetime | None:
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


def should_run_prediction(
    *,
    event_name: str,
    predictions_path: Path,
    threshold: timedelta,
    now: datetime | None = None,
) -> bool:
    """Manual runs always proceed; scheduled runs proceed only when stale."""
    if event_name == "workflow_dispatch":
        return True

    current_time = now or datetime.now(timezone.utc)
    if current_time.tzinfo is None:
        current_time = current_time.replace(tzinfo=timezone.utc)

    latest = latest_prediction_time(predictions_path)
    return latest is None or current_time - latest >= threshold


def _write_output(path: Path, should_run: bool) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(f"should_run={'true' if should_run else 'false'}\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Decide whether the predict workflow should do a full run."
    )
    parser.add_argument("--event-name", default="schedule")
    parser.add_argument("--predictions-path", default="predictions.log")
    parser.add_argument("--threshold-minutes", type=int, default=25)
    parser.add_argument("--output", help="GitHub Actions output file path")
    args = parser.parse_args()

    threshold = timedelta(minutes=args.threshold_minutes)
    should_run = should_run_prediction(
        event_name=args.event_name,
        predictions_path=Path(args.predictions_path),
        threshold=threshold,
    )

    status = "stale or manual" if should_run else "fresh"
    print(f"Prediction workflow freshness: {status}; should_run={should_run}")
    if args.output:
        _write_output(Path(args.output), should_run)


if __name__ == "__main__":
    main()
