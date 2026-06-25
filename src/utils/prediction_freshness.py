"""Utilities for detecting stale prediction pipeline output.

GitHub scheduled workflows can be delayed or skipped under load. The watchdog
workflow uses this module to decide whether the repository's latest prediction
artifact is old enough to warrant dispatching a recovery run.
"""

from __future__ import annotations

import argparse
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, Optional

PREDICTION_HEADER_RE = re.compile(
    r"^\[(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}) UTC\] -- Prediction Run #\d+"
)
PREDICTION_TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M"


def parse_prediction_timestamp(line: str) -> Optional[datetime]:
    """Parse a prediction-run header timestamp as an aware UTC datetime."""
    match = PREDICTION_HEADER_RE.match(line.strip())
    if not match:
        return None

    parsed = datetime.strptime(match.group("timestamp"), PREDICTION_TIMESTAMP_FORMAT)
    return parsed.replace(tzinfo=timezone.utc)


def latest_prediction_timestamp(lines: Iterable[str]) -> Optional[datetime]:
    """Return the most recent prediction-run timestamp in ``lines``."""
    latest: Optional[datetime] = None
    for line in lines:
        timestamp = parse_prediction_timestamp(line)
        if timestamp is not None and (latest is None or timestamp > latest):
            latest = timestamp
    return latest


def latest_prediction_timestamp_from_file(path: Path) -> Optional[datetime]:
    """Read ``path`` and return the latest prediction timestamp, if present."""
    try:
        with path.open("r", encoding="utf-8") as handle:
            return latest_prediction_timestamp(handle)
    except FileNotFoundError:
        return None


def prediction_age(
    latest_timestamp: Optional[datetime], *, now: Optional[datetime] = None
) -> Optional[timedelta]:
    """Return age of the latest prediction, or ``None`` if no timestamp exists."""
    if latest_timestamp is None:
        return None

    current = now or datetime.now(timezone.utc)
    if current.tzinfo is None:
        current = current.replace(tzinfo=timezone.utc)
    return current.astimezone(timezone.utc) - latest_timestamp.astimezone(timezone.utc)


def is_prediction_stale(
    path: Path,
    *,
    max_age: timedelta,
    now: Optional[datetime] = None,
) -> bool:
    """Return ``True`` when ``path`` is missing, unparseable, or too old."""
    age = prediction_age(latest_prediction_timestamp_from_file(path), now=now)
    return age is None or age > max_age


def _format_age(age: Optional[timedelta]) -> str:
    if age is None:
        return "unknown"
    total_seconds = max(0, int(age.total_seconds()))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, _ = divmod(remainder, 60)
    return f"{hours}h {minutes}m"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Check whether predictions.log is older than a configured threshold."
    )
    parser.add_argument(
        "--path",
        default="predictions.log",
        type=Path,
        help="Path to the prediction log to inspect.",
    )
    parser.add_argument(
        "--max-age-hours",
        default=3.0,
        type=float,
        help="Maximum acceptable prediction age in hours.",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    max_age = timedelta(hours=args.max_age_hours)
    latest = latest_prediction_timestamp_from_file(args.path)
    age = prediction_age(latest)

    if age is None:
        print(f"stale: no prediction timestamp found in {args.path}")
        return 1

    latest_text = latest.isoformat()
    age_text = _format_age(age)
    if age > max_age:
        print(f"stale: latest prediction at {latest_text} is {age_text} old")
        return 1

    print(f"fresh: latest prediction at {latest_text} is {age_text} old")
    return 0


if __name__ == "__main__":
    sys.exit(main())
