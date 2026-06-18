"""Check whether the latest prediction log entry is recent enough."""

from __future__ import annotations

import argparse
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

PREDICTION_HEADER_RE = re.compile(
    r"\[(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2} UTC)\]\s+--\s+Prediction Run #\d+"
)
PREDICTION_TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M UTC"


def parse_prediction_timestamp(value: str) -> datetime:
    """Parse the timestamp format written in ``predictions.log`` headers."""
    parsed = datetime.strptime(value, PREDICTION_TIMESTAMP_FORMAT)
    return parsed.replace(tzinfo=timezone.utc)


def latest_prediction_timestamp(log_path: Path | str) -> datetime | None:
    """Return the most recent prediction timestamp found in a text log."""
    path = Path(log_path)
    try:
        content = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None

    latest: datetime | None = None
    for match in PREDICTION_HEADER_RE.finditer(content):
        try:
            timestamp = parse_prediction_timestamp(match.group("timestamp"))
        except ValueError:
            continue
        if latest is None or timestamp > latest:
            latest = timestamp
    return latest


def prediction_age(
    log_path: Path | str,
    *,
    now: datetime | None = None,
) -> timedelta | None:
    """Return the age of the latest prediction, or ``None`` if none exists."""
    latest = latest_prediction_timestamp(log_path)
    if latest is None:
        return None

    current = now or datetime.now(timezone.utc)
    if current.tzinfo is None:
        current = current.replace(tzinfo=timezone.utc)
    else:
        current = current.astimezone(timezone.utc)
    return current - latest


def is_prediction_fresh(
    log_path: Path | str,
    *,
    max_age: timedelta,
    now: datetime | None = None,
) -> bool:
    """Return true when the latest prediction is no older than ``max_age``."""
    age = prediction_age(log_path, now=now)
    return age is not None and age <= max_age


def _format_age(age: timedelta) -> str:
    total_seconds = max(0, int(age.total_seconds()))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, _ = divmod(remainder, 60)
    return f"{hours}h {minutes}m"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Fail when predictions.log has no recent prediction run."
    )
    parser.add_argument(
        "--log-path",
        "--path",
        default="predictions.log",
        help="Path to the text prediction log (default: predictions.log)",
    )
    parser.add_argument(
        "--max-age-hours",
        type=float,
        default=3.0,
        help="Maximum allowed age for the latest prediction (default: 3)",
    )
    args = parser.parse_args(argv)

    latest = latest_prediction_timestamp(args.log_path)
    if latest is None:
        print(f"::warning::No prediction entries found in {args.log_path}")
        return 1

    max_age = timedelta(hours=args.max_age_hours)
    now = datetime.now(timezone.utc)
    age = now - latest
    if age <= max_age:
        print(
            "Latest prediction is fresh: "
            f"{latest.strftime(PREDICTION_TIMESTAMP_FORMAT)} ({_format_age(age)} old)"
        )
        return 0

    print(
        "::warning::Latest prediction is stale: "
        f"{latest.strftime(PREDICTION_TIMESTAMP_FORMAT)} ({_format_age(age)} old, "
        f"limit {args.max_age_hours:g}h)"
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
