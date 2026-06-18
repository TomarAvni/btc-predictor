"""Helpers for checking whether prediction output is fresh enough."""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path


DEFAULT_PREDICTIONS_PATH = Path("predictions.log")
PREDICTION_HEADER_RE = re.compile(
    r"^\[(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}) UTC\] -- Prediction Run #\d+",
    re.MULTILINE,
)


@dataclass(frozen=True)
class FreshnessResult:
    """Result of checking a prediction log for recent output."""

    is_fresh: bool
    latest_timestamp: datetime | None
    age: timedelta | None
    reason: str


def parse_prediction_timestamp(value: str) -> datetime:
    """Parse a prediction log UTC timestamp into an aware datetime."""

    return datetime.strptime(value, "%Y-%m-%d %H:%M").replace(tzinfo=timezone.utc)


def latest_prediction_timestamp(text: str) -> datetime | None:
    """Return the most recent prediction timestamp in a log body."""

    latest: datetime | None = None
    for match in PREDICTION_HEADER_RE.finditer(text):
        timestamp = parse_prediction_timestamp(match.group("timestamp"))
        if latest is None or timestamp > latest:
            latest = timestamp
    return latest


def check_prediction_freshness(
    path: Path | str = DEFAULT_PREDICTIONS_PATH,
    *,
    max_age: timedelta = timedelta(hours=3),
    now: datetime | None = None,
) -> FreshnessResult:
    """Check whether the latest prediction entry is within ``max_age``."""

    log_path = Path(path)
    if not log_path.exists():
        return FreshnessResult(False, None, None, f"{log_path} does not exist")

    latest = latest_prediction_timestamp(log_path.read_text(encoding="utf-8", errors="replace"))
    if latest is None:
        return FreshnessResult(False, None, None, f"{log_path} has no prediction run headers")

    current_time = now or datetime.now(timezone.utc)
    if current_time.tzinfo is None:
        current_time = current_time.replace(tzinfo=timezone.utc)
    else:
        current_time = current_time.astimezone(timezone.utc)

    age = current_time - latest
    if age < timedelta(0):
        return FreshnessResult(True, latest, age, "latest prediction timestamp is in the future")

    is_fresh = age <= max_age
    reason = (
        f"latest prediction is {format_timedelta(age)} old"
        if is_fresh
        else f"latest prediction is stale: {format_timedelta(age)} old"
    )
    return FreshnessResult(is_fresh, latest, age, reason)


def format_timedelta(value: timedelta) -> str:
    """Format a timedelta for logs without fractional seconds."""

    total_seconds = int(abs(value.total_seconds()))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    sign = "-" if value.total_seconds() < 0 else ""
    if hours:
        return f"{sign}{hours}h {minutes}m {seconds}s"
    if minutes:
        return f"{sign}{minutes}m {seconds}s"
    return f"{sign}{seconds}s"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Check whether prediction output is fresh.")
    parser.add_argument("--path", default=str(DEFAULT_PREDICTIONS_PATH), help="Prediction log path")
    parser.add_argument(
        "--max-age-hours",
        type=float,
        default=3.0,
        help="Maximum acceptable age for the latest prediction entry",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = check_prediction_freshness(
        args.path,
        max_age=timedelta(hours=args.max_age_hours),
    )

    if result.latest_timestamp is not None:
        print(f"Latest prediction: {result.latest_timestamp.isoformat()}")
    print(result.reason)
    return 0 if result.is_fresh else 1


if __name__ == "__main__":
    sys.exit(main())
