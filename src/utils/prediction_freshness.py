"""Freshness checks for the scheduled prediction pipeline.

The GitHub schedule can occasionally skip or delay runs. This module lets
automation check the committed prediction artifact and trigger recovery when
the latest successful prediction is too old.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

PREDICTION_HEADER_RE = re.compile(
    r"^\[(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2} UTC)\] -- Prediction Run #\d+",
    re.MULTILINE,
)
UTC_LOG_FORMAT = "%Y-%m-%d %H:%M UTC"


@dataclass(frozen=True)
class FreshnessResult:
    """Result of checking a prediction log for freshness."""

    is_fresh: bool
    latest_timestamp: datetime | None
    age: timedelta | None
    reason: str


def parse_prediction_timestamp(value: str) -> datetime:
    """Parse a prediction log timestamp into an aware UTC ``datetime``."""
    return datetime.strptime(value, UTC_LOG_FORMAT).replace(tzinfo=timezone.utc)


def latest_prediction_timestamp(log_text: str) -> datetime | None:
    """Return the newest prediction timestamp found in ``log_text``."""
    latest: datetime | None = None
    for match in PREDICTION_HEADER_RE.finditer(log_text):
        timestamp = parse_prediction_timestamp(match.group("timestamp"))
        if latest is None or timestamp > latest:
            latest = timestamp
    return latest


def check_prediction_freshness(
    path: str | Path = "predictions.log",
    *,
    max_age: timedelta = timedelta(hours=3),
    now: datetime | None = None,
) -> FreshnessResult:
    """Check whether the newest prediction log entry is within ``max_age``."""
    log_path = Path(path)
    if not log_path.exists():
        return FreshnessResult(False, None, None, f"{log_path} does not exist")

    try:
        latest = latest_prediction_timestamp(log_path.read_text(encoding="utf-8"))
    except OSError as exc:
        return FreshnessResult(False, None, None, f"could not read {log_path}: {exc}")

    if latest is None:
        return FreshnessResult(False, None, None, f"{log_path} has no prediction headers")

    current_time = now or datetime.now(timezone.utc)
    if current_time.tzinfo is None:
        current_time = current_time.replace(tzinfo=timezone.utc)
    else:
        current_time = current_time.astimezone(timezone.utc)

    age = current_time - latest
    if age < timedelta(0):
        return FreshnessResult(True, latest, age, "latest prediction timestamp is in the future")

    if age <= max_age:
        return FreshnessResult(True, latest, age, "latest prediction is fresh")

    return FreshnessResult(False, latest, age, "latest prediction is stale")


def _format_age(age: timedelta | None) -> str:
    if age is None:
        return "unknown"

    total_seconds = int(age.total_seconds())
    sign = "-" if total_seconds < 0 else ""
    total_seconds = abs(total_seconds)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{sign}{hours}h {minutes}m {seconds}s"


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(description="Check prediction log freshness")
    parser.add_argument(
        "--path",
        default="predictions.log",
        help="Path to predictions.log (default: predictions.log)",
    )
    parser.add_argument(
        "--max-age-hours",
        type=float,
        default=3.0,
        help="Maximum acceptable age in hours (default: 3)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns 0 when fresh and 1 when stale."""
    args = build_parser().parse_args(argv)
    result = check_prediction_freshness(
        args.path,
        max_age=timedelta(hours=args.max_age_hours),
    )

    if result.latest_timestamp is None:
        print(f"STALE: {result.reason}")
    else:
        latest = result.latest_timestamp.strftime(UTC_LOG_FORMAT)
        print(
            f"{'FRESH' if result.is_fresh else 'STALE'}: {result.reason}; "
            f"latest={latest}; age={_format_age(result.age)}"
        )

    return 0 if result.is_fresh else 1


if __name__ == "__main__":
    sys.exit(main())
