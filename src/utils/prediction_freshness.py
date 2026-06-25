"""Helpers for detecting stale prediction pipeline output."""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, Optional

_HEADER_RE = re.compile(
    r"^\[(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}) UTC\] -- Prediction Run #\d+",
    re.MULTILINE,
)
_LOG_TIME_FORMAT = "%Y-%m-%d %H:%M"


@dataclass(frozen=True)
class FreshnessResult:
    """Freshness details for the prediction log."""

    latest_timestamp: Optional[datetime]
    max_age: timedelta
    now: datetime
    path: Path

    @property
    def age(self) -> Optional[timedelta]:
        if self.latest_timestamp is None:
            return None
        return self.now - self.latest_timestamp

    @property
    def is_fresh(self) -> bool:
        age = self.age
        return age is not None and age <= self.max_age

    @property
    def reason(self) -> str:
        if self.latest_timestamp is None:
            return f"no prediction timestamps found in {self.path}"

        age = self.age
        if age is None:
            return f"no prediction timestamps found in {self.path}"

        age_minutes = age.total_seconds() / 60
        max_minutes = self.max_age.total_seconds() / 60
        state = "fresh" if self.is_fresh else "stale"
        latest = self.latest_timestamp.strftime("%Y-%m-%d %H:%M UTC")
        return (
            f"latest prediction at {latest} is {state}: "
            f"age {age_minutes:.1f} minutes, threshold {max_minutes:.1f} minutes"
        )


def _ensure_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def parse_prediction_timestamps(text: str) -> list[datetime]:
    """Return all prediction run timestamps found in log text."""
    timestamps: list[datetime] = []
    for match in _HEADER_RE.finditer(text):
        parsed = datetime.strptime(match.group("timestamp"), _LOG_TIME_FORMAT)
        timestamps.append(parsed.replace(tzinfo=timezone.utc))
    return timestamps


def latest_prediction_timestamp(timestamps: Iterable[datetime]) -> Optional[datetime]:
    """Return the newest prediction timestamp from an iterable."""
    latest: Optional[datetime] = None
    for timestamp in timestamps:
        normalized = _ensure_utc(timestamp)
        if latest is None or normalized > latest:
            latest = normalized
    return latest


def check_prediction_freshness(
    path: str | Path = "predictions.log",
    *,
    max_age: timedelta = timedelta(hours=3),
    now: Optional[datetime] = None,
) -> FreshnessResult:
    """Check whether the latest prediction log entry is within max_age."""
    log_path = Path(path)
    current_time = _ensure_utc(now or datetime.now(timezone.utc))

    try:
        text = log_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        latest = None
    else:
        latest = latest_prediction_timestamp(parse_prediction_timestamps(text))

    return FreshnessResult(
        latest_timestamp=latest,
        max_age=max_age,
        now=current_time,
        path=log_path,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fail when predictions.log has no recent prediction entries."
    )
    parser.add_argument(
        "--path",
        default="predictions.log",
        help="Path to the prediction log to inspect.",
    )
    parser.add_argument(
        "--max-age-hours",
        type=float,
        default=3.0,
        help="Maximum allowed age for the latest prediction entry.",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    result = check_prediction_freshness(
        args.path,
        max_age=timedelta(hours=args.max_age_hours),
    )
    print(result.reason)
    return 0 if result.is_fresh else 1


if __name__ == "__main__":
    raise SystemExit(main())
