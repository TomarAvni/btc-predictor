"""Check whether the prediction log has been updated recently.

The GitHub Actions watchdog uses this module as a small, dependency-free
freshness probe before deciding whether to dispatch a replacement Predict run.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, Optional


PREDICTION_HEADER_RE = re.compile(
    r"^\[(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}) UTC\] -- Prediction Run #\d+",
    re.MULTILINE,
)
UTC_LOG_FORMAT = "%Y-%m-%d %H:%M"


@dataclass(frozen=True)
class FreshnessResult:
    """Freshness status for a prediction log."""

    latest: Optional[datetime]
    max_age: timedelta
    now: datetime
    path: Path

    @property
    def age(self) -> Optional[timedelta]:
        if self.latest is None:
            return None
        return self.now - self.latest

    @property
    def is_fresh(self) -> bool:
        age = self.age
        return age is not None and age <= self.max_age

    def message(self) -> str:
        if self.latest is None:
            return f"No prediction runs found in {self.path}."
        age = self.age
        assert age is not None
        minutes = int(age.total_seconds() // 60)
        status = "fresh" if self.is_fresh else "stale"
        return (
            f"Latest prediction run: {self.latest.strftime('%Y-%m-%d %H:%M UTC')} "
            f"({minutes} minutes old, max {int(self.max_age.total_seconds() // 60)} "
            f"minutes) -- {status}."
        )


def _ensure_utc(dt: Optional[datetime]) -> datetime:
    if dt is None:
        return datetime.now(timezone.utc)
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def parse_prediction_timestamps(text: str) -> Iterable[datetime]:
    """Yield UTC datetimes from prediction run headers in log text."""
    for match in PREDICTION_HEADER_RE.finditer(text):
        parsed = datetime.strptime(match.group("timestamp"), UTC_LOG_FORMAT)
        yield parsed.replace(tzinfo=timezone.utc)


def latest_prediction_time(path: Path) -> Optional[datetime]:
    """Return the newest prediction timestamp in ``path``, or ``None``."""
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None

    latest: Optional[datetime] = None
    for timestamp in parse_prediction_timestamps(text):
        if latest is None or timestamp > latest:
            latest = timestamp
    return latest


def check_freshness(path: Path, *, max_age: timedelta, now: Optional[datetime] = None) -> FreshnessResult:
    """Check whether ``path`` contains a prediction run newer than ``max_age``."""
    current_time = _ensure_utc(now)
    return FreshnessResult(
        latest=latest_prediction_time(path),
        max_age=max_age,
        now=current_time,
        path=path,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--path",
        type=Path,
        default=Path("predictions.log"),
        help="Prediction log path to inspect (default: predictions.log).",
    )
    parser.add_argument(
        "--max-age-hours",
        type=float,
        default=3.0,
        help="Maximum acceptable age in hours before the log is considered stale.",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    result = check_freshness(
        args.path,
        max_age=timedelta(hours=args.max_age_hours),
    )
    print(result.message())
    return 0 if result.is_fresh else 1


if __name__ == "__main__":
    sys.exit(main())
