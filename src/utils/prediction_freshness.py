"""Freshness checks for the scheduled prediction pipeline.

The GitHub Actions watchdog uses this module to decide whether the latest
prediction output is recent enough or whether it should dispatch a recovery
Predict run.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable

from src import PREDICTIONS_LOG

_PREDICTION_HEADER_RE = re.compile(
    r"^\[(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2} UTC)\]\s+--\s+Prediction Run #\d+"
)
_LOG_TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M UTC"


@dataclass(frozen=True)
class PredictionFreshness:
    """Freshness status for the newest prediction run in the log."""

    latest_timestamp: datetime | None
    age: timedelta | None
    max_age: timedelta

    @property
    def is_fresh(self) -> bool:
        return self.age is not None and self.age <= self.max_age


def _parse_prediction_timestamps(lines: Iterable[str]) -> list[datetime]:
    timestamps: list[datetime] = []
    for line in lines:
        match = _PREDICTION_HEADER_RE.match(line.strip())
        if not match:
            continue
        parsed = datetime.strptime(match.group("timestamp"), _LOG_TIMESTAMP_FORMAT)
        timestamps.append(parsed.replace(tzinfo=timezone.utc))
    return timestamps


def check_prediction_freshness(
    path: Path = PREDICTIONS_LOG,
    *,
    max_age: timedelta = timedelta(hours=3),
    now: datetime | None = None,
) -> PredictionFreshness:
    """Return freshness information for the newest prediction log entry."""

    reference_time = now or datetime.now(timezone.utc)
    if reference_time.tzinfo is None:
        reference_time = reference_time.replace(tzinfo=timezone.utc)
    else:
        reference_time = reference_time.astimezone(timezone.utc)

    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        return PredictionFreshness(None, None, max_age)

    timestamps = _parse_prediction_timestamps(lines)
    if not timestamps:
        return PredictionFreshness(None, None, max_age)

    latest = max(timestamps)
    return PredictionFreshness(latest, reference_time - latest, max_age)


def _format_timedelta(value: timedelta | None) -> str:
    if value is None:
        return "unknown"

    total_seconds = max(0, int(value.total_seconds()))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours}h {minutes}m {seconds}s"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check whether predictions.log is fresh.")
    parser.add_argument(
        "--path",
        type=Path,
        default=PREDICTIONS_LOG,
        help=f"Prediction log path (default: {PREDICTIONS_LOG})",
    )
    parser.add_argument(
        "--max-age-hours",
        type=float,
        default=3.0,
        help="Maximum allowed age in hours before the prediction output is stale.",
    )
    args = parser.parse_args(argv)

    max_age = timedelta(hours=args.max_age_hours)
    freshness = check_prediction_freshness(args.path, max_age=max_age)

    if freshness.latest_timestamp is None:
        print(f"No prediction run timestamp found in {args.path}.")
        return 1

    latest = freshness.latest_timestamp.strftime(_LOG_TIMESTAMP_FORMAT)
    age = _format_timedelta(freshness.age)
    max_age_text = _format_timedelta(max_age)
    if freshness.is_fresh:
        print(f"Latest prediction at {latest} is fresh (age {age}, max {max_age_text}).")
        return 0

    print(f"Latest prediction at {latest} is stale (age {age}, max {max_age_text}).")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
