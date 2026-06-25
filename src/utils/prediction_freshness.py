"""Freshness checks for the prediction pipeline output log.

The scheduled GitHub Actions cron can occasionally skip or delay runs. This
module gives automation a deterministic signal from the committed
``predictions.log`` file so a watchdog workflow can recover stale pipelines.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

_PREDICTION_HEADER_RE = re.compile(
    r"^\[(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2} UTC)\] -- Prediction Run #\d+",
    re.MULTILINE,
)
_LOG_TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M UTC"


@dataclass(frozen=True)
class PredictionFreshness:
    """Freshness result for the latest prediction log entry."""

    latest_run_at: datetime | None
    max_age: timedelta
    now: datetime

    @property
    def age(self) -> timedelta | None:
        if self.latest_run_at is None:
            return None
        return self.now - self.latest_run_at

    @property
    def is_fresh(self) -> bool:
        age = self.age
        return age is not None and age <= self.max_age


def parse_prediction_timestamp(value: str) -> datetime:
    """Parse a ``predictions.log`` UTC timestamp into an aware datetime."""

    return datetime.strptime(value, _LOG_TIMESTAMP_FORMAT).replace(tzinfo=timezone.utc)


def latest_prediction_run_at(log_path: str | Path = "predictions.log") -> datetime | None:
    """Return the newest prediction run timestamp from ``log_path``.

    ``None`` is returned when the file is missing, unreadable, or does not
    contain any valid prediction headers.
    """

    path = Path(log_path)
    try:
        content = path.read_text(encoding="utf-8")
    except OSError:
        return None

    latest: datetime | None = None
    for match in _PREDICTION_HEADER_RE.finditer(content):
        try:
            run_at = parse_prediction_timestamp(match.group("timestamp"))
        except ValueError:
            continue
        if latest is None or run_at > latest:
            latest = run_at
    return latest


def check_prediction_freshness(
    log_path: str | Path = "predictions.log",
    *,
    max_age: timedelta = timedelta(hours=3),
    now: datetime | None = None,
) -> PredictionFreshness:
    """Check whether the prediction pipeline has produced recent output."""

    current_time = now or datetime.now(timezone.utc)
    if current_time.tzinfo is None:
        current_time = current_time.replace(tzinfo=timezone.utc)
    else:
        current_time = current_time.astimezone(timezone.utc)

    latest = latest_prediction_run_at(log_path)
    return PredictionFreshness(
        latest_run_at=latest,
        max_age=max_age,
        now=current_time,
    )


def _format_age(age: timedelta | None) -> str:
    if age is None:
        return "unknown"

    total_seconds = max(0, int(age.total_seconds()))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, _ = divmod(remainder, 60)
    return f"{hours}h {minutes}m"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Check whether predictions.log has a recent prediction run."
    )
    parser.add_argument(
        "--log-path",
        default="predictions.log",
        help="Path to the prediction text log (default: predictions.log)",
    )
    parser.add_argument(
        "--max-age-hours",
        type=float,
        default=3.0,
        help="Maximum acceptable age for the latest run (default: 3)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = check_prediction_freshness(
        args.log_path,
        max_age=timedelta(hours=args.max_age_hours),
    )

    if result.latest_run_at is None:
        print(f"Prediction log is stale: no prediction runs found in {args.log_path}.")
        return 1

    latest = result.latest_run_at.strftime(_LOG_TIMESTAMP_FORMAT)
    age = _format_age(result.age)
    max_age = _format_age(result.max_age)
    if result.is_fresh:
        print(f"Prediction log is fresh: latest run at {latest} (age {age}, max {max_age}).")
        return 0

    print(f"Prediction log is stale: latest run at {latest} (age {age}, max {max_age}).")
    return 1


if __name__ == "__main__":
    sys.exit(main())
