"""Freshness checks for the scheduled prediction pipeline."""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

from src import PREDICTIONS_LOG

_RUN_HEADER = re.compile(
    r"^\[(\d{4}-\d{2}-\d{2}\s\d{2}:\d{2}\sUTC)\]\s*--\s*Prediction Run #\d+"
)
_UTC_LOG_FMT = "%Y-%m-%d %H:%M UTC"


@dataclass(frozen=True)
class FreshnessStatus:
    """Status of the latest prediction log entry."""

    path: Path
    latest_run_at: datetime | None
    checked_at: datetime
    max_age: timedelta

    @property
    def age(self) -> timedelta | None:
        if self.latest_run_at is None:
            return None
        return self.checked_at - self.latest_run_at

    @property
    def is_fresh(self) -> bool:
        age = self.age
        return age is not None and age <= self.max_age


def _parse_run_timestamp(value: str) -> datetime:
    return datetime.strptime(value, _UTC_LOG_FMT).replace(tzinfo=timezone.utc)


def latest_prediction_run_at(path: Path = PREDICTIONS_LOG) -> datetime | None:
    """Return the latest prediction run timestamp found in ``path``."""
    if not path.exists():
        return None

    latest: datetime | None = None
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = _RUN_HEADER.search(line)
        if not match:
            continue
        run_at = _parse_run_timestamp(match.group(1))
        if latest is None or run_at > latest:
            latest = run_at
    return latest


def check_freshness(
    path: Path = PREDICTIONS_LOG,
    *,
    max_age: timedelta = timedelta(hours=3),
    now: datetime | None = None,
) -> FreshnessStatus:
    """Check whether the latest prediction run is within ``max_age``."""
    checked_at = now or datetime.now(timezone.utc)
    if checked_at.tzinfo is None:
        checked_at = checked_at.replace(tzinfo=timezone.utc)
    else:
        checked_at = checked_at.astimezone(timezone.utc)

    return FreshnessStatus(
        path=path,
        latest_run_at=latest_prediction_run_at(path),
        checked_at=checked_at,
        max_age=max_age,
    )


def _format_timedelta(value: timedelta) -> str:
    total_seconds = max(0, int(value.total_seconds()))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours}h {minutes}m {seconds}s"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Exit successfully when predictions.log has a fresh run."
    )
    parser.add_argument(
        "--log-path",
        type=Path,
        default=PREDICTIONS_LOG,
        help="Path to predictions.log (default: repository predictions.log)",
    )
    parser.add_argument(
        "--max-age-hours",
        type=float,
        default=3.0,
        help="Maximum allowed age in hours before the log is stale",
    )
    args = parser.parse_args(argv)

    status = check_freshness(
        args.log_path,
        max_age=timedelta(hours=args.max_age_hours),
    )

    if status.latest_run_at is None:
        print(f"No prediction runs found in {status.path}")
        return 1

    age = status.age or timedelta.max
    print(
        "Latest prediction run: "
        f"{status.latest_run_at.strftime(_UTC_LOG_FMT)} "
        f"(age {_format_timedelta(age)}, max {_format_timedelta(status.max_age)})"
    )
    return 0 if status.is_fresh else 1


if __name__ == "__main__":
    raise SystemExit(main())
