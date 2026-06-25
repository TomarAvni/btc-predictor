"""Freshness checks for the rolling ``predictions.log`` file.

The Predict GitHub Actions schedule can be delayed or skipped by GitHub. This
module gives CI a small, dependency-free way to decide whether the last
successful prediction artifact is too old and needs a manual recovery dispatch.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

from src import PREDICTIONS_LOG

_RUN_HEADER = re.compile(
    r"^\[(?P<timestamp>\d{4}-\d{2}-\d{2}\s\d{2}:\d{2}\sUTC)\]\s*--\s*Prediction Run #(?P<run>\d+)"
)
_LOG_TS_FORMAT = "%Y-%m-%d %H:%M UTC"


@dataclass(frozen=True)
class PredictionFreshness:
    """Freshness summary for the latest prediction log entry."""

    log_path: Path
    latest_timestamp: datetime | None
    run_number: int | None
    now: datetime

    @property
    def age(self) -> timedelta | None:
        if self.latest_timestamp is None:
            return None
        return self.now - self.latest_timestamp

    def is_stale(self, max_age: timedelta) -> bool:
        age = self.age
        return age is None or age > max_age


def _ensure_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def parse_prediction_timestamp(value: str) -> datetime:
    """Parse a ``predictions.log`` UTC timestamp into an aware datetime."""

    return datetime.strptime(value, _LOG_TS_FORMAT).replace(tzinfo=timezone.utc)


def latest_prediction_run(log_path: Path = PREDICTIONS_LOG) -> tuple[datetime, int] | None:
    """Return the latest prediction timestamp/run number found in ``log_path``."""

    if not log_path.exists():
        return None

    latest: tuple[datetime, int] | None = None
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = _RUN_HEADER.search(line)
        if not match:
            continue
        timestamp = parse_prediction_timestamp(match.group("timestamp"))
        run_number = int(match.group("run"))
        if latest is None or timestamp >= latest[0]:
            latest = (timestamp, run_number)

    return latest


def get_prediction_freshness(
    log_path: Path = PREDICTIONS_LOG,
    *,
    now: datetime | None = None,
) -> PredictionFreshness:
    """Build a freshness summary for the latest prediction in ``log_path``."""

    current_time = _ensure_utc(now or datetime.now(timezone.utc))
    latest = latest_prediction_run(log_path)
    if latest is None:
        return PredictionFreshness(log_path=log_path, latest_timestamp=None, run_number=None, now=current_time)

    return PredictionFreshness(
        log_path=log_path,
        latest_timestamp=latest[0],
        run_number=latest[1],
        now=current_time,
    )


def _format_age(age: timedelta | None) -> str:
    if age is None:
        return "unknown"

    total_seconds = max(0, int(age.total_seconds()))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, _ = divmod(remainder, 60)
    return f"{hours}h {minutes}m"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Check whether predictions.log is fresh enough.")
    parser.add_argument("--path", type=Path, default=PREDICTIONS_LOG, help="Path to predictions.log")
    parser.add_argument(
        "--max-age-hours",
        type=float,
        default=3.0,
        help="Maximum allowed age for the latest prediction run",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    max_age = timedelta(hours=args.max_age_hours)
    freshness = get_prediction_freshness(args.path)

    if freshness.latest_timestamp is None:
        print(f"No prediction runs found in {freshness.log_path}.")
        return 1

    latest = freshness.latest_timestamp.strftime(_LOG_TS_FORMAT)
    age = _format_age(freshness.age)
    print(
        f"Latest prediction run #{freshness.run_number} at {latest} "
        f"(age {age}, max {args.max_age_hours:g}h)."
    )

    if freshness.is_stale(max_age):
        print("Prediction pipeline is stale.")
        return 1

    print("Prediction pipeline is fresh.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
