"""Check whether prediction output has been updated recently enough."""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional, Sequence


DEFAULT_LOG_PATH = Path("predictions.log")
DEFAULT_MAX_AGE_HOURS = 3.0
_RUN_HEADER_RE = re.compile(r"^\[(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}) UTC\] -- Prediction Run #\d+")
_RUN_HEADER_FORMAT = "%Y-%m-%d %H:%M"


@dataclass(frozen=True)
class FreshnessResult:
    """Result of checking the latest prediction timestamp."""

    is_fresh: bool
    latest_run_at: Optional[datetime]
    max_age: timedelta
    checked_at: datetime
    reason: str

    @property
    def age(self) -> Optional[timedelta]:
        if self.latest_run_at is None:
            return None
        return self.checked_at - self.latest_run_at


def _ensure_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def parse_prediction_run_timestamp(line: str) -> Optional[datetime]:
    """Parse a prediction run header from ``predictions.log``."""

    match = _RUN_HEADER_RE.match(line.strip())
    if not match:
        return None

    parsed = datetime.strptime(match.group("timestamp"), _RUN_HEADER_FORMAT)
    return parsed.replace(tzinfo=timezone.utc)


def latest_prediction_run_at(log_path: Path = DEFAULT_LOG_PATH) -> Optional[datetime]:
    """Return the newest prediction run timestamp in a log file, if present."""

    try:
        lines = log_path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        return None

    latest: Optional[datetime] = None
    for line in lines:
        run_at = parse_prediction_run_timestamp(line)
        if run_at is not None and (latest is None or run_at > latest):
            latest = run_at
    return latest


def check_prediction_freshness(
    log_path: Path = DEFAULT_LOG_PATH,
    *,
    max_age: timedelta = timedelta(hours=DEFAULT_MAX_AGE_HOURS),
    now: Optional[datetime] = None,
) -> FreshnessResult:
    """Check whether the latest prediction run is within ``max_age``."""

    checked_at = _ensure_utc(now or datetime.now(timezone.utc))
    latest = latest_prediction_run_at(log_path)
    if latest is None:
        return FreshnessResult(
            is_fresh=False,
            latest_run_at=None,
            max_age=max_age,
            checked_at=checked_at,
            reason=f"No prediction run timestamp found in {log_path}",
        )

    age = checked_at - latest
    if age < timedelta(0):
        return FreshnessResult(
            is_fresh=True,
            latest_run_at=latest,
            max_age=max_age,
            checked_at=checked_at,
            reason="Latest prediction timestamp is in the future",
        )

    if age <= max_age:
        return FreshnessResult(
            is_fresh=True,
            latest_run_at=latest,
            max_age=max_age,
            checked_at=checked_at,
            reason="Latest prediction is fresh",
        )

    return FreshnessResult(
        is_fresh=False,
        latest_run_at=latest,
        max_age=max_age,
        checked_at=checked_at,
        reason="Latest prediction is stale",
    )


def _format_timedelta(value: timedelta) -> str:
    total_seconds = max(0, int(value.total_seconds()))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours}h {minutes}m {seconds}s"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--path",
        "--log-path",
        dest="log_path",
        default=str(DEFAULT_LOG_PATH),
        help="Path to predictions.log (default: predictions.log)",
    )
    parser.add_argument(
        "--max-age-hours",
        type=float,
        default=DEFAULT_MAX_AGE_HOURS,
        help=f"Maximum acceptable age in hours (default: {DEFAULT_MAX_AGE_HOURS:g})",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.max_age_hours <= 0:
        parser.error("--max-age-hours must be greater than zero")

    result = check_prediction_freshness(
        Path(args.log_path),
        max_age=timedelta(hours=args.max_age_hours),
    )

    if result.latest_run_at is None:
        print(f"STALE: {result.reason}")
    else:
        age = result.age or timedelta(0)
        print(
            f"{'FRESH' if result.is_fresh else 'STALE'}: latest prediction at "
            f"{result.latest_run_at.strftime('%Y-%m-%d %H:%M UTC')} "
            f"(age {_format_timedelta(age)}, max {_format_timedelta(result.max_age)})"
        )

    return 0 if result.is_fresh else 1


if __name__ == "__main__":
    sys.exit(main())
