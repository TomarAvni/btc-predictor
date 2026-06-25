"""Check whether the prediction pipeline has produced recent output."""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable

from src import PREDICTIONS_LOG

_HEADER_RE = re.compile(
    r"^\[(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2} UTC)\] -- Prediction Run #\d+"
)
_LOG_TS_FMT = "%Y-%m-%d %H:%M UTC"


@dataclass(frozen=True)
class FreshnessResult:
    """Outcome of checking the latest prediction log timestamp."""

    is_fresh: bool
    latest_timestamp: datetime | None
    age: timedelta | None
    reason: str


def _parse_log_timestamp(value: str) -> datetime:
    parsed = datetime.strptime(value, _LOG_TS_FMT)
    return parsed.replace(tzinfo=timezone.utc)


def iter_prediction_timestamps(lines: Iterable[str]) -> Iterable[datetime]:
    """Yield prediction run timestamps found in log header lines."""

    for line in lines:
        match = _HEADER_RE.match(line.strip())
        if not match:
            continue
        try:
            yield _parse_log_timestamp(match.group("timestamp"))
        except ValueError:
            continue


def latest_prediction_timestamp(log_path: Path = PREDICTIONS_LOG) -> datetime | None:
    """Return the newest timestamp recorded in ``predictions.log``."""

    try:
        with log_path.open("r", encoding="utf-8") as handle:
            return max(iter_prediction_timestamps(handle), default=None)
    except FileNotFoundError:
        return None


def check_prediction_freshness(
    *,
    log_path: Path = PREDICTIONS_LOG,
    max_age: timedelta = timedelta(hours=3),
    now: datetime | None = None,
) -> FreshnessResult:
    """Check whether the latest prediction log entry is newer than ``max_age``."""

    if now is None:
        now = datetime.now(timezone.utc)
    elif now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    else:
        now = now.astimezone(timezone.utc)

    latest = latest_prediction_timestamp(log_path)
    if latest is None:
        return FreshnessResult(
            is_fresh=False,
            latest_timestamp=None,
            age=None,
            reason=f"No prediction timestamps found in {log_path}",
        )

    age = now - latest
    if age <= max_age:
        return FreshnessResult(
            is_fresh=True,
            latest_timestamp=latest,
            age=age,
            reason=f"Latest prediction is {age} old",
        )

    return FreshnessResult(
        is_fresh=False,
        latest_timestamp=latest,
        age=age,
        reason=f"Latest prediction is stale: {age} old (max {max_age})",
    )


def _format_timedelta(value: timedelta | None) -> str:
    if value is None:
        return "unknown"
    total_seconds = int(value.total_seconds())
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours}h {minutes}m {seconds}s"


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--path",
        "--log-path",
        dest="path",
        type=Path,
        default=PREDICTIONS_LOG,
        help="Path to predictions.log",
    )
    parser.add_argument(
        "--max-age-hours",
        type=float,
        default=3.0,
        help="Maximum allowed age before the prediction pipeline is stale",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    result = check_prediction_freshness(
        log_path=args.path,
        max_age=timedelta(hours=args.max_age_hours),
    )

    latest = (
        result.latest_timestamp.strftime(_LOG_TS_FMT)
        if result.latest_timestamp is not None
        else "none"
    )
    print(f"fresh={str(result.is_fresh).lower()}")
    print(f"latest={latest}")
    print(f"age={_format_timedelta(result.age)}")
    print(result.reason)
    return 0 if result.is_fresh else 1


if __name__ == "__main__":
    raise SystemExit(main())
