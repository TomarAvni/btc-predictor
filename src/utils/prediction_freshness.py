"""Freshness checks for the GitHub Actions prediction pipeline."""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, Optional

_LOG_TIME_FORMAT = "%Y-%m-%d %H:%M UTC"
_RUN_HEADER_RE = re.compile(
    r"^\[(?P<timestamp>\d{4}-\d{2}-\d{2}\s\d{2}:\d{2}\sUTC)\]\s*--\s*Prediction Run #\d+"
)


@dataclass(frozen=True)
class FreshnessStatus:
    """Result of checking a prediction log against a maximum age."""

    path: Path
    now: datetime
    max_age: timedelta
    latest_prediction_at: Optional[datetime]

    @property
    def age(self) -> Optional[timedelta]:
        if self.latest_prediction_at is None:
            return None
        return self.now - self.latest_prediction_at

    @property
    def is_fresh(self) -> bool:
        return self.age is not None and self.age <= self.max_age

    @property
    def reason(self) -> str:
        if self.latest_prediction_at is None:
            return "no prediction run timestamp found"
        assert self.age is not None
        if self.is_fresh:
            return f"latest prediction is {self.age} old"
        return f"latest prediction is stale at {self.age} old"


def _ensure_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def parse_prediction_timestamps(lines: Iterable[str]) -> list[datetime]:
    """Parse UTC timestamps from prediction run headers."""

    timestamps: list[datetime] = []
    for line in lines:
        match = _RUN_HEADER_RE.match(line)
        if not match:
            continue
        parsed = datetime.strptime(match.group("timestamp"), _LOG_TIME_FORMAT)
        timestamps.append(parsed.replace(tzinfo=timezone.utc))
    return timestamps


def latest_prediction_timestamp(path: str | Path) -> Optional[datetime]:
    """Return the newest prediction run timestamp in ``path``, if any."""

    log_path = Path(path)
    if not log_path.exists():
        return None
    timestamps = parse_prediction_timestamps(
        log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    )
    if not timestamps:
        return None
    return max(timestamps)


def check_prediction_freshness(
    path: str | Path = "predictions.log",
    *,
    max_age: timedelta = timedelta(hours=3),
    now: Optional[datetime] = None,
) -> FreshnessStatus:
    """Check whether the latest prediction in ``path`` is within ``max_age``."""

    checked_at = _ensure_utc(now or datetime.now(timezone.utc))
    return FreshnessStatus(
        path=Path(path),
        now=checked_at,
        max_age=max_age,
        latest_prediction_at=latest_prediction_timestamp(path),
    )


def _format_dt(dt: Optional[datetime]) -> str:
    if dt is None:
        return "none"
    return dt.strftime(_LOG_TIME_FORMAT)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Exit successfully only when predictions.log is fresh."
    )
    parser.add_argument(
        "--path",
        default="predictions.log",
        help="Prediction log path to inspect (default: predictions.log)",
    )
    parser.add_argument(
        "--max-age-hours",
        type=float,
        default=3.0,
        help="Maximum allowed age in hours before the log is stale (default: 3)",
    )
    args = parser.parse_args(argv)

    status = check_prediction_freshness(
        args.path,
        max_age=timedelta(hours=args.max_age_hours),
    )
    state = "fresh" if status.is_fresh else "stale"
    print(
        f"Prediction log is {state}: {status.reason}; "
        f"latest={_format_dt(status.latest_prediction_at)}; "
        f"checked_at={status.now.strftime(_LOG_TIME_FORMAT)}; "
        f"max_age={status.max_age}."
    )
    return 0 if status.is_fresh else 1


if __name__ == "__main__":
    raise SystemExit(main())
