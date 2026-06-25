"""Check whether the prediction log has been updated recently.

The GitHub scheduled workflow can be delayed or skipped under load.  This
module gives the watchdog workflow a deterministic way to decide whether the
pipeline is stale based on the newest prediction timestamp committed to
``predictions.log``.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

PREDICTION_HEADER_RE = re.compile(
    r"^\[(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}) UTC\] -- Prediction Run #\d+"
)
PREDICTION_TIME_FORMAT = "%Y-%m-%d %H:%M"


@dataclass(frozen=True)
class FreshnessStatus:
    """Freshness result for a prediction log."""

    path: Path
    latest_prediction_at: datetime | None
    checked_at: datetime
    max_age: timedelta

    @property
    def age(self) -> timedelta | None:
        if self.latest_prediction_at is None:
            return None
        return self.checked_at - self.latest_prediction_at

    @property
    def is_fresh(self) -> bool:
        age = self.age
        return age is not None and timedelta(0) <= age <= self.max_age

    @property
    def message(self) -> str:
        if self.latest_prediction_at is None:
            return f"No prediction entries found in {self.path}"

        age = self.age
        assert age is not None
        state = "fresh" if self.is_fresh else "stale"
        return (
            f"Latest prediction is {state}: "
            f"{self.latest_prediction_at.isoformat()} "
            f"({age.total_seconds() / 3600:.2f} hours old, "
            f"limit {self.max_age.total_seconds() / 3600:.2f} hours)"
        )


def _ensure_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def parse_prediction_timestamp(line: str) -> datetime | None:
    """Parse a ``predictions.log`` header line into an aware UTC datetime."""
    match = PREDICTION_HEADER_RE.match(line.strip())
    if match is None:
        return None

    parsed = datetime.strptime(match.group("timestamp"), PREDICTION_TIME_FORMAT)
    return parsed.replace(tzinfo=timezone.utc)


def latest_prediction_timestamp(path: str | Path) -> datetime | None:
    """Return the newest prediction timestamp found in ``path``."""
    log_path = Path(path)
    try:
        lines = log_path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        return None

    latest: datetime | None = None
    for line in lines:
        timestamp = parse_prediction_timestamp(line)
        if timestamp is None:
            continue
        if latest is None or timestamp > latest:
            latest = timestamp
    return latest


def check_freshness(
    path: str | Path = "predictions.log",
    *,
    max_age: timedelta = timedelta(hours=3),
    now: datetime | None = None,
) -> FreshnessStatus:
    """Check whether ``path`` has a recent prediction entry."""
    checked_at = _ensure_utc(now or datetime.now(timezone.utc))
    latest = latest_prediction_timestamp(path)
    return FreshnessStatus(
        path=Path(path),
        latest_prediction_at=latest,
        checked_at=checked_at,
        max_age=max_age,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--path",
        "--log-path",
        dest="path",
        default="predictions.log",
        help="Prediction log to inspect (default: predictions.log)",
    )
    parser.add_argument(
        "--max-age-hours",
        type=float,
        default=3.0,
        help="Maximum allowed age for the latest prediction entry",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    status = check_freshness(
        args.path,
        max_age=timedelta(hours=args.max_age_hours),
    )
    print(status.message)
    return 0 if status.is_fresh else 1


if __name__ == "__main__":
    sys.exit(main())
