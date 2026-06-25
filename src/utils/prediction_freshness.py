"""Freshness checks for the committed prediction log.

GitHub scheduled workflows can be delayed or dropped under load. This module
lets a watchdog workflow decide whether the predictor has stopped producing
committed runs and should be dispatched manually.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

from src import PREDICTIONS_LOG

_RUN_HEADER = re.compile(
    r"^\[(?P<timestamp>\d{4}-\d{2}-\d{2}\s\d{2}:\d{2}\sUTC)\]\s*--\s*"
    r"Prediction Run #(?P<run_number>\d+)"
)
_LOG_TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M UTC"


@dataclass(frozen=True)
class PredictionFreshness:
    """Status returned by :func:`check_prediction_freshness`."""

    path: Path
    max_age: timedelta
    now: datetime
    latest_run_at: datetime | None
    latest_run_number: int | None
    reason: str

    @property
    def age(self) -> timedelta | None:
        if self.latest_run_at is None:
            return None
        return self.now - self.latest_run_at

    @property
    def is_fresh(self) -> bool:
        age = self.age
        return age is not None and timedelta(0) <= age <= self.max_age


def _ensure_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _parse_log_timestamp(value: str) -> datetime:
    return datetime.strptime(value, _LOG_TIMESTAMP_FORMAT).replace(tzinfo=timezone.utc)


def iter_prediction_headers(path: Path) -> list[tuple[datetime, int]]:
    """Return all parseable prediction headers from *path*."""

    headers: list[tuple[datetime, int]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = _RUN_HEADER.search(line)
        if not match:
            continue
        headers.append((
            _parse_log_timestamp(match.group("timestamp")),
            int(match.group("run_number")),
        ))
    return headers


def check_prediction_freshness(
    path: Path | str = PREDICTIONS_LOG,
    *,
    max_age: timedelta = timedelta(hours=1),
    now: datetime | None = None,
) -> PredictionFreshness:
    """Check whether *path* contains a recent prediction run."""

    log_path = Path(path)
    current_time = _ensure_utc(now or datetime.now(timezone.utc))

    if not log_path.exists():
        return PredictionFreshness(
            path=log_path,
            max_age=max_age,
            now=current_time,
            latest_run_at=None,
            latest_run_number=None,
            reason=f"{log_path} does not exist",
        )

    headers = iter_prediction_headers(log_path)
    if not headers:
        return PredictionFreshness(
            path=log_path,
            max_age=max_age,
            now=current_time,
            latest_run_at=None,
            latest_run_number=None,
            reason=f"{log_path} contains no prediction run headers",
        )

    latest_run_at, latest_run_number = max(headers, key=lambda item: (item[0], item[1]))
    status = PredictionFreshness(
        path=log_path,
        max_age=max_age,
        now=current_time,
        latest_run_at=latest_run_at,
        latest_run_number=latest_run_number,
        reason="latest prediction is within freshness window",
    )
    if status.is_fresh:
        return status

    age = status.age
    if age is not None and age < timedelta(0):
        reason = "latest prediction timestamp is in the future"
    else:
        reason = "latest prediction is older than freshness window"
    return PredictionFreshness(
        path=log_path,
        max_age=max_age,
        now=current_time,
        latest_run_at=latest_run_at,
        latest_run_number=latest_run_number,
        reason=reason,
    )


def _resolve_path(args: argparse.Namespace) -> Path:
    selected = args.log_path or args.option_path or args.positional_path
    return Path(selected) if selected else PREDICTIONS_LOG


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Check prediction log freshness.")
    parser.add_argument(
        "positional_path",
        nargs="?",
        help="Path to predictions.log (defaults to repository predictions.log)",
    )
    parser.add_argument("--path", dest="option_path", help="Alias for --log-path")
    parser.add_argument("--log-path", help="Path to predictions.log")
    parser.add_argument(
        "--max-age-hours",
        type=float,
        default=1.0,
        help="Maximum allowed age of the latest prediction run (default: 1)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    status = check_prediction_freshness(
        _resolve_path(args),
        max_age=timedelta(hours=args.max_age_hours),
    )

    if status.latest_run_at is None:
        print(f"STALE: {status.reason}", file=sys.stderr)
        return 1

    age_minutes = status.age.total_seconds() / 60 if status.age else 0.0
    latest = status.latest_run_at.strftime(_LOG_TIMESTAMP_FORMAT)
    prefix = "FRESH" if status.is_fresh else "STALE"
    print(
        f"{prefix}: latest prediction run #{status.latest_run_number} at {latest} "
        f"({age_minutes:.1f} minutes old); {status.reason}"
    )
    return 0 if status.is_fresh else 1


if __name__ == "__main__":
    raise SystemExit(main())
