"""Check whether the prediction log has been updated recently.

The scheduled Predict workflow can be delayed or skipped by GitHub Actions.
This module gives the watchdog workflow a cheap, deterministic freshness check
against the committed prediction log before it dispatches a recovery run.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from src import PREDICTIONS_LOG

_RUN_HEADER = re.compile(
    r"^\[(?P<timestamp>\d{4}-\d{2}-\d{2}\s\d{2}:\d{2}\sUTC)\]\s*--\s*Prediction Run #(?P<run>\d+)"
)
_LOG_TS_FORMAT = "%Y-%m-%d %H:%M UTC"


@dataclass(frozen=True)
class PredictionFreshness:
    """Freshness details for the newest prediction run in the log."""

    latest_timestamp: datetime | None
    latest_run_number: int | None
    age_hours: float | None
    max_age_hours: float
    reason: str

    @property
    def is_fresh(self) -> bool:
        return self.age_hours is not None and self.age_hours <= self.max_age_hours


def _parse_log_timestamp(value: str) -> datetime:
    return datetime.strptime(value, _LOG_TS_FORMAT).replace(tzinfo=timezone.utc)


def latest_prediction_run(path: Path = PREDICTIONS_LOG) -> tuple[datetime, int] | None:
    """Return the newest prediction timestamp and run number from ``path``."""

    if not path.exists():
        return None

    latest: tuple[datetime, int] | None = None
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = _RUN_HEADER.search(line)
        if not match:
            continue
        timestamp = _parse_log_timestamp(match.group("timestamp"))
        run_number = int(match.group("run"))
        if latest is None or timestamp > latest[0]:
            latest = (timestamp, run_number)

    return latest


def check_prediction_freshness(
    path: Path = PREDICTIONS_LOG,
    *,
    max_age_hours: float = 3.0,
    now: datetime | None = None,
) -> PredictionFreshness:
    """Measure how old the newest prediction log entry is."""

    current_time = now or datetime.now(timezone.utc)
    if current_time.tzinfo is None:
        current_time = current_time.replace(tzinfo=timezone.utc)
    else:
        current_time = current_time.astimezone(timezone.utc)

    latest = latest_prediction_run(path)
    if latest is None:
        reason = f"No prediction runs found in {path}"
        return PredictionFreshness(None, None, None, max_age_hours, reason)

    latest_timestamp, run_number = latest
    age_hours = (current_time - latest_timestamp).total_seconds() / 3600
    reason = (
        f"latest run #{run_number} at {latest_timestamp.strftime(_LOG_TS_FORMAT)} "
        f"is {age_hours:.2f}h old (limit {max_age_hours:.2f}h)"
    )
    return PredictionFreshness(latest_timestamp, run_number, age_hours, max_age_hours, reason)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--path",
        type=Path,
        default=PREDICTIONS_LOG,
        help="Path to predictions.log (default: repository predictions.log)",
    )
    parser.add_argument(
        "--max-age-hours",
        type=float,
        default=3.0,
        help="Maximum acceptable age for the latest prediction run",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    freshness = check_prediction_freshness(args.path, max_age_hours=args.max_age_hours)
    print(freshness.reason)
    if freshness.is_fresh:
        print("Prediction log is fresh.")
        return 0
    print("Prediction log is stale.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
