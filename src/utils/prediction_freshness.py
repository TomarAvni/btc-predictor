"""Freshness checks for the GitHub Actions prediction pipeline."""

from __future__ import annotations

import argparse
import re
from datetime import datetime, timezone
from pathlib import Path

PREDICTION_HEADER_RE = re.compile(
    r"^\[(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}) UTC\] -- Prediction Run #(?P<run>\d+)\s*$"
)


def parse_prediction_timestamp(line: str) -> datetime | None:
    """Parse a prediction log header timestamp, returning UTC-aware datetimes."""
    match = PREDICTION_HEADER_RE.match(line.strip())
    if not match:
        return None

    parsed = datetime.strptime(match.group("timestamp"), "%Y-%m-%d %H:%M")
    return parsed.replace(tzinfo=timezone.utc)


def latest_prediction_timestamp(path: str | Path = "predictions.log") -> datetime | None:
    """Return the newest prediction timestamp in the log, or None if absent."""
    log_path = Path(path)
    if not log_path.exists():
        return None

    latest: datetime | None = None
    with log_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            timestamp = parse_prediction_timestamp(line)
            if timestamp is not None and (latest is None or timestamp > latest):
                latest = timestamp
    return latest


def prediction_age_hours(
    path: str | Path = "predictions.log",
    *,
    now: datetime | None = None,
) -> float | None:
    """Return hours since the latest prediction, or None when no prediction exists."""
    latest = latest_prediction_timestamp(path)
    if latest is None:
        return None

    reference = now or datetime.now(timezone.utc)
    if reference.tzinfo is None:
        reference = reference.replace(tzinfo=timezone.utc)
    else:
        reference = reference.astimezone(timezone.utc)

    return (reference - latest).total_seconds() / 3600


def is_prediction_stale(
    path: str | Path = "predictions.log",
    *,
    max_age_hours: float = 3.0,
    now: datetime | None = None,
) -> bool:
    """Return True when no prediction exists or the latest one is too old."""
    age = prediction_age_hours(path, now=now)
    return age is None or age > max_age_hours


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Check whether predictions.log is fresh enough.")
    parser.add_argument("--path", default="predictions.log", help="Prediction log path to inspect.")
    parser.add_argument(
        "--max-age-hours",
        type=float,
        default=3.0,
        help="Maximum allowed age for the latest prediction before it is considered stale.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    latest = latest_prediction_timestamp(args.path)
    if latest is None:
        print(f"No prediction timestamp found in {args.path}; prediction pipeline is stale.")
        return 1

    age = prediction_age_hours(args.path)
    assert age is not None
    if age > args.max_age_hours:
        print(
            "Latest prediction is stale: "
            f"{latest.isoformat()} ({age:.2f}h old, limit {args.max_age_hours:.2f}h)."
        )
        return 1

    print(
        "Latest prediction is fresh: "
        f"{latest.isoformat()} ({age:.2f}h old, limit {args.max_age_hours:.2f}h)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
