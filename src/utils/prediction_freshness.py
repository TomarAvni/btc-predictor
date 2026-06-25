"""Helpers for detecting stale prediction pipeline output."""

from __future__ import annotations

import argparse
import os
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PREDICTIONS_LOG = PROJECT_ROOT / "predictions.log"

_RUN_HEADER = re.compile(
    r"^\[(\d{4}-\d{2}-\d{2}\s\d{2}:\d{2}\sUTC)\]\s*--\s*Prediction Run #\d+"
)


@dataclass(frozen=True)
class PredictionFreshness:
    """Freshness status for the latest logged prediction."""

    latest_at: datetime | None
    max_age: timedelta
    now: datetime

    @property
    def age(self) -> timedelta | None:
        if self.latest_at is None:
            return None
        return self.now - self.latest_at

    @property
    def is_stale(self) -> bool:
        age = self.age
        return age is None or age >= self.max_age

    @property
    def reason(self) -> str:
        age = self.age
        if age is None:
            return "no prediction runs found"
        if self.is_stale:
            return f"latest prediction is {age.total_seconds() / 3600:.2f} hours old"
        return f"latest prediction is fresh at {age.total_seconds() / 60:.1f} minutes old"


def _as_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def parse_prediction_timestamp(value: str) -> datetime:
    """Parse the UTC timestamp format written in predictions.log headers."""

    return datetime.strptime(value, "%Y-%m-%d %H:%M UTC").replace(tzinfo=timezone.utc)


def latest_prediction_timestamp(path: Path = DEFAULT_PREDICTIONS_LOG) -> datetime | None:
    """Return the newest prediction timestamp from *path*, or ``None``."""

    if not path.exists():
        return None

    latest: datetime | None = None
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = _RUN_HEADER.search(line)
        if match:
            latest = parse_prediction_timestamp(match.group(1))
    return latest


def check_prediction_freshness(
    path: Path = DEFAULT_PREDICTIONS_LOG,
    *,
    max_age: timedelta = timedelta(hours=1),
    now: datetime | None = None,
) -> PredictionFreshness:
    """Return freshness status for the latest logged prediction."""

    now_utc = _as_utc(now or datetime.now(timezone.utc))
    latest = latest_prediction_timestamp(path)
    return PredictionFreshness(latest_at=latest, max_age=max_age, now=now_utc)


def _write_github_output(path: str, freshness: PredictionFreshness) -> None:
    latest = freshness.latest_at.isoformat() if freshness.latest_at else ""
    age_seconds = "" if freshness.age is None else str(int(freshness.age.total_seconds()))
    with open(path, "a", encoding="utf-8") as f:
        f.write(f"stale={'true' if freshness.is_stale else 'false'}\n")
        f.write(f"latest_at={latest}\n")
        f.write(f"age_seconds={age_seconds}\n")
        f.write(f"reason={freshness.reason}\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Check whether predictions.log has gone stale."
    )
    parser.add_argument(
        "--path",
        type=Path,
        default=DEFAULT_PREDICTIONS_LOG,
        help="Path to predictions.log",
    )
    parser.add_argument(
        "--max-age-hours",
        type=float,
        default=1.0,
        help="Maximum allowed prediction age before stale recovery is needed",
    )
    parser.add_argument(
        "--github-output",
        default=os.environ.get("GITHUB_OUTPUT"),
        help="Optional GitHub Actions output file to populate",
    )
    args = parser.parse_args(argv)

    freshness = check_prediction_freshness(
        args.path,
        max_age=timedelta(hours=args.max_age_hours),
    )
    status = "stale" if freshness.is_stale else "fresh"
    latest = freshness.latest_at.isoformat() if freshness.latest_at else "none"
    print(f"Prediction freshness: {status}; latest={latest}; {freshness.reason}")

    if args.github_output:
        _write_github_output(args.github_output, freshness)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
