"""Freshness checks for the prediction pipeline.

Supports two automation modes:

- Watchdog (``--max-age-hours``): exit 0 when fresh, 1 when stale.
- Workflow gate (``--event-name``, ``--threshold-minutes``, ``--output``):
  write ``should_run`` to a GitHub Actions output file.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

_PREDICTION_HEADER_RE = re.compile(
    r"^\[(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2} UTC)\] -- Prediction Run #\d+",
    re.MULTILINE,
)
_LOG_TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M UTC"
_JSONL_TIMESTAMP_KEY = "timestamp"
_DEFAULT_JSONL_PATH = "data/predictions/predictions.jsonl"


@dataclass(frozen=True)
class PredictionFreshness:
    """Freshness result for the latest prediction log entry."""

    latest_run_at: datetime | None
    max_age: timedelta
    now: datetime

    @property
    def age(self) -> timedelta | None:
        if self.latest_run_at is None:
            return None
        return self.now - self.latest_run_at

    @property
    def is_fresh(self) -> bool:
        age = self.age
        return age is not None and age <= self.max_age


def _ensure_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def parse_prediction_timestamp(value: str) -> datetime:
    """Parse a ``predictions.log`` UTC timestamp into an aware datetime."""

    return datetime.strptime(value, _LOG_TIMESTAMP_FORMAT).replace(tzinfo=timezone.utc)


def _parse_jsonl_timestamp(value: object) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None

    text = value.strip().replace("Z", "+00:00")
    try:
        return _ensure_utc(datetime.fromisoformat(text))
    except ValueError:
        return None


def latest_prediction_jsonl_time(path: str | Path) -> datetime | None:
    """Return the newest UTC timestamp from prediction JSONL records."""

    jsonl_path = Path(path)
    if not jsonl_path.exists():
        return None

    latest: datetime | None = None
    with jsonl_path.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            ts = _parse_jsonl_timestamp(record.get(_JSONL_TIMESTAMP_KEY))
            if ts is not None and (latest is None or ts > latest):
                latest = ts
    return latest


def latest_prediction_run_at(log_path: str | Path = "predictions.log") -> datetime | None:
    """Return the newest prediction run timestamp from ``log_path``.

    ``None`` is returned when the file is missing, unreadable, or does not
    contain any valid prediction headers.
    """

    path = Path(log_path)
    try:
        content = path.read_text(encoding="utf-8")
    except OSError:
        return None

    latest: datetime | None = None
    for match in _PREDICTION_HEADER_RE.finditer(content):
        try:
            run_at = parse_prediction_timestamp(match.group("timestamp"))
        except ValueError:
            continue
        if latest is None or run_at > latest:
            latest = run_at
    return latest


def latest_prediction_time(
    jsonl_path: str | Path = _DEFAULT_JSONL_PATH,
    log_path: str | Path = "predictions.log",
) -> datetime | None:
    """Return the newest prediction timestamp from machine or text logs."""

    candidates = [
        latest_prediction_jsonl_time(jsonl_path),
        latest_prediction_run_at(log_path),
    ]
    return max((ts for ts in candidates if ts is not None), default=None)


def should_run_prediction(
    *,
    event_name: str,
    jsonl_path: str | Path = _DEFAULT_JSONL_PATH,
    log_path: str | Path = "predictions.log",
    threshold: timedelta,
    now: datetime | None = None,
) -> bool:
    """Manual runs always proceed; scheduled runs proceed only when stale."""

    if event_name == "workflow_dispatch":
        return True

    current_time = _ensure_utc(now or datetime.now(timezone.utc))
    latest = latest_prediction_time(jsonl_path, log_path)
    return latest is None or current_time - latest >= threshold


def check_prediction_freshness(
    log_path: str | Path = "predictions.log",
    *,
    jsonl_path: str | Path | None = None,
    max_age: timedelta = timedelta(hours=3),
    now: datetime | None = None,
) -> PredictionFreshness:
    """Check whether the prediction pipeline has produced recent output."""

    current_time = _ensure_utc(now or datetime.now(timezone.utc))
    if jsonl_path is None:
        latest = latest_prediction_run_at(log_path)
    else:
        latest = latest_prediction_time(jsonl_path, log_path)
    return PredictionFreshness(
        latest_run_at=latest,
        max_age=max_age,
        now=current_time,
    )


def _format_age(age: timedelta | None) -> str:
    if age is None:
        return "unknown"

    total_seconds = max(0, int(age.total_seconds()))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, _ = divmod(remainder, 60)
    return f"{hours}h {minutes}m"


def _write_github_output(path: Path, should_run: bool) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"should_run={'true' if should_run else 'false'}\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Check prediction pipeline freshness for CI automation."
    )
    parser.add_argument(
        "--log-path",
        default="predictions.log",
        help="Path to the prediction text log (default: predictions.log)",
    )
    parser.add_argument(
        "--jsonl-path",
        default=_DEFAULT_JSONL_PATH,
        help="Path to the prediction JSONL store",
    )
    parser.add_argument(
        "--max-age-hours",
        type=float,
        help="Watchdog mode: maximum acceptable age for the latest run",
    )
    parser.add_argument(
        "--max-age-minutes",
        type=int,
        help="Watchdog mode: maximum acceptable age in minutes (overrides --max-age-hours)",
    )
    parser.add_argument(
        "--event-name",
        help="Workflow gate mode: GitHub event name (schedule or workflow_dispatch)",
    )
    parser.add_argument(
        "--threshold-minutes",
        type=int,
        default=25,
        help="Workflow gate mode: minimum age before a scheduled run proceeds",
    )
    parser.add_argument(
        "--output",
        help="Workflow gate mode: GitHub Actions output file path",
    )
    return parser


def _resolve_watchdog_max_age(args: argparse.Namespace) -> timedelta:
    if args.max_age_minutes is not None:
        return timedelta(minutes=args.max_age_minutes)
    max_age_hours = 3.0 if args.max_age_hours is None else args.max_age_hours
    return timedelta(hours=max_age_hours)


def _run_watchdog_mode(args: argparse.Namespace) -> int:
    result = check_prediction_freshness(
        args.log_path,
        jsonl_path=args.jsonl_path,
        max_age=_resolve_watchdog_max_age(args),
    )

    if result.latest_run_at is None:
        print(
            "Prediction log is stale: no prediction runs found in "
            f"{args.log_path} or {args.jsonl_path}."
        )
        return 1

    latest = result.latest_run_at.strftime(_LOG_TIMESTAMP_FORMAT)
    age = _format_age(result.age)
    max_age = _format_age(result.max_age)
    if result.is_fresh:
        print(f"Prediction log is fresh (latest run at {latest}, age {age}, max {max_age}).")
        return 0

    print(f"Prediction log is stale: latest run at {latest} (age {age}, max {max_age}).")
    return 1


def _run_workflow_gate_mode(args: argparse.Namespace) -> int:
    should_run = should_run_prediction(
        event_name=args.event_name or "schedule",
        jsonl_path=args.jsonl_path,
        log_path=args.log_path,
        threshold=timedelta(minutes=args.threshold_minutes),
    )
    status = "stale or manual" if should_run else "fresh"
    print(f"Prediction workflow freshness: {status}; should_run={should_run}")
    if args.output:
        _write_github_output(Path(args.output), should_run)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.output is not None or args.event_name is not None:
        return _run_workflow_gate_mode(args)
    return _run_watchdog_mode(args)


if __name__ == "__main__":
    sys.exit(main())
