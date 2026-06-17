"""Check whether the prediction pipeline has produced recent output."""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

DEFAULT_JSONL_PATH = Path("data/predictions/predictions.jsonl")
DEFAULT_TEXT_LOG_PATH = Path("predictions.log")

_TEXT_LOG_TIMESTAMP_RE = re.compile(
    r"\[(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2} UTC)\]\s+--\s+Prediction Run #"
)
_TEXT_LOG_TIMESTAMP_FMT = "%Y-%m-%d %H:%M UTC"


@dataclass(frozen=True)
class PredictionFreshness:
    """Result of a freshness check."""

    is_fresh: bool
    latest_timestamp: datetime | None
    age: timedelta | None
    max_age: timedelta
    source: str | None

    @property
    def status(self) -> str:
        return "fresh" if self.is_fresh else "stale"


def _ensure_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _parse_iso_utc(value: object) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        return _ensure_utc(datetime.fromisoformat(value.strip().replace("Z", "+00:00")))
    except ValueError:
        return None


def latest_jsonl_timestamp(path: Path | str = DEFAULT_JSONL_PATH) -> datetime | None:
    """Return the newest timestamp from the JSONL prediction log."""

    log_path = Path(path)
    if not log_path.exists():
        return None

    latest: datetime | None = None
    try:
        with log_path.open(encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                ts = _parse_iso_utc(record.get("timestamp"))
                if ts and (latest is None or ts > latest):
                    latest = ts
    except OSError:
        return None
    return latest


def latest_text_log_timestamp(path: Path | str = DEFAULT_TEXT_LOG_PATH) -> datetime | None:
    """Return the newest timestamp from the human-readable prediction log."""

    log_path = Path(path)
    if not log_path.exists():
        return None

    latest: datetime | None = None
    try:
        content = log_path.read_text(encoding="utf-8")
    except OSError:
        return None

    for match in _TEXT_LOG_TIMESTAMP_RE.finditer(content):
        try:
            ts = datetime.strptime(match.group("timestamp"), _TEXT_LOG_TIMESTAMP_FMT)
        except ValueError:
            continue
        ts = ts.replace(tzinfo=timezone.utc)
        if latest is None or ts > latest:
            latest = ts
    return latest


def latest_prediction_timestamp(
    *,
    jsonl_path: Path | str = DEFAULT_JSONL_PATH,
    text_log_path: Path | str = DEFAULT_TEXT_LOG_PATH,
) -> tuple[datetime | None, str | None]:
    """Return the newest prediction timestamp and the artifact it came from."""

    candidates: list[tuple[datetime, str]] = []
    jsonl_ts = latest_jsonl_timestamp(jsonl_path)
    if jsonl_ts:
        candidates.append((jsonl_ts, str(jsonl_path)))
    text_ts = latest_text_log_timestamp(text_log_path)
    if text_ts:
        candidates.append((text_ts, str(text_log_path)))

    if not candidates:
        return None, None
    return max(candidates, key=lambda item: item[0])


def check_prediction_freshness(
    *,
    max_age_hours: float = 3.0,
    now: datetime | None = None,
    jsonl_path: Path | str = DEFAULT_JSONL_PATH,
    text_log_path: Path | str = DEFAULT_TEXT_LOG_PATH,
) -> PredictionFreshness:
    """Check whether the newest prediction artifact is within ``max_age_hours``."""

    current_time = _ensure_utc(now or datetime.now(timezone.utc))
    max_age = timedelta(hours=max_age_hours)
    latest, source = latest_prediction_timestamp(
        jsonl_path=jsonl_path,
        text_log_path=text_log_path,
    )
    if latest is None:
        return PredictionFreshness(False, None, None, max_age, None)

    age = current_time - latest
    return PredictionFreshness(age <= max_age, latest, age, max_age, source)


def _format_timedelta(value: timedelta | None) -> str:
    if value is None:
        return "unknown"
    total_seconds = int(value.total_seconds())
    sign = "-" if total_seconds < 0 else ""
    total_seconds = abs(total_seconds)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{sign}{hours}h {minutes}m {seconds}s"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Check whether prediction output is fresh enough."
    )
    parser.add_argument(
        "--max-age-hours",
        type=float,
        default=3.0,
        help="Maximum acceptable age for the latest prediction artifact.",
    )
    parser.add_argument(
        "--jsonl-path",
        default=str(DEFAULT_JSONL_PATH),
        help="Path to data/predictions/predictions.jsonl.",
    )
    parser.add_argument(
        "--text-log-path",
        default=str(DEFAULT_TEXT_LOG_PATH),
        help="Path to predictions.log.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    result = check_prediction_freshness(
        max_age_hours=args.max_age_hours,
        jsonl_path=args.jsonl_path,
        text_log_path=args.text_log_path,
    )

    if result.latest_timestamp is None:
        print(
            "Prediction pipeline is stale: no prediction artifacts were found "
            f"(max age {_format_timedelta(result.max_age)})."
        )
    else:
        latest = result.latest_timestamp.strftime("%Y-%m-%dT%H:%M:%SZ")
        print(
            f"Prediction pipeline is {result.status}: latest={latest}, "
            f"age={_format_timedelta(result.age)}, "
            f"max_age={_format_timedelta(result.max_age)}, source={result.source}"
        )
    return 0 if result.is_fresh else 1


if __name__ == "__main__":
    sys.exit(main())
