"""Helpers for deciding whether the prediction pipeline is stale.

The GitHub Actions watchdog uses this module before installing project
dependencies, so keep it limited to the Python standard library.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable

JSONL_PATH = Path("data/predictions/predictions.jsonl")
TEXT_LOG_PATH = Path("predictions.log")
LOG_HEADER_RE = re.compile(r"\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2} UTC)\]")


@dataclass(frozen=True)
class FreshnessStatus:
    """Freshness details for the latest committed prediction."""

    latest_timestamp: datetime | None
    max_age: timedelta
    now: datetime
    source: str | None = None

    @property
    def age(self) -> timedelta | None:
        if self.latest_timestamp is None:
            return None
        return self.now - self.latest_timestamp

    @property
    def is_stale(self) -> bool:
        return self.age is None or self.age > self.max_age


def _ensure_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _parse_timestamp(value: object) -> datetime | None:
    if not isinstance(value, str):
        return None

    text = value.strip()
    if not text:
        return None

    if text.endswith(" UTC"):
        try:
            return datetime.strptime(text, "%Y-%m-%d %H:%M UTC").replace(
                tzinfo=timezone.utc
            )
        except ValueError:
            return None

    try:
        return _ensure_utc(datetime.fromisoformat(text.replace("Z", "+00:00")))
    except ValueError:
        return None


def _iter_jsonl_timestamps(path: Path) -> Iterable[datetime]:
    if not path.exists():
        return

    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            ts = _parse_timestamp(row.get("timestamp"))
            if ts is not None:
                yield ts


def latest_prediction_timestamp(
    *,
    jsonl_path: Path = JSONL_PATH,
    text_log_path: Path = TEXT_LOG_PATH,
) -> tuple[datetime | None, str | None]:
    """Return the newest prediction timestamp and the file it came from."""

    jsonl_latest = max(_iter_jsonl_timestamps(jsonl_path), default=None)
    if jsonl_latest is not None:
        return jsonl_latest, str(jsonl_path)

    if not text_log_path.exists():
        return None, None

    latest: datetime | None = None
    with text_log_path.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            match = LOG_HEADER_RE.search(line)
            if not match:
                continue
            ts = _parse_timestamp(match.group(1))
            if ts is not None and (latest is None or ts > latest):
                latest = ts

    if latest is None:
        return None, None
    return latest, str(text_log_path)


def check_prediction_freshness(
    *,
    max_age: timedelta,
    now: datetime | None = None,
    jsonl_path: Path = JSONL_PATH,
    text_log_path: Path = TEXT_LOG_PATH,
) -> FreshnessStatus:
    """Compute whether the latest prediction is older than ``max_age``."""

    checked_at = _ensure_utc(now or datetime.now(timezone.utc))
    latest, source = latest_prediction_timestamp(
        jsonl_path=jsonl_path,
        text_log_path=text_log_path,
    )
    return FreshnessStatus(
        latest_timestamp=latest,
        max_age=max_age,
        now=checked_at,
        source=source,
    )


def _format_dt(dt: datetime | None) -> str:
    if dt is None:
        return ""
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _write_github_output(status: FreshnessStatus) -> None:
    output_path = os.environ.get("GITHUB_OUTPUT")
    if not output_path:
        return

    age_seconds = "" if status.age is None else str(int(status.age.total_seconds()))
    lines = [
        f"stale={'true' if status.is_stale else 'false'}",
        f"latest_timestamp={_format_dt(status.latest_timestamp)}",
        f"age_seconds={age_seconds}",
        f"source={status.source or ''}",
    ]
    with open(output_path, "a", encoding="utf-8") as f:
        f.write("\n".join(lines))
        f.write("\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Check whether committed prediction output is stale."
    )
    parser.add_argument(
        "--max-age-hours",
        type=float,
        default=3.0,
        help="Maximum fresh age in hours before the pipeline should run.",
    )
    parser.add_argument(
        "--github-output",
        action="store_true",
        help="Write stale/latest/age details to $GITHUB_OUTPUT.",
    )
    parser.add_argument(
        "--exit-code",
        action="store_true",
        help="Exit 1 when stale instead of always exiting 0.",
    )
    args = parser.parse_args(argv)

    status = check_prediction_freshness(
        max_age=timedelta(hours=args.max_age_hours),
    )
    if args.github_output:
        _write_github_output(status)

    latest = _format_dt(status.latest_timestamp) or "none"
    age = "unknown" if status.age is None else str(status.age).split(".")[0]
    state = "stale" if status.is_stale else "fresh"
    print(
        f"Prediction freshness: {state}; latest={latest}; "
        f"age={age}; source={status.source or 'none'}"
    )

    if args.exit_code and status.is_stale:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
