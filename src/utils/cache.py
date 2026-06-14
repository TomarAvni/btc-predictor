"""Simple file-based cache with TTL.

Keys are hashed from request parameters; values are stored as JSON files
in data/cache/ with timestamp metadata for expiry.
"""

import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src import DATA_DIR

CACHE_DIR = DATA_DIR / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def cache_key(identifier: str, params: dict | None = None) -> str:
    """Deterministic hash for a request identifier + params."""
    raw = identifier + (json.dumps(params, sort_keys=True) if params else "")
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def get_cached(
    identifier: str,
    params: dict | None = None,
    max_age_minutes: int = 60,
) -> Any | None:
    """Return cached value if it exists and hasn't expired, else None."""
    key = cache_key(identifier, params)
    cache_file = CACHE_DIR / f"{key}.json"

    if not cache_file.exists():
        return None

    try:
        data = json.loads(cache_file.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None

    cached_at = datetime.fromisoformat(data["cached_at"])
    age_seconds = (datetime.now(timezone.utc) - cached_at).total_seconds()

    if age_seconds > max_age_minutes * 60:
        return None

    return data["payload"]


def set_cached(
    identifier: str,
    payload: Any,
    params: dict | None = None,
) -> None:
    """Store a value in the file cache."""
    key = cache_key(identifier, params)
    cache_file = CACHE_DIR / f"{key}.json"

    data = {
        "identifier": identifier,
        "params": params,
        "cached_at": datetime.now(timezone.utc).isoformat(),
        "payload": payload,
    }
    cache_file.write_text(json.dumps(data, default=str), encoding="utf-8")


def clear_expired(max_age_minutes: int = 1440) -> int:
    """Remove cache files older than max_age_minutes. Returns count removed."""
    removed = 0
    cutoff = time.time() - max_age_minutes * 60

    for f in CACHE_DIR.glob("*.json"):
        try:
            if f.stat().st_mtime < cutoff:
                f.unlink()
                removed += 1
        except OSError:
            continue

    return removed


class DiskCache:
    """Object-oriented wrapper around the file-based cache functions."""

    def __init__(self, namespace: str = "", max_age_minutes: int = 60) -> None:
        self.namespace = namespace
        self.max_age_minutes = max_age_minutes

    def get(self, key: str, params: dict | None = None) -> Any | None:
        identifier = f"{self.namespace}:{key}" if self.namespace else key
        return get_cached(identifier, params, self.max_age_minutes)

    def set(self, key: str, payload: Any, params: dict | None = None) -> None:
        identifier = f"{self.namespace}:{key}" if self.namespace else key
        set_cached(identifier, payload, params)

    def clear(self) -> int:
        return clear_expired(self.max_age_minutes)


Cache = DiskCache
