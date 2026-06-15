"""Timezone helpers for displaying timestamps in Israel local time.

All data is stored and logged in UTC (the source of truth). These helpers
convert UTC timestamps to Israel local time (Asia/Jerusalem) *for display
only*, handling DST automatically via the IANA timezone database so the
label is correctly "IDT" (summer) or "IST" (winter).

A fixed-offset fallback is used only if the IANA database is unavailable
(e.g. a Windows host without the ``tzdata`` package), so the dashboard
never crashes -- it just shows a best-effort offset.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Optional

try:  # Prefer the real IANA zone so DST is handled correctly.
    from zoneinfo import ZoneInfo

    ISRAEL_TZ = ZoneInfo("Asia/Jerusalem")
except Exception:  # pragma: no cover - only on hosts without tzdata
    ISRAEL_TZ = timezone(timedelta(hours=3), name="IDT")

# Format used across human-facing displays.
_DISPLAY_FMT = "%Y-%m-%d %H:%M"
# Format the predictor/loggers write into predictions.log (always UTC).
_UTC_LOG_FMT = "%Y-%m-%d %H:%M UTC"


def _ensure_aware_utc(dt: datetime) -> datetime:
    """Treat a naive datetime as UTC; leave aware datetimes untouched."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt


def to_israel(dt: datetime) -> datetime:
    """Convert any datetime (naive assumed UTC) to Israel local time."""
    return _ensure_aware_utc(dt).astimezone(ISRAEL_TZ)


def now_israel() -> datetime:
    """Current time in Israel local time."""
    return datetime.now(timezone.utc).astimezone(ISRAEL_TZ)


def _tz_label(dt: datetime) -> str:
    """Return the timezone abbreviation (IDT/IST), with a sane fallback."""
    label = dt.strftime("%Z")
    return label or "Israel"


def format_israel(dt: datetime, *, fmt: str = _DISPLAY_FMT, with_label: bool = True) -> str:
    """Format a datetime in Israel local time with a clear tz label."""
    local = to_israel(dt)
    text = local.strftime(fmt)
    if with_label:
        text += f" {_tz_label(local)}"
    return text


def now_israel_str(*, fmt: str = _DISPLAY_FMT, with_label: bool = True) -> str:
    """Current Israel-local time as a labelled string."""
    return format_israel(datetime.now(timezone.utc), fmt=fmt, with_label=with_label)


def _parse_to_utc(value: str) -> Optional[datetime]:
    """Best-effort parse of common timestamp strings into an aware UTC dt."""
    text = value.strip()
    if not text:
        return None

    # "2026-06-15 06:00 UTC" (predictions.log header format)
    if text.endswith(" UTC"):
        try:
            naive = datetime.strptime(text, _UTC_LOG_FMT)
            return naive.replace(tzinfo=timezone.utc)
        except ValueError:
            pass

    # ISO 8601 (trade/journal timestamps), tolerating a trailing Z.
    iso = text.replace("Z", "+00:00")
    try:
        return _ensure_aware_utc(datetime.fromisoformat(iso))
    except ValueError:
        return None


def utc_str_to_israel(
    value: Optional[str],
    *,
    fmt: str = _DISPLAY_FMT,
    with_label: bool = True,
    fallback: str = "—",
) -> str:
    """Convert a stored UTC timestamp string to an Israel-local display string.

    Accepts both the predictions.log header format ("YYYY-MM-DD HH:MM UTC")
    and ISO 8601 timestamps. Returns ``fallback`` (or the original string)
    if parsing fails, so display code never raises.
    """
    if not value or not isinstance(value, str):
        return fallback

    dt = _parse_to_utc(value)
    if dt is None:
        return value  # Unknown format -- show as-is rather than hiding it.
    return format_israel(dt, fmt=fmt, with_label=with_label)
