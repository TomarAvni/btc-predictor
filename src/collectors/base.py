"""Abstract base class for all data collectors.

Every collector must implement async collect/get_latest/get_historical.
The base class provides built-in caching, graceful error wrapping,
and a declared update interval so the scheduler knows how often to poll.
"""

from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from src.utils.cache import get_cached, set_cached
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class BaseCollector(ABC):
    """Abstract base for all data collectors.

    Subclasses define *_collect*, *_get_latest*, and *_get_historical*
    (the "inner" methods).  The public API wraps them with error handling
    and optional caching so a single collector failure never crashes the
    rest of the pipeline.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable identifier used in logs and cache keys."""
        ...

    @property
    @abstractmethod
    def update_interval_seconds(self) -> int:
        """How often this collector should be polled, in seconds.

        Typical tiers: fast=900 (15 min), medium=3600 (1 h),
        slow=21600 (6 h), daily=86400.
        """
        ...

    # ------------------------------------------------------------------
    # Public API (wraps inner methods with error handling + caching)
    # ------------------------------------------------------------------

    async def collect(self) -> pd.DataFrame:
        """Fetch the latest data.  Returns an empty DataFrame on failure."""
        try:
            return await self._collect()
        except Exception as exc:
            logger.error("[%s] collect() failed: %s", self.name, exc, exc_info=True)
            return pd.DataFrame()

    async def get_latest(self) -> dict[str, Any]:
        """Return the most recent snapshot as a dict.

        Checks the cache first (keyed on collector name); falls back to
        a live fetch via *_get_latest*.
        """
        cache_minutes = max(self.update_interval_seconds // 60, 1)
        cached = get_cached(f"{self.name}:latest", max_age_minutes=cache_minutes)
        if cached is not None:
            return cached

        try:
            result = await self._get_latest()
            set_cached(f"{self.name}:latest", result)
            return result
        except Exception as exc:
            logger.error("[%s] get_latest() failed: %s", self.name, exc, exc_info=True)
            return {}

    async def get_historical(
        self,
        start: str,
        end: str | None = None,
    ) -> pd.DataFrame:
        """Fetch historical data for the given date range.

        Args:
            start: ISO-8601 date string for the range start.
            end:   ISO-8601 date string for the range end (None = now).
        """
        try:
            return await self._get_historical(start, end)
        except Exception as exc:
            logger.error("[%s] get_historical() failed: %s", self.name, exc, exc_info=True)
            return pd.DataFrame()

    # ------------------------------------------------------------------
    # Inner methods -- subclasses implement these
    # ------------------------------------------------------------------

    @abstractmethod
    async def _collect(self) -> pd.DataFrame:
        """Core collection logic.  Must return a DatetimeIndex DataFrame."""
        ...

    async def _get_latest(self) -> dict[str, Any]:
        """Return a summary dict of the most recent data point.

        Default implementation collects and takes the last row.
        """
        df = await self._collect()
        if df.empty:
            return {}
        row = df.iloc[-1]
        result = row.to_dict()
        result["timestamp"] = str(df.index[-1])
        return result

    async def _get_historical(
        self,
        start: str,
        end: str | None = None,
    ) -> pd.DataFrame:
        """Override to supply historical data for backtesting."""
        return pd.DataFrame()

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def health_check(self) -> dict[str, Any]:
        """Quick status info for monitoring dashboards."""
        return {
            "collector": self.name,
            "interval_s": self.update_interval_seconds,
            "status": "ok",
        }

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} name={self.name!r}>"
