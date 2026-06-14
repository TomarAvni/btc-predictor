"""
Sentiment collector.

Sources: Fear & Greed Index, Google Trends for "Bitcoin".
Both are strong contrarian indicators at extremes.
"""

from typing import Any

import httpx
import pandas as pd

from src.collectors import BaseCollector
from src.utils.logger import setup_logger

logger = setup_logger(__name__)
from src.utils.cache import get_cached, set_cached


class SentimentCollector(BaseCollector):
    """Collects market sentiment data."""

    name = "sentiment"
    tier = 2
    update_interval_seconds = 3600

    def __init__(self, config: dict):
        self.fng_url = config.get("collectors", {}).get("fear_greed", {}).get(
            "url", "https://api.alternative.me/fng/"
        )

    async def _collect(self) -> pd.DataFrame:
        data = await self._collect_dict()
        return pd.DataFrame([data]) if data else pd.DataFrame()

    async def _get_latest(self) -> dict[str, Any]:
        return await self._collect_dict()

    async def _collect_dict(self) -> dict[str, Any]:
        results = {}

        fng = await self._fetch_fear_greed()
        results.update(fng)

        return results

    async def _fetch_fear_greed(self) -> dict:
        """Fetch Fear & Greed Index (0-100 scale)."""
        url = f"{self.fng_url}?limit=30"
        cached = get_cached(url, max_age_minutes=60)

        if not cached:
            try:
                async with httpx.AsyncClient(timeout=15) as client:
                    resp = await client.get(url)
                    resp.raise_for_status()
                    cached = resp.json()
                    set_cached(url, cached)
            except Exception as e:
                logger.warning(f"Failed to fetch Fear & Greed: {e}")
                return {"fear_greed_value": None, "fear_greed_label": "unknown"}

        data = cached.get("data", [])
        if not data:
            return {"fear_greed_value": None, "fear_greed_label": "unknown"}

        current = data[0]
        value = int(current.get("value", 50))
        label = current.get("value_classification", "Neutral")

        # Compute 7-day average for trend
        values_7d = [int(d["value"]) for d in data[:7] if "value" in d]
        avg_7d = sum(values_7d) / len(values_7d) if values_7d else value

        # Compute 30-day average
        values_30d = [int(d["value"]) for d in data[:30] if "value" in d]
        avg_30d = sum(values_30d) / len(values_30d) if values_30d else value

        # Contrarian signal: extreme fear = buy, extreme greed = sell
        if value <= 20:
            contrarian = "strong_buy"
        elif value <= 35:
            contrarian = "buy"
        elif value >= 80:
            contrarian = "strong_sell"
        elif value >= 65:
            contrarian = "sell"
        else:
            contrarian = "neutral"

        return {
            "fear_greed_value": value,
            "fear_greed_label": label,
            "fear_greed_7d_avg": round(avg_7d, 1),
            "fear_greed_30d_avg": round(avg_30d, 1),
            "fear_greed_trend": "rising" if avg_7d > avg_30d else "falling",
            "fear_greed_contrarian_signal": contrarian,
        }
