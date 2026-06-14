"""Long-term holder (LTH) vs Short-term holder (STH) supply analysis.

LTH = coins unmoved for 155+ days
STH = coins moved within 155 days

Key signals:
- LTH distributing to STH = cycle top approaching
- STH capitulating (selling at loss) = cycle bottom forming
- LTH/STH ratio at extremes = major turning points
"""

import httpx
import pandas as pd

from src.collectors.base import BaseCollector
from src.utils.cache import DiskCache
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class HoldersCollector(BaseCollector):
    """Tracks LTH/STH supply distribution from on-chain data."""

    name = "holders"
    tier = 3
    update_interval_seconds = 21600

    def __init__(self):
        self.cache = DiskCache()

    async def get_utxo_age_proxy(self) -> dict:
        """Approximate LTH/STH from UTXO data.

        Full LTH/STH data requires Glassnode or CryptoQuant (paid).
        This uses free blockchain.info UTXO data as a proxy.
        """
        cache_key = "utxo_age_proxy"
        cached = self.cache.get(cache_key, max_age_seconds=43200)  # 12h cache
        if cached:
            return cached

        try:
            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.get(
                    "https://api.blockchain.info/charts/utxo-count",
                    params={"timespan": "60days", "format": "json"},
                )
                if resp.status_code == 200:
                    data = resp.json()
                    values = data.get("values", [])

                    if len(values) >= 2:
                        recent = values[-1]["y"]
                        month_ago = values[max(0, len(values) - 30)]["y"]
                        growth_pct = ((recent - month_ago) / month_ago * 100) if month_ago > 0 else 0

                        result = {
                            "utxo_count": recent,
                            "utxo_growth_30d_pct": round(growth_pct, 2),
                            "accumulation_signal": growth_pct > 1.0,
                            "data_source": "blockchain.info (proxy)",
                        }
                        self.cache.set(cache_key, result)
                        return result

        except Exception as e:
            logger.warning(f"UTXO age proxy fetch failed: {e}")

        return {"accumulation_signal": None, "data_source": "unavailable"}

    def interpret_lth_behavior(self, lth_supply_change_pct: float) -> str:
        """Interpret LTH supply change.

        Positive = LTH accumulating (bullish, supply squeeze forming)
        Negative = LTH distributing (bearish, selling into strength)
        """
        if lth_supply_change_pct > 1.0:
            return "strong_accumulation_bullish"
        elif lth_supply_change_pct > 0.2:
            return "mild_accumulation"
        elif lth_supply_change_pct > -0.2:
            return "neutral_holding"
        elif lth_supply_change_pct > -1.0:
            return "mild_distribution_caution"
        else:
            return "heavy_distribution_bearish"

    async def _collect(self) -> pd.DataFrame:
        data = await self.get_utxo_age_proxy()
        if not data or data.get("data_source") == "unavailable":
            return pd.DataFrame()

        df = pd.DataFrame([data], index=[pd.Timestamp.now(tz="UTC")])
        return df

    async def collect_history(self, start: str, end: str | None = None) -> pd.DataFrame:
        # Historical LTH/STH requires paid APIs
        return pd.DataFrame()
