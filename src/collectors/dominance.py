"""Bitcoin dominance and stablecoin supply collector.

BTC.D (dominance) shows capital rotation between BTC and altcoins.
Stablecoin supply growth indicates new capital entering the market.
"""

import httpx
import pandas as pd

from src.collectors import BaseCollector
from src.utils.cache import DiskCache
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

COINGECKO_BASE = "https://api.coingecko.com/api/v3"


class DominanceCollector(BaseCollector):
    """Tracks BTC dominance and stablecoin supply metrics."""

    name = "dominance"
    tier = 4
    update_interval_seconds = 86400

    def __init__(self):
        self.cache = DiskCache()

    async def get_btc_dominance(self) -> dict:
        """Fetch current BTC market dominance percentage.

        Rising dominance = money flowing into BTC (risk-off in crypto).
        Falling dominance = money flowing to alts (risk-on, euphoria phase).
        """
        cache_key = "btc_dominance"
        cached = self.cache.get(cache_key, max_age_seconds=3600)
        if cached:
            return cached

        try:
            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.get(f"{COINGECKO_BASE}/global")
                if resp.status_code == 200:
                    data = resp.json()["data"]
                    result = {
                        "btc_dominance_pct": data["market_cap_percentage"]["btc"],
                        "eth_dominance_pct": data["market_cap_percentage"].get("eth", 0),
                        "total_market_cap_usd": data["total_market_cap"]["usd"],
                        "total_volume_24h_usd": data["total_volume"]["usd"],
                    }
                    self.cache.set(cache_key, result)
                    return result
        except Exception as e:
            logger.warning(f"BTC dominance fetch failed: {e}")
        return {}

    async def get_stablecoin_market_cap(self) -> dict:
        """Fetch total stablecoin market cap (USDT + USDC + others).

        Growing stablecoin supply = new capital entering crypto (bullish).
        Shrinking supply = capital leaving (bearish).
        """
        cache_key = "stablecoin_mcap"
        cached = self.cache.get(cache_key, max_age_seconds=14400)  # 4h cache
        if cached:
            return cached

        stablecoins = ["tether", "usd-coin", "dai"]
        total_mcap = 0.0

        try:
            async with httpx.AsyncClient(timeout=30) as client:
                ids = ",".join(stablecoins)
                resp = await client.get(
                    f"{COINGECKO_BASE}/simple/price",
                    params={"ids": ids, "vs_currencies": "usd", "include_market_cap": "true"},
                )
                if resp.status_code == 200:
                    data = resp.json()
                    for coin in stablecoins:
                        if coin in data:
                            total_mcap += data[coin].get("usd_market_cap", 0)

            result = {"stablecoin_total_mcap_usd": total_mcap}
            self.cache.set(cache_key, result)
            return result
        except Exception as e:
            logger.warning(f"Stablecoin market cap fetch failed: {e}")
        return {}

    def interpret_dominance(self, dominance_pct: float) -> str:
        """Interpret BTC dominance level."""
        if dominance_pct > 65:
            return "high_dominance_btc_season"
        elif dominance_pct > 55:
            return "moderate_dominance_btc_favored"
        elif dominance_pct > 45:
            return "balanced_market"
        elif dominance_pct > 35:
            return "alt_season_starting"
        else:
            return "extreme_alt_season_caution"

    async def _collect(self) -> pd.DataFrame:
        """Collect latest dominance and stablecoin data."""
        dom = await self.get_btc_dominance()
        stable = await self.get_stablecoin_market_cap()

        row = {**dom, **stable}
        if not row:
            return pd.DataFrame()

        df = pd.DataFrame([row], index=[pd.Timestamp.now(tz="UTC")])
        return df

    async def collect_history(self, start: str, end: str | None = None) -> pd.DataFrame:
        # CoinGecko free tier doesn't provide historical dominance easily
        # Would need to use their /coins/bitcoin/market_chart endpoint
        return pd.DataFrame()
