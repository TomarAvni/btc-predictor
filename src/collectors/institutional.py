"""Institutional flow collector: BTC ETF flows, Coinbase premium, Korean premium.

ETF flows are the dominant new signal post-2024 that didn't exist in prior cycles.
"""

import httpx
import pandas as pd

from src.collectors import BaseCollector
from src.utils.cache import DiskCache
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

# Coinbase premium: compare Coinbase BTC/USD vs Binance BTC/USDT
BINANCE_TICKER_URL = "https://api.binance.com/api/v3/ticker/price"
COINBASE_TICKER_URL = "https://api.coinbase.com/v2/prices/BTC-USD/spot"


class InstitutionalCollector(BaseCollector):
    """Tracks institutional demand signals: ETF flows and exchange premiums."""

    name = "institutional"
    tier = 2
    update_interval_seconds = 3600

    def __init__(self):
        self.cache = DiskCache()

    async def get_coinbase_premium(self) -> dict:
        """Calculate Coinbase premium (Coinbase price - Binance price) / Binance price.

        Positive premium = US institutional buying pressure.
        Negative premium = US selling pressure or Asian demand dominance.
        """
        try:
            async with httpx.AsyncClient(timeout=15) as client:
                # Binance BTC/USDT price
                binance_resp = await client.get(
                    BINANCE_TICKER_URL, params={"symbol": "BTCUSDT"}
                )
                binance_price = float(binance_resp.json()["price"])

                # Coinbase BTC/USD price
                coinbase_resp = await client.get(COINBASE_TICKER_URL)
                coinbase_price = float(coinbase_resp.json()["data"]["amount"])

            premium_pct = ((coinbase_price - binance_price) / binance_price) * 100

            interpretation = "neutral"
            if premium_pct > 0.15:
                interpretation = "strong_us_demand_bullish"
            elif premium_pct > 0.05:
                interpretation = "mild_us_demand"
            elif premium_pct < -0.15:
                interpretation = "us_selling_pressure_bearish"
            elif premium_pct < -0.05:
                interpretation = "mild_us_selling"

            return {
                "coinbase_premium_pct": round(premium_pct, 4),
                "coinbase_price": coinbase_price,
                "binance_price": binance_price,
                "interpretation": interpretation,
            }

        except Exception as e:
            logger.warning(f"Coinbase premium fetch failed: {e}")
            return {}

    async def get_etf_flow_estimate(self) -> dict:
        """Estimate ETF flow direction from price/volume patterns.

        Note: Real ETF flow data requires paid APIs (e.g., SoSoValue, Farside).
        This provides a framework that can be upgraded to real data sources.
        For now, returns a placeholder structure.
        """
        # This is a placeholder for real ETF flow integration.
        # Real implementation would use:
        # - Farside Investors (farside.co.uk) for daily ETF flow data
        # - SoSoValue API for real-time estimates
        # - SEC filings for AUM changes
        return {
            "source": "placeholder",
            "note": "Integrate farside.co.uk or SoSoValue API for real ETF flow data",
            "net_flow_btc": None,
            "interpretation": "no_data",
        }

    async def _collect(self) -> pd.DataFrame:
        """Collect latest institutional signals."""
        premium = await self.get_coinbase_premium()

        if not premium:
            return pd.DataFrame()

        row = {
            "timestamp": pd.Timestamp.now(tz="UTC"),
            "coinbase_premium_pct": premium.get("coinbase_premium_pct", 0),
        }

        df = pd.DataFrame([row])
        df.set_index("timestamp", inplace=True)
        return df

    async def collect_history(self, start: str, end: str | None = None) -> pd.DataFrame:
        """Historical institutional data (limited without paid APIs)."""
        # Premium is point-in-time; no free historical source
        return pd.DataFrame()
