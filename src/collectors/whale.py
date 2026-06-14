"""Whale wallet movement tracker.

Monitors large BTC transactions and wallet balance changes to detect
accumulation or distribution by major holders.
"""

import httpx
import pandas as pd

from src.collectors import BaseCollector
from src.utils.cache import DiskCache
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

BLOCKCHAIR_BASE = "https://api.blockchair.com/bitcoin"
WHALE_THRESHOLD_BTC = 100  # Transactions >= 100 BTC considered "whale"


class WhaleCollector(BaseCollector):
    """Tracks large BTC wallet movements and accumulation patterns."""

    name = "whale"
    tier = 3
    update_interval_seconds = 21600

    def __init__(self):
        self.cache = DiskCache()

    async def get_large_transactions(self, min_btc: float = WHALE_THRESHOLD_BTC) -> pd.DataFrame:
        """Fetch recent large BTC transactions from Blockchair.

        Large inflows to exchanges = potential selling.
        Large outflows from exchanges = accumulation (bullish).
        """
        cache_key = f"whale_txs_{min_btc}"
        cached = self.cache.get(cache_key, max_age_seconds=3600)
        if cached:
            df = pd.DataFrame(cached)
            if not df.empty and "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
                df.set_index("timestamp", inplace=True)
            return df

        try:
            min_satoshi = int(min_btc * 1e8)
            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.get(
                    f"{BLOCKCHAIR_BASE}/transactions",
                    params={
                        "q": f"output_total({min_satoshi}..)",
                        "s": "time(desc)",
                        "limit": 50,
                    },
                )
                if resp.status_code == 200:
                    data = resp.json()
                    txs = data.get("data", [])
                    records = []
                    for tx in txs:
                        records.append({
                            "timestamp": pd.to_datetime(tx.get("time"), utc=True),
                            "tx_hash": tx.get("hash", ""),
                            "total_btc": tx.get("output_total", 0) / 1e8,
                            "fee_btc": tx.get("fee", 0) / 1e8,
                            "input_count": tx.get("input_count", 0),
                            "output_count": tx.get("output_count", 0),
                        })

                    df = pd.DataFrame(records)
                    if not df.empty:
                        df.set_index("timestamp", inplace=True)
                        df.sort_index(inplace=True)
                        self.cache.set(cache_key, df.reset_index().to_dict(orient="records"))
                    return df

        except Exception as e:
            logger.warning(f"Whale transaction fetch failed: {e}")

        return pd.DataFrame()

    async def get_whale_accumulation_score(self) -> dict:
        """Calculate a whale accumulation vs distribution score.

        Positive = net accumulation (bullish).
        Negative = net distribution (bearish).
        """
        txs = await self.get_large_transactions()
        if txs.empty:
            return {"score": 0.0, "interpretation": "no_data"}

        # Heuristic: many outputs = distribution, few outputs = consolidation/accumulation
        # Transactions with 1-2 outputs are likely accumulation (moving to cold storage)
        # Transactions with many outputs are likely distribution (exchange deposits, selling)
        accumulation = txs[txs["output_count"] <= 2]["total_btc"].sum()
        distribution = txs[txs["output_count"] > 5]["total_btc"].sum()

        total = accumulation + distribution
        if total == 0:
            return {"score": 0.0, "interpretation": "neutral"}

        score = (accumulation - distribution) / total  # -1 to +1

        if score > 0.3:
            interpretation = "strong_accumulation_bullish"
        elif score > 0.1:
            interpretation = "mild_accumulation"
        elif score > -0.1:
            interpretation = "neutral"
        elif score > -0.3:
            interpretation = "mild_distribution"
        else:
            interpretation = "strong_distribution_bearish"

        return {
            "score": round(score, 4),
            "interpretation": interpretation,
            "accumulation_btc": round(accumulation, 2),
            "distribution_btc": round(distribution, 2),
            "tx_count": len(txs),
        }

    async def _collect(self) -> pd.DataFrame:
        """Collect latest whale activity summary."""
        score_data = await self.get_whale_accumulation_score()
        df = pd.DataFrame([score_data], index=[pd.Timestamp.now(tz="UTC")])
        return df

    async def collect_history(self, start: str, end: str | None = None) -> pd.DataFrame:
        """Historical whale data (limited by API availability)."""
        return await self.get_large_transactions()
