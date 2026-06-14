"""Miner metrics collector: hash price, miner outflows, miner reserves.

Post-halving miner capitulation is a known cycle pattern. When miners
can't cover costs, they sell BTC, creating temporary selling pressure
that often marks local bottoms.
"""

import httpx
import pandas as pd

from src.collectors import BaseCollector
from src.utils.cache import DiskCache
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

BLOCKCHAIN_INFO_BASE = "https://api.blockchain.info"


class MinerCollector(BaseCollector):
    """Tracks miner health and behavior."""

    name = "miner"
    tier = 3
    update_interval_seconds = 21600

    def __init__(self):
        self.cache = DiskCache()

    async def get_miner_revenue(self) -> pd.DataFrame:
        """Fetch daily miner revenue (block reward + fees in BTC)."""
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.get(
                    f"{BLOCKCHAIN_INFO_BASE}/charts/miners-revenue",
                    params={"timespan": "180days", "format": "json"},
                )
                if resp.status_code == 200:
                    data = resp.json()
                    records = []
                    for point in data.get("values", []):
                        records.append({
                            "timestamp": pd.to_datetime(point["x"], unit="s", utc=True),
                            "miner_revenue_usd": point["y"],
                        })
                    df = pd.DataFrame(records)
                    if not df.empty:
                        df.set_index("timestamp", inplace=True)
                    return df
        except Exception as e:
            logger.warning(f"Miner revenue fetch failed: {e}")
        return pd.DataFrame()

    async def get_hash_rate(self) -> pd.DataFrame:
        """Fetch network hash rate -- declining hash rate signals miner capitulation."""
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.get(
                    f"{BLOCKCHAIN_INFO_BASE}/charts/hash-rate",
                    params={"timespan": "365days", "format": "json"},
                )
                if resp.status_code == 200:
                    data = resp.json()
                    records = []
                    for point in data.get("values", []):
                        records.append({
                            "timestamp": pd.to_datetime(point["x"], unit="s", utc=True),
                            "hash_rate_eh": point["y"] / 1e6,  # Convert to EH/s
                        })
                    df = pd.DataFrame(records)
                    if not df.empty:
                        df.set_index("timestamp", inplace=True)
                    return df
        except Exception as e:
            logger.warning(f"Hash rate fetch failed: {e}")
        return pd.DataFrame()

    async def get_difficulty(self) -> pd.DataFrame:
        """Fetch mining difficulty history."""
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.get(
                    f"{BLOCKCHAIN_INFO_BASE}/charts/difficulty",
                    params={"timespan": "365days", "format": "json"},
                )
                if resp.status_code == 200:
                    data = resp.json()
                    records = []
                    for point in data.get("values", []):
                        records.append({
                            "timestamp": pd.to_datetime(point["x"], unit="s", utc=True),
                            "difficulty": point["y"],
                        })
                    df = pd.DataFrame(records)
                    if not df.empty:
                        df.set_index("timestamp", inplace=True)
                    return df
        except Exception as e:
            logger.warning(f"Difficulty fetch failed: {e}")
        return pd.DataFrame()

    def compute_hash_price(self, revenue_df: pd.DataFrame, hashrate_df: pd.DataFrame) -> pd.DataFrame:
        """Compute hash price = daily revenue / hash rate.

        Hash price below production cost = miner capitulation signal.
        """
        if revenue_df.empty or hashrate_df.empty:
            return pd.DataFrame()

        combined = revenue_df.join(hashrate_df, how="inner")
        combined["hash_price"] = combined["miner_revenue_usd"] / (combined["hash_rate_eh"] * 1e6)
        return combined[["hash_price"]]

    def detect_capitulation(self, hashrate_df: pd.DataFrame, window: int = 30) -> bool:
        """Detect miner capitulation: hash rate declining for extended period."""
        if hashrate_df.empty or len(hashrate_df) < window:
            return False
        recent = hashrate_df["hash_rate_eh"].tail(window)
        return recent.iloc[-1] < recent.iloc[0] * 0.95  # 5% decline over window

    async def _collect(self) -> pd.DataFrame:
        """Collect latest miner metrics."""
        revenue = await self.get_miner_revenue()
        hashrate = await self.get_hash_rate()

        results = {}
        if not hashrate.empty:
            results["hash_rate_eh"] = hashrate["hash_rate_eh"].iloc[-1]
            results["hash_rate_30d_change"] = (
                hashrate["hash_rate_eh"].iloc[-1] / hashrate["hash_rate_eh"].iloc[-30] - 1
            ) * 100 if len(hashrate) >= 30 else 0
            results["miner_capitulation"] = self.detect_capitulation(hashrate)

        if not revenue.empty:
            results["miner_revenue_usd"] = revenue["miner_revenue_usd"].iloc[-1]

        if not results:
            return pd.DataFrame()

        df = pd.DataFrame([results], index=[pd.Timestamp.now(tz="UTC")])
        return df

    async def collect_history(self, start: str, end: str | None = None) -> pd.DataFrame:
        revenue = await self.get_miner_revenue()
        hashrate = await self.get_hash_rate()

        frames = [f for f in [revenue, hashrate] if not f.empty]
        if not frames:
            return pd.DataFrame()

        combined = pd.concat(frames, axis=1)
        mask = combined.index >= pd.Timestamp(start, tz="UTC")
        if end:
            mask &= combined.index <= pd.Timestamp(end, tz="UTC")
        return combined[mask]
