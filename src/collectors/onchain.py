"""On-chain metrics collector: exchange flows, MVRV, active addresses, realized price."""

import httpx
import pandas as pd

from src.collectors import BaseCollector
from src.utils.cache import DiskCache
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

# Free on-chain data sources
BLOCKCHAIN_INFO_BASE = "https://api.blockchain.info"
BLOCKCHAIR_BASE = "https://api.blockchair.com/bitcoin"


class OnChainCollector(BaseCollector):
    """Collects on-chain BTC metrics from public APIs."""

    name = "onchain"
    tier = 3
    update_interval_seconds = 21600

    def __init__(self):
        self.cache = DiskCache()

    async def get_exchange_reserves(self) -> pd.DataFrame:
        """Estimate exchange reserves via known exchange addresses.

        High exchange reserves = selling pressure potential.
        Declining reserves = supply squeeze (bullish).
        """
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                # Blockchain.info provides aggregate exchange balance estimates
                resp = await client.get(
                    f"{BLOCKCHAIN_INFO_BASE}/charts/balance",
                    params={"timespan": "30days", "format": "json"},
                )
                if resp.status_code == 200:
                    data = resp.json()
                    records = []
                    for point in data.get("values", []):
                        records.append({
                            "timestamp": pd.to_datetime(point["x"], unit="s", utc=True),
                            "total_balance": point["y"] / 1e8,  # satoshi to BTC
                        })
                    df = pd.DataFrame(records)
                    if not df.empty:
                        df.set_index("timestamp", inplace=True)
                    return df

        except Exception as e:
            logger.warning(f"Exchange reserves fetch failed: {e}")

        return pd.DataFrame()

    async def get_active_addresses(self) -> pd.DataFrame:
        """Fetch daily active address count.

        Rising active addresses = growing network usage (bullish).
        """
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.get(
                    f"{BLOCKCHAIN_INFO_BASE}/charts/n-unique-addresses",
                    params={"timespan": "180days", "format": "json"},
                )
                if resp.status_code == 200:
                    data = resp.json()
                    records = []
                    for point in data.get("values", []):
                        records.append({
                            "timestamp": pd.to_datetime(point["x"], unit="s", utc=True),
                            "active_addresses": point["y"],
                        })
                    df = pd.DataFrame(records)
                    if not df.empty:
                        df.set_index("timestamp", inplace=True)
                    return df

        except Exception as e:
            logger.warning(f"Active addresses fetch failed: {e}")

        return pd.DataFrame()

    async def get_hash_rate(self) -> pd.DataFrame:
        """Fetch network hash rate (security and miner health indicator)."""
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.get(
                    f"{BLOCKCHAIN_INFO_BASE}/charts/hash-rate",
                    params={"timespan": "180days", "format": "json"},
                )
                if resp.status_code == 200:
                    data = resp.json()
                    records = []
                    for point in data.get("values", []):
                        records.append({
                            "timestamp": pd.to_datetime(point["x"], unit="s", utc=True),
                            "hash_rate_th": point["y"],
                        })
                    df = pd.DataFrame(records)
                    if not df.empty:
                        df.set_index("timestamp", inplace=True)
                    return df

        except Exception as e:
            logger.warning(f"Hash rate fetch failed: {e}")

        return pd.DataFrame()

    async def get_transaction_volume(self) -> pd.DataFrame:
        """Fetch daily transaction volume in BTC."""
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.get(
                    f"{BLOCKCHAIN_INFO_BASE}/charts/estimated-transaction-volume",
                    params={"timespan": "180days", "format": "json"},
                )
                if resp.status_code == 200:
                    data = resp.json()
                    records = []
                    for point in data.get("values", []):
                        records.append({
                            "timestamp": pd.to_datetime(point["x"], unit="s", utc=True),
                            "tx_volume_btc": point["y"] / 1e8,
                        })
                    df = pd.DataFrame(records)
                    if not df.empty:
                        df.set_index("timestamp", inplace=True)
                    return df

        except Exception as e:
            logger.warning(f"Transaction volume fetch failed: {e}")

        return pd.DataFrame()

    async def get_mempool_size(self) -> pd.DataFrame:
        """Fetch mempool transaction count (network congestion)."""
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.get(
                    f"{BLOCKCHAIN_INFO_BASE}/charts/mempool-count",
                    params={"timespan": "30days", "format": "json"},
                )
                if resp.status_code == 200:
                    data = resp.json()
                    records = []
                    for point in data.get("values", []):
                        records.append({
                            "timestamp": pd.to_datetime(point["x"], unit="s", utc=True),
                            "mempool_tx_count": point["y"],
                        })
                    df = pd.DataFrame(records)
                    if not df.empty:
                        df.set_index("timestamp", inplace=True)
                    return df

        except Exception as e:
            logger.warning(f"Mempool size fetch failed: {e}")

        return pd.DataFrame()

    async def _collect(self) -> pd.DataFrame:
        """Collect latest on-chain snapshot from all available sources."""
        results = {}

        active = await self.get_active_addresses()
        if not active.empty:
            results["active_addresses"] = active["active_addresses"].iloc[-1]

        hashrate = await self.get_hash_rate()
        if not hashrate.empty:
            results["hash_rate_th"] = hashrate["hash_rate_th"].iloc[-1]

        mempool = await self.get_mempool_size()
        if not mempool.empty:
            results["mempool_tx_count"] = mempool["mempool_tx_count"].iloc[-1]

        if not results:
            return pd.DataFrame()

        df = pd.DataFrame([results], index=[pd.Timestamp.now(tz="UTC")])
        return df

    async def collect_history(self, start: str, end: str | None = None) -> pd.DataFrame:
        """Collect historical on-chain data."""
        active = await self.get_active_addresses()
        hashrate = await self.get_hash_rate()
        tx_vol = await self.get_transaction_volume()

        frames = [f for f in [active, hashrate, tx_vol] if not f.empty]
        if not frames:
            return pd.DataFrame()

        combined = pd.concat(frames, axis=1)
        mask = combined.index >= pd.Timestamp(start, tz="UTC")
        if end:
            mask &= combined.index <= pd.Timestamp(end, tz="UTC")
        return combined[mask]
