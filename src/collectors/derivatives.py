"""
Derivatives data collector.

Sources: Binance Futures API (free)
Metrics: Funding rates, open interest, long/short ratio.

High funding rate + high OI = overleveraged, correction risk.
Negative funding = shorts paying longs, potential squeeze up.
"""

from typing import Any

import httpx
import pandas as pd

from src.collectors import BaseCollector
from src.utils.logger import setup_logger

logger = setup_logger(__name__)
from src.utils.cache import get_cached, set_cached


BINANCE_FUTURES_BASE = "https://fapi.binance.com"


class DerivativesCollector(BaseCollector):
    """Collects crypto derivatives market data."""

    name = "derivatives"
    tier = 2
    update_interval_seconds = 3600

    def __init__(self, config: dict):
        self.config = config

    async def _collect(self) -> pd.DataFrame:
        data = await self._collect_dict()
        return pd.DataFrame([data]) if data else pd.DataFrame()

    async def _get_latest(self) -> dict[str, Any]:
        return await self._collect_dict()

    async def _collect_dict(self) -> dict[str, Any]:
        results = {}

        funding = await self._fetch_funding_rate()
        results.update(funding)

        oi = await self._fetch_open_interest()
        results.update(oi)

        ls_ratio = await self._fetch_long_short_ratio()
        results.update(ls_ratio)

        return results

    async def _fetch_funding_rate(self) -> dict:
        """Fetch current and recent funding rates for BTC perpetual."""
        url = f"{BINANCE_FUTURES_BASE}/fapi/v1/fundingRate"
        params = {"symbol": "BTCUSDT", "limit": 8}
        cache_key = f"{url}?symbol=BTCUSDT"
        cached = get_cached(cache_key, max_age_minutes=60)

        if not cached:
            try:
                async with httpx.AsyncClient(timeout=15) as client:
                    resp = await client.get(url, params=params)
                    resp.raise_for_status()
                    cached = resp.json()
                    set_cached(cache_key, cached)
            except Exception as e:
                logger.warning(f"Failed to fetch funding rate: {e}")
                return {"funding_rate": None}

        if not cached:
            return {"funding_rate": None}

        current_rate = float(cached[-1]["fundingRate"])
        rates = [float(r["fundingRate"]) for r in cached]
        avg_rate = sum(rates) / len(rates)

        # Interpret funding rate
        if current_rate > 0.01:
            leverage_signal = "overleveraged_longs"
        elif current_rate < -0.005:
            leverage_signal = "overleveraged_shorts"
        else:
            leverage_signal = "neutral"

        return {
            "funding_rate_current": round(current_rate * 100, 4),
            "funding_rate_avg_24h": round(avg_rate * 100, 4),
            "funding_leverage_signal": leverage_signal,
            "funding_annualized_pct": round(current_rate * 3 * 365 * 100, 2),
        }

    async def _fetch_open_interest(self) -> dict:
        """Fetch open interest for BTC futures."""
        url = f"{BINANCE_FUTURES_BASE}/fapi/v1/openInterest"
        params = {"symbol": "BTCUSDT"}
        cached = get_cached(f"{url}?BTCUSDT", max_age_minutes=60)

        if not cached:
            try:
                async with httpx.AsyncClient(timeout=15) as client:
                    resp = await client.get(url, params=params)
                    resp.raise_for_status()
                    cached = resp.json()
                    set_cached(f"{url}?BTCUSDT", cached)
            except Exception as e:
                logger.warning(f"Failed to fetch open interest: {e}")
                return {"open_interest_btc": None}

        oi_btc = float(cached.get("openInterest", 0))
        return {
            "open_interest_btc": round(oi_btc, 2),
        }

    async def _fetch_long_short_ratio(self) -> dict:
        """Fetch global long/short account ratio."""
        url = f"{BINANCE_FUTURES_BASE}/futures/data/globalLongShortAccountRatio"
        params = {"symbol": "BTCUSDT", "period": "1h", "limit": 1}
        cached = get_cached(f"{url}?BTCUSDT_1h", max_age_minutes=60)

        if not cached:
            try:
                async with httpx.AsyncClient(timeout=15) as client:
                    resp = await client.get(url, params=params)
                    resp.raise_for_status()
                    cached = resp.json()
                    set_cached(f"{url}?BTCUSDT_1h", cached)
            except Exception as e:
                logger.warning(f"Failed to fetch long/short ratio: {e}")
                return {"long_short_ratio": None}

        if not cached:
            return {"long_short_ratio": None}

        ratio = float(cached[0]["longShortRatio"])
        long_pct = float(cached[0]["longAccount"])
        short_pct = float(cached[0]["shortAccount"])

        return {
            "long_short_ratio": round(ratio, 3),
            "long_account_pct": round(long_pct * 100, 1),
            "short_account_pct": round(short_pct * 100, 1),
            "crowd_positioning": "long_heavy" if ratio > 1.5 else ("short_heavy" if ratio < 0.7 else "balanced"),
        }
