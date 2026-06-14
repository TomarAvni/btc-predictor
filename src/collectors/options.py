"""Options market data collector: put/call ratio, max pain, expiry calendar.

Options expiry dates (monthly/quarterly) cause predictable volatility.
Max pain price = the price at which the most options expire worthless,
acting as a gravity point for price.
"""

import httpx
import pandas as pd

from src.collectors import BaseCollector
from src.utils.cache import DiskCache
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

# Deribit is the dominant BTC options exchange
DERIBIT_BASE = "https://www.deribit.com/api/v2/public"


class OptionsCollector(BaseCollector):
    """Collects BTC options market data from Deribit."""

    name = "options"
    tier = 2
    update_interval_seconds = 3600

    def __init__(self):
        self.cache = DiskCache()

    async def get_options_summary(self) -> dict:
        """Fetch BTC options market overview: OI, volume, put/call ratio."""
        cache_key = "options_summary"
        cached = self.cache.get(cache_key, max_age_seconds=3600)
        if cached:
            return cached

        try:
            async with httpx.AsyncClient(timeout=30) as client:
                # Get all BTC option instruments
                resp = await client.get(
                    f"{DERIBIT_BASE}/get_book_summary_by_currency",
                    params={"currency": "BTC", "kind": "option"},
                )
                if resp.status_code != 200:
                    return {}

                data = resp.json().get("result", [])

            total_call_oi = 0.0
            total_put_oi = 0.0
            total_call_volume = 0.0
            total_put_volume = 0.0

            for instrument in data:
                name = instrument.get("instrument_name", "")
                oi = instrument.get("open_interest", 0)
                vol = instrument.get("volume", 0)

                if "-C" in name:
                    total_call_oi += oi
                    total_call_volume += vol
                elif "-P" in name:
                    total_put_oi += oi
                    total_put_volume += vol

            total_oi = total_call_oi + total_put_oi
            put_call_ratio = total_put_oi / total_call_oi if total_call_oi > 0 else 0

            result = {
                "put_call_ratio": round(put_call_ratio, 4),
                "total_call_oi_btc": round(total_call_oi, 2),
                "total_put_oi_btc": round(total_put_oi, 2),
                "total_options_oi_btc": round(total_oi, 2),
                "total_call_volume_btc": round(total_call_volume, 2),
                "total_put_volume_btc": round(total_put_volume, 2),
            }
            self.cache.set(cache_key, result)
            return result

        except Exception as e:
            logger.warning(f"Options summary fetch failed: {e}")
            return {}

    async def get_max_pain(self) -> dict:
        """Calculate max pain for nearest expiry.

        Max pain = strike price where most options expire worthless.
        Price tends to gravitate toward max pain near expiry.
        """
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                # Get instruments to find nearest expiry
                resp = await client.get(
                    f"{DERIBIT_BASE}/get_instruments",
                    params={"currency": "BTC", "kind": "option", "expired": "false"},
                )
                if resp.status_code != 200:
                    return {}

                instruments = resp.json().get("result", [])

            if not instruments:
                return {}

            # Find nearest expiry
            expiries = set()
            for inst in instruments:
                exp = inst.get("expiration_timestamp")
                if exp:
                    expiries.add(exp)

            if not expiries:
                return {}

            nearest_expiry = min(expiries)
            nearest_instruments = [
                i for i in instruments
                if i.get("expiration_timestamp") == nearest_expiry
            ]

            # Collect strikes and OI
            strikes = {}
            for inst in nearest_instruments:
                strike = inst.get("strike")
                if strike is None:
                    continue
                if strike not in strikes:
                    strikes[strike] = {"call_oi": 0, "put_oi": 0}

                name = inst.get("instrument_name", "")
                # We'd need another API call to get OI per instrument
                # For now, note the available strikes
                if "-C" in name:
                    strikes[strike]["call_oi"] += 1
                elif "-P" in name:
                    strikes[strike]["put_oi"] += 1

            expiry_dt = pd.to_datetime(nearest_expiry, unit="ms", utc=True)
            days_to_expiry = (expiry_dt - pd.Timestamp.now(tz="UTC")).total_seconds() / 86400

            return {
                "nearest_expiry": expiry_dt.isoformat(),
                "days_to_expiry": round(days_to_expiry, 1),
                "num_strikes": len(strikes),
                "note": "Full max pain calculation requires per-instrument OI data",
            }

        except Exception as e:
            logger.warning(f"Max pain calculation failed: {e}")
            return {}

    def interpret_put_call_ratio(self, ratio: float) -> str:
        """Interpret put/call OI ratio."""
        if ratio > 0.8:
            return "high_put_demand_hedging_or_bearish"
        elif ratio > 0.6:
            return "moderate_put_interest_cautious"
        elif ratio > 0.4:
            return "balanced_neutral"
        elif ratio > 0.2:
            return "call_dominated_bullish"
        else:
            return "extreme_call_demand_euphoric"

    async def _collect(self) -> pd.DataFrame:
        summary = await self.get_options_summary()
        max_pain = await self.get_max_pain()

        row = {**summary, **max_pain}
        if not row:
            return pd.DataFrame()

        df = pd.DataFrame([row], index=[pd.Timestamp.now(tz="UTC")])
        return df

    async def collect_history(self, start: str, end: str | None = None) -> pd.DataFrame:
        # Options data is mostly point-in-time; historical requires paid APIs
        return pd.DataFrame()
