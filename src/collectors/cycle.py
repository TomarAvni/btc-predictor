"""BTC halving cycle analysis and power law corridor.

Tracks the current position within the ~4-year halving cycle, compares
performance to prior cycles, and computes the power-law regression band
(log price vs. log days since genesis).
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from src.collectors import BaseCollector
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

HALVING_DATES: list[datetime] = [
    datetime(2012, 11, 28, tzinfo=timezone.utc),
    datetime(2016, 7, 9, tzinfo=timezone.utc),
    datetime(2020, 5, 11, tzinfo=timezone.utc),
    datetime(2024, 4, 19, tzinfo=timezone.utc),
]

AVG_CYCLE_DAYS = 1460  # ~4 years between halvings
GENESIS = datetime(2009, 1, 3, tzinfo=timezone.utc)


class CycleCollector(BaseCollector):
    """Halving-cycle position tracker and power-law corridor calculator."""

    name = "cycle"
    update_interval_seconds = 86400  # daily

    # ------------------------------------------------------------------
    # Cycle position
    # ------------------------------------------------------------------

    def get_cycle_position(self, date: datetime | None = None) -> dict[str, Any]:
        """Where are we in the current halving cycle?

        Returns a dict with cycle_number, days_since_halving,
        pct_through_cycle, phase label, and estimated next halving.
        """
        if date is None:
            date = datetime.now(timezone.utc)

        past = [h for h in HALVING_DATES if h <= date]
        if not past:
            return {
                "cycle_number": 0,
                "days_since_halving": 0,
                "pct_through_cycle": 0.0,
                "phase": "pre_halving",
                "estimated_next_halving": HALVING_DATES[0].isoformat(),
                "last_halving": None,
            }

        last_halving = past[-1]
        cycle_number = HALVING_DATES.index(last_halving) + 1
        days_since = (date - last_halving).days
        pct = days_since / AVG_CYCLE_DAYS
        next_est = last_halving + pd.Timedelta(days=AVG_CYCLE_DAYS)

        return {
            "cycle_number": cycle_number,
            "days_since_halving": days_since,
            "pct_through_cycle": round(pct, 4),
            "phase": self._classify_phase(pct),
            "estimated_next_halving": next_est.isoformat(),
            "last_halving": last_halving.isoformat(),
        }

    @staticmethod
    def _classify_phase(pct: float) -> str:
        """Map cycle percentage to a named phase.

        Historical pattern:
          0-15 %  post-halving accumulation
         15-50 %  bull-run acceleration
         50-75 %  euphoria / blow-off top
         75-100%  bear market / late-cycle accumulation
        """
        if pct < 0.15:
            return "post_halving_accumulation"
        if pct < 0.50:
            return "bull_acceleration"
        if pct < 0.75:
            return "euphoria_zone"
        return "bear_accumulation"

    # ------------------------------------------------------------------
    # Historical cycle comparison
    # ------------------------------------------------------------------

    def get_historical_comparison(self, price_df: pd.DataFrame) -> pd.DataFrame:
        """Align each cycle to day-0 = halving date and compute ROI curves.

        Returns a DataFrame where each column is a cycle and the index
        is days since halving.
        """
        if price_df.empty:
            return pd.DataFrame()

        daily = price_df.resample("1D").agg({"close": "last"}).dropna()
        cycles: dict[str, pd.Series] = {}

        for i, halving in enumerate(HALVING_DATES):
            cycle_end = (
                HALVING_DATES[i + 1]
                if i + 1 < len(HALVING_DATES)
                else datetime.now(timezone.utc)
            )
            mask = (daily.index >= halving) & (daily.index < cycle_end)
            chunk = daily[mask]
            if chunk.empty:
                continue

            base = float(chunk["close"].iloc[0])
            roi = (chunk["close"] / base - 1) * 100
            days = (chunk.index - halving).days
            cycles[f"cycle_{i + 1}"] = pd.Series(roi.values, index=days)

        return pd.DataFrame(cycles) if cycles else pd.DataFrame()

    # ------------------------------------------------------------------
    # Power-law corridor
    # ------------------------------------------------------------------

    def compute_power_law(self, price_df: pd.DataFrame) -> dict[str, Any]:
        """Fit log(price) ~ a * log(days_since_genesis) + b.

        Returns fair value, upper/lower bands (+-2 sigma), and the
        current price's position within the band (0 = floor, 1 = ceiling).
        """
        if price_df.empty:
            return {}

        daily = price_df.resample("1D").agg({"close": "last"}).dropna()

        days_since = (daily.index - GENESIS).total_seconds() / 86400
        log_days = np.log10(days_since.values)
        log_price = np.log10(daily["close"].values)

        valid = np.isfinite(log_days) & np.isfinite(log_price) & (log_days > 0)
        if valid.sum() < 100:
            return {}

        slope, intercept = np.polyfit(log_days[valid], log_price[valid], 1)

        now_days = (datetime.now(timezone.utc) - GENESIS).total_seconds() / 86400
        log_now = math.log10(now_days)
        fair_value = 10 ** (slope * log_now + intercept)

        residuals = log_price[valid] - (slope * log_days[valid] + intercept)
        std = float(np.std(residuals))

        upper = 10 ** (slope * log_now + intercept + 2 * std)
        lower = 10 ** (slope * log_now + intercept - 2 * std)

        current_price = float(daily["close"].iloc[-1])
        position = (
            (math.log10(current_price) - (slope * log_now + intercept - 2 * std))
            / (4 * std)
        )

        return {
            "fair_value": round(fair_value, 2),
            "upper_band": round(upper, 2),
            "lower_band": round(lower, 2),
            "current_price": round(current_price, 2),
            "position_in_band": round(position, 4),
            "slope": round(slope, 6),
        }

    # ------------------------------------------------------------------
    # BaseCollector interface
    # ------------------------------------------------------------------

    async def _collect(self) -> pd.DataFrame:
        pos = self.get_cycle_position()
        row = {
            "cycle_number": pos["cycle_number"],
            "days_since_halving": pos["days_since_halving"],
            "pct_through_cycle": pos["pct_through_cycle"],
            "phase": pos["phase"],
        }
        return pd.DataFrame([row], index=[pd.Timestamp.now(tz="UTC")])

    async def _get_latest(self) -> dict[str, Any]:
        return self.get_cycle_position()

    async def _get_historical(
        self, start: str, end: str | None = None
    ) -> pd.DataFrame:
        start_dt = pd.Timestamp(start, tz="UTC")
        end_dt = pd.Timestamp(end, tz="UTC") if end else pd.Timestamp.now(tz="UTC")

        dates = pd.date_range(start_dt, end_dt, freq="1D")
        rows = [
            {
                "timestamp": d,
                **{
                    k: v
                    for k, v in self.get_cycle_position(d.to_pydatetime()).items()
                    if k in ("cycle_number", "days_since_halving", "pct_through_cycle", "phase")
                },
            }
            for d in dates
        ]
        df = pd.DataFrame(rows)
        if not df.empty:
            df.set_index("timestamp", inplace=True)
        return df
