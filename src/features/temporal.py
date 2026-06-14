"""Temporal / calendar features.

Extracts cyclical time-of-day, day-of-week, month-of-year, and
session-based features from a DatetimeIndex.  Uses sine/cosine
encoding so that e.g. hour-23 and hour-0 are adjacent in feature space.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class TemporalFeatures:
    """Generate temporal feature columns from a DatetimeIndex."""

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add temporal columns to *df* (must have a DatetimeIndex).

        Columns added:
          hour_sin, hour_cos            -- time of day (cyclical)
          dow_sin, dow_cos              -- day of week (cyclical)
          month_sin, month_cos          -- month of year (cyclical)
          is_weekend                    -- 1 if Sat/Sun
          is_us_session                 -- 1 during NYSE hours (13:30-20:00 UTC)
          is_asia_session               -- 1 during Asian session (00:00-08:00 UTC)
          quarter                       -- calendar quarter 1-4
        """
        if df.empty:
            return df

        idx = df.index
        if not isinstance(idx, pd.DatetimeIndex):
            logger.warning("Expected DatetimeIndex; skipping temporal features")
            return df

        out = df.copy()

        hour = idx.hour.values.astype(float)
        out["hour_sin"] = np.sin(2 * math.pi * hour / 24)
        out["hour_cos"] = np.cos(2 * math.pi * hour / 24)

        dow = idx.dayofweek.values.astype(float)
        out["dow_sin"] = np.sin(2 * math.pi * dow / 7)
        out["dow_cos"] = np.cos(2 * math.pi * dow / 7)

        month = idx.month.values.astype(float)
        out["month_sin"] = np.sin(2 * math.pi * month / 12)
        out["month_cos"] = np.cos(2 * math.pi * month / 12)

        out["is_weekend"] = (idx.dayofweek >= 5).astype(int)
        out["is_us_session"] = ((idx.hour >= 13) & (idx.hour < 20)).astype(int)
        out["is_asia_session"] = ((idx.hour >= 0) & (idx.hour < 8)).astype(int)
        out["quarter"] = idx.quarter

        return out
