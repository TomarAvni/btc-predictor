"""Technical analysis indicators computed from price data.

Computes RSI, MACD, Bollinger Bands, EMA crossovers, ATR, volume profile,
and support/resistance levels using the `ta` library.  All indicators are
aligned to the price DataFrame's DatetimeIndex.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import ta

from src.collectors import BaseCollector
from src.collectors.price import PriceCollector
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class TechnicalCollector(BaseCollector):
    """Computes TA indicators from hourly BTC price data."""

    name = "technical"
    update_interval_seconds = 900  # 15 min (fast tier)

    def __init__(self, price_collector: PriceCollector) -> None:
        self.price = price_collector

    # ------------------------------------------------------------------
    # Indicator computation
    # ------------------------------------------------------------------

    def compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute all technical indicators on a price OHLCV DataFrame.

        Requires at least 200 rows so the longest-window indicators
        (EMA-200) have enough data.
        """
        if df.empty or len(df) < 200:
            logger.warning("Not enough data for TA (%d rows, need 200+)", len(df))
            return df

        out = df.copy()

        # --- RSI (multiple periods) ---
        out["rsi_14"] = ta.momentum.RSIIndicator(out["close"], window=14).rsi()
        out["rsi_7"] = ta.momentum.RSIIndicator(out["close"], window=7).rsi()

        # --- MACD (12, 26, 9) ---
        macd = ta.trend.MACD(out["close"], window_slow=26, window_fast=12, window_sign=9)
        out["macd"] = macd.macd()
        out["macd_signal"] = macd.macd_signal()
        out["macd_histogram"] = macd.macd_diff()

        # --- Bollinger Bands (20, 2) ---
        bb = ta.volatility.BollingerBands(out["close"], window=20, window_dev=2)
        out["bb_upper"] = bb.bollinger_hband()
        out["bb_lower"] = bb.bollinger_lband()
        out["bb_middle"] = bb.bollinger_mavg()
        out["bb_width"] = (out["bb_upper"] - out["bb_lower"]) / out["bb_middle"]

        # --- EMAs ---
        for window in (9, 21, 50, 200):
            out[f"ema_{window}"] = (
                ta.trend.EMAIndicator(out["close"], window=window).ema_indicator()
            )

        # --- EMA crossover signals (1 = fast above slow) ---
        out["ema_9_21_cross"] = (out["ema_9"] > out["ema_21"]).astype(int)
        out["ema_50_200_cross"] = (out["ema_50"] > out["ema_200"]).astype(int)

        # --- ATR (14-period) ---
        out["atr_14"] = ta.volatility.AverageTrueRange(
            out["high"], out["low"], out["close"], window=14
        ).average_true_range()

        # --- Volume profile ---
        out["volume_sma_20"] = out["volume"].rolling(window=20).mean()
        out["volume_ratio"] = out["volume"] / out["volume_sma_20"]

        # --- Stochastic RSI ---
        stoch = ta.momentum.StochRSIIndicator(out["close"])
        out["stoch_rsi_k"] = stoch.stochrsi_k()
        out["stoch_rsi_d"] = stoch.stochrsi_d()

        # --- ADX (trend strength) ---
        adx = ta.trend.ADXIndicator(out["high"], out["low"], out["close"])
        out["adx"] = adx.adx()

        # --- OBV ---
        out["obv"] = ta.volume.OnBalanceVolumeIndicator(
            out["close"], out["volume"]
        ).on_balance_volume()

        # --- VWAP (resets daily) ---
        out["vwap"] = self._compute_vwap(out)

        # --- Price position within recent range (0 = bottom, 1 = top) ---
        out["price_position_24h"] = self._price_position(out["close"], 24)
        out["price_position_168h"] = self._price_position(out["close"], 168)

        return out

    # ------------------------------------------------------------------
    # Support / resistance
    # ------------------------------------------------------------------

    def compute_support_resistance(
        self, df: pd.DataFrame, lookback: int = 720
    ) -> dict[str, list[float]]:
        """Identify support/resistance from recent pivot points."""
        recent = df.tail(lookback)
        if recent.empty:
            return {"support": [], "resistance": []}

        current_price = float(recent["close"].iloc[-1])

        highs = recent["high"].rolling(24).max().dropna()
        lows = recent["low"].rolling(24).min().dropna()

        resistance = self._cluster_levels(
            highs[highs > current_price].values, current_price
        )
        support = self._cluster_levels(
            lows[lows < current_price].values, current_price
        )

        return {
            "support": sorted(support, reverse=True)[:5],
            "resistance": sorted(resistance)[:5],
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_vwap(df: pd.DataFrame) -> pd.Series:
        """VWAP that resets each UTC day."""
        typical = (df["high"] + df["low"] + df["close"]) / 3
        tp_vol = typical * df["volume"]

        day_groups = df.index.date
        vwap = pd.Series(index=df.index, dtype=float)

        for day in pd.unique(day_groups):
            mask = day_groups == day
            cum_tp_vol = tp_vol[mask].cumsum()
            cum_vol = df["volume"][mask].cumsum()
            vwap[mask] = cum_tp_vol / cum_vol.replace(0, float("nan"))

        return vwap

    @staticmethod
    def _price_position(close: pd.Series, window: int) -> pd.Series:
        high = close.rolling(window).max()
        low = close.rolling(window).min()
        rng = high - low
        return (close - low) / rng.replace(0, float("nan"))

    @staticmethod
    def _cluster_levels(
        prices, reference: float, threshold_pct: float = 0.02
    ) -> list[float]:
        """Merge nearby price levels into averaged clusters."""
        if len(prices) == 0:
            return []

        sorted_p = sorted(prices)
        threshold = reference * threshold_pct
        clusters: list[float] = []
        current: list[float] = [sorted_p[0]]

        for p in sorted_p[1:]:
            if p - current[-1] <= threshold:
                current.append(p)
            else:
                clusters.append(sum(current) / len(current))
                current = [p]
        clusters.append(sum(current) / len(current))
        return clusters

    # ------------------------------------------------------------------
    # BaseCollector interface
    # ------------------------------------------------------------------

    async def _collect(self) -> pd.DataFrame:
        hourly = self.price.load_history()
        if hourly.empty:
            return pd.DataFrame()
        return self.compute_indicators(hourly.tail(1000))

    async def _get_latest(self) -> dict[str, Any]:
        df = await self._collect()
        if df.empty:
            return {}
        row = df.iloc[-1]
        return {
            "timestamp": str(df.index[-1]),
            "rsi_14": _safe_float(row.get("rsi_14")),
            "macd": _safe_float(row.get("macd")),
            "macd_signal": _safe_float(row.get("macd_signal")),
            "bb_width": _safe_float(row.get("bb_width")),
            "ema_9_21_cross": int(row.get("ema_9_21_cross", 0)),
            "ema_50_200_cross": int(row.get("ema_50_200_cross", 0)),
            "atr_14": _safe_float(row.get("atr_14")),
            "volume_ratio": _safe_float(row.get("volume_ratio")),
            "adx": _safe_float(row.get("adx")),
        }

    async def _get_historical(
        self, start: str, end: str | None = None
    ) -> pd.DataFrame:
        hourly = await self.price.get_historical(start, end)
        return self.compute_indicators(hourly)


def _safe_float(val) -> float | None:
    try:
        v = float(val)
        return None if pd.isna(v) else round(v, 4)
    except (TypeError, ValueError):
        return None
