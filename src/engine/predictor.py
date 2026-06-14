"""Prediction engine -- orchestrates collect -> features -> predict -> output.

Phase 1 runs the full data pipeline (price + TA + cycle + temporal) and
produces a signal summary.  Model-based predictions are stubbed with
simple heuristics until Phase 2 delivers trained models.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.collectors.cycle import CycleCollector
from src.collectors.price import PriceCollector
from src.collectors.technical import TechnicalCollector
from src.features.engineer import FeatureEngineer
from src.features.temporal import TemporalFeatures
from src.output.formatter import PredictionFormatter
from src.output.text_logger import PredictionLogger
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class PredictionEngine:
    """Top-level orchestrator.

    Usage::

        engine = PredictionEngine()
        await engine.initialize()      # download history if needed
        await engine.run_prediction()  # single prediction cycle
    """

    def __init__(self, config: dict | None = None) -> None:
        cfg = config or {}
        storage = cfg.get("data", {}).get("price_path", "data/price")
        pred_file = cfg.get("app", {}).get("predictions_file", "predictions.log")

        self.price = PriceCollector(storage_path=storage)
        self.technical = TechnicalCollector(self.price)
        self.cycle = CycleCollector()
        self.temporal = TemporalFeatures()
        self.engineer = FeatureEngineer()
        self.formatter = PredictionFormatter()
        self.logger = PredictionLogger(output_path=pred_file)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def initialize(self) -> None:
        """Download full price history (first run or resume)."""
        logger.info("Initializing -- downloading price history ...")
        await self.price.download_full_history()
        logger.info("Initialization complete")

    async def run_prediction(self) -> dict[str, Any]:
        """Execute a single prediction cycle and log the result.

        Steps:
          1. Incremental price update
          2. Compute TA indicators
          3. Gather cycle metrics
          4. Build feature matrix
          5. Generate predictions (Phase 1: heuristic stub)
          6. Format and log output
        """
        # 1 -- Price update
        logger.info("Running prediction cycle ...")
        price_df = await self.price.collect()
        if price_df.empty:
            logger.error("No price data available; aborting prediction")
            return {}

        # 2 -- TA indicators
        ta_df = await self.technical.collect()

        # 3 -- Cycle position
        cycle_info = self.cycle.get_cycle_position()
        cycle_df = await self.cycle.collect()

        # 4 -- Temporal features
        temporal_df = self.temporal.compute(price_df.tail(1000))

        # 5 -- Build signals summary
        latest_price = await self.price.get_latest()
        latest_ta = await self.technical.get_latest()

        signals = self._build_signal_summary(latest_price, latest_ta, cycle_info)

        # 6 -- Phase 1 heuristic predictions
        predictions = self._heuristic_predictions(latest_ta, cycle_info)

        # 7 -- Log output
        self.logger.log_prediction(predictions, signals)

        report = self.formatter.format_report(predictions, signals)
        print(report)

        return {
            "predictions": predictions,
            "signals": signals,
            "cycle": cycle_info,
        }

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def get_status(self) -> dict[str, Any]:
        """Return a summary of available data."""
        df = self.price.load_history()
        if df.empty:
            return {"status": "no_data", "candles": 0}
        return {
            "status": "ok",
            "candles": len(df),
            "first": str(df.index[0]),
            "last": str(df.index[-1]),
            "timespan_days": (df.index[-1] - df.index[0]).days,
        }

    # ------------------------------------------------------------------
    # Phase 1 heuristic predictions (replaced by ML in Phase 2)
    # ------------------------------------------------------------------

    @staticmethod
    def _heuristic_predictions(
        ta_signals: dict[str, Any],
        cycle_info: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Produce directional predictions from simple TA heuristics.

        This is intentionally naive -- real predictions come from the
        trained ensemble in Phase 2.
        """
        rsi = ta_signals.get("rsi_14")
        macd = ta_signals.get("macd")
        macd_sig = ta_signals.get("macd_signal")
        ema_cross = ta_signals.get("ema_9_21_cross", 0)
        cycle_phase = cycle_info.get("phase", "")

        bullish_signals = 0
        total_signals = 0

        if rsi is not None:
            total_signals += 1
            if rsi < 70:
                bullish_signals += 1

        if macd is not None and macd_sig is not None:
            total_signals += 1
            if macd > macd_sig:
                bullish_signals += 1

        if ema_cross is not None:
            total_signals += 1
            bullish_signals += int(ema_cross)

        total_signals += 1
        if cycle_phase in ("post_halving_accumulation", "bull_acceleration"):
            bullish_signals += 1

        ratio = bullish_signals / max(total_signals, 1)
        direction = "UP" if ratio >= 0.5 else "DOWN"

        horizons = [
            {"label": "24h", "mag_scale": 1.0, "conf_base": 65},
            {"label": "7d", "mag_scale": 3.0, "conf_base": 52},
            {"label": "30d", "mag_scale": 8.0, "conf_base": 40},
            {"label": "90d", "mag_scale": 15.0, "conf_base": 30},
        ]

        predictions = []
        for h in horizons:
            magnitude = round(h["mag_scale"] * abs(ratio - 0.5) * 2, 1)
            confidence = max(
                h["conf_base"] + int((ratio - 0.5) * 20), 20
            )
            predictions.append({
                "timeframe": h["label"],
                "direction": direction,
                "magnitude": magnitude,
                "confidence": confidence,
            })

        return predictions

    @staticmethod
    def _build_signal_summary(
        price: dict, ta: dict, cycle: dict
    ) -> dict[str, dict[str, Any]]:
        """Assemble a human-readable signal summary dict."""
        signals: dict[str, dict[str, Any]] = {}

        if price.get("close"):
            signals["Price"] = {
                "value": f"${price['close']:,.2f}",
                "interpretation": "",
            }

        rsi = ta.get("rsi_14")
        if rsi is not None:
            interp = "Overbought" if rsi > 70 else "Oversold" if rsi < 30 else "Neutral"
            signals["RSI (14)"] = {"value": f"{rsi:.1f}", "interpretation": interp}

        macd_val = ta.get("macd")
        macd_sig = ta.get("macd_signal")
        if macd_val is not None and macd_sig is not None:
            cross = "Bullish crossover" if macd_val > macd_sig else "Bearish crossover"
            signals["MACD"] = {"value": f"{macd_val:.2f}", "interpretation": cross}

        adx = ta.get("adx")
        if adx is not None:
            strength = "Strong trend" if adx > 25 else "Weak/no trend"
            signals["ADX"] = {"value": f"{adx:.1f}", "interpretation": strength}

        vol = ta.get("volume_ratio")
        if vol is not None:
            interp = "Above average" if vol > 1.2 else "Below average" if vol < 0.8 else "Normal"
            signals["Volume Ratio"] = {"value": f"{vol:.2f}", "interpretation": interp}

        ema_cross = ta.get("ema_50_200_cross")
        if ema_cross is not None:
            signals["EMA 50/200"] = {
                "value": "Golden cross" if ema_cross else "Death cross",
                "interpretation": "",
            }

        phase = cycle.get("phase", "")
        pct = cycle.get("pct_through_cycle", 0)
        if phase:
            signals["Cycle Phase"] = {
                "value": phase.replace("_", " ").title(),
                "interpretation": f"{pct*100:.0f}% through cycle",
            }

        return signals


Predictor = PredictionEngine
