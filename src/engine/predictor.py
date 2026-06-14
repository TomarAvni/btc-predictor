"""Prediction engine -- orchestrates collect -> features -> predict -> output.

Loads trained ML models from data/validation/models/ when available and
falls back to TA heuristics only if models are missing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.collectors.cycle import CycleCollector
from src.collectors.price import PriceCollector
from src.collectors.technical import TechnicalCollector
from src.features.engineer import FeatureEngineer
from src.features.temporal import TemporalFeatures
from src.models.baseline_model import BaselineModel
from src.models.xgboost_model import XGBoostPredictor
from src.output.formatter import PredictionFormatter
from src.output.text_logger import PredictionLogger
from src.training.feature_builder import TrainingFeatureBuilder
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

TIMEFRAMES = ["24h", "7d", "30d", "90d"]
ENSEMBLE_WEIGHTS = {"baseline": 0.35, "xgboost": 0.65}
# Minimum display magnitude per horizon when regressor/heuristic returns ~0
MIN_MAGNITUDE = {"24h": 0.3, "7d": 0.8, "30d": 2.0, "90d": 4.0}


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
        models_path = cfg.get("data", {}).get(
            "models_path", "data/validation/models"
        )

        self.price = PriceCollector(storage_path=storage)
        self.technical = TechnicalCollector(self.price)
        self.cycle = CycleCollector()
        self.temporal = TemporalFeatures()
        self.engineer = FeatureEngineer()
        self.formatter = PredictionFormatter()
        self.logger = PredictionLogger(output_path=pred_file)

        self.model_dir = Path(models_path)
        self.feature_builder = TrainingFeatureBuilder(model_dir=self.model_dir)
        self._baseline_models: dict[str, BaselineModel] = {}
        self._xgboost_models: dict[str, XGBoostPredictor] = {}
        self._models_loaded = False
        self._using_ml = False

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
          5. Generate predictions (ML models or heuristic fallback)
          6. Format and log output
        """
        logger.info("Running prediction cycle ...")
        price_df = await self.price.collect()
        if price_df.empty:
            logger.error("No price data available; aborting prediction")
            return {}

        ta_df = await self.technical.collect()
        cycle_info = self.cycle.get_cycle_position()
        await self.cycle.collect()
        self.temporal.compute(price_df.tail(1000))

        latest_price = await self.price.get_latest()
        latest_ta = await self.technical.get_latest()
        signals = self._build_signal_summary(latest_price, latest_ta, cycle_info)

        predictions = self._generate_predictions(price_df, ta_df)
        self.logger.log_prediction(predictions, signals)

        report = self.formatter.format_report(predictions, signals)
        print(report)

        return {
            "predictions": predictions,
            "signals": signals,
            "cycle": cycle_info,
            "using_ml": self._using_ml,
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
            "ml_models": self._models_available(),
        }

    # ------------------------------------------------------------------
    # Model loading & prediction
    # ------------------------------------------------------------------

    def _models_available(self) -> bool:
        if not self.model_dir.exists():
            return False
        for tf in TIMEFRAMES:
            baseline_path = self.model_dir / f"baseline_{tf}" / "baseline_model.pkl"
            xgb_path = self.model_dir / f"xgb_{tf}" / "direction.pkl"
            if baseline_path.exists() or xgb_path.exists():
                return True
        return False

    def _load_models(self) -> bool:
        if self._models_loaded:
            return self._using_ml

        self._models_loaded = True
        if not self._models_available():
            logger.warning(
                "No trained models in %s — using heuristic predictions",
                self.model_dir,
            )
            self._using_ml = False
            return False

        loaded_any = False
        for tf in TIMEFRAMES:
            baseline_path = self.model_dir / f"baseline_{tf}"
            xgb_path = self.model_dir / f"xgb_{tf}"

            if (baseline_path / "baseline_model.pkl").exists():
                baseline = BaselineModel()
                baseline.load(baseline_path)
                self._baseline_models[tf] = baseline
                loaded_any = True

            xgb_dir = xgb_path / "direction.pkl"
            xgb_mag = xgb_path / "magnitude.pkl"
            if xgb_dir.exists() and xgb_mag.exists():
                xgb = XGBoostPredictor(timeframe=tf)
                xgb._model_path = xgb_path
                xgb.load()
                if xgb.model_direction is not None and xgb.model_magnitude is not None:
                    self._xgboost_models[tf] = xgb
                    loaded_any = True
                elif xgb_dir.exists():
                    logger.warning(
                        "XGBoost %s: direction model present but magnitude regressor "
                        "missing or failed to load — skipping timeframe",
                        tf,
                    )

        self._using_ml = loaded_any
        if loaded_any:
            logger.info("Using ML models from %s", self.model_dir)
        else:
            logger.warning("Model files found but none loaded — using heuristics")
        return loaded_any

    def _generate_predictions(
        self,
        price_df: pd.DataFrame,
        ta_df: pd.DataFrame,
    ) -> list[dict[str, Any]]:
        if self._load_models():
            ml_preds = self._ml_predictions(price_df, ta_df)
            if ml_preds:
                return ml_preds

        logger.info("Falling back to heuristic predictions")
        latest_ta = ta_df.iloc[-1].to_dict() if not ta_df.empty else {}
        cycle_info = self.cycle.get_cycle_position()
        return self._heuristic_predictions(latest_ta, cycle_info)

    def _ml_predictions(
        self,
        price_df: pd.DataFrame,
        ta_df: pd.DataFrame,
    ) -> list[dict[str, Any]]:
        merged = self._build_merged_data(price_df, ta_df)
        if merged.empty:
            return []

        features_df = self.feature_builder.build_features(merged)
        if features_df.empty:
            return []

        numeric = features_df.select_dtypes(include=[np.number]).fillna(0)
        latest = numeric.iloc[-1]
        features_dict = {col: float(latest[col]) for col in numeric.columns}

        predictions: list[dict[str, Any]] = []
        for tf in TIMEFRAMES:
            baseline = self._baseline_models.get(tf)
            xgb = self._xgboost_models.get(tf)
            if not baseline and not xgb:
                continue

            if baseline and xgb:
                bp = baseline.predict(features_dict)
                xp = xgb.predict(features_dict)
                ens_prob = (
                    ENSEMBLE_WEIGHTS["baseline"] * bp["direction_prob"]
                    + ENSEMBLE_WEIGHTS["xgboost"] * xp["direction_prob"]
                )
                ens_mag = (
                    ENSEMBLE_WEIGHTS["baseline"] * abs(bp["predicted_magnitude"])
                    + ENSEMBLE_WEIGHTS["xgboost"] * abs(xp["predicted_magnitude"])
                )
                raw_conf = abs(ens_prob - 0.5) * 2
            elif xgb:
                xp = xgb.predict(features_dict)
                ens_prob = xp["direction_prob"]
                ens_mag = abs(xp["predicted_magnitude"])
                raw_conf = xp.get("raw_confidence", abs(ens_prob - 0.5) * 2)
            else:
                bp = baseline.predict(features_dict)
                ens_prob = bp["direction_prob"]
                ens_mag = abs(bp["predicted_magnitude"])
                raw_conf = bp.get("confidence", abs(ens_prob - 0.5) * 2)

            direction = "UP" if ens_prob > 0.5 else "DOWN"
            confidence = raw_conf * 100 if raw_conf <= 1.0 else raw_conf
            confidence = max(10.0, min(95.0, confidence))

            predictions.append({
                "timeframe": tf,
                "direction": direction,
                "magnitude": self._display_magnitude(ens_mag, ens_prob, tf),
                "confidence": int(round(confidence)),
            })

        return predictions

    @staticmethod
    def _build_merged_data(
        price_df: pd.DataFrame,
        ta_df: pd.DataFrame,
    ) -> pd.DataFrame:
        merged = price_df.copy()
        if ta_df.empty:
            return merged

        ta_cols = [c for c in ta_df.columns if c not in merged.columns]
        if ta_cols:
            merged = merged.join(ta_df[ta_cols], how="left")
        return merged

    # ------------------------------------------------------------------
    # Heuristic fallback
    # ------------------------------------------------------------------

    @staticmethod
    def _heuristic_predictions(
        ta_signals: dict[str, Any],
        cycle_info: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Produce directional predictions from simple TA heuristics."""
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
            strength = max(abs(ratio - 0.5) * 2, 0.15)
            raw_mag = h["mag_scale"] * strength
            prob = ratio if direction == "UP" else 1.0 - ratio
            magnitude = PredictionEngine._display_magnitude(raw_mag, prob, h["label"])
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
    def _display_magnitude(
        raw_magnitude: float,
        direction_prob: float,
        timeframe: str,
    ) -> float:
        """Return a display-ready magnitude, never rounding a directional call to 0.0%."""
        mag = abs(raw_magnitude)
        if round(mag, 1) >= 0.1:
            return round(mag, 1)

        floor = MIN_MAGNITUDE.get(timeframe, 1.0)
        strength = max(abs(direction_prob - 0.5) * 2, 0.15)
        estimated = floor * strength
        return round(max(mag, estimated), 1)

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
