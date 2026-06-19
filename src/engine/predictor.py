"""Prediction engine -- orchestrates collect -> features -> predict -> output.

Loads trained ML models from data/validation/models/ when available and
falls back to TA heuristics only if models are missing.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.collectors.cycle import CycleCollector
from src.collectors.onchain_flows import LATEST_PATH as ONCHAIN_FLOW_LATEST_PATH
from src.collectors.onchain_flows import HISTORY_PATH as ONCHAIN_FLOW_HISTORY_PATH
from src.collectors.onchain_flows import OnChainFlowCollector
from src.collectors.price import PriceCollector
from src.collectors.technical import TechnicalCollector
from src.features.engineer import FeatureEngineer
from src.features.temporal import TemporalFeatures
from src.horizons import HORIZON_HOURS, LEGACY_MODEL_ALIASES, TIMEFRAMES
from src.models.baseline_model import BaselineModel
from src.models.calibration import ProbabilityCalibrator
from src.models.xgboost_model import XGBoostPredictor
from src.output.formatter import PredictionFormatter
from src.output.jsonl_logger import PredictionJSONLLogger
from src.output.text_logger import PredictionLogger
from src.training.feature_builder import TrainingFeatureBuilder
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

ENSEMBLE_WEIGHTS = {"baseline": 0.35, "xgboost": 0.65}


def _min_magnitude(timeframe: str) -> float:
    """Minimum display magnitude (%) per horizon when the model returns ~0.

    Scales with the horizon length so a longer horizon never shows an
    implausibly tiny floor.  The power fit is anchored to the historical
    values (24h ~ 0.3%, 168h ~ 0.8%, 30d ~ 2.0%).
    """
    hrs = HORIZON_HOURS.get(timeframe, 24)
    return round(0.0495 * hrs ** 0.558, 2)


class PredictionEngine:
    """Top-level orchestrator.

    Usage::

        engine = PredictionEngine()
        await engine.initialize()      # download history if needed
        await engine.run_prediction()  # single prediction cycle
    """

    def __init__(self, config: dict | None = None) -> None:
        cfg = config or {}
        self.config = cfg
        storage = cfg.get("data", {}).get("price_path", "data/price")
        pred_file = cfg.get("app", {}).get("predictions_file", "predictions.log")
        models_path = cfg.get("data", {}).get(
            "models_path", "data/validation/models"
        )
        # Second prediction module (X/Twitter sentiment). Off by setting
        # BTC_DISABLE_SENTIMENT_MODULE; otherwise runs (mock mode without keys).
        self._enable_sentiment = (
            os.environ.get("BTC_DISABLE_SENTIMENT_MODULE", "").lower()
            not in ("1", "true", "yes")
        )
        self._sentiment_manager = None

        self.price = PriceCollector(storage_path=storage)
        self.technical = TechnicalCollector(self.price)
        self.cycle = CycleCollector()
        self.onchain_flows = OnChainFlowCollector()
        self.temporal = TemporalFeatures()
        self.engineer = FeatureEngineer()
        self.formatter = PredictionFormatter()
        self.logger = PredictionLogger(output_path=pred_file)
        self.jsonl_logger = PredictionJSONLLogger()

        self.model_dir = Path(models_path)
        self.feature_builder = TrainingFeatureBuilder(model_dir=self.model_dir)
        self._baseline_models: dict[str, BaselineModel] = {}
        self._xgboost_models: dict[str, XGBoostPredictor] = {}
        self._models_loaded = False
        self._using_ml = False
        # Calibrated confidence: maps raw P(up) -> calibrated P(up) so the
        # emitted confidence is an honest P(correct direction). Disabled if the
        # artifact is missing or BTC_DISABLE_CALIBRATED_CONFIDENCE is set.
        self._calibrator = ProbabilityCalibrator(model_dir=self.model_dir).load()
        self._use_calibration = (
            os.environ.get("BTC_DISABLE_CALIBRATED_CONFIDENCE", "").lower()
            not in ("1", "true", "yes")
        )

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
        flow_df = await self.onchain_flows.collect()
        self.temporal.compute(price_df.tail(1000))

        latest_price = await self.price.get_latest()
        latest_ta = await self.technical.get_latest()
        latest_flow = flow_df.iloc[-1].to_dict() if not flow_df.empty else {}
        signals = self._build_signal_summary(latest_price, latest_ta, cycle_info, latest_flow)

        predictions, features_dict, model_source = self._generate_predictions(
            price_df, ta_df
        )
        self.logger.log_prediction(predictions, signals)
        run_number = self.logger._run_counter

        # Build a flat signals summary for JSONL (value + interpretation per signal).
        signals_flat = {
            k: f"{v.get('value', '')} -- {v.get('interpretation', '')}".strip(" --")
            for k, v in signals.items()
        }
        self.jsonl_logger.log(
            run_number=run_number,
            timestamp=datetime.now(timezone.utc),
            btc_price=float(latest_price.get("close", 0)),
            used_ml=self._using_ml,
            model_source=model_source,
            predictions=predictions,
            features=features_dict,
            signals_summary=signals_flat,
        )

        # Second module: produce + log the tweet-sentiment tracks (llm_direct now;
        # llm_calibrated / blended once those models are trained). Never fatal.
        await self._log_sentiment_tracks(run_number, float(latest_price.get("close", 0)))

        report = self.formatter.format_report(predictions, signals)
        print(report)

        return {
            "predictions": predictions,
            "signals": signals,
            "cycle": cycle_info,
            "using_ml": self._using_ml,
            "run_number": run_number,
        }

    async def _log_sentiment_tracks(self, run_number: int, btc_price: float) -> None:
        """Run the sentiment manager and log its tracks as separate JSONL lines.

        Each track is logged with its own ``model_source`` (e.g. ``llm_direct``)
        so the scorer compares it head-to-head against the ``numbers`` track.
        Any failure here is swallowed so the core predict path is unaffected.
        """
        if not self._enable_sentiment:
            return
        try:
            if self._sentiment_manager is None:
                from src.engine.sentiment_manager import SentimentManager

                self._sentiment_manager = SentimentManager(config=self.config)

            result = await self._sentiment_manager.run_tick()
            preds = result.get("predictions") or []
            if not preds:
                return
            self.jsonl_logger.log(
                run_number=run_number,
                timestamp=datetime.now(timezone.utc),
                btc_price=btc_price,
                used_ml=not result.get("mock", True),
                model_source=result.get("model_source", "llm_direct"),
                predictions=preds,
                features={},
                signals_summary={"sentiment_state": result.get("state", {})},
            )
            logger.info(
                "Sentiment track '%s' logged (%d tweets, %d extractions)",
                result.get("model_source"), result.get("n_tweets", 0),
                result.get("n_extractions", 0),
            )
        except Exception as exc:  # pragma: no cover - never block core predict
            logger.warning("Sentiment module skipped: %s", exc)

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

    def _resolve_model_dir(self, timeframe: str, prefix: str) -> Path:
        """Return the model directory for *timeframe*, honoring legacy aliases.

        Prefers ``<prefix>_<timeframe>``; if that directory is absent it falls
        back to a legacy-aliased horizon (e.g. the old ``7d`` artifact serves
        the new ``168h`` point) so already-trained models are reused without
        being moved on disk.
        """
        primary = self.model_dir / f"{prefix}_{timeframe}"
        if primary.exists():
            return primary
        legacy = LEGACY_MODEL_ALIASES.get(timeframe)
        if legacy:
            alt = self.model_dir / f"{prefix}_{legacy}"
            if alt.exists():
                return alt
        return primary

    def _models_available(self) -> bool:
        if not self.model_dir.exists():
            return False
        for tf in TIMEFRAMES:
            baseline_path = self._resolve_model_dir(tf, "baseline") / "baseline_model.pkl"
            xgb_path = self._resolve_model_dir(tf, "xgb") / "direction.pkl"
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
            baseline_path = self._resolve_model_dir(tf, "baseline")
            xgb_path = self._resolve_model_dir(tf, "xgb")

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
    ) -> tuple[list[dict[str, Any]], dict[str, float], str]:
        """Return (predictions, features_dict, model_source).

        Produces a value for *every* horizon in :data:`TIMEFRAMES`.  Horizons
        with a trained ML model use it; horizons whose model is not yet trained
        fall back cleanly to the heuristic estimate so the full curve is always
        populated and a missing ``xgb_<h>`` artifact never crashes a run.
        """
        latest_ta = ta_df.iloc[-1].to_dict() if not ta_df.empty else {}
        cycle_info = self.cycle.get_cycle_position()
        heuristic = self._heuristic_predictions(latest_ta, cycle_info)

        if self._load_models():
            ml_preds, features_dict = self._ml_predictions(price_df, ta_df)
            if ml_preds:
                ml_by_tf = {p["timeframe"]: p for p in ml_preds}
                merged: list[dict[str, Any]] = []
                for h in heuristic:
                    tf = h["timeframe"]
                    if tf in ml_by_tf:
                        merged.append(ml_by_tf[tf])
                    else:
                        filled = dict(h)
                        filled["model_source"] = "heuristic"
                        merged.append(filled)
                n_ml = len(ml_by_tf)
                source = (
                    "ensemble"
                    if n_ml == len(merged)
                    else f"hybrid ({n_ml}/{len(merged)} ML)"
                )
                return merged, features_dict, source

        logger.info("Falling back to heuristic predictions")
        return heuristic, {}, "heuristic"

    def _ml_predictions(
        self,
        price_df: pd.DataFrame,
        ta_df: pd.DataFrame,
    ) -> tuple[list[dict[str, Any]], dict[str, float]]:
        """Return (predictions, features_dict).

        Each prediction dict includes the standard display fields plus
        ``direction_prob`` (raw ensemble P(UP)) and ``calibrated`` (bool)
        for post-hoc calibration analysis.
        """
        merged = self._build_merged_data(price_df, ta_df)
        merged = self._merge_tweet_signal(merged)
        merged = self._merge_onchain_flow_signal(merged)
        if merged.empty:
            return [], {}

        features_df = self.feature_builder.build_features(merged)
        if features_df.empty:
            return [], {}

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

            is_calibrated = self._use_calibration and self._calibrator.has_horizon(tf)
            if is_calibrated:
                # Honest confidence = calibrated P(correct direction), in [50, 95].
                confidence = self._calibrator.confidence(ens_prob, tf)
            else:
                # Legacy fallback: margin-derived pseudo-confidence.
                confidence = raw_conf * 100 if raw_conf <= 1.0 else raw_conf
                confidence = max(10.0, min(95.0, confidence))

            predictions.append({
                "timeframe": tf,
                "direction": direction,
                "direction_prob": round(float(ens_prob), 4),
                "magnitude": self._display_magnitude(ens_mag, ens_prob, tf),
                "confidence": int(round(confidence)),
                "calibrated": is_calibrated,
            })

        return predictions, features_dict

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

    @staticmethod
    def _merge_tweet_signal(merged: pd.DataFrame) -> pd.DataFrame:
        """Forward-fill the tweet-sentiment signal onto the price grid.

        Closes the train/serve parity gap: training merges signal history from
        ``data/history/``, so live predict must too. No-op when the signal file
        is absent (columns are added by the feature builder as placeholders).
        """
        if merged.empty:
            return merged
        try:
            from src.engine.sentiment_memory import SentimentMemory

            signal = SentimentMemory().load_signal()
            if signal is None or signal.empty:
                return merged
            new_cols = [c for c in signal.columns if c not in merged.columns]
            if not new_cols:
                return merged
            reindexed = signal[new_cols].reindex(merged.index, method="ffill")
            return merged.join(reindexed, how="left")
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Tweet signal merge skipped: %s", exc)
            return merged

    @staticmethod
    def _merge_onchain_flow_signal(merged: pd.DataFrame) -> pd.DataFrame:
        """Forward-fill on-chain flow metrics onto the price grid."""
        if merged.empty or not ONCHAIN_FLOW_HISTORY_PATH.exists():
            return merged
        try:
            signal = pd.read_parquet(ONCHAIN_FLOW_HISTORY_PATH)
            if signal.empty:
                return merged
            if not isinstance(signal.index, pd.DatetimeIndex):
                if "timestamp" in signal.columns:
                    signal = signal.set_index("timestamp")
                signal.index = pd.to_datetime(signal.index, utc=True)
            new_cols = [c for c in signal.columns if c not in merged.columns]
            if not new_cols:
                return merged
            reindexed = signal[new_cols].reindex(merged.index, method="ffill")
            return merged.join(reindexed, how="left")
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("On-chain flow signal merge skipped: %s", exc)
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

        predictions = []
        for tf in TIMEFRAMES:
            strength = max(abs(ratio - 0.5) * 2, 0.15)
            raw_mag = PredictionEngine._heuristic_mag_scale(tf) * strength
            prob = ratio if direction == "UP" else 1.0 - ratio
            magnitude = PredictionEngine._display_magnitude(raw_mag, prob, tf)
            confidence = max(
                PredictionEngine._heuristic_conf_base(tf) + int((ratio - 0.5) * 20), 20
            )
            predictions.append({
                "timeframe": tf,
                "direction": direction,
                "magnitude": magnitude,
                "confidence": confidence,
            })

        return predictions

    @staticmethod
    def _heuristic_mag_scale(timeframe: str) -> float:
        """Expected move scale (multiplier) per horizon for the TA fallback.

        Power fit anchored to the original hand-tuned scales
        (24h ~ 1.0, 168h ~ 3.0, 30d ~ 8.0); grows with the horizon length.
        """
        hrs = HORIZON_HOURS.get(timeframe, 24)
        return 0.1437 * hrs ** 0.611

    @staticmethod
    def _heuristic_conf_base(timeframe: str) -> int:
        """Baseline confidence per horizon for the TA fallback.

        Monotonically decays with the horizon length (shorter horizons carry
        more conviction); clamped to a sane [30, 70] band.
        """
        import math

        hrs = HORIZON_HOURS.get(timeframe, 24)
        base = 66 - 3.2 * math.log2(max(hrs, 1) / 6)
        return int(round(max(30, min(70, base))))

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

        floor = _min_magnitude(timeframe)
        strength = max(abs(direction_prob - 0.5) * 2, 0.15)
        estimated = floor * strength
        return round(max(mag, estimated), 1)

    @staticmethod
    def _build_signal_summary(
        price: dict, ta: dict, cycle: dict, onchain_flow: dict | None = None
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

        flow = onchain_flow or {}
        btc_24h = flow.get("whale_btc_moved_24h")
        if btc_24h is not None:
            net_exchange = float(flow.get("net_exchange_flow_btc_24h", 0.0) or 0.0)
            score = float(flow.get("flow_accumulation_score", 0.0) or 0.0)
            if score > 0.2:
                interp = "cold-storage-like accumulation"
            elif score < -0.2:
                interp = "distribution-like whale flow"
            else:
                interp = "mixed/neutral large flows"
            signals["On-chain Whale Flow"] = {
                "value": f"{float(btc_24h):,.0f} BTC moved (24h)",
                "interpretation": f"{interp}; net exchange labeled flow {net_exchange:+,.0f} BTC",
            }

        if ONCHAIN_FLOW_LATEST_PATH.exists():
            signals["On-chain Data Coverage"] = {
                "value": "free public APIs + local labels",
                "interpretation": "exact exchange/miner labels require address labels or paid entity API",
            }

        return signals


Predictor = PredictionEngine
