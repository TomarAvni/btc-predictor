"""Stacking ensemble combining Baseline, XGBoost, and TFT predictions.

Uses a meta-learner (logistic regression) trained on validation set predictions
to optimally combine the three models. Weights are per-timeframe since different
models excel at different horizons.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge

from src import DATA_DIR
from src.horizons import HORIZON_HOURS, TIMEFRAMES
from src.models.baseline_model import BaselineModel
from src.models.xgboost_model import XGBoostPredictor
from src.models.tft_model import TFTPredictor
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

MODEL_DIR = DATA_DIR / "models"


def _default_weights(timeframe: str) -> dict[str, float]:
    """Per-horizon base ensemble weights, scaled by horizon length.

    Short horizons lean on XGBoost (tabular signal), longer horizons give the
    sequence model (TFT) more say.  Used until the meta-learner is trained.
    """
    hrs = HORIZON_HOURS.get(timeframe, 24)
    if hrs <= 48:
        return {"baseline": 0.15, "xgboost": 0.55, "tft": 0.30}
    if hrs <= 168:
        return {"baseline": 0.10, "xgboost": 0.45, "tft": 0.45}
    return {"baseline": 0.10, "xgboost": 0.40, "tft": 0.50}


def _confidence_cap(timeframe: str) -> int:
    """Max confidence per horizon (decays with horizon length).

    Anchored to the historical caps (24h ~ 90, 168h ~ 79, 30d ~ 70).
    """
    import math

    hrs = HORIZON_HOURS.get(timeframe, 24)
    cap = 108.7 - 5.88 * math.log(max(hrs, 1))
    return int(round(max(60, min(90, cap))))


class EnsemblePredictor:
    """Stacking ensemble combining Baseline + XGBoost + TFT.

    Architecture:
    - Level 0: Three base models generate independent predictions
    - Level 1: Meta-learner combines their outputs per timeframe
    - Confidence incorporates model agreement, historical accuracy,
      and current volatility regime

    The meta-learner is trained on out-of-fold validation predictions
    to avoid overfitting.
    """

    def __init__(self, model_dir: Optional[Path] = None) -> None:
        self.model_dir = model_dir or MODEL_DIR
        self.model_dir.mkdir(parents=True, exist_ok=True)

        self.baseline = BaselineModel()
        self.xgboost: dict[str, XGBoostPredictor] = {}
        self.tft = TFTPredictor(model_dir=self.model_dir / "tft")

        # Per-timeframe meta-learners
        self._meta_direction: dict[str, LogisticRegression] = {}
        self._meta_magnitude: dict[str, Ridge] = {}
        self._weights: dict[str, dict[str, float]] = {
            tf: _default_weights(tf) for tf in TIMEFRAMES
        }
        self._is_fitted = False

    def train_meta_learner(
        self,
        val_predictions: dict[str, pd.DataFrame],
        val_actuals: dict[str, pd.Series],
    ) -> dict[str, float]:
        """Train the meta-learner on stacked validation predictions.

        Args:
            val_predictions: Per-timeframe DataFrame with columns for each
                model's direction_prob and magnitude prediction.
            val_actuals: Per-timeframe Series of actual returns.

        Returns:
            Dict of meta-learner performance metrics per timeframe.
        """
        metrics = {}

        for tf in TIMEFRAMES:
            if tf not in val_predictions or tf not in val_actuals:
                continue

            preds = val_predictions[tf]
            actuals = val_actuals[tf]

            # Build meta-features from base model outputs
            meta_features = self._build_meta_features(preds)
            y_direction = (actuals > 0).astype(int)

            if len(meta_features) < 50:
                logger.warning("Insufficient data for meta-learner on %s", tf)
                continue

            # Train direction meta-learner
            self._meta_direction[tf] = LogisticRegression(C=1.0, max_iter=1000)
            self._meta_direction[tf].fit(meta_features, y_direction)

            # Train magnitude meta-learner
            self._meta_magnitude[tf] = Ridge(alpha=1.0)
            self._meta_magnitude[tf].fit(meta_features, actuals.values)

            # Evaluate
            dir_acc = float(self._meta_direction[tf].score(meta_features, y_direction))
            mag_pred = self._meta_magnitude[tf].predict(meta_features)
            mae = float(np.abs(mag_pred - actuals.values).mean())

            metrics[tf] = {"direction_accuracy": dir_acc, "mae": mae}

            # Update weights from meta-learner coefficients
            self._update_weights_from_meta(tf)

            logger.info("Meta-learner %s: accuracy=%.3f, MAE=%.2f%%", tf, dir_acc, mae)

        self._is_fitted = True
        self.save()
        return metrics

    def predict(
        self,
        features: dict[str, float],
        timeframe: str = "24h",
        volatility_regime: str = "normal",
    ) -> dict[str, Any]:
        """Generate ensemble prediction for a single timeframe.

        Returns:
            Dict with direction, magnitude, confidence, and component predictions.
        """
        # Get base model predictions
        baseline_pred = self.baseline.predict(features)
        xgb_pred = self._get_xgb_prediction(features, timeframe)
        tft_pred = self.tft.predict(pd.DataFrame([features]))

        # Combine using meta-learner or weighted average
        if timeframe in self._meta_direction and self._is_fitted:
            result = self._meta_predict(
                baseline_pred, xgb_pred, tft_pred, timeframe
            )
        else:
            result = self._weighted_predict(
                baseline_pred, xgb_pred, tft_pred, timeframe
            )

        # Compute confidence
        confidence = self._compute_confidence(
            baseline_pred, xgb_pred, tft_pred, timeframe, volatility_regime
        )
        result["confidence"] = confidence

        result["components"] = {
            "baseline": baseline_pred,
            "xgboost": xgb_pred,
            "tft": tft_pred,
        }

        return result

    def predict_all_horizons(
        self,
        features: dict[str, float],
        volatility_regime: str = "normal",
    ) -> list[dict[str, Any]]:
        """Generate predictions for all configured timeframes."""
        results = []
        for tf in TIMEFRAMES:
            pred = self.predict(features, tf, volatility_regime)
            pred["timeframe"] = tf
            results.append(pred)
        return results

    def _get_xgb_prediction(
        self, features: dict[str, float], timeframe: str
    ) -> dict[str, float]:
        """Get XGBoost prediction, loading model if needed."""
        if timeframe not in self.xgboost:
            self.xgboost[timeframe] = XGBoostPredictor(timeframe=timeframe)
        return self.xgboost[timeframe].predict(features)

    def _meta_predict(
        self,
        baseline: dict,
        xgb: dict,
        tft: dict,
        timeframe: str,
    ) -> dict[str, Any]:
        """Use trained meta-learner for final prediction."""
        meta_features = np.array([[
            baseline.get("direction_prob", 0.5),
            baseline.get("predicted_magnitude", 0.0),
            xgb.get("direction_prob", 0.5),
            xgb.get("predicted_magnitude", 0.0),
            tft.get("direction_prob", 0.5),
            tft.get("predicted_magnitude", 0.0),
        ]])

        direction_prob = float(
            self._meta_direction[timeframe].predict_proba(meta_features)[0][1]
        )
        magnitude = float(
            self._meta_magnitude[timeframe].predict(meta_features)[0]
        )

        direction = "UP" if direction_prob > 0.5 else "DOWN"

        return {
            "direction": direction,
            "direction_prob": direction_prob,
            "magnitude": abs(magnitude),
            "predicted_return": magnitude,
        }

    def _weighted_predict(
        self,
        baseline: dict,
        xgb: dict,
        tft: dict,
        timeframe: str,
    ) -> dict[str, Any]:
        """Fallback: simple weighted average when meta-learner isn't trained."""
        weights = self._weights.get(timeframe, {"baseline": 0.15, "xgboost": 0.55, "tft": 0.30})

        direction_prob = (
            weights["baseline"] * baseline.get("direction_prob", 0.5)
            + weights["xgboost"] * xgb.get("direction_prob", 0.5)
            + weights["tft"] * tft.get("direction_prob", 0.5)
        )

        magnitude = (
            weights["baseline"] * abs(baseline.get("predicted_magnitude", 0))
            + weights["xgboost"] * abs(xgb.get("predicted_magnitude", 0))
            + weights["tft"] * abs(tft.get("predicted_magnitude", 0))
        )

        direction = "UP" if direction_prob > 0.5 else "DOWN"
        signed_return = magnitude * (1 if direction_prob > 0.5 else -1)

        return {
            "direction": direction,
            "direction_prob": direction_prob,
            "magnitude": magnitude,
            "predicted_return": signed_return,
        }

    def _compute_confidence(
        self,
        baseline: dict,
        xgb: dict,
        tft: dict,
        timeframe: str,
        volatility_regime: str,
    ) -> float:
        """Compute final confidence score factoring in multiple signals.

        Factors:
        1. Model agreement (all same direction = higher)
        2. Average raw confidence from models
        3. Volatility regime adjustment
        4. Timeframe decay (longer = lower max confidence)
        """
        # Agreement check
        directions = []
        for pred in [baseline, xgb, tft]:
            prob = pred.get("direction_prob", 0.5)
            if prob != 0.5:
                directions.append("UP" if prob > 0.5 else "DOWN")

        if len(directions) >= 2:
            agreement = directions.count(directions[0]) / len(directions)
        else:
            agreement = 0.5

        # Average confidence from individual models
        avg_confidence = np.mean([
            baseline.get("confidence", 0.5) * 100 if baseline.get("confidence", 0) <= 1 else baseline.get("confidence", 50),
            xgb.get("raw_confidence", 0.5) * 100 if xgb.get("raw_confidence", 0) <= 1 else xgb.get("raw_confidence", 50),
            tft.get("confidence", 50),
        ])

        # Start with average confidence, scale by agreement
        confidence = avg_confidence * (0.5 + 0.5 * agreement)

        # Volatility regime adjustment
        vol_multipliers = {"low": 1.1, "normal": 1.0, "high": 0.75}
        confidence *= vol_multipliers.get(volatility_regime, 1.0)

        # Timeframe decay
        max_conf = _confidence_cap(timeframe)
        confidence = min(confidence, max_conf)

        return max(10.0, min(95.0, confidence))

    def _build_meta_features(self, predictions_df: pd.DataFrame) -> np.ndarray:
        """Build feature matrix for meta-learner from base model outputs."""
        feature_cols = [
            col for col in predictions_df.columns
            if any(x in col for x in ["direction_prob", "magnitude", "confidence"])
        ]

        if not feature_cols:
            feature_cols = predictions_df.columns.tolist()

        return predictions_df[feature_cols].fillna(0).values

    def _update_weights_from_meta(self, timeframe: str) -> None:
        """Update model weights from meta-learner coefficients."""
        if timeframe not in self._meta_direction:
            return

        coefs = self._meta_direction[timeframe].coef_[0]
        # Coefficients correspond to: baseline_dir, baseline_mag, xgb_dir, xgb_mag, tft_dir, tft_mag
        if len(coefs) >= 6:
            raw_weights = {
                "baseline": abs(coefs[0]) + abs(coefs[1]),
                "xgboost": abs(coefs[2]) + abs(coefs[3]),
                "tft": abs(coefs[4]) + abs(coefs[5]),
            }
            total = sum(raw_weights.values())
            if total > 0:
                self._weights[timeframe] = {
                    k: v / total for k, v in raw_weights.items()
                }

    def save(self) -> None:
        """Save ensemble state to disk."""
        state = {
            "weights": self._weights,
            "meta_direction": self._meta_direction,
            "meta_magnitude": self._meta_magnitude,
            "is_fitted": self._is_fitted,
        }
        path = self.model_dir / "ensemble_state.pkl"
        with open(path, "wb") as f:
            pickle.dump(state, f)
        logger.info("Ensemble saved to %s", path)

    def load(self) -> None:
        """Load ensemble state from disk."""
        path = self.model_dir / "ensemble_state.pkl"
        if not path.exists():
            logger.warning("No ensemble state found at %s", path)
            return

        with open(path, "rb") as f:
            state = pickle.load(f)

        self._weights = state["weights"]
        self._meta_direction = state["meta_direction"]
        self._meta_magnitude = state["meta_magnitude"]
        self._is_fitted = state["is_fitted"]
        logger.info("Ensemble loaded from %s", path)


EnsembleModel = EnsemblePredictor
