"""Confidence calibration module.

Ensures that when the model says "70% confidence", it's actually right
~70% of the time. Uses both Platt scaling and isotonic regression,
with volatility-adjusted and timeframe-decayed confidence bounds.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from src import DATA_DIR
from src.horizons import HORIZON_HOURS
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

MODEL_DIR = DATA_DIR / "models"


def _confidence_cap(horizon_label: str) -> float:
    """Max calibrated confidence per horizon (decays with horizon length).

    Anchored to the historical caps (24h ~ 92, 168h ~ 81, 30d ~ 72).
    """
    import math

    hrs = HORIZON_HOURS.get(horizon_label, 24)
    cap = 110.7 - 5.88 * math.log(max(hrs, 1))
    return max(62.0, min(92.0, cap))


def _horizon_penalty(horizon_label: str) -> float:
    """Heuristic-confidence penalty per horizon (larger for longer horizons).

    Anchored to the historical penalties (24h ~ 0, 168h ~ -5, 30d ~ -12).
    """
    import math

    hrs = HORIZON_HOURS.get(horizon_label, 24)
    return -3.53 * math.log(max(hrs, 24) / 24)


class ConfidenceCalibrator:
    """Calibrates raw model outputs into honest probability estimates.

    Two calibration methods:
    - Platt scaling: logistic regression on raw prediction magnitudes
    - Isotonic regression: non-parametric monotonic calibration

    Additional adjustments:
    - Volatility regime scaling (high vol → lower confidence)
    - Timeframe decay (longer horizons → lower max confidence)
    """

    def __init__(self, model_dir: Optional[Path] = None) -> None:
        self.model_dir = model_dir or MODEL_DIR
        self._platt_models: dict[str, LogisticRegression] = {}
        self._isotonic_models: dict[str, IsotonicRegression] = {}
        self._method: str = "platt"
        self._historical_accuracy: dict[str, list[tuple[float, bool]]] = {}

    def fit(
        self,
        raw_predictions: np.ndarray,
        actual_outcomes: np.ndarray,
        horizon_label: str,
        method: str = "platt",
    ) -> dict[str, float]:
        """Fit calibration model on historical predictions vs outcomes.

        Args:
            raw_predictions: Raw model output magnitudes (or probabilities).
            actual_outcomes: 1 if prediction was correct, 0 if wrong.
            horizon_label: Which timeframe this calibrates.
            method: 'platt' for logistic regression, 'isotonic' for non-parametric.

        Returns:
            Dict with calibration metrics.
        """
        if len(raw_predictions) < 50:
            logger.warning(
                "Not enough data to calibrate %s: %d samples (need 50+)",
                horizon_label, len(raw_predictions),
            )
            return {"error": "insufficient_data"}

        self._method = method
        X = np.abs(raw_predictions).reshape(-1, 1)
        y = actual_outcomes.astype(int)

        if method == "platt":
            calibrator = LogisticRegression(C=1.0, max_iter=1000)
            calibrator.fit(X, y)
            self._platt_models[horizon_label] = calibrator

            probs = calibrator.predict_proba(X)[:, 1]

        elif method == "isotonic":
            calibrator = IsotonicRegression(
                y_min=0.0, y_max=1.0, out_of_bounds="clip"
            )
            calibrator.fit(X.ravel(), y)
            self._isotonic_models[horizon_label] = calibrator

            probs = calibrator.predict(X.ravel())

        else:
            raise ValueError(f"Unknown calibration method: {method}")

        # Compute calibration quality metrics
        brier_score = float(np.mean((probs - y) ** 2))
        log_loss = float(-np.mean(y * np.log(probs + 1e-10) + (1 - y) * np.log(1 - probs + 1e-10)))

        logger.info(
            "Calibrator fitted for %s (%s): brier=%.4f, logloss=%.4f",
            horizon_label, method, brier_score, log_loss,
        )

        self.save()
        return {
            "brier_score": brier_score,
            "log_loss": log_loss,
            "n_samples": len(raw_predictions),
            "method": method,
        }

    def calibrate(
        self,
        raw_magnitude: float,
        horizon_label: str,
        volatility_regime: str = "normal",
    ) -> float:
        """Convert raw prediction magnitude to calibrated confidence %.

        Args:
            raw_magnitude: Absolute value of model's raw prediction.
            horizon_label: Prediction timeframe.
            volatility_regime: 'low', 'normal', or 'high'.

        Returns:
            Calibrated confidence as percentage (10-95%).
        """
        X = np.array([[abs(raw_magnitude)]])

        # Try isotonic first, then Platt, then heuristic
        if horizon_label in self._isotonic_models:
            confidence = float(self._isotonic_models[horizon_label].predict(X.ravel())[0]) * 100
        elif horizon_label in self._platt_models:
            confidence = float(self._platt_models[horizon_label].predict_proba(X)[0][1]) * 100
        else:
            confidence = self._heuristic_confidence(raw_magnitude, horizon_label)

        # Volatility adjustment
        vol_multipliers = {"low": 1.1, "normal": 1.0, "high": 0.75}
        confidence *= vol_multipliers.get(volatility_regime, 1.0)

        # Timeframe cap
        max_confidence = _confidence_cap(horizon_label)
        confidence = min(confidence, max_confidence)

        return max(10.0, min(95.0, confidence))

    def calibrate_batch(
        self,
        raw_magnitudes: np.ndarray,
        horizon_label: str,
        volatility_regime: str = "normal",
    ) -> np.ndarray:
        """Calibrate a batch of predictions."""
        return np.array([
            self.calibrate(float(m), horizon_label, volatility_regime)
            for m in raw_magnitudes
        ])

    def get_volatility_regime(
        self,
        returns: pd.Series,
        window: int = 168,
    ) -> str:
        """Classify current volatility regime from recent hourly returns.

        Compares recent volatility to historical norm.
        """
        if returns.empty or len(returns) < window:
            return "normal"

        current_vol = float(returns.tail(window).std())
        historical_vol = float(returns.std())

        if historical_vol == 0:
            return "normal"

        ratio = current_vol / historical_vol

        if ratio < 0.7:
            return "low"
        elif ratio > 1.5:
            return "high"
        return "normal"

    def track_accuracy(
        self, confidence: float, was_correct: bool, horizon_label: str
    ) -> None:
        """Track prediction outcomes for ongoing calibration monitoring."""
        if horizon_label not in self._historical_accuracy:
            self._historical_accuracy[horizon_label] = []
        self._historical_accuracy[horizon_label].append((confidence, was_correct))

    def get_calibration_report(
        self, horizon_label: str, n_bins: int = 10
    ) -> dict:
        """Generate calibration reliability diagram data.

        Returns binned accuracy vs confidence for visualization.
        """
        history = self._historical_accuracy.get(horizon_label, [])
        if len(history) < 50:
            return {"error": "insufficient_data", "samples": len(history)}

        confidences = np.array([h[0] for h in history])
        outcomes = np.array([h[1] for h in history])

        bins = np.linspace(0, 100, n_bins + 1)
        report: dict = {"bins": [], "ece": 0.0}
        weighted_errors = []

        for i in range(n_bins):
            mask = (confidences >= bins[i]) & (confidences < bins[i + 1])
            if mask.sum() == 0:
                continue

            avg_confidence = float(confidences[mask].mean())
            actual_accuracy = float(outcomes[mask].mean() * 100)
            n_samples = int(mask.sum())
            cal_error = abs(avg_confidence - actual_accuracy)

            report["bins"].append({
                "confidence_range": f"{bins[i]:.0f}-{bins[i+1]:.0f}%",
                "avg_confidence": round(avg_confidence, 1),
                "actual_accuracy": round(actual_accuracy, 1),
                "n_samples": n_samples,
                "calibration_error": round(cal_error, 1),
            })
            weighted_errors.append(cal_error * n_samples)

        report["ece"] = sum(weighted_errors) / len(history) if history else 0
        report["total_samples"] = len(history)
        return report

    def save(self) -> None:
        """Persist calibration models to disk."""
        path = self.model_dir / "confidence_calibrator.pkl"
        self.model_dir.mkdir(parents=True, exist_ok=True)
        state = {
            "platt_models": self._platt_models,
            "isotonic_models": self._isotonic_models,
            "method": self._method,
            "historical_accuracy": self._historical_accuracy,
        }
        with open(path, "wb") as f:
            pickle.dump(state, f)

    def load(self) -> None:
        """Load calibration models from disk."""
        path = self.model_dir / "confidence_calibrator.pkl"
        if not path.exists():
            return

        with open(path, "rb") as f:
            state = pickle.load(f)

        self._platt_models = state.get("platt_models", {})
        self._isotonic_models = state.get("isotonic_models", {})
        self._method = state.get("method", "platt")
        self._historical_accuracy = state.get("historical_accuracy", {})
        logger.info("Confidence calibrator loaded from %s", path)

    def _heuristic_confidence(self, magnitude: float, horizon_label: str) -> float:
        """Simple confidence estimate when no calibrator is available."""
        base = 50.0 + min(magnitude * 3, 20.0)
        base += _horizon_penalty(horizon_label)
        return base
