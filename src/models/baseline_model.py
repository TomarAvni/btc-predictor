"""Baseline model for sanity-check comparison.

Uses only lagged price returns as features with logistic regression for
direction and linear regression for magnitude. If complex models can't
beat this, something is wrong with the pipeline.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge

from src.utils.logger import setup_logger

logger = setup_logger(__name__)

BASELINE_FEATURES = [
    "return_1h", "return_4h", "return_24h", "return_7d", "return_30d",
    "volatility_24h", "volatility_7d",
]


class BaselineModel:
    """Simple baseline model using only price-derived features.

    Serves as a lower bound: if XGBoost/TFT can't beat this, the added
    complexity isn't justified. Uses logistic regression for direction
    and ridge regression for magnitude.
    """

    def __init__(self, feature_columns: Optional[list[str]] = None) -> None:
        self.feature_columns = feature_columns or BASELINE_FEATURES
        self.direction_model: Optional[LogisticRegression] = None
        self.magnitude_model: Optional[Ridge] = None
        self._actual_features: list[str] = []

    def train(
        self,
        X: pd.DataFrame,
        y_direction: pd.Series,
        y_magnitude: pd.Series,
    ) -> dict[str, float]:
        """Train baseline direction and magnitude models.

        Args:
            X: Feature matrix (uses only baseline features if available).
            y_direction: Binary target (1=up, 0=down).
            y_magnitude: Continuous target (percentage return).

        Returns:
            Dict with training metrics.
        """
        available = [c for c in self.feature_columns if c in X.columns]
        if not available:
            available = X.columns[:7].tolist()
            logger.warning(
                "No baseline features found. Using first %d columns.", len(available)
            )

        self._actual_features = available
        X_subset = X[available].fillna(0).values

        # Direction model
        self.direction_model = LogisticRegression(
            C=1.0, max_iter=1000, random_state=42
        )
        self.direction_model.fit(X_subset, y_direction.values)

        # Magnitude model
        self.magnitude_model = Ridge(alpha=1.0)
        self.magnitude_model.fit(X_subset, y_magnitude.values)

        # Compute training accuracy
        dir_accuracy = float(self.direction_model.score(X_subset, y_direction.values))
        mag_predictions = self.magnitude_model.predict(X_subset)
        mae = float(np.abs(mag_predictions - y_magnitude.values).mean())

        logger.info(
            "Baseline trained: direction_acc=%.3f, magnitude_MAE=%.2f%%",
            dir_accuracy, mae,
        )
        return {"direction_accuracy": dir_accuracy, "magnitude_mae": mae}

    def predict(self, features: dict[str, float]) -> dict[str, float]:
        """Predict direction and magnitude from a feature dictionary.

        Returns:
            Dict with direction_prob, predicted_magnitude, confidence.
        """
        if self.direction_model is None:
            return {"direction_prob": 0.5, "predicted_magnitude": 0.0, "confidence": 0.0}

        X = np.array([[features.get(f, 0.0) for f in self._actual_features]])

        direction_prob = float(self.direction_model.predict_proba(X)[0][1])
        magnitude = float(self.magnitude_model.predict(X)[0])
        confidence = abs(direction_prob - 0.5) * 2

        return {
            "direction_prob": direction_prob,
            "predicted_magnitude": magnitude,
            "confidence": confidence,
        }

    def predict_batch(self, X: pd.DataFrame) -> pd.DataFrame:
        """Batch prediction for walk-forward validation.

        Returns DataFrame with direction_prob, predicted_return, confidence.
        """
        if self.direction_model is None:
            return pd.DataFrame(
                {"direction_prob": 0.5, "predicted_return": 0.0, "confidence": 0.0},
                index=X.index,
            )

        available = [c for c in self._actual_features if c in X.columns]
        X_subset = X[available].fillna(0).values

        direction_probs = self.direction_model.predict_proba(X_subset)[:, 1]
        magnitudes = self.magnitude_model.predict(X_subset)
        confidences = np.abs(direction_probs - 0.5) * 2

        # Signed prediction (positive=up, negative=down)
        signs = np.where(direction_probs > 0.5, 1, -1)
        predicted_returns = np.abs(magnitudes) * signs

        return pd.DataFrame({
            "direction_prob": direction_probs,
            "predicted_return": predicted_returns,
            "confidence": confidences,
        }, index=X.index)

    def save(self, path: Path) -> None:
        """Save model to disk."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        with open(path / "baseline_model.pkl", "wb") as f:
            pickle.dump({
                "direction_model": self.direction_model,
                "magnitude_model": self.magnitude_model,
                "features": self._actual_features,
            }, f)
        logger.info("Baseline model saved to %s", path)

    def load(self, path: Path) -> None:
        """Load model from disk."""
        path = Path(path)
        pkl_path = path / "baseline_model.pkl"
        if not pkl_path.exists():
            logger.warning("No baseline model found at %s", pkl_path)
            return

        with open(pkl_path, "rb") as f:
            data = pickle.load(f)
            self.direction_model = data["direction_model"]
            self.magnitude_model = data["magnitude_model"]
            self._actual_features = data["features"]
        logger.info("Baseline model loaded from %s", path)
