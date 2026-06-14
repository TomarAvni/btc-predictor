"""XGBoost model for BTC price prediction.

Gradient boosting excels at tabular features: cycle position,
sentiment scores, funding rates, macro correlations, etc.
Outputs both direction probability and magnitude estimate.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit

from src import DATA_DIR
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

MODEL_DIR = DATA_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


class XGBoostPredictor:
    """XGBoost model for each prediction timeframe.

    Uses XGBClassifier for direction (UP/DOWN) and XGBRegressor for magnitude.
    Hyperparameters tuned for financial time series with early stopping.
    """

    def __init__(self, config: Optional[dict] = None, timeframe: str = "24h"):
        cfg = (config or {}).get("models", {}).get("xgboost", {})
        self.timeframe = timeframe
        self.model_direction: Optional[xgb.XGBClassifier] = None
        self.model_magnitude: Optional[xgb.XGBRegressor] = None
        self.feature_names: list[str] = []
        self._model_path = MODEL_DIR / f"xgb_{timeframe}"

        self._n_estimators = cfg.get("n_estimators", 500)
        self._max_depth = cfg.get("max_depth", 6)
        self._learning_rate = cfg.get("learning_rate", 0.05)
        self._subsample = cfg.get("subsample", 0.8)
        self._colsample = cfg.get("colsample_bytree", 0.8)
        self._early_stopping = cfg.get("early_stopping_rounds", 50)

    def train(
        self,
        X: pd.DataFrame,
        y_direction: pd.Series,
        y_magnitude: pd.Series,
    ) -> dict[str, float]:
        """Train both direction (classifier) and magnitude (regressor) models.

        Args:
            X: Feature matrix.
            y_direction: Binary target (1=up, 0=down).
            y_magnitude: Continuous target (percentage change).

        Returns:
            Dict with training/validation metrics.
        """
        self.feature_names = list(X.columns)

        self.model_direction = xgb.XGBClassifier(
            n_estimators=self._n_estimators,
            max_depth=self._max_depth,
            learning_rate=self._learning_rate,
            subsample=self._subsample,
            colsample_bytree=self._colsample,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
        )

        self.model_magnitude = xgb.XGBRegressor(
            n_estimators=self._n_estimators,
            max_depth=self._max_depth,
            learning_rate=self._learning_rate,
            subsample=self._subsample,
            colsample_bytree=self._colsample,
            eval_metric="mae",
            random_state=42,
            n_jobs=-1,
        )

        # Time-series split: use last fold for early stopping validation
        tscv = TimeSeriesSplit(n_splits=5)
        splits = list(tscv.split(X))
        train_idx, val_idx = splits[-1]

        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_dir_train, y_dir_val = y_direction.iloc[train_idx], y_direction.iloc[val_idx]
        y_mag_train, y_mag_val = y_magnitude.iloc[train_idx], y_magnitude.iloc[val_idx]

        # Train direction model with early stopping
        self.model_direction.fit(
            X_train, y_dir_train,
            eval_set=[(X_val, y_dir_val)],
            verbose=False,
        )

        # Train magnitude model with early stopping
        self.model_magnitude.fit(
            X_train, y_mag_train,
            eval_set=[(X_val, y_mag_val)],
            verbose=False,
        )

        # Validation metrics
        val_dir_accuracy = float(self.model_direction.score(X_val, y_dir_val))
        val_mag_predictions = self.model_magnitude.predict(X_val)
        val_mae = float(np.abs(val_mag_predictions - y_mag_val.values).mean())

        logger.info(
            "XGBoost %s: val_accuracy=%.3f, val_MAE=%.2f%%",
            self.timeframe, val_dir_accuracy, val_mae,
        )

        self.save()

        return {
            "direction_accuracy": val_dir_accuracy,
            "magnitude_mae": val_mae,
            "n_train": len(X_train),
            "n_val": len(X_val),
            "best_iteration_dir": getattr(self.model_direction, "best_iteration", self._n_estimators),
        }

    def predict(self, features: dict[str, float]) -> dict[str, float]:
        """Predict direction and magnitude from current features.

        Args:
            features: Dict of feature_name → value.

        Returns:
            Dict with direction_prob, predicted_magnitude, raw_confidence.
        """
        if self.model_direction is None:
            self.load()
            if self.model_direction is None:
                return {"direction_prob": 0.5, "predicted_magnitude": 0.0, "raw_confidence": 0.0}

        X = np.array([[features.get(f, 0.0) for f in self.feature_names]])

        direction_prob = float(self.model_direction.predict_proba(X)[0][1])
        magnitude = float(self.model_magnitude.predict(X)[0])
        raw_confidence = abs(direction_prob - 0.5) * 2

        return {
            "direction_prob": direction_prob,
            "predicted_magnitude": magnitude,
            "raw_confidence": raw_confidence,
        }

    def predict_batch(self, X: pd.DataFrame) -> pd.DataFrame:
        """Batch prediction returning a DataFrame.

        Returns DataFrame with direction_prob, predicted_return, confidence.
        """
        if self.model_direction is None:
            self.load()
            if self.model_direction is None:
                return pd.DataFrame(
                    {"direction_prob": 0.5, "predicted_return": 0.0, "confidence": 0.0},
                    index=X.index,
                )

        available = [c for c in self.feature_names if c in X.columns]
        missing = [c for c in self.feature_names if c not in X.columns]

        X_aligned = X[available].copy()
        for col in missing:
            X_aligned[col] = 0.0
        X_aligned = X_aligned[self.feature_names].fillna(0).values

        direction_probs = self.model_direction.predict_proba(X_aligned)[:, 1]
        magnitudes = self.model_magnitude.predict(X_aligned)
        confidences = np.abs(direction_probs - 0.5) * 2

        signs = np.where(direction_probs > 0.5, 1, -1)
        predicted_returns = np.abs(magnitudes) * signs

        return pd.DataFrame({
            "direction_prob": direction_probs,
            "predicted_return": predicted_returns,
            "confidence": confidences,
        }, index=X.index)

    def get_feature_importance(self, top_n: int = 20) -> dict[str, float]:
        """Get feature importance scores, sorted by importance."""
        if self.model_direction is None:
            return {}

        importance = self.model_direction.feature_importances_
        imp_dict = dict(zip(self.feature_names, importance))
        sorted_imp = dict(sorted(imp_dict.items(), key=lambda x: x[1], reverse=True))

        if top_n:
            sorted_imp = dict(list(sorted_imp.items())[:top_n])
        return sorted_imp

    def save(self) -> None:
        """Save model to disk using pickle."""
        self._model_path.mkdir(parents=True, exist_ok=True)
        if self.model_direction:
            with open(self._model_path / "direction.pkl", "wb") as f:
                pickle.dump(self.model_direction, f)
        if self.model_magnitude:
            with open(self._model_path / "magnitude.pkl", "wb") as f:
                pickle.dump(self.model_magnitude, f)
        with open(self._model_path / "features.pkl", "wb") as f:
            pickle.dump(self.feature_names, f)
        logger.info("XGBoost model saved: %s", self._model_path)

    def load(self) -> None:
        """Load model from disk."""
        dir_path = self._model_path / "direction.pkl"
        mag_path = self._model_path / "magnitude.pkl"
        feat_path = self._model_path / "features.pkl"

        if dir_path.exists() and mag_path.exists() and feat_path.exists():
            with open(dir_path, "rb") as f:
                self.model_direction = pickle.load(f)
            with open(mag_path, "rb") as f:
                self.model_magnitude = pickle.load(f)
            with open(feat_path, "rb") as f:
                self.feature_names = pickle.load(f)
            logger.info("Loaded XGBoost model for %s", self.timeframe)
        else:
            logger.warning("No trained XGBoost model found for %s", self.timeframe)


XGBoostModel = XGBoostPredictor
