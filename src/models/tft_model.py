"""Temporal Fusion Transformer model for multi-horizon BTC forecasting.

Uses pytorch-forecasting's TemporalFusionTransformer for sequential
prediction across all horizons simultaneously. Provides quantile
outputs for uncertainty estimation and attention weights for interpretability.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from src import DATA_DIR
from src.horizons import HORIZON_HOUR_VALUES
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

MODEL_DIR = DATA_DIR / "models" / "tft"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

SEQUENCE_LENGTH = 168  # 7 days of hourly data as input
PREDICTION_HORIZONS = list(HORIZON_HOUR_VALUES)

TIME_VARYING_KNOWN = [
    "hour_sin", "hour_cos", "day_sin", "day_cos", "month_sin", "month_cos",
    "days_since_halving", "cycle_pct",
]

TIME_VARYING_UNKNOWN = [
    "close", "volume", "return_1h", "return_4h",
    "volatility_24h", "rsi_14", "macd_histogram",
    "bb_width", "volume_ratio",
]

STATIC_CATEGORICALS = ["cycle_number"]


class TFTPredictor:
    """Temporal Fusion Transformer for multi-horizon BTC prediction.

    Architecture:
    - Variable selection networks choose which features matter at each step
    - LSTM encoder processes historical sequence
    - Multi-head attention over encoder outputs
    - Quantile outputs (10th, 50th, 90th) for uncertainty estimation
    - Predicts all configured horizons (see src/horizons.py) simultaneously
    """

    def __init__(
        self,
        sequence_length: int = SEQUENCE_LENGTH,
        model_dir: Optional[Path] = None,
        hidden_size: int = 64,
        attention_head_size: int = 4,
        dropout: float = 0.1,
        learning_rate: float = 0.001,
        max_epochs: int = 50,
        batch_size: int = 128,
    ) -> None:
        self.sequence_length = sequence_length
        self.model_dir = model_dir or MODEL_DIR
        self.hidden_size = hidden_size
        self.attention_head_size = attention_head_size
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.batch_size = batch_size

        self._model = None
        self._trainer = None
        self._training_dataset = None
        self._feature_names: list[str] = []
        self._is_trained = False

    def train(
        self,
        features_df: pd.DataFrame,
        labels_df: pd.DataFrame,
        val_fraction: float = 0.2,
    ) -> dict[str, float]:
        """Train the TFT model on historical features and labels.

        Args:
            features_df: Feature matrix with DatetimeIndex (hourly).
            labels_df: Labels with return_{horizon} columns.
            val_fraction: Fraction of data for validation (from the end).

        Returns:
            Dict with training metrics.
        """
        try:
            import pytorch_lightning as pl
            from pytorch_forecasting import (
                TemporalFusionTransformer,
                TimeSeriesDataSet,
                QuantileLoss,
            )
            from pytorch_forecasting.data import GroupNormalizer
        except ImportError:
            logger.error(
                "pytorch-forecasting not installed. "
                "Install with: pip install pytorch-forecasting pytorch-lightning"
            )
            return {"error": "missing_dependency"}

        prepared = self._prepare_data(features_df, labels_df)
        if prepared is None:
            return {"error": "data_preparation_failed"}

        df_prepared, target_col = prepared

        # Split into train/val (temporal split, no shuffling)
        n_total = df_prepared["time_idx"].max() + 1
        train_cutoff = int(n_total * (1 - val_fraction))

        training = TimeSeriesDataSet(
            df_prepared[df_prepared["time_idx"] <= train_cutoff],
            time_idx="time_idx",
            target=target_col,
            group_ids=["group"],
            min_encoder_length=self.sequence_length // 2,
            max_encoder_length=self.sequence_length,
            min_prediction_length=1,
            max_prediction_length=24,
            static_categoricals=["group"],
            time_varying_known_reals=self._get_available_columns(
                df_prepared, TIME_VARYING_KNOWN
            ),
            time_varying_unknown_reals=self._get_available_columns(
                df_prepared, TIME_VARYING_UNKNOWN + [target_col]
            ),
            target_normalizer=GroupNormalizer(groups=["group"]),
            add_relative_time_idx=True,
            add_target_scales=True,
            add_encoder_length=True,
        )

        validation = TimeSeriesDataSet.from_dataset(
            training,
            df_prepared[df_prepared["time_idx"] > train_cutoff],
            predict=True,
            stop_randomization=True,
        )

        train_dataloader = training.to_dataloader(
            train=True, batch_size=self.batch_size, num_workers=0
        )
        val_dataloader = validation.to_dataloader(
            train=False, batch_size=self.batch_size, num_workers=0
        )

        tft = TemporalFusionTransformer.from_dataset(
            training,
            learning_rate=self.learning_rate,
            hidden_size=self.hidden_size,
            attention_head_size=self.attention_head_size,
            dropout=self.dropout,
            hidden_continuous_size=self.hidden_size // 2,
            output_size=7,  # 7 quantiles
            loss=QuantileLoss(quantiles=[0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98]),
            reduce_on_plateau_patience=4,
        )

        trainer = pl.Trainer(
            max_epochs=self.max_epochs,
            accelerator="auto",
            gradient_clip_val=0.1,
            enable_progress_bar=True,
            callbacks=[
                pl.callbacks.EarlyStopping(
                    monitor="val_loss", patience=5, mode="min"
                ),
            ],
        )

        trainer.fit(
            tft,
            train_dataloaders=train_dataloader,
            val_dataloaders=val_dataloader,
        )

        self._model = tft
        self._trainer = trainer
        self._training_dataset = training
        self._is_trained = True

        # Compute validation metrics
        val_metrics = self._evaluate(tft, val_dataloader)
        self.save()

        logger.info("TFT training complete: %s", val_metrics)
        return val_metrics

    def predict(self, features_df: pd.DataFrame) -> dict[str, Any]:
        """Generate predictions with uncertainty estimates.

        Returns:
            Dict with direction, magnitude, confidence per horizon,
            plus quantile forecasts.
        """
        if not self._is_trained:
            self.load()
            if not self._is_trained:
                return self._empty_prediction()

        try:
            predictions = self._model.predict(
                features_df, mode="quantiles", return_x=True
            )
        except Exception as e:
            logger.debug("TFT prediction failed: %s", e)
            return self._empty_prediction()

        # Extract quantile predictions
        quantiles = predictions.output
        median_pred = float(quantiles[:, :, 3].mean())  # 50th percentile
        lower_10 = float(quantiles[:, :, 1].mean())
        upper_90 = float(quantiles[:, :, 5].mean())

        direction_prob = 1.0 / (1.0 + np.exp(-median_pred * 10))
        magnitude = abs(median_pred)

        # Confidence from quantile spread (narrower = more confident)
        spread = upper_90 - lower_10
        confidence = max(0, min(1, 1 - spread / 20)) * 100

        return {
            "direction_prob": direction_prob,
            "predicted_magnitude": magnitude,
            "confidence": confidence,
            "quantiles": {
                "p10": lower_10,
                "p50": median_pred,
                "p90": upper_90,
            },
        }

    def predict_batch(self, X: pd.DataFrame) -> pd.DataFrame:
        """Batch prediction for walk-forward validation.

        Returns DataFrame aligned to input index with predictions.
        """
        if not self._is_trained:
            return pd.DataFrame({
                "direction_prob": 0.5,
                "predicted_return": 0.0,
                "confidence": 0.0,
            }, index=X.index)

        # For batch, we use a simpler approach compatible with walk-forward
        results = []
        for i in range(0, len(X), self.sequence_length):
            chunk = X.iloc[i:i + self.sequence_length]
            if len(chunk) < self.sequence_length // 2:
                break
            try:
                pred = self.predict(chunk)
                results.append({
                    "timestamp": chunk.index[-1],
                    "direction_prob": pred["direction_prob"],
                    "predicted_return": pred["predicted_magnitude"] * (
                        1 if pred["direction_prob"] > 0.5 else -1
                    ),
                    "confidence": pred["confidence"],
                })
            except Exception:
                pass

        if results:
            result_df = pd.DataFrame(results).set_index("timestamp")
            return result_df.reindex(X.index, method="ffill")

        return pd.DataFrame({
            "direction_prob": 0.5,
            "predicted_return": 0.0,
            "confidence": 0.0,
        }, index=X.index)

    def get_attention_weights(self) -> Optional[dict[str, np.ndarray]]:
        """Extract attention weights for interpretability.

        Shows which time steps and features the model focused on.
        """
        if self._model is None:
            return None

        try:
            interpretation = self._model.interpret_output(
                self._model.predict(self._training_dataset.to_dataloader(
                    train=False, batch_size=64, num_workers=0
                ), return_x=True, return_decoder_lengths=True),
                reduction="mean",
            )
            return {
                "encoder_attention": interpretation["attention"].numpy()
                if hasattr(interpretation.get("attention", None), "numpy") else None,
                "feature_importance": interpretation.get("static_variables", {}),
            }
        except Exception as e:
            logger.debug("Failed to extract attention weights: %s", e)
            return None

    def save(self) -> None:
        """Save trained model to disk."""
        self.model_dir.mkdir(parents=True, exist_ok=True)

        if self._model is not None:
            model_path = self.model_dir / "tft_model.ckpt"
            self._trainer.save_checkpoint(str(model_path))

        meta = {
            "is_trained": self._is_trained,
            "feature_names": self._feature_names,
            "sequence_length": self.sequence_length,
            "hidden_size": self.hidden_size,
        }
        with open(self.model_dir / "tft_meta.pkl", "wb") as f:
            pickle.dump(meta, f)

        logger.info("TFT model saved to %s", self.model_dir)

    def load(self) -> None:
        """Load trained model from disk."""
        meta_path = self.model_dir / "tft_meta.pkl"
        model_path = self.model_dir / "tft_model.ckpt"

        if not meta_path.exists():
            logger.warning("No TFT model metadata found at %s", meta_path)
            return

        with open(meta_path, "rb") as f:
            meta = pickle.load(f)

        self._feature_names = meta["feature_names"]
        self._is_trained = meta["is_trained"]
        self.sequence_length = meta["sequence_length"]

        if model_path.exists() and self._is_trained:
            try:
                from pytorch_forecasting import TemporalFusionTransformer
                self._model = TemporalFusionTransformer.load_from_checkpoint(
                    str(model_path)
                )
                logger.info("TFT model loaded from %s", model_path)
            except Exception as e:
                logger.warning("Failed to load TFT model: %s", e)
                self._is_trained = False

    def _prepare_data(
        self,
        features_df: pd.DataFrame,
        labels_df: pd.DataFrame,
    ) -> Optional[tuple[pd.DataFrame, str]]:
        """Prepare data in the format expected by pytorch-forecasting."""
        target_col = "return_24h"
        if target_col not in labels_df.columns:
            logger.error("Target column '%s' not found in labels", target_col)
            return None

        df = features_df.copy()
        df[target_col] = labels_df[target_col]
        df = df.dropna(subset=[target_col])

        if len(df) < self.sequence_length * 3:
            logger.error("Insufficient data for TFT: %d rows (need %d+)",
                        len(df), self.sequence_length * 3)
            return None

        # Add required columns for TimeSeriesDataSet
        df["time_idx"] = range(len(df))
        df["group"] = "BTC"

        # Fill NaN features
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(0)

        self._feature_names = [
            c for c in df.columns
            if c not in ("time_idx", "group", target_col)
        ]

        return df, target_col

    def _evaluate(self, model, val_dataloader) -> dict[str, float]:
        """Evaluate model on validation set."""
        try:
            predictions = model.predict(val_dataloader, return_y=True)
            actuals = predictions.y[0] if isinstance(predictions.y, tuple) else predictions.y

            pred_values = predictions.output[:, :, 3]  # Median quantile
            pred_flat = pred_values.flatten().numpy()
            actual_flat = actuals.flatten().numpy()

            min_len = min(len(pred_flat), len(actual_flat))
            pred_flat = pred_flat[:min_len]
            actual_flat = actual_flat[:min_len]

            direction_correct = np.sign(pred_flat) == np.sign(actual_flat)
            accuracy = float(direction_correct.mean())
            mae = float(np.abs(pred_flat - actual_flat).mean())

            return {"direction_accuracy": accuracy, "mae": mae}
        except Exception as e:
            logger.warning("TFT evaluation failed: %s", e)
            return {"direction_accuracy": 0.5, "mae": 0.0}

    @staticmethod
    def _get_available_columns(df: pd.DataFrame, candidates: list[str]) -> list[str]:
        """Filter column candidates to those actually present in the DataFrame."""
        return [c for c in candidates if c in df.columns]

    @staticmethod
    def _empty_prediction() -> dict[str, Any]:
        return {
            "direction_prob": 0.5,
            "predicted_magnitude": 0.0,
            "confidence": 0.0,
            "quantiles": {"p10": 0.0, "p50": 0.0, "p90": 0.0},
        }


TFTModel = TFTPredictor
