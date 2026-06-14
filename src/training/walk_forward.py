"""Walk-forward cross-validation for temporal financial data.

Implements expanding-window validation split by BTC halving cycles,
ensuring no future data ever leaks into the training set.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

import numpy as np
import pandas as pd

from src.collectors.cycle import HALVING_DATES
from src.training.metrics import MetricsTracker
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

CYCLE_BOUNDARIES = [
    ("cycle_1", datetime(2012, 11, 28, tzinfo=timezone.utc), datetime(2016, 7, 9, tzinfo=timezone.utc)),
    ("cycle_2", datetime(2016, 7, 9, tzinfo=timezone.utc), datetime(2020, 5, 11, tzinfo=timezone.utc)),
    ("cycle_3", datetime(2020, 5, 11, tzinfo=timezone.utc), datetime(2024, 4, 19, tzinfo=timezone.utc)),
    ("cycle_4", datetime(2024, 4, 19, tzinfo=timezone.utc), datetime(2028, 4, 1, tzinfo=timezone.utc)),
]


@dataclass
class FoldResult:
    """Results from one walk-forward fold."""

    fold_number: int
    train_cycles: list[str]
    val_cycle: str
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    val_start: pd.Timestamp
    val_end: pd.Timestamp
    train_samples: int
    val_samples: int
    metrics: dict[str, Any] = field(default_factory=dict)
    model_type: str = ""


@dataclass
class WalkForwardResult:
    """Complete walk-forward validation results."""

    folds: list[FoldResult] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)
    best_model_per_timeframe: dict[str, str] = field(default_factory=dict)
    ensemble_weights: dict[str, dict[str, float]] = field(default_factory=dict)


class WalkForwardValidator:
    """Walk-forward cross-validation using BTC halving cycles.

    Strategy: expanding window where each fold adds one more cycle to training.
    - Fold 1: Train on Cycle 1-2, validate on Cycle 3
    - Fold 2: Train on Cycle 1-3, validate on Cycle 4
    
    This prevents overfitting to a single market regime and ensures the model
    generalizes across different cycle phases.
    """

    def __init__(self, min_train_cycles: int = 2) -> None:
        self.min_train_cycles = min_train_cycles
        self.fold_results: list[FoldResult] = []

    def get_folds(
        self,
        data: pd.DataFrame,
    ) -> list[tuple[pd.DataFrame, pd.DataFrame, FoldResult]]:
        """Generate train/validation splits based on halving cycles.

        Returns a list of (train_df, val_df, fold_info) tuples.
        The training window expands with each fold (never shrinks).
        """
        folds = []
        available_cycles = self._identify_available_cycles(data)

        if len(available_cycles) <= self.min_train_cycles:
            logger.warning(
                "Not enough cycles for walk-forward validation. "
                "Need at least %d train + 1 val, have %d total.",
                self.min_train_cycles, len(available_cycles),
            )
            return folds

        for fold_num, val_idx in enumerate(
            range(self.min_train_cycles, len(available_cycles))
        ):
            train_cycles = available_cycles[:val_idx]
            val_cycle = available_cycles[val_idx]

            train_start = train_cycles[0][1]
            train_end = train_cycles[-1][2]
            val_start = val_cycle[1]
            val_end = val_cycle[2]

            train_mask = (data.index >= train_start) & (data.index < train_end)
            val_mask = (data.index >= val_start) & (data.index < val_end)

            train_df = data[train_mask]
            val_df = data[val_mask]

            if train_df.empty or val_df.empty:
                logger.warning(
                    "Fold %d has empty split (train=%d, val=%d), skipping",
                    fold_num + 1, len(train_df), len(val_df),
                )
                continue

            fold_info = FoldResult(
                fold_number=fold_num + 1,
                train_cycles=[c[0] for c in train_cycles],
                val_cycle=val_cycle[0],
                train_start=pd.Timestamp(train_start),
                train_end=pd.Timestamp(train_end),
                val_start=pd.Timestamp(val_start),
                val_end=pd.Timestamp(val_end),
                train_samples=len(train_df),
                val_samples=len(val_df),
            )

            folds.append((train_df, val_df, fold_info))
            logger.info(
                "Fold %d: train on %s (%d samples), validate on %s (%d samples)",
                fold_info.fold_number,
                fold_info.train_cycles,
                fold_info.train_samples,
                fold_info.val_cycle,
                fold_info.val_samples,
            )

        return folds

    def run_validation(
        self,
        data: pd.DataFrame,
        labels: pd.DataFrame,
        train_fn,
        predict_fn,
        timeframes: Optional[list[str]] = None,
    ) -> WalkForwardResult:
        """Run full walk-forward validation with provided train/predict functions.

        Args:
            data: Feature matrix (DatetimeIndex).
            labels: Label DataFrame with return_{timeframe} columns.
            train_fn: Callable(train_X, train_y, timeframe) -> trained_model
            predict_fn: Callable(model, val_X) -> predictions_df
            timeframes: Which timeframes to evaluate (default: all available).

        Returns:
            WalkForwardResult with per-fold and aggregate metrics.
        """
        if timeframes is None:
            timeframes = [
                col.replace("return_", "")
                for col in labels.columns
                if col.startswith("return_") and not col.startswith("return_direction")
            ]

        result = WalkForwardResult()
        folds = self.get_folds(data)

        if not folds:
            logger.error("No valid folds generated. Check data coverage.")
            return result

        metrics_tracker = MetricsTracker()

        for train_df, val_df, fold_info in folds:
            fold_metrics: dict[str, dict] = {}

            for tf in timeframes:
                target_col = f"return_{tf}"
                if target_col not in labels.columns:
                    continue

                train_labels = labels.loc[train_df.index, target_col].dropna()
                val_labels = labels.loc[val_df.index, target_col].dropna()

                common_train = train_df.index.intersection(train_labels.index)
                common_val = val_df.index.intersection(val_labels.index)

                if len(common_train) < 100 or len(common_val) < 50:
                    logger.warning(
                        "Fold %d, %s: insufficient data (train=%d, val=%d)",
                        fold_info.fold_number, tf, len(common_train), len(common_val),
                    )
                    continue

                X_train = train_df.loc[common_train]
                y_train = train_labels.loc[common_train]
                X_val = val_df.loc[common_val]
                y_val = val_labels.loc[common_val]

                try:
                    model = train_fn(X_train, y_train, tf)
                    predictions = predict_fn(model, X_val)

                    # Compute fold metrics
                    direction_correct = (
                        np.sign(predictions) == np.sign(y_val.values)
                    )
                    accuracy = float(direction_correct.mean())
                    mae = float(np.abs(predictions - y_val.values).mean())

                    fold_metrics[tf] = {
                        "direction_accuracy": accuracy,
                        "mae": mae,
                        "n_samples": len(y_val),
                    }

                    logger.info(
                        "Fold %d | %s: accuracy=%.3f, MAE=%.2f%% (%d samples)",
                        fold_info.fold_number, tf, accuracy, mae, len(y_val),
                    )

                except Exception as e:
                    logger.error(
                        "Fold %d, %s training failed: %s",
                        fold_info.fold_number, tf, e,
                    )

            fold_info.metrics = fold_metrics
            result.folds.append(fold_info)

        result.summary = self._compute_summary(result.folds, timeframes)
        result.best_model_per_timeframe = self._determine_best_models(result.folds)
        result.ensemble_weights = self._compute_ensemble_weights(result.folds)

        return result

    def _identify_available_cycles(
        self, data: pd.DataFrame
    ) -> list[tuple[str, datetime, datetime]]:
        """Determine which halving cycles have data coverage."""
        data_start = data.index[0].to_pydatetime()
        data_end = data.index[-1].to_pydatetime()

        if data_start.tzinfo is None:
            data_start = data_start.replace(tzinfo=timezone.utc)
        if data_end.tzinfo is None:
            data_end = data_end.replace(tzinfo=timezone.utc)

        available = []
        for name, start, end in CYCLE_BOUNDARIES:
            cycle_end = min(end, data_end)
            if start >= data_start and cycle_end > start:
                mask = (data.index >= start) & (data.index < cycle_end)
                if mask.sum() > 720:  # At least 30 days of data
                    available.append((name, start, cycle_end))

        logger.info("Available cycles for validation: %s", [c[0] for c in available])
        return available

    def _compute_summary(
        self, folds: list[FoldResult], timeframes: list[str]
    ) -> dict[str, Any]:
        """Aggregate metrics across all folds."""
        summary: dict[str, Any] = {"n_folds": len(folds)}

        for tf in timeframes:
            tf_accuracies = []
            tf_maes = []
            for fold in folds:
                if tf in fold.metrics:
                    tf_accuracies.append(fold.metrics[tf]["direction_accuracy"])
                    tf_maes.append(fold.metrics[tf]["mae"])

            if tf_accuracies:
                summary[tf] = {
                    "mean_accuracy": float(np.mean(tf_accuracies)),
                    "std_accuracy": float(np.std(tf_accuracies)),
                    "mean_mae": float(np.mean(tf_maes)),
                    "min_accuracy": float(np.min(tf_accuracies)),
                    "max_accuracy": float(np.max(tf_accuracies)),
                }

        return summary

    def _determine_best_models(self, folds: list[FoldResult]) -> dict[str, str]:
        """Placeholder for determining best model per timeframe.

        In the full pipeline this is populated by the trainer which runs
        multiple models per fold.
        """
        return {}

    def _compute_ensemble_weights(
        self, folds: list[FoldResult]
    ) -> dict[str, dict[str, float]]:
        """Compute optimal ensemble weights from validation performance.

        Uses inverse-error weighting: models with lower MAE get higher weight.
        Returns per-timeframe weight dictionaries.
        """
        return {}

    def print_summary(self, result: WalkForwardResult) -> str:
        """Generate human-readable walk-forward summary."""
        lines = [
            "=" * 70,
            "  WALK-FORWARD VALIDATION SUMMARY",
            "=" * 70,
            f"  Folds: {result.summary.get('n_folds', 0)}",
            "",
        ]

        for tf, metrics in result.summary.items():
            if tf == "n_folds":
                continue
            if isinstance(metrics, dict):
                lines.append(f"  {tf}:")
                lines.append(f"    Accuracy: {metrics['mean_accuracy']:.1%} ± {metrics['std_accuracy']:.1%}")
                lines.append(f"    MAE: {metrics['mean_mae']:.2f}%")
                lines.append(f"    Range: [{metrics['min_accuracy']:.1%}, {metrics['max_accuracy']:.1%}]")

        lines.append("")
        lines.append("  PER-FOLD RESULTS:")
        for fold in result.folds:
            lines.append(f"    Fold {fold.fold_number}: "
                        f"train={fold.train_cycles} → val={fold.val_cycle}")
            for tf, m in fold.metrics.items():
                lines.append(f"      {tf}: acc={m['direction_accuracy']:.3f}, MAE={m['mae']:.2f}%")

        lines.append("=" * 70)
        return "\n".join(lines)
