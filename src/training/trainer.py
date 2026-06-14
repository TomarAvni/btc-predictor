"""Main training orchestrator.

Coordinates the full pipeline: data loading, feature engineering,
walk-forward validation across halving cycles, model training, and
evaluation. Saves trained models and performance reports.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from src import DATA_DIR
from src.simulation.data_loader import HistoricalDataLoader
from src.simulation.labeler import ForwardReturnLabeler
from src.training.feature_builder import TrainingFeatureBuilder
from src.training.metrics import MetricsTracker
from src.training.walk_forward import WalkForwardValidator, WalkForwardResult
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

TIMEFRAMES = ["24h", "7d", "30d", "90d"]
MODEL_DIR = DATA_DIR / "models"


class TrainingOrchestrator:
    """Coordinates the complete model training pipeline.

    Pipeline:
    1. Load data from Parquet files
    2. Build features and labels
    3. Split by halving cycles (walk-forward)
    4. Train all models on each fold
    5. Validate on next cycle
    6. Compute ensemble weights from validation
    7. Save models and performance report
    """

    def __init__(
        self,
        data_dir: Optional[Path] = None,
        output_dir: Optional[Path] = None,
        cycles: Optional[list[int]] = None,
    ) -> None:
        self.data_dir = data_dir or DATA_DIR / "price"
        self.output_dir = output_dir or MODEL_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.cycles = cycles or [2, 3, 4]
        self.data_loader = HistoricalDataLoader(price_dir=self.data_dir)
        self.labeler = ForwardReturnLabeler()
        self.feature_builder = TrainingFeatureBuilder(model_dir=self.output_dir)
        self.walk_forward = WalkForwardValidator(min_train_cycles=2)
        self.metrics = MetricsTracker()

        self._models: dict[str, Any] = {}
        self._ensemble_weights: dict[str, dict[str, float]] = {}

    def run(self) -> dict[str, Any]:
        """Execute the full training pipeline.

        Returns a summary dict with metrics for each model and timeframe.
        """
        logger.info("=" * 70)
        logger.info("TRAINING PIPELINE START")
        logger.info("=" * 70)

        # Step 1: Load data
        logger.info("Step 1: Loading historical data...")
        price_df = self.data_loader.load_price_data()
        merged_df = self.data_loader.get_merged_dataset()
        logger.info("Data loaded: %d rows, %d columns", len(merged_df), len(merged_df.columns))

        # Step 2: Build features and labels
        logger.info("Step 2: Building features and labels...")
        features_df = self.feature_builder.build_features(merged_df)
        labels_df = self.labeler.compute_labels(price_df)

        # Align features and labels
        common_idx = features_df.index.intersection(labels_df.index)
        features_df = features_df.loc[common_idx]
        labels_df = labels_df.loc[common_idx]

        # Drop rows with all-NaN labels (end of dataset)
        return_cols = [c for c in labels_df.columns if c.startswith("return_")]
        valid_mask = labels_df[return_cols].notna().any(axis=1)
        features_df = features_df[valid_mask]
        labels_df = labels_df[valid_mask]

        logger.info(
            "Training data: %d samples with %d features",
            len(features_df), len(features_df.columns),
        )

        # Step 3: Walk-forward validation
        logger.info("Step 3: Running walk-forward validation...")
        results = self._run_walk_forward(features_df, labels_df)

        # Step 4: Train final models on all available data
        logger.info("Step 4: Training final models on full dataset...")
        final_metrics = self._train_final_models(features_df, labels_df)

        # Step 5: Save models and report
        logger.info("Step 5: Saving models and generating report...")
        report = self._generate_final_report(results, final_metrics)
        self._save_report(report)

        logger.info("=" * 70)
        logger.info("TRAINING PIPELINE COMPLETE")
        logger.info("=" * 70)

        return {
            "walk_forward": results.summary if results else {},
            "final_metrics": final_metrics,
            "ensemble_weights": self._ensemble_weights,
            "report": report,
        }

    def _run_walk_forward(
        self,
        features_df: pd.DataFrame,
        labels_df: pd.DataFrame,
    ) -> Optional[WalkForwardResult]:
        """Run walk-forward validation across all model types."""
        from src.models.baseline_model import BaselineModel
        from src.models.xgboost_model import XGBoostPredictor

        numeric_features = features_df.select_dtypes(include=[np.number])
        numeric_features = numeric_features.fillna(0)

        model_results: dict[str, WalkForwardResult] = {}

        # Baseline model
        logger.info("Walk-forward: Baseline model...")
        baseline_result = self.walk_forward.run_validation(
            data=numeric_features,
            labels=labels_df,
            train_fn=self._train_baseline,
            predict_fn=self._predict_baseline,
            timeframes=TIMEFRAMES,
        )
        model_results["baseline"] = baseline_result

        # XGBoost model
        logger.info("Walk-forward: XGBoost model...")
        xgb_result = self.walk_forward.run_validation(
            data=numeric_features,
            labels=labels_df,
            train_fn=self._train_xgboost,
            predict_fn=self._predict_xgboost,
            timeframes=TIMEFRAMES,
        )
        model_results["xgboost"] = xgb_result

        # Compute ensemble weights from validation performance
        self._ensemble_weights = self._compute_weights(model_results)
        logger.info("Ensemble weights: %s", self._ensemble_weights)

        return xgb_result

    def _train_final_models(
        self,
        features_df: pd.DataFrame,
        labels_df: pd.DataFrame,
    ) -> dict[str, Any]:
        """Train final models on all available data and save them."""
        from src.models.baseline_model import BaselineModel
        from src.models.xgboost_model import XGBoostPredictor

        numeric_features = features_df.select_dtypes(include=[np.number]).fillna(0)
        final_metrics: dict[str, Any] = {}

        for tf in TIMEFRAMES:
            target_col = f"return_{tf}"
            if target_col not in labels_df.columns:
                continue

            valid_mask = labels_df[target_col].notna()
            X = numeric_features[valid_mask]
            y = labels_df.loc[valid_mask, target_col]

            if len(X) < 500:
                logger.warning("Insufficient data for %s: %d samples", tf, len(X))
                continue

            # Train direction targets
            y_direction = (y > 0).astype(int)

            # Baseline
            baseline = BaselineModel()
            baseline.train(X, y_direction, y)
            baseline.save(self.output_dir / f"baseline_{tf}")

            # XGBoost
            xgb_config = {"models": {"xgboost": {}}}
            xgb = XGBoostPredictor(config=xgb_config, timeframe=tf)
            xgb.train(X, y_direction, y)

            final_metrics[tf] = {
                "n_samples": len(X),
                "pct_up": float(y_direction.mean()),
                "mean_return": float(y.mean()),
            }

        # Fit feature scaler on all training data
        self.feature_builder.fit_scaler(numeric_features)

        return final_metrics

    def _train_baseline(
        self, X: pd.DataFrame, y: pd.Series, timeframe: str
    ) -> Any:
        """Train a baseline model for walk-forward validation."""
        from src.models.baseline_model import BaselineModel

        model = BaselineModel()
        y_direction = (y > 0).astype(int)
        model.train(X, y_direction, y)
        return model

    def _predict_baseline(self, model: Any, X: pd.DataFrame) -> np.ndarray:
        """Generate predictions from a baseline model."""
        predictions = model.predict_batch(X)
        return predictions["predicted_return"].values

    def _train_xgboost(
        self, X: pd.DataFrame, y: pd.Series, timeframe: str
    ) -> Any:
        """Train an XGBoost model for walk-forward validation."""
        from src.models.xgboost_model import XGBoostPredictor

        config = {"models": {"xgboost": {
            "n_estimators": 300,
            "max_depth": 6,
            "learning_rate": 0.05,
        }}}
        model = XGBoostPredictor(config=config, timeframe=timeframe)
        y_direction = (y > 0).astype(int)
        model.train(X, y_direction, y)
        return model

    def _predict_xgboost(self, model: Any, X: pd.DataFrame) -> np.ndarray:
        """Generate predictions from an XGBoost model."""
        predictions = []
        for idx in range(len(X)):
            row = X.iloc[idx]
            features = {col: float(row[col]) for col in X.columns}
            pred = model.predict(features)
            mag = pred["predicted_magnitude"]
            direction = 1 if pred["direction_prob"] > 0.5 else -1
            predictions.append(mag * direction)
        return np.array(predictions)

    def _compute_weights(
        self, model_results: dict[str, WalkForwardResult]
    ) -> dict[str, dict[str, float]]:
        """Compute ensemble weights from walk-forward validation results."""
        weights: dict[str, dict[str, float]] = {}

        for tf in TIMEFRAMES:
            tf_weights: dict[str, float] = {}
            accuracies: dict[str, float] = {}

            for model_name, result in model_results.items():
                tf_accs = []
                for fold in result.folds:
                    if tf in fold.metrics:
                        tf_accs.append(fold.metrics[tf]["direction_accuracy"])
                if tf_accs:
                    accuracies[model_name] = np.mean(tf_accs)

            if accuracies:
                total = sum(accuracies.values())
                for model_name, acc in accuracies.items():
                    tf_weights[model_name] = acc / total if total > 0 else 1.0 / len(accuracies)

            weights[tf] = tf_weights

        return weights

    def _generate_final_report(
        self,
        wf_result: Optional[WalkForwardResult],
        final_metrics: dict,
    ) -> str:
        """Generate the final training report."""
        lines = [
            "=" * 70,
            "  BTC PREDICTION MODEL - TRAINING REPORT",
            "=" * 70,
            "",
            "DATA SUMMARY:",
        ]

        for tf, m in final_metrics.items():
            lines.append(f"  {tf}: {m['n_samples']} samples, "
                        f"{m['pct_up']:.1%} up, mean return={m['mean_return']:+.2f}%")

        lines.append("")
        lines.append("WALK-FORWARD VALIDATION:")

        if wf_result and wf_result.summary:
            for tf, metrics in wf_result.summary.items():
                if tf == "n_folds" or not isinstance(metrics, dict):
                    continue
                lines.append(
                    f"  {tf}: accuracy={metrics.get('mean_accuracy', 0):.1%} "
                    f"± {metrics.get('std_accuracy', 0):.1%}, "
                    f"MAE={metrics.get('mean_mae', 0):.2f}%"
                )

        lines.append("")
        lines.append("ENSEMBLE WEIGHTS:")
        for tf, weights in self._ensemble_weights.items():
            weight_str = ", ".join(f"{k}={v:.2f}" for k, v in weights.items())
            lines.append(f"  {tf}: {weight_str}")

        lines.append("")
        lines.append("=" * 70)
        return "\n".join(lines)

    def _save_report(self, report: str) -> None:
        """Save the training report to disk."""
        report_path = self.output_dir / "training_report.txt"
        report_path.write_text(report, encoding="utf-8")
        logger.info("Report saved to %s", report_path)
        print(report)


Trainer = TrainingOrchestrator
