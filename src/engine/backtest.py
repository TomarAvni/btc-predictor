"""Comprehensive backtesting framework.

Runs trained models through historical data, compares predictions to actual
outcomes, and generates detailed performance reports including accuracy by
regime, confidence calibration curves, and best/worst prediction analysis.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from src import DATA_DIR
from src.horizons import TIMEFRAMES
from src.models.confidence import ConfidenceCalibrator
from src.simulation.data_loader import HistoricalDataLoader
from src.simulation.labeler import ForwardReturnLabeler
from src.simulation.market_replay import MarketReplay
from src.training.feature_builder import TrainingFeatureBuilder
from src.training.metrics import MetricsTracker
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


@dataclass
class BacktestPrediction:
    """A single backtest prediction record."""

    timestamp: pd.Timestamp
    timeframe: str
    predicted_direction: str
    predicted_magnitude: float
    confidence: float
    actual_return: float
    actual_direction: str
    was_correct: bool
    regime: str


@dataclass
class BacktestReport:
    """Complete backtest results."""

    predictions: list[BacktestPrediction] = field(default_factory=list)
    overall_accuracy: dict[str, float] = field(default_factory=dict)
    regime_accuracy: dict[str, dict[str, float]] = field(default_factory=dict)
    calibration_data: dict[str, Any] = field(default_factory=dict)
    best_predictions: list[BacktestPrediction] = field(default_factory=list)
    worst_predictions: list[BacktestPrediction] = field(default_factory=list)
    summary_text: str = ""


class Backtester:
    """Comprehensive backtesting with regime-aware evaluation.

    Given a trained model and date range, replays the market simulation
    and evaluates prediction quality across multiple dimensions.
    """

    def __init__(
        self,
        data_dir: Optional[Path] = None,
        model_dir: Optional[Path] = None,
    ) -> None:
        self.data_dir = data_dir or DATA_DIR / "price"
        self.model_dir = model_dir or DATA_DIR / "models"

        self.data_loader = HistoricalDataLoader(price_dir=self.data_dir)
        self.labeler = ForwardReturnLabeler()
        self.feature_builder = TrainingFeatureBuilder(model_dir=self.model_dir)
        self.metrics = MetricsTracker()
        self.calibrator = ConfidenceCalibrator(model_dir=self.model_dir)

        self._predictions: list[BacktestPrediction] = []

    def run(
        self,
        model,
        start_date: str | pd.Timestamp,
        end_date: str | pd.Timestamp,
        timeframes: Optional[list[str]] = None,
        step_hours: int = 4,
    ) -> BacktestReport:
        """Run a comprehensive backtest of a trained model.

        Args:
            model: Any model with a predict(features: dict) method.
            start_date: Backtest start date.
            end_date: Backtest end date.
            timeframes: Which prediction horizons to evaluate.
            step_hours: Hours between predictions (4 = every 4 hours).

        Returns:
            BacktestReport with comprehensive analysis.
        """
        start = pd.Timestamp(start_date, tz="UTC")
        end = pd.Timestamp(end_date, tz="UTC")
        timeframes = timeframes or TIMEFRAMES

        logger.info("Backtesting from %s to %s (step=%dh)", start.date(), end.date(), step_hours)

        # Load data
        price_df = self.data_loader.load_price_data()
        merged_df = self.data_loader.get_merged_dataset()
        labels_df = self.labeler.compute_labels(price_df)

        # Build features for the backtest range
        features_df = self.feature_builder.build_features(merged_df)

        # Set up the market replay for regime classification
        replay = MarketReplay(data_loader=self.data_loader, labeler=self.labeler)
        replay._prepare_data(start, end)

        # Get simulation timestamps
        available = features_df.loc[start:end].index
        test_timestamps = available[::step_hours]

        logger.info("Backtest will evaluate %d timestamps", len(test_timestamps))

        self._predictions = []

        for ts in test_timestamps:
            if ts not in features_df.index:
                continue

            # Get features at this timestamp
            row = features_df.loc[ts]
            features = {col: float(row[col]) if pd.notna(row[col]) else 0.0 for col in features_df.columns}

            # Get model prediction
            try:
                prediction = model.predict(features)
            except Exception as e:
                logger.debug("Prediction failed at %s: %s", ts, e)
                continue

            # Get regime
            regime = replay._classify_market_regime(ts)

            # Record predictions for each timeframe
            for tf in timeframes:
                actual_col = f"return_{tf}"
                if ts not in labels_df.index or actual_col not in labels_df.columns:
                    continue

                actual_return = labels_df.loc[ts, actual_col]
                if pd.isna(actual_return):
                    continue

                pred_direction = "UP" if prediction.get("direction_prob", 0.5) > 0.5 else "DOWN"
                actual_direction = "UP" if actual_return > 0 else "DOWN"
                was_correct = pred_direction == actual_direction

                bp = BacktestPrediction(
                    timestamp=ts,
                    timeframe=tf,
                    predicted_direction=pred_direction,
                    predicted_magnitude=prediction.get("predicted_magnitude", 0),
                    confidence=prediction.get("raw_confidence", 0.5) * 100,
                    actual_return=float(actual_return),
                    actual_direction=actual_direction,
                    was_correct=was_correct,
                    regime=regime,
                )
                self._predictions.append(bp)

                # Track in metrics
                self.metrics.record(
                    timestamp=ts,
                    timeframe=tf,
                    predicted_direction=1 if pred_direction == "UP" else 0,
                    predicted_magnitude=prediction.get("predicted_magnitude", 0),
                    confidence=bp.confidence,
                    actual_return=float(actual_return),
                    regime=regime,
                )

        # Generate report
        report = self._generate_report(timeframes)
        logger.info("Backtest complete: %d predictions recorded", len(self._predictions))
        return report

    def _generate_report(self, timeframes: list[str]) -> BacktestReport:
        """Generate comprehensive backtest report from predictions."""
        report = BacktestReport(predictions=self._predictions)

        if not self._predictions:
            report.summary_text = "No predictions generated during backtest."
            return report

        df = pd.DataFrame([vars(p) for p in self._predictions])

        # Overall accuracy per timeframe
        for tf in timeframes:
            tf_data = df[df["timeframe"] == tf]
            if not tf_data.empty:
                report.overall_accuracy[tf] = float(tf_data["was_correct"].mean())

        # Accuracy by regime
        for regime in df["regime"].unique():
            regime_data = df[df["regime"] == regime]
            report.regime_accuracy[regime] = {
                "overall": float(regime_data["was_correct"].mean()),
                "count": len(regime_data),
            }
            for tf in timeframes:
                tf_regime = regime_data[regime_data["timeframe"] == tf]
                if not tf_regime.empty:
                    report.regime_accuracy[regime][tf] = float(tf_regime["was_correct"].mean())

        # Confidence calibration
        report.calibration_data = self._compute_calibration(df)

        # Best predictions (highest confidence AND correct)
        correct = df[df["was_correct"]]
        if not correct.empty:
            best = correct.nlargest(10, "confidence")
            report.best_predictions = [
                BacktestPrediction(**row) for _, row in best.iterrows()
            ]

        # Worst predictions (highest confidence AND wrong)
        incorrect = df[~df["was_correct"]]
        if not incorrect.empty:
            worst = incorrect.nlargest(10, "confidence")
            report.worst_predictions = [
                BacktestPrediction(**row) for _, row in worst.iterrows()
            ]

        # Generate summary text
        report.summary_text = self._format_report(report)
        return report

    def _compute_calibration(self, df: pd.DataFrame, n_bins: int = 5) -> dict:
        """Compute confidence calibration data."""
        bins = np.linspace(0, 100, n_bins + 1)
        calibration: dict = {"bins": []}

        for i in range(n_bins):
            mask = (df["confidence"] >= bins[i]) & (df["confidence"] < bins[i + 1])
            bin_data = df[mask]
            if bin_data.empty:
                continue

            calibration["bins"].append({
                "range": f"{bins[i]:.0f}-{bins[i+1]:.0f}%",
                "avg_confidence": float(bin_data["confidence"].mean()),
                "actual_accuracy": float(bin_data["was_correct"].mean() * 100),
                "count": len(bin_data),
            })

        return calibration

    def _format_report(self, report: BacktestReport) -> str:
        """Generate human-readable backtest report."""
        lines = [
            "=" * 70,
            "  BACKTEST RESULTS",
            "=" * 70,
            "",
            f"  Total predictions: {len(report.predictions)}",
            "",
            "ACCURACY BY TIMEFRAME:",
        ]

        for tf, acc in report.overall_accuracy.items():
            n = sum(1 for p in report.predictions if p.timeframe == tf)
            lines.append(f"  {tf:>4s}: {acc:.1%} ({n} predictions)")

        lines.append("")
        lines.append("ACCURACY BY MARKET REGIME:")
        for regime, data in report.regime_accuracy.items():
            lines.append(f"  {regime:>10s}: {data['overall']:.1%} (n={data['count']})")

        lines.append("")
        lines.append("CONFIDENCE CALIBRATION:")
        for bin_data in report.calibration_data.get("bins", []):
            lines.append(
                f"  Conf {bin_data['range']:>10s}: "
                f"actual={bin_data['actual_accuracy']:.1f}% "
                f"(n={bin_data['count']})"
            )

        if report.worst_predictions:
            lines.append("")
            lines.append("WORST PREDICTIONS (highest confidence, wrong):")
            for p in report.worst_predictions[:5]:
                lines.append(
                    f"  {p.timestamp.date()} {p.timeframe}: "
                    f"predicted {p.predicted_direction} ({p.confidence:.0f}% conf), "
                    f"actual {p.actual_return:+.1f}%"
                )

        if report.best_predictions:
            lines.append("")
            lines.append("BEST PREDICTIONS (highest confidence, correct):")
            for p in report.best_predictions[:5]:
                lines.append(
                    f"  {p.timestamp.date()} {p.timeframe}: "
                    f"predicted {p.predicted_direction} ({p.confidence:.0f}% conf), "
                    f"actual {p.actual_return:+.1f}%"
                )

        lines.append("")
        lines.append("=" * 70)
        return "\n".join(lines)
