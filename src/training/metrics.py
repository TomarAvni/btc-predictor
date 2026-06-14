"""Performance tracking for model evaluation.

Computes directional accuracy, magnitude error, confidence calibration,
per-regime performance, and Sharpe-like metrics from model predictions.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class MetricsTracker:
    """Tracks and reports model performance across multiple dimensions.

    Metrics computed:
    - Directional accuracy per timeframe
    - Magnitude MAE per timeframe
    - Confidence calibration (predicted confidence vs actual accuracy)
    - Per-regime performance (bull/bear/sideways)
    - Sharpe-like return metric (if following model predictions)
    """

    def __init__(self) -> None:
        self.predictions: list[dict] = []
        self._report_cache: Optional[dict] = None

    def record(
        self,
        timestamp: pd.Timestamp,
        timeframe: str,
        predicted_direction: int,
        predicted_magnitude: float,
        confidence: float,
        actual_return: float,
        regime: str = "unknown",
    ) -> None:
        """Record a single prediction-vs-actual pair."""
        actual_direction = 1 if actual_return > 0 else 0
        self.predictions.append({
            "timestamp": timestamp,
            "timeframe": timeframe,
            "predicted_direction": predicted_direction,
            "predicted_magnitude": predicted_magnitude,
            "confidence": confidence,
            "actual_return": actual_return,
            "actual_direction": actual_direction,
            "direction_correct": int(predicted_direction == actual_direction),
            "magnitude_error": abs(predicted_magnitude - abs(actual_return)),
            "regime": regime,
        })
        self._report_cache = None

    def record_batch(
        self,
        predictions_df: pd.DataFrame,
        actuals_df: pd.DataFrame,
        timeframe: str,
        regimes: Optional[pd.Series] = None,
    ) -> None:
        """Record a batch of predictions for a single timeframe.

        Args:
            predictions_df: Must have columns: direction, magnitude, confidence
            actuals_df: Must have column: return_{timeframe}
            timeframe: Label like '24h', '7d', etc.
            regimes: Optional Series of regime labels aligned to the index.
        """
        return_col = f"return_{timeframe}"
        if return_col not in actuals_df.columns:
            logger.warning("Column '%s' not found in actuals", return_col)
            return

        common_idx = predictions_df.index.intersection(actuals_df.index)
        if common_idx.empty:
            return

        for ts in common_idx:
            pred = predictions_df.loc[ts]
            actual = float(actuals_df.loc[ts, return_col])

            if pd.isna(actual):
                continue

            regime = "unknown"
            if regimes is not None and ts in regimes.index:
                regime = str(regimes.loc[ts])

            self.record(
                timestamp=ts,
                timeframe=timeframe,
                predicted_direction=int(pred.get("direction", 0)),
                predicted_magnitude=float(pred.get("magnitude", 0)),
                confidence=float(pred.get("confidence", 50)),
                actual_return=actual,
                regime=regime,
            )

    def compute_metrics(self, timeframe: Optional[str] = None) -> dict:
        """Compute all metrics, optionally filtered to a specific timeframe."""
        df = pd.DataFrame(self.predictions)
        if df.empty:
            return {"error": "no_predictions_recorded"}

        if timeframe:
            df = df[df["timeframe"] == timeframe]
            if df.empty:
                return {"error": f"no_predictions_for_{timeframe}"}

        return {
            "directional_accuracy": self._directional_accuracy(df),
            "magnitude_metrics": self._magnitude_metrics(df),
            "confidence_calibration": self._confidence_calibration(df),
            "regime_performance": self._regime_performance(df),
            "sharpe_metric": self._sharpe_metric(df),
            "total_predictions": len(df),
        }

    def generate_report(self, timeframe: Optional[str] = None) -> str:
        """Generate a human-readable performance report."""
        metrics = self.compute_metrics(timeframe)
        if "error" in metrics:
            return f"Cannot generate report: {metrics['error']}"

        lines = [
            "=" * 70,
            f"  MODEL PERFORMANCE REPORT{f' ({timeframe})' if timeframe else ''}",
            "=" * 70,
            "",
        ]

        # Directional accuracy
        dir_acc = metrics["directional_accuracy"]
        lines.append("DIRECTIONAL ACCURACY:")
        lines.append(f"  Overall: {dir_acc['overall']:.1%} ({metrics['total_predictions']} predictions)")
        if "by_timeframe" in dir_acc:
            for tf, acc in dir_acc["by_timeframe"].items():
                lines.append(f"  {tf:>4s}: {acc:.1%}")
        lines.append("")

        # Magnitude
        mag = metrics["magnitude_metrics"]
        lines.append("MAGNITUDE METRICS:")
        lines.append(f"  Mean Absolute Error: {mag['mae']:.2f}%")
        lines.append(f"  Median Absolute Error: {mag['median_ae']:.2f}%")
        lines.append(f"  RMSE: {mag['rmse']:.2f}%")
        lines.append("")

        # Confidence calibration
        cal = metrics["confidence_calibration"]
        lines.append("CONFIDENCE CALIBRATION:")
        if "bins" in cal:
            for b in cal["bins"]:
                lines.append(
                    f"  Conf {b['range']:>10s}: actual {b['accuracy']:.1%} "
                    f"(n={b['count']}, error={b['calibration_error']:.1f}pp)"
                )
        lines.append(f"  Expected Calibration Error: {cal.get('ece', 0):.2f}pp")
        lines.append("")

        # Regime performance
        regime = metrics["regime_performance"]
        lines.append("PERFORMANCE BY REGIME:")
        for r, data in regime.items():
            lines.append(
                f"  {r:>10s}: accuracy={data['accuracy']:.1%}, "
                f"n={data['count']}, avg_return={data['avg_actual_return']:+.2f}%"
            )
        lines.append("")

        # Sharpe
        sharpe = metrics["sharpe_metric"]
        lines.append("RISK-ADJUSTED METRICS:")
        lines.append(f"  Strategy Return (annualized): {sharpe.get('annual_return', 0):.1%}")
        lines.append(f"  Sharpe Ratio: {sharpe.get('sharpe', 0):.2f}")
        lines.append(f"  Max Drawdown: {sharpe.get('max_drawdown', 0):.1%}")
        lines.append("")
        lines.append("=" * 70)

        return "\n".join(lines)

    def _directional_accuracy(self, df: pd.DataFrame) -> dict:
        """Compute directional accuracy overall and by timeframe."""
        overall = float(df["direction_correct"].mean())

        by_tf: dict[str, float] = {}
        for tf in df["timeframe"].unique():
            tf_data = df[df["timeframe"] == tf]
            by_tf[tf] = float(tf_data["direction_correct"].mean())

        return {"overall": overall, "by_timeframe": by_tf}

    def _magnitude_metrics(self, df: pd.DataFrame) -> dict:
        """Compute magnitude prediction accuracy."""
        errors = df["magnitude_error"]
        return {
            "mae": float(errors.mean()),
            "median_ae": float(errors.median()),
            "rmse": float(np.sqrt((errors ** 2).mean())),
            "std": float(errors.std()),
        }

    def _confidence_calibration(self, df: pd.DataFrame, n_bins: int = 5) -> dict:
        """Check if confidence scores match actual accuracy."""
        bins = np.linspace(0, 100, n_bins + 1)
        calibration_bins = []
        weighted_errors = []

        for i in range(n_bins):
            mask = (df["confidence"] >= bins[i]) & (df["confidence"] < bins[i + 1])
            bin_data = df[mask]
            if len(bin_data) == 0:
                continue

            avg_conf = float(bin_data["confidence"].mean())
            accuracy = float(bin_data["direction_correct"].mean())
            cal_error = abs(avg_conf / 100 - accuracy)

            calibration_bins.append({
                "range": f"{bins[i]:.0f}-{bins[i+1]:.0f}%",
                "avg_confidence": avg_conf,
                "accuracy": accuracy,
                "count": len(bin_data),
                "calibration_error": cal_error * 100,
            })
            weighted_errors.append(cal_error * len(bin_data))

        ece = sum(weighted_errors) / len(df) * 100 if len(df) > 0 else 0

        return {"bins": calibration_bins, "ece": ece}

    def _regime_performance(self, df: pd.DataFrame) -> dict:
        """Break down performance by market regime."""
        results = {}
        for regime in df["regime"].unique():
            regime_data = df[df["regime"] == regime]
            results[regime] = {
                "accuracy": float(regime_data["direction_correct"].mean()),
                "count": len(regime_data),
                "avg_actual_return": float(regime_data["actual_return"].mean()),
                "avg_confidence": float(regime_data["confidence"].mean()),
            }
        return results

    def _sharpe_metric(self, df: pd.DataFrame) -> dict:
        """Compute a Sharpe-like metric assuming we follow the model's predictions.

        Strategy: go long when model predicts UP, short when DOWN.
        Return = actual_return * sign(predicted_direction - 0.5)
        """
        if len(df) < 10:
            return {"sharpe": 0, "annual_return": 0, "max_drawdown": 0}

        strategy_returns = df["actual_return"] * np.where(
            df["predicted_direction"] == 1, 1, -1
        )

        # Annualize (assume hourly data → ~8760 observations/year)
        periods_per_year = 8760 / max(1, len(df["timeframe"].unique()))
        n_observations = len(strategy_returns)
        scale = periods_per_year / n_observations if n_observations > 0 else 1

        annual_return = float(strategy_returns.sum() * scale)
        annual_vol = float(strategy_returns.std() * np.sqrt(periods_per_year))
        sharpe = annual_return / annual_vol if annual_vol > 0 else 0

        cumulative = (1 + strategy_returns / 100).cumprod()
        peak = cumulative.expanding().max()
        drawdown = (cumulative - peak) / peak
        max_dd = float(drawdown.min())

        return {
            "sharpe": round(sharpe, 3),
            "annual_return": annual_return / 100,
            "max_drawdown": max_dd,
            "total_strategy_return": float(strategy_returns.sum()),
        }
