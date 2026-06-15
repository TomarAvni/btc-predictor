"""80/20 Chronological Train-Test Validation Pipeline.

Loads all available historical data, splits at 80% chronologically with a
90-day gap buffer, trains all models on the training set, evaluates on the
holdout, runs the trading agent through the test period, and produces a
comprehensive validation report.

Usage:
    python validate.py --split 0.8 --output data/validation/
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from src import DATA_DIR
from src.models.baseline_model import BaselineModel
from src.models.calibration import (
    ProbabilityCalibrator,
    directional_confidence,
    expected_calibration_error,
)
from src.models.confidence import ConfidenceCalibrator
from src.models.xgboost_model import XGBoostPredictor
from src.simulation.data_loader import HistoricalDataLoader
from src.simulation.labeler import ForwardReturnLabeler
from src.training.feature_builder import TrainingFeatureBuilder
from src.training.wf_evaluation import (
    fit_calibrator_from_oof,
    run_walk_forward,
    tune_xgboost,
)
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

TIMEFRAMES = ["24h", "7d", "30d", "90d"]
HORIZON_HOURS = {"24h": 24, "7d": 168, "30d": 720, "90d": 2160}
GAP_DAYS = 90


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="BTC Predictor 80/20 Validation")
    parser.add_argument(
        "--split", type=float, default=0.8,
        help="Training fraction (default: 0.8 = 80%% train / 20%% test)",
    )
    parser.add_argument(
        "--output", type=str, default="data/validation/",
        help="Directory for validation outputs",
    )
    parser.add_argument(
        "--step-hours", type=int, default=4,
        help="Hours between test predictions (default: 4)",
    )
    parser.add_argument(
        "--no-trading", action="store_true",
        help="Skip the trading agent evaluation",
    )
    parser.add_argument(
        "--skip-tft", action="store_true",
        help="Skip optional TFT training (faster, CPU-only CI runs)",
    )
    parser.add_argument(
        "--no-walk-forward", action="store_true",
        help="Skip purged walk-forward validation + calibration fitting",
    )
    parser.add_argument(
        "--wf-splits", type=int, default=5,
        help="Number of purged walk-forward folds (default: 5)",
    )
    parser.add_argument(
        "--embargo-hours", type=int, default=0,
        help="Extra embargo gap (hours) beyond the per-horizon label purge",
    )
    parser.add_argument(
        "--wf-min-train-frac", type=float, default=0.4,
        help="Fraction of the series reserved before the first walk-forward test block",
    )
    parser.add_argument(
        "--tune", action="store_true",
        help="Run a modest time-series-aware XGBoost grid search (24h horizon) and report",
    )
    return parser.parse_args()


# ═══════════════════════════════════════════════════════════════════════════════
# Data Loading & Splitting
# ═══════════════════════════════════════════════════════════════════════════════


def load_all_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load price data and full merged dataset."""
    loader = HistoricalDataLoader()
    price_df = loader.load_price_data()
    merged_df = loader.get_merged_dataset()
    return price_df, merged_df


def chronological_split(
    price_df: pd.DataFrame,
    merged_df: pd.DataFrame,
    train_fraction: float,
    gap_days: int = GAP_DAYS,
) -> dict[str, Any]:
    """Split data chronologically into train, gap, and test sets."""
    n = len(price_df)
    train_end_idx = int(n * train_fraction)

    train_end_ts = price_df.index[train_end_idx]
    gap_end_ts = train_end_ts + pd.Timedelta(days=gap_days)

    # Find actual gap end in the index
    test_start_candidates = price_df.index[price_df.index >= gap_end_ts]
    if test_start_candidates.empty:
        raise ValueError(
            f"Not enough data for a {gap_days}-day gap buffer. "
            f"Data ends at {price_df.index[-1]} but gap requires {gap_end_ts}."
        )
    test_start_ts = test_start_candidates[0]

    data_start = price_df.index[0]
    data_end = price_df.index[-1]

    train_price = price_df.loc[:train_end_ts]
    test_price = price_df.loc[test_start_ts:]
    train_merged = merged_df.loc[:train_end_ts]
    test_merged = merged_df.loc[test_start_ts:]

    return {
        "train_price": train_price,
        "test_price": test_price,
        "train_merged": train_merged,
        "test_merged": test_merged,
        "data_start": data_start,
        "data_end": data_end,
        "train_end": train_end_ts,
        "gap_end": test_start_ts,
        "test_start": test_start_ts,
        "train_n": len(train_price),
        "test_n": len(test_price),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Model Training
# ═══════════════════════════════════════════════════════════════════════════════


def train_models(
    train_merged: pd.DataFrame,
    train_price: pd.DataFrame,
    output_dir: Path,
    skip_tft: bool = False,
) -> dict[str, Any]:
    """Train all models on the training set and return trained model objects."""
    model_dir = output_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    feature_builder = TrainingFeatureBuilder(model_dir=model_dir)
    labeler = ForwardReturnLabeler()

    logger.info("Building features for training data...")
    features_df = feature_builder.build_features(train_merged)
    labels_df = labeler.compute_labels(train_price)

    # Align indices
    common_idx = features_df.index.intersection(labels_df.index)
    features_df = features_df.loc[common_idx]
    labels_df = labels_df.loc[common_idx]

    # Drop rows where all return labels are NaN
    return_cols = [c for c in labels_df.columns if c.startswith("return_")]
    valid_mask = labels_df[return_cols].notna().any(axis=1)
    features_df = features_df[valid_mask]
    labels_df = labels_df[valid_mask]

    numeric_features = features_df.select_dtypes(include=[np.number]).fillna(0)

    logger.info(
        "Training data prepared: %d samples × %d features",
        len(numeric_features), len(numeric_features.columns),
    )

    models: dict[str, dict[str, Any]] = {"baseline": {}, "xgboost": {}}
    training_metrics: dict[str, Any] = {}

    for tf in TIMEFRAMES:
        target_col = f"return_{tf}"
        if target_col not in labels_df.columns:
            continue

        tf_valid = labels_df[target_col].notna()
        X = numeric_features[tf_valid]
        y = labels_df.loc[tf_valid, target_col]

        if len(X) < 500:
            logger.warning("Skipping %s: only %d samples", tf, len(X))
            continue

        y_direction = (y > 0).astype(int)

        # Baseline
        baseline = BaselineModel()
        baseline.train(X, y_direction, y)
        baseline.save(model_dir / f"baseline_{tf}")
        models["baseline"][tf] = baseline

        # XGBoost
        xgb_config = {"models": {"xgboost": {
            "n_estimators": 300, "max_depth": 6, "learning_rate": 0.05,
        }}}
        xgb = XGBoostPredictor(config=xgb_config, timeframe=tf)
        xgb._model_path = model_dir / f"xgb_{tf}"
        xgb.train(X, y_direction, y)
        models["xgboost"][tf] = xgb

        training_metrics[tf] = {
            "n_samples": len(X),
            "pct_up": float(y_direction.mean()),
            "mean_return": float(y.mean()),
        }

        logger.info("Trained models for %s (%d samples)", tf, len(X))

    # Fit scaler
    feature_builder.fit_scaler(numeric_features)

    # Try TFT (optional dependency)
    tft_available = False
    if skip_tft:
        logger.info("TFT training skipped (--skip-tft)")
    else:
        try:
            from src.models.tft_model import TFTPredictor
            tft = TFTPredictor(model_dir=model_dir / "tft", max_epochs=10)
            # Only train TFT if pytorch_forecasting is available
            target_col = "return_24h"
            if target_col in labels_df.columns:
                tf_valid = labels_df[target_col].notna()
                tft_features = features_df[tf_valid]
                tft_labels = labels_df[tf_valid]
                tft_result = tft.train(tft_features, tft_labels)
                if "error" not in tft_result:
                    tft_available = True
                    models["tft"] = tft
                    logger.info("TFT model trained successfully")
                else:
                    logger.info("TFT training skipped: %s", tft_result.get("error"))
        except Exception as e:
            logger.info("TFT not available: %s", e)

    # Calibrate confidence
    calibrator = ConfidenceCalibrator(model_dir=model_dir)

    return {
        "models": models,
        "feature_builder": feature_builder,
        "labeler": labeler,
        "training_metrics": training_metrics,
        "tft_available": tft_available,
        "calibrator": calibrator,
        "feature_columns": numeric_features.columns.tolist(),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Test Evaluation
# ═══════════════════════════════════════════════════════════════════════════════


def evaluate_on_test(
    test_merged: pd.DataFrame,
    test_price: pd.DataFrame,
    trained: dict[str, Any],
    step_hours: int = 4,
) -> dict[str, Any]:
    """Evaluate all models on the test set without lookahead."""
    feature_builder = trained["feature_builder"]
    labeler = trained["labeler"]
    models = trained["models"]

    logger.info("Building features for test data...")
    test_features = feature_builder.build_features(test_merged)
    test_labels = labeler.compute_labels(test_price)

    common_idx = test_features.index.intersection(test_labels.index)
    test_features = test_features.loc[common_idx]
    test_labels = test_labels.loc[common_idx]

    numeric_test = test_features.select_dtypes(include=[np.number]).fillna(0)

    # Step through test period
    test_timestamps = numeric_test.index[::step_hours]
    logger.info("Evaluating %d test timestamps...", len(test_timestamps))

    results: dict[str, list[dict]] = {
        "baseline": [], "xgboost": [], "ensemble": [],
    }

    for ts in test_timestamps:
        if ts not in numeric_test.index or ts not in test_labels.index:
            continue

        row = numeric_test.loc[ts]
        features_dict = {col: float(row[col]) for col in numeric_test.columns}

        for tf in TIMEFRAMES:
            actual_col = f"return_{tf}"
            direction_col = f"direction_{tf}"

            if actual_col not in test_labels.columns:
                continue
            actual_return = test_labels.loc[ts, actual_col]
            if pd.isna(actual_return):
                continue

            actual_direction = 1 if actual_return > 0 else 0

            # Baseline prediction
            if tf in models.get("baseline", {}):
                pred = models["baseline"][tf].predict(features_dict)
                pred_dir = 1 if pred["direction_prob"] > 0.5 else 0
                results["baseline"].append({
                    "timestamp": ts,
                    "timeframe": tf,
                    "pred_direction": pred_dir,
                    "pred_magnitude": pred["predicted_magnitude"],
                    "direction_prob": pred["direction_prob"],
                    "confidence": pred.get("confidence", 0.5),
                    "actual_return": float(actual_return),
                    "actual_direction": actual_direction,
                    "correct": pred_dir == actual_direction,
                })

            # XGBoost prediction
            if tf in models.get("xgboost", {}):
                pred = models["xgboost"][tf].predict(features_dict)
                pred_dir = 1 if pred["direction_prob"] > 0.5 else 0
                results["xgboost"].append({
                    "timestamp": ts,
                    "timeframe": tf,
                    "pred_direction": pred_dir,
                    "pred_magnitude": pred["predicted_magnitude"],
                    "direction_prob": pred["direction_prob"],
                    "confidence": pred.get("raw_confidence", 0.5),
                    "actual_return": float(actual_return),
                    "actual_direction": actual_direction,
                    "correct": pred_dir == actual_direction,
                })

            # Ensemble (weighted average of baseline + XGBoost)
            baseline_pred = models["baseline"].get(tf)
            xgb_pred_model = models["xgboost"].get(tf)
            if baseline_pred and xgb_pred_model:
                bp = baseline_pred.predict(features_dict)
                xp = xgb_pred_model.predict(features_dict)
                ens_prob = 0.35 * bp["direction_prob"] + 0.65 * xp["direction_prob"]
                ens_mag = 0.35 * abs(bp["predicted_magnitude"]) + 0.65 * abs(xp["predicted_magnitude"])
                ens_dir = 1 if ens_prob > 0.5 else 0
                results["ensemble"].append({
                    "timestamp": ts,
                    "timeframe": tf,
                    "pred_direction": ens_dir,
                    "pred_magnitude": ens_mag,
                    "direction_prob": ens_prob,
                    "confidence": abs(ens_prob - 0.5) * 2,
                    "actual_return": float(actual_return),
                    "actual_direction": actual_direction,
                    "correct": ens_dir == actual_direction,
                })

    return results


def compute_metrics(results: dict[str, list[dict]]) -> dict[str, Any]:
    """Compute accuracy, MAE, and calibration metrics from evaluation results."""
    metrics: dict[str, Any] = {}

    for model_name, preds in results.items():
        if not preds:
            continue

        df = pd.DataFrame(preds)
        model_metrics: dict[str, Any] = {}

        for tf in TIMEFRAMES:
            tf_df = df[df["timeframe"] == tf]
            if tf_df.empty:
                continue

            direction_acc = float(tf_df["correct"].mean())
            mae = float((tf_df["pred_magnitude"] - tf_df["actual_return"]).abs().mean())
            mae_pct = float(tf_df["actual_return"].abs().mean())

            model_metrics[tf] = {
                "direction_accuracy": round(direction_acc * 100, 1),
                "mae": round(mae, 2),
                "mae_pct": round(mae_pct, 2),
                "n_predictions": len(tf_df),
            }

        metrics[model_name] = model_metrics

    return metrics


def compute_calibration(results: dict[str, list[dict]]) -> dict[str, Any]:
    """Compute confidence calibration data (single-split holdout).

    Confidence here is the *directional* confidence ``max(P, 1-P)`` -- i.e. the
    probability the model assigns to the side it actually predicted. This is the
    quantity that should match realized accuracy, and is the correct semantics
    for the downstream Kelly sizing. (The old code binned ``|P-0.5|*2``, a margin
    that almost never exceeds 0.5, so every bin below 50% was empty.)
    """
    calibration: dict[str, Any] = {}

    model_key = "ensemble" if results.get("ensemble") else "xgboost"
    preds = results.get(model_key, [])
    if not preds:
        return calibration

    df = pd.DataFrame(preds)
    if "direction_prob" not in df.columns:
        return calibration

    p = df["direction_prob"].astype(float).to_numpy()
    df["confidence_pct"] = np.maximum(p, 1.0 - p) * 100

    bins = [50, 60, 70, 80, 90, 100]
    for i in range(len(bins) - 1):
        low, high = bins[i], bins[i + 1]
        upper = high + 0.001 if high == 100 else high
        mask = (df["confidence_pct"] >= low) & (df["confidence_pct"] < upper)
        bin_df = df[mask]
        if len(bin_df) >= 10:
            actual_acc = float(bin_df["correct"].mean() * 100)
            calibration[f"{low}%"] = {
                "stated_confidence": (low + high) / 2,
                "actual_accuracy": round(actual_acc, 1),
                "n_samples": len(bin_df),
                "well_calibrated": abs(actual_acc - (low + high) / 2) < 10,
            }

    return calibration


def compute_regime_accuracy(
    results: dict[str, list[dict]],
    test_price: pd.DataFrame,
) -> dict[str, float]:
    """Compute accuracy broken down by market regime."""
    model_key = "ensemble" if results.get("ensemble") else "xgboost"
    preds = results.get(model_key, [])
    if not preds:
        return {}

    df = pd.DataFrame(preds)

    close = test_price["close"]
    ema_200 = close.ewm(span=200 * 24, adjust=False).mean()

    def classify_regime(ts: pd.Timestamp) -> str:
        if ts not in ema_200.index:
            return "unknown"
        price = close.loc[:ts].iloc[-1]
        ema = ema_200.loc[:ts].iloc[-1]
        ratio = (price - ema) / ema if ema > 0 else 0
        if ratio > 0.05:
            return "bull"
        elif ratio < -0.05:
            return "bear"
        return "sideways"

    df["regime"] = df["timestamp"].apply(classify_regime)

    regime_acc: dict[str, float] = {}
    for regime in ["bull", "bear", "sideways"]:
        regime_df = df[df["regime"] == regime]
        if len(regime_df) >= 10:
            regime_acc[regime] = round(float(regime_df["correct"].mean() * 100), 1)

    return regime_acc


def get_feature_importance(models: dict[str, dict]) -> list[tuple[str, float]]:
    """Get top feature importances from XGBoost models."""
    importances: dict[str, float] = {}

    for tf, model in models.get("xgboost", {}).items():
        tf_imp = model.get_feature_importance(top_n=None)
        for feat, imp in tf_imp.items():
            importances[feat] = importances.get(feat, 0) + imp

    if not importances:
        return []

    total = sum(importances.values())
    if total > 0:
        importances = {k: v / total * 100 for k, v in importances.items()}

    sorted_imp = sorted(importances.items(), key=lambda x: x[1], reverse=True)
    return sorted_imp[:10]


# ═══════════════════════════════════════════════════════════════════════════════
# Purged Walk-Forward Validation + Calibration (out-of-sample, honest)
# ═══════════════════════════════════════════════════════════════════════════════


def run_walk_forward_section(
    price_df: pd.DataFrame,
    merged_df: pd.DataFrame,
    output_dir: Path,
    n_splits: int,
    embargo_hours: int,
    min_train_frac: float,
    tune: bool,
) -> dict[str, Any]:
    """Run purged walk-forward CV across the full series, fit + persist the
    isotonic calibrator on out-of-fold predictions, and (optionally) tune.

    This is the methodologically-correct estimate of live performance: every
    prediction is strictly out-of-sample relative to its training window, with
    a per-horizon purge (+ optional embargo) removing label overlap at the seam.
    """
    model_dir = output_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    feature_builder = TrainingFeatureBuilder(model_dir=model_dir)
    labeler = ForwardReturnLabeler()

    features_df = feature_builder.build_features(merged_df)
    labels_df = labeler.compute_labels(price_df)

    common_idx = features_df.index.intersection(labels_df.index)
    features_df = features_df.loc[common_idx].sort_index()
    labels_df = labels_df.loc[common_idx].sort_index()

    numeric = features_df.select_dtypes(include=[np.number]).fillna(0)

    logger.info(
        "Walk-forward: %d samples × %d features, %d folds, embargo=%dh",
        len(numeric), numeric.shape[1], n_splits, embargo_hours,
    )

    wf_results = run_walk_forward(
        features=numeric,
        labels=labels_df,
        timeframes=TIMEFRAMES,
        horizon_hours=HORIZON_HOURS,
        n_splits=n_splits,
        embargo_hours=embargo_hours,
        min_train_frac=min_train_frac,
    )

    calibrator, calib_report = fit_calibrator_from_oof(wf_results, model_dir)

    summary: dict[str, Any] = {}
    for tf, res in wf_results.items():
        summary[tf] = {
            "accuracy": round(res["accuracy"] * 100, 1),
            "baseline_accuracy": (
                round(res["baseline_accuracy"] * 100, 1)
                if res["baseline_accuracy"] is not None else None
            ),
            "auc": round(res["auc"], 4) if res["auc"] is not None else None,
            "up_rate": round(res["up_rate"] * 100, 1),
            "n": res["n"],
            "n_folds": res["n_folds"],
            "fold_accuracies": [round(a * 100, 1) for a in res["fold_accuracies"]],
            "confusion": res["confusion"],
        }

    tune_result = None
    if tune:
        logger.info("Running XGBoost hyperparameter search (24h horizon)...")
        tune_result = tune_xgboost(
            features=numeric,
            labels=labels_df,
            timeframe="24h",
            horizon_hours=HORIZON_HOURS["24h"],
            n_splits=max(3, n_splits - 1),
            embargo_hours=embargo_hours,
            min_train_frac=min_train_frac,
        )

    return {
        "summary": summary,
        "calibration": calib_report,
        "tuning": tune_result,
        "calibrated_horizons": calibrator.fitted_horizons,
    }


def generate_wf_report(wf_section: dict[str, Any]) -> list[str]:
    """Render the walk-forward + calibration section of the text report."""
    lines: list[str] = []
    summary = wf_section.get("summary", {})
    calib = wf_section.get("calibration", {})

    lines.append("PURGED WALK-FORWARD VALIDATION (out-of-sample, XGBoost):")
    if not summary:
        lines.append("  No walk-forward folds produced (insufficient data).")
        lines.append("")
        return lines

    lines.append(
        f"  {'Horizon':<10s}{'WF Acc':>9s}{'Baseline':>10s}{'AUC':>8s}"
        f"{'Up-rate':>9s}{'Folds':>7s}{'N':>9s}"
    )
    for tf in TIMEFRAMES:
        if tf not in summary:
            continue
        s = summary[tf]
        auc = f"{s['auc']:.3f}" if s["auc"] is not None else "N/A"
        base = f"{s['baseline_accuracy']:.1f}%" if s["baseline_accuracy"] is not None else "N/A"
        lines.append(
            f"  {tf:<10s}{s['accuracy']:>8.1f}%{base:>10s}{auc:>8s}"
            f"{s['up_rate']:>8.1f}%{s['n_folds']:>7d}{s['n']:>9d}"
        )
    lines.append("")
    lines.append("  Per-fold WF accuracy (chronological):")
    for tf in TIMEFRAMES:
        if tf in summary and summary[tf]["fold_accuracies"]:
            accs = ", ".join(f"{a:.1f}%" for a in summary[tf]["fold_accuracies"])
            lines.append(f"    {tf:<6s} {accs}")
    lines.append("")
    lines.append("  NOTE: longer-horizon accuracy is inflated by overlapping labels")
    lines.append("  and BTC's upward drift (high up-rate); treat 30d/90d AUC, not raw")
    lines.append("  accuracy, as the edge signal, and watch the LAST fold for decay.")
    lines.append("")

    lines.append("PROBABILITY CALIBRATION (isotonic, fit on OOF predictions):")
    if calib:
        lines.append(f"  {'Horizon':<10s}{'ECE before':>12s}{'ECE after':>12s}{'AUC':>8s}")
        for tf in TIMEFRAMES:
            if tf not in calib:
                continue
            c = calib[tf]
            auc = f"{c['auc']:.3f}" if c["auc"] is not None else "N/A"
            lines.append(
                f"  {tf:<10s}{c['ece_before_pp']:>10.2f}pp{c['ece_after_pp']:>10.2f}pp{auc:>8s}"
            )
        lines.append("")
        lines.append("  (ECE = Expected Calibration Error in percentage points; lower is better)")
        lines.append("  ECE-after is measured in-sample to the isotonic fit, so it is a")
        lines.append("  slightly optimistic floor; the before/after gap is the honest signal")
        lines.append("  that the raw model probabilities were over-confident.")
    else:
        lines.append("  Calibration not fit (insufficient out-of-fold data).")
    lines.append("")

    tuning = wf_section.get("tuning")
    if tuning and "best" in tuning:
        best = tuning["best"]
        lines.append("HYPERPARAMETER SEARCH (24h, purged walk-forward):")
        lines.append(
            f"  Best: {best['params']} -> logloss={best['mean_logloss']}, "
            f"acc={best['mean_accuracy']}%"
        )
        lines.append("")

    return lines


# ═══════════════════════════════════════════════════════════════════════════════
# Trading Agent Evaluation
# ═══════════════════════════════════════════════════════════════════════════════


def run_trading_backtest(
    test_merged: pd.DataFrame,
    test_price: pd.DataFrame,
    trained: dict[str, Any],
    step_hours: int = 4,
) -> dict[str, Any]:
    """Run the trading agent through the test period."""
    from src.trading.agent import TradingAgent
    from src.trading.portfolio import Portfolio

    portfolio = Portfolio(load_existing=False)
    agent = TradingAgent(portfolio=portfolio)
    agent.reset()

    models = trained["models"]
    feature_builder = trained["feature_builder"]

    test_features = feature_builder.build_features(test_merged)
    numeric_test = test_features.select_dtypes(include=[np.number]).fillna(0)

    test_timestamps = test_price.index[::step_hours]
    equity_curve: list[dict] = []

    logger.info("Running trading agent on %d test timestamps...", len(test_timestamps))

    for ts in test_timestamps:
        if ts not in test_price.index:
            continue

        current_price = float(test_price.loc[ts, "close"])
        high = float(test_price.loc[ts, "high"])
        low = float(test_price.loc[ts, "low"])

        # Price update for SL/TP checks
        agent.on_price_update(current_price, high=high, low=low, timestamp=ts.to_pydatetime())

        # Generate predictions from models
        if ts in numeric_test.index:
            row = numeric_test.loc[ts]
            features_dict = {col: float(row[col]) for col in numeric_test.columns}

            predictions = []
            for tf in TIMEFRAMES:
                xgb_model = models.get("xgboost", {}).get(tf)
                baseline_model = models.get("baseline", {}).get(tf)

                if xgb_model:
                    pred = xgb_model.predict(features_dict)
                elif baseline_model:
                    pred = baseline_model.predict(features_dict)
                else:
                    continue

                direction = "UP" if pred["direction_prob"] > 0.5 else "DOWN"
                magnitude = abs(pred.get("predicted_magnitude", 0))
                confidence = pred.get("raw_confidence", pred.get("confidence", 0.5))
                if confidence <= 1.0:
                    confidence *= 100

                predictions.append({
                    "timeframe": tf,
                    "direction": direction,
                    "magnitude": magnitude,
                    "confidence": confidence,
                })

            if predictions:
                agent.on_new_prediction(
                    predictions, current_price, timestamp=ts.to_pydatetime()
                )

        equity_curve.append({
            "timestamp": ts.isoformat(),
            "portfolio_value": portfolio.total_value_usd,
            "btc_price": current_price,
        })

    # Compute trading metrics
    perf_metrics = agent.get_performance_summary()

    # Buy & hold comparison
    if len(test_price) >= 2:
        start_price = float(test_price.iloc[0]["close"])
        end_price = float(test_price.iloc[-1]["close"])
        buy_hold_return = (end_price - start_price) / start_price * 100
    else:
        buy_hold_return = 0.0

    perf_metrics["buy_and_hold_return_pct"] = round(buy_hold_return, 2)
    perf_metrics["equity_curve"] = equity_curve

    return perf_metrics


# ═══════════════════════════════════════════════════════════════════════════════
# Report Generation
# ═══════════════════════════════════════════════════════════════════════════════


def generate_report(
    split_info: dict,
    metrics: dict[str, Any],
    calibration: dict[str, Any],
    regime_accuracy: dict[str, float],
    feature_importance: list[tuple[str, float]],
    trading_metrics: Optional[dict[str, Any]],
    wf_section: Optional[dict[str, Any]] = None,
) -> str:
    """Generate the human-readable validation report."""
    lines = [
        "=" * 80,
        "BTC PREDICTOR -- VALIDATION REPORT",
        "=" * 80,
        "",
        "Data Split:",
        f"  Training: {split_info['data_start'].strftime('%Y-%m-%d')} to "
        f"{split_info['train_end'].strftime('%Y-%m-%d')} "
        f"({split_info['train_n']:,} hourly candles)",
        f"  Buffer:   {split_info['train_end'].strftime('%Y-%m-%d')} to "
        f"{split_info['gap_end'].strftime('%Y-%m-%d')} "
        f"({GAP_DAYS} days, excluded)",
        f"  Test:     {split_info['test_start'].strftime('%Y-%m-%d')} to "
        f"{split_info['data_end'].strftime('%Y-%m-%d')} "
        f"({split_info['test_n']:,} hourly candles)",
        "",
    ]

    # Model accuracy table
    lines.append("MODEL ACCURACY (Test Set):")
    model_names = [m for m in ["baseline", "xgboost", "ensemble"] if m in metrics]

    # Header
    header = "  {:>16s}".format("")
    for mn in model_names:
        header += f"  {mn:>10s}"
    lines.append(header)

    # Direction accuracy
    for tf in TIMEFRAMES:
        row = f"  {tf + ' Direction':<16s}"
        for mn in model_names:
            val = metrics.get(mn, {}).get(tf, {}).get("direction_accuracy", "N/A")
            if isinstance(val, (int, float)):
                row += f"  {val:>9.1f}%"
            else:
                row += f"  {'N/A':>10s}"
        lines.append(row)

    lines.append("")

    # MAE
    for tf in TIMEFRAMES:
        row = f"  {tf + ' MAE':<16s}"
        for mn in model_names:
            val = metrics.get(mn, {}).get(tf, {}).get("mae", "N/A")
            if isinstance(val, (int, float)):
                row += f"  {val:>9.2f}%"
            else:
                row += f"  {'N/A':>10s}"
        lines.append(row)

    lines.append("")

    # Purged walk-forward + calibration (the honest, out-of-sample numbers)
    if wf_section:
        lines.extend(generate_wf_report(wf_section))

    # Confidence calibration (single 80/20 holdout)
    lines.append("CONFIDENCE CALIBRATION (single 80/20 holdout, directional):")
    if calibration:
        for conf_level, data in calibration.items():
            status = "well calibrated" if data["well_calibrated"] else "needs adjustment"
            lines.append(
                f"  When model said {conf_level}: actually right "
                f"{data['actual_accuracy']:.1f}% ({status}, n={data['n_samples']})"
            )
    else:
        lines.append("  Insufficient data for calibration analysis")
    lines.append("")

    # Trading results
    if trading_metrics:
        lines.append("TRADING AGENT RESULTS (Test Period):")
        lines.append(f"  Starting Balance:  ${trading_metrics.get('starting_balance', 2000):,.2f}")
        lines.append(f"  Ending Balance:    ${trading_metrics.get('current_value', 0):,.2f}")
        lines.append(f"  Total Return:      {trading_metrics.get('total_return_pct', 0):+.1f}%")
        lines.append(f"  BTC Buy & Hold:    {trading_metrics.get('buy_and_hold_return_pct', 0):+.1f}% (comparison benchmark)")
        lines.append(f"  Total Trades:      {trading_metrics.get('total_trades', 0)}")
        lines.append(f"  Win Rate:          {trading_metrics.get('win_rate_pct', 0):.1f}%")
        lines.append(f"  Max Drawdown:      {trading_metrics.get('max_drawdown_pct', 0):.1f}%")
        lines.append(f"  Sharpe Ratio:      {trading_metrics.get('sharpe_ratio', 0):.2f}")
        lines.append("")

    # Market regime breakdown
    lines.append("MARKET REGIME BREAKDOWN:")
    if regime_accuracy:
        for regime, acc in regime_accuracy.items():
            lines.append(f"  {regime.capitalize():<10s} accuracy: {acc:.1f}%")
    else:
        lines.append("  Insufficient data for regime analysis")
    lines.append("")

    # Feature importance
    lines.append("FEATURE IMPORTANCE (Top 10):")
    if feature_importance:
        for i, (feat, imp) in enumerate(feature_importance, 1):
            lines.append(f"  {i:>2d}. {feat:<30s} {imp:.1f}%")
    else:
        lines.append("  No feature importance available")
    lines.append("")

    # Conclusion
    lines.append("CONCLUSION:")
    conclusions = []

    # Does ensemble beat baseline?
    for tf in TIMEFRAMES:
        ens_acc = metrics.get("ensemble", {}).get(tf, {}).get("direction_accuracy")
        base_acc = metrics.get("baseline", {}).get(tf, {}).get("direction_accuracy")
        if ens_acc and base_acc and ens_acc > base_acc:
            conclusions.append(
                f"  Ensemble beats baseline on {tf}: {ens_acc:.1f}% vs {base_acc:.1f}%"
            )
            break
    else:
        conclusions.append("  Ensemble does not consistently beat baseline -- "
                          "more data or feature engineering needed")

    # Trading agent vs buy-and-hold
    if trading_metrics:
        agent_ret = trading_metrics.get("total_return_pct", 0)
        bh_ret = trading_metrics.get("buy_and_hold_return_pct", 0)
        if agent_ret > bh_ret:
            conclusions.append(
                f"  Trading agent outperforms buy-and-hold: "
                f"{agent_ret:+.1f}% vs {bh_ret:+.1f}%"
            )
        else:
            conclusions.append(
                f"  Trading agent underperforms buy-and-hold: "
                f"{agent_ret:+.1f}% vs {bh_ret:+.1f}%"
            )

    # Calibration assessment
    if calibration:
        n_well = sum(1 for d in calibration.values() if d["well_calibrated"])
        n_total = len(calibration)
        if n_well == n_total:
            conclusions.append("  Confidence calibration is well-calibrated across all bins")
        else:
            conclusions.append(
                f"  Confidence needs adjustment: {n_well}/{n_total} bins well-calibrated"
            )

    for c in conclusions:
        lines.append(c)
    lines.append("")
    lines.append("=" * 80)

    return "\n".join(lines)


def _json_default(obj: Any) -> Any:
    """Convert numpy scalars/arrays to native Python types for json.dumps."""
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.bool_):
        return bool(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def save_results_json(
    output_dir: Path,
    split_info: dict,
    metrics: dict[str, Any],
    calibration: dict[str, Any],
    regime_accuracy: dict[str, float],
    feature_importance: list[tuple[str, float]],
    trading_metrics: Optional[dict[str, Any]],
    wf_section: Optional[dict[str, Any]] = None,
) -> None:
    """Save machine-readable results to JSON."""
    results = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "split": {
            "data_start": split_info["data_start"].isoformat(),
            "data_end": split_info["data_end"].isoformat(),
            "train_end": split_info["train_end"].isoformat(),
            "test_start": split_info["test_start"].isoformat(),
            "gap_days": GAP_DAYS,
            "train_samples": split_info["train_n"],
            "test_samples": split_info["test_n"],
        },
        "model_accuracy": metrics,
        "confidence_calibration": calibration,
        "regime_accuracy": regime_accuracy,
        "feature_importance": [
            {"feature": feat, "importance_pct": imp}
            for feat, imp in feature_importance
        ],
    }

    if wf_section:
        results["walk_forward"] = wf_section.get("summary", {})
        results["walk_forward_calibration"] = wf_section.get("calibration", {})
        if wf_section.get("tuning"):
            results["hyperparameter_search"] = wf_section["tuning"]

    if trading_metrics:
        # Remove equity curve from JSON summary (it can be large)
        trading_summary = {
            k: v for k, v in trading_metrics.items() if k != "equity_curve"
        }
        results["trading"] = trading_summary

        # Save equity curve separately
        equity = trading_metrics.get("equity_curve", [])
        if equity:
            equity_path = output_dir / "equity_curve.json"
            equity_path.write_text(
                json.dumps(equity, indent=2, default=_json_default), encoding="utf-8"
            )

    results_path = output_dir / "results.json"
    results_path.write_text(
        json.dumps(results, indent=2, default=_json_default), encoding="utf-8"
    )
    logger.info("Results saved to %s", results_path)


# ═══════════════════════════════════════════════════════════════════════════════
# Main Pipeline
# ═══════════════════════════════════════════════════════════════════════════════


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.time()

    print("""
╔══════════════════════════════════════════════════════════════════════════╗
║          BTC PREDICTOR -- 80/20 VALIDATION PIPELINE                    ║
╠══════════════════════════════════════════════════════════════════════════╣
║  1. Load all historical data                                           ║
║  2. Split 80/20 chronologically (with 90-day gap buffer)               ║
║  3. Train models on training set                                       ║
║  4. Evaluate on holdout test set                                       ║
║  5. Run trading agent on test period                                   ║
║  6. Generate comprehensive validation report                           ║
╚══════════════════════════════════════════════════════════════════════════╝
    """)

    # Step 1: Load data
    print(f"\n{'─'*70}")
    print("  Step 1/6: Loading historical data...")
    print(f"{'─'*70}")
    price_df, merged_df = load_all_data()
    print(f"  Loaded {len(price_df):,} hourly candles")
    print(f"  Date range: {price_df.index[0].strftime('%Y-%m-%d')} to "
          f"{price_df.index[-1].strftime('%Y-%m-%d')}")
    print(f"  Columns: {len(merged_df.columns)} (OHLCV + signals)")

    # Step 2: Split
    print(f"\n{'─'*70}")
    print(f"  Step 2/6: Splitting data ({args.split:.0%} train / "
          f"{1-args.split:.0%} test)...")
    print(f"{'─'*70}")
    split_info = chronological_split(price_df, merged_df, args.split)
    print(f"  Training: {split_info['data_start'].strftime('%Y-%m-%d')} to "
          f"{split_info['train_end'].strftime('%Y-%m-%d')} "
          f"({split_info['train_n']:,} candles)")
    print(f"  Buffer:   {GAP_DAYS} days excluded (prevents 90d label leakage)")
    print(f"  Test:     {split_info['test_start'].strftime('%Y-%m-%d')} to "
          f"{split_info['data_end'].strftime('%Y-%m-%d')} "
          f"({split_info['test_n']:,} candles)")

    # Step 3: Train
    print(f"\n{'─'*70}")
    print("  Step 3/6: Training models on training set...")
    print(f"{'─'*70}")
    trained = train_models(
        split_info["train_merged"],
        split_info["train_price"],
        output_dir,
        skip_tft=args.skip_tft,
    )
    if not trained["models"]["baseline"] and not trained["models"]["xgboost"]:
        raise RuntimeError(
            "No models were trained. Check that data/price/ has enough hourly candles."
        )
    print(f"  Models trained: baseline, XGBoost"
          f"{', TFT' if trained['tft_available'] else ''}")
    print(f"  Training metrics: {json.dumps(trained['training_metrics'], indent=4)}")

    # Step 4: Evaluate
    print(f"\n{'─'*70}")
    print("  Step 4/6: Evaluating on test set...")
    print(f"{'─'*70}")
    results = evaluate_on_test(
        split_info["test_merged"],
        split_info["test_price"],
        trained,
        step_hours=args.step_hours,
    )
    metrics = compute_metrics(results)
    calibration = compute_calibration(results)
    regime_accuracy = compute_regime_accuracy(results, split_info["test_price"])
    feature_importance = get_feature_importance(trained["models"])

    # Step 4b: Purged walk-forward validation + calibration (honest OOS numbers)
    wf_section = None
    if not args.no_walk_forward:
        print(f"\n{'─'*70}")
        print("  Step 4b: Purged walk-forward validation + calibration...")
        print(f"{'─'*70}")
        try:
            wf_section = run_walk_forward_section(
                price_df, merged_df, output_dir,
                n_splits=args.wf_splits,
                embargo_hours=args.embargo_hours,
                min_train_frac=args.wf_min_train_frac,
                tune=args.tune,
            )
            for tf in TIMEFRAMES:
                s = wf_section["summary"].get(tf)
                if s:
                    c = wf_section["calibration"].get(tf, {})
                    print(
                        f"    {tf}: WF acc={s['accuracy']:.1f}% "
                        f"(baseline={s['baseline_accuracy']}), "
                        f"AUC={s['auc']}, "
                        f"ECE {c.get('ece_before_pp','?')}->{c.get('ece_after_pp','?')}pp"
                    )
            print(f"    Calibrator fit for: {wf_section['calibrated_horizons']}")
        except Exception as e:
            logger.exception("Walk-forward validation failed: %s", e)
            print(f"  [WARN] Walk-forward validation failed (non-fatal): {e}")
    else:
        print(f"\n{'─'*70}")
        print("  Step 4b: Walk-forward validation skipped (--no-walk-forward)")
        print(f"{'─'*70}")

    for model_name, model_metrics in metrics.items():
        print(f"\n  {model_name.upper()}:")
        for tf, tf_metrics in model_metrics.items():
            print(f"    {tf}: accuracy={tf_metrics['direction_accuracy']:.1f}%, "
                  f"MAE={tf_metrics['mae']:.2f}%, "
                  f"n={tf_metrics['n_predictions']}")

    # Step 5: Trading backtest
    trading_metrics = None
    if not args.no_trading:
        print(f"\n{'─'*70}")
        print("  Step 5/6: Running trading agent on test period...")
        print(f"{'─'*70}")
        try:
            trading_metrics = run_trading_backtest(
                split_info["test_merged"],
                split_info["test_price"],
                trained,
                step_hours=args.step_hours,
            )
            print(f"  Starting: ${trading_metrics.get('starting_balance', 2000):,.2f}")
            print(f"  Ending:   ${trading_metrics.get('current_value', 0):,.2f}")
            print(f"  Return:   {trading_metrics.get('total_return_pct', 0):+.1f}%")
            print(f"  Buy&Hold: {trading_metrics.get('buy_and_hold_return_pct', 0):+.1f}%")
            print(f"  Trades:   {trading_metrics.get('total_trades', 0)}")
        except Exception as e:
            logger.warning("Trading backtest failed: %s", e)
            print(f"  [WARN] Trading backtest failed: {e}")
    else:
        print(f"\n{'─'*70}")
        print("  Step 5/6: Trading backtest skipped (--no-trading)")
        print(f"{'─'*70}")

    # Step 6: Generate report
    print(f"\n{'─'*70}")
    print("  Step 6/6: Generating validation report...")
    print(f"{'─'*70}")

    report = generate_report(
        split_info, metrics, calibration, regime_accuracy,
        feature_importance, trading_metrics, wf_section,
    )

    # Save report
    report_path = output_dir / "report.txt"
    report_path.write_text(report, encoding="utf-8")
    print(f"\n  Report saved to: {report_path}")

    # Save JSON results
    save_results_json(
        output_dir, split_info, metrics, calibration,
        regime_accuracy, feature_importance, trading_metrics, wf_section,
    )

    elapsed = time.time() - start_time
    print(f"\n{'═'*80}")
    print(f"  Validation complete in {elapsed:.1f}s")
    print(f"  Output directory: {output_dir}")
    print(f"{'═'*80}")

    # Print the full report
    print("\n")
    print(report)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception as exc:
        logger.exception("Validation failed: %s", exc)
        print(f"\n[ERROR] Validation failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
