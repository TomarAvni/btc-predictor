"""Out-of-sample walk-forward evaluation + calibration fitting.

Trains a fresh model on each purged walk-forward fold and predicts the held-out
block, producing genuinely out-of-fold (OOF) predictions across the tail of the
series. These OOF predictions are the honest estimate of live performance and
the *only* correct data to fit a probability calibrator on.

Used by ``validate.py`` to report walk-forward accuracy/AUC and to fit + persist
the per-horizon isotonic calibrator.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd
import xgboost as xgb

from src.models.baseline_model import BaselineModel
from src.models.calibration import (
    ProbabilityCalibrator,
    directional_confidence,
    expected_calibration_error,
    safe_auc,
)
from src.training.purged_cv import PurgedWalkForwardCV
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

DEFAULT_XGB_PARAMS = {
    "n_estimators": 300,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
}


def _make_classifier(params: dict[str, Any]) -> xgb.XGBClassifier:
    return xgb.XGBClassifier(
        n_estimators=params.get("n_estimators", 300),
        max_depth=params.get("max_depth", 6),
        learning_rate=params.get("learning_rate", 0.05),
        subsample=params.get("subsample", 0.8),
        colsample_bytree=params.get("colsample_bytree", 0.8),
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )


def run_walk_forward(
    features: pd.DataFrame,
    labels: pd.DataFrame,
    timeframes: list[str],
    horizon_hours: dict[str, int],
    n_splits: int = 5,
    embargo_hours: int = 0,
    min_train_frac: float = 0.4,
    xgb_params: Optional[dict[str, Any]] = None,
    include_baseline: bool = True,
) -> dict[str, Any]:
    """Run purged walk-forward CV and collect OOF predictions per horizon.

    Args:
        features: Numeric feature matrix (DatetimeIndex, already NaN-filled).
        labels: Label frame with ``return_{tf}`` columns aligned to ``features``.
        timeframes: Horizons to evaluate (e.g. ``["24h", "7d", ...]``).
        horizon_hours: Map timeframe -> forward horizon in hours (for purging).
        n_splits: Number of walk-forward folds.
        embargo_hours: Extra embargo beyond the per-horizon purge.
        min_train_frac: Fraction of the series reserved for the first train set.
        xgb_params: XGBoost hyperparameters (defaults to ``DEFAULT_XGB_PARAMS``).
        include_baseline: Also evaluate the logistic baseline for comparison.

    Returns:
        Dict keyed by timeframe with OOF arrays and honest metrics.
    """
    params = {**DEFAULT_XGB_PARAMS, **(xgb_params or {})}
    cv = PurgedWalkForwardCV(
        n_splits=n_splits,
        embargo_hours=embargo_hours,
        min_train_frac=min_train_frac,
        mode="expanding",
    )

    index = features.index
    feature_matrix = features.to_numpy(dtype=float)
    results: dict[str, Any] = {}

    for tf in timeframes:
        return_col = f"return_{tf}"
        if return_col not in labels.columns:
            continue

        ret = labels[return_col]
        label_valid = ret.notna()
        y_up_full = (ret > 0).astype(int)

        oof_prob: list[np.ndarray] = []
        oof_y: list[np.ndarray] = []
        oof_base_prob: list[np.ndarray] = []
        fold_accuracies: list[float] = []
        fold_meta: list[dict[str, Any]] = []

        for fold in cv.split(index, horizon_hours[tf], label_valid=label_valid):
            X_tr = feature_matrix[fold.train_positions]
            y_tr = y_up_full.to_numpy()[fold.train_positions]
            X_te = feature_matrix[fold.test_positions]
            y_te = y_up_full.to_numpy()[fold.test_positions]

            if len(np.unique(y_tr)) < 2:
                logger.warning("Fold %d (%s): single-class train, skipping", fold.fold_number, tf)
                continue

            clf = _make_classifier(params)
            clf.fit(X_tr, y_tr)
            prob = clf.predict_proba(X_te)[:, 1]

            oof_prob.append(prob)
            oof_y.append(y_te)
            acc = float(((prob > 0.5).astype(int) == y_te).mean())
            fold_accuracies.append(acc)

            base_acc = None
            if include_baseline:
                base = BaselineModel()
                X_tr_df = features.iloc[fold.train_positions]
                y_tr_dir = pd.Series(y_tr, index=X_tr_df.index)
                y_tr_mag = ret.iloc[fold.train_positions]
                base.train(X_tr_df, y_tr_dir, y_tr_mag)
                base_pred = base.predict_batch(features.iloc[fold.test_positions])
                bprob = base_pred["direction_prob"].to_numpy()
                oof_base_prob.append(bprob)
                base_acc = float(((bprob > 0.5).astype(int) == y_te).mean())

            fold_meta.append({
                "fold": fold.fold_number,
                "train_start": str(fold.train_start.date()),
                "train_end": str(fold.train_end.date()),
                "test_start": str(fold.test_start.date()),
                "test_end": str(fold.test_end.date()),
                "n_train": fold.n_train,
                "n_test": fold.n_test,
                "xgb_accuracy": round(acc * 100, 1),
                "baseline_accuracy": round(base_acc * 100, 1) if base_acc is not None else None,
            })
            logger.info(
                "WF %s fold %d: train[%s..%s] n=%d -> test[%s..%s] n=%d | xgb_acc=%.3f",
                tf, fold.fold_number, fold.train_start.date(), fold.train_end.date(),
                fold.n_train, fold.test_start.date(), fold.test_end.date(),
                fold.n_test, acc,
            )

        if not oof_prob:
            logger.warning("No usable folds for %s", tf)
            continue

        prob_all = np.concatenate(oof_prob)
        y_all = np.concatenate(oof_y)
        pred_dir = (prob_all > 0.5).astype(int)

        tp = int(((pred_dir == 1) & (y_all == 1)).sum())
        tn = int(((pred_dir == 0) & (y_all == 0)).sum())
        fp = int(((pred_dir == 1) & (y_all == 0)).sum())
        fn = int(((pred_dir == 0) & (y_all == 1)).sum())

        base_prob_all = np.concatenate(oof_base_prob) if oof_base_prob else None

        results[tf] = {
            "oof_prob_up": prob_all,
            "oof_y_up": y_all,
            "oof_baseline_prob_up": base_prob_all,
            "n": int(len(y_all)),
            "accuracy": float((pred_dir == y_all).mean()),
            "auc": safe_auc(prob_all, y_all),
            "baseline_accuracy": (
                float(((base_prob_all > 0.5).astype(int) == y_all).mean())
                if base_prob_all is not None else None
            ),
            "up_rate": float(y_all.mean()),
            "confusion": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
            "fold_accuracies": fold_accuracies,
            "folds": fold_meta,
            "n_folds": len(fold_accuracies),
        }

    return results


def fit_calibrator_from_oof(
    wf_results: dict[str, Any],
    model_dir,
) -> tuple[ProbabilityCalibrator, dict[str, Any]]:
    """Fit + persist an isotonic calibrator on OOF predictions.

    Returns the calibrator and a per-horizon report of ECE before/after and AUC.
    """
    calibrator = ProbabilityCalibrator(model_dir=model_dir)
    report: dict[str, Any] = {}

    for tf, res in wf_results.items():
        prob = res["oof_prob_up"]
        y = res["oof_y_up"]

        conf_before = directional_confidence(prob)
        correct = ((prob > 0.5).astype(int) == y).astype(int)
        ece_before = expected_calibration_error(conf_before, correct)

        fit_stats = calibrator.fit(prob, y, tf)

        if "error" not in fit_stats:
            cal_p = calibrator.calibrate_prob_array(prob, tf)
            conf_after = directional_confidence(cal_p)
            ece_after = expected_calibration_error(conf_after, correct)
        else:
            ece_after = ece_before

        report[tf] = {
            "n": res["n"],
            "auc": res["auc"],
            "ece_before_pp": ece_before["ece"],
            "ece_after_pp": ece_after["ece"],
            "reliability_after": ece_after["bins"],
            "reliability_before": ece_before["bins"],
            "brier": fit_stats,
        }

    if calibrator.fitted_horizons:
        calibrator.save()
    return calibrator, report


def tune_xgboost(
    features: pd.DataFrame,
    labels: pd.DataFrame,
    timeframe: str,
    horizon_hours: int,
    n_splits: int = 4,
    embargo_hours: int = 0,
    min_train_frac: float = 0.4,
    grid: Optional[list[dict[str, Any]]] = None,
) -> dict[str, Any]:
    """Small, time-series-aware grid search for XGBoost on one horizon.

    Scores candidates by mean out-of-fold log-loss (lower is better) over the
    purged walk-forward folds. Deliberately modest to avoid data-snooping.
    """
    from sklearn.metrics import log_loss

    if grid is None:
        grid = [
            {"max_depth": d, "learning_rate": lr, "n_estimators": 300}
            for d in (3, 5, 7)
            for lr in (0.03, 0.06)
        ]

    return_col = f"return_{timeframe}"
    ret = labels[return_col]
    label_valid = ret.notna()
    y_up_full = (ret > 0).astype(int).to_numpy()
    feature_matrix = features.to_numpy(dtype=float)
    index = features.index

    cv = PurgedWalkForwardCV(
        n_splits=n_splits, embargo_hours=embargo_hours,
        min_train_frac=min_train_frac, mode="expanding",
    )
    folds = list(cv.split(index, horizon_hours, label_valid=label_valid))

    candidates: list[dict[str, Any]] = []
    for cand in grid:
        params = {**DEFAULT_XGB_PARAMS, **cand}
        losses, accs = [], []
        for fold in folds:
            X_tr = feature_matrix[fold.train_positions]
            y_tr = y_up_full[fold.train_positions]
            X_te = feature_matrix[fold.test_positions]
            y_te = y_up_full[fold.test_positions]
            if len(np.unique(y_tr)) < 2 or len(np.unique(y_te)) < 2:
                continue
            clf = _make_classifier(params)
            clf.fit(X_tr, y_tr)
            prob = clf.predict_proba(X_te)[:, 1]
            losses.append(log_loss(y_te, prob, labels=[0, 1]))
            accs.append(float(((prob > 0.5).astype(int) == y_te).mean()))
        if losses:
            candidates.append({
                "params": cand,
                "mean_logloss": round(float(np.mean(losses)), 5),
                "mean_accuracy": round(float(np.mean(accs)) * 100, 2),
            })

    if not candidates:
        return {"error": "no_candidates"}

    best = min(candidates, key=lambda c: c["mean_logloss"])
    return {"timeframe": timeframe, "best": best, "all": candidates}
