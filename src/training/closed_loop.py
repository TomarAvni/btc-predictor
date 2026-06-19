"""Closed-loop learning from realized prediction outcomes.

The offline pipeline (``validate.py``) always rebuilds models + calibrator from
raw price history and never looks at how the live model actually did. This
module closes that loop by consuming the accumulated labeled live outcomes (the
``labeled_store``) and feeding them back into:

  1. **Probability calibration** -- refit the per-horizon isotonic calibrator on
     real live ``(direction_prob -> realized up/down)`` pairs so the surfaced
     confidence reflects the model's *actual* live hit rate. This is the most
     direct "learn from your mistakes" win and works with modest data.

  2. **Model retraining** -- where enough live data has matured, blend the live
     labeled rows into the historical training set and retrain per horizon.

Both paths are governed by a hard SAFETY GUARD: a candidate artifact is trained
into a staging location and only promoted to the serving location if it is
**not worse** than the current serving artifact on a held-out slice of recent
live outcomes (see :func:`is_not_worse`). Otherwise the current artifact is kept
and the reason is logged. The loop also no-ops cleanly when little/no data has
matured yet, so an early cloud run never crashes.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Optional

import numpy as np

from src.models.calibration import ProbabilityCalibrator, brier_score
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

# ── Closed-loop thresholds (overridable via autotrain.py CLI / env) ──────────
# Minimum live labeled rows for a horizon before we attempt a live calibrator
# refit. Must comfortably exceed the isotonic fit's own 50-sample floor so the
# train split (after holding out a recent slice) still has enough signal.
MIN_CALIBRATION_ROWS = 100

# Minimum live labeled rows for a horizon before we attempt to blend it into
# model retraining. Model retraining is expensive and noisy, so this is set
# high; below it we keep serving the historical model unchanged.
MIN_RETRAIN_ROWS = 400

# Fraction of the most-recent live rows held out (by time) to judge promotion.
DEFAULT_HOLDOUT_FRAC = 0.3

# Brier-score tolerance (calibrator): promote only if candidate Brier is no more
# than this much worse than the current calibrator on the holdout. 0.0 = strict.
DEFAULT_CALIBRATION_TOLERANCE = 0.0

# Directional-accuracy tolerance (models), in fraction: promote a retrained
# model only if its holdout accuracy is within this much of the current model.
DEFAULT_ACCURACY_TOLERANCE = 0.01


def is_not_worse(
    candidate: float,
    current: float,
    tolerance: float,
    higher_is_better: bool,
) -> bool:
    """Promotion predicate: is *candidate* "not worse" than *current*?

    For higher-is-better metrics (e.g. directional accuracy) a candidate passes
    when ``candidate >= current - tolerance``. For lower-is-better metrics (e.g.
    Brier score) it passes when ``candidate <= current + tolerance``. ``NaN``
    metrics never pass.
    """
    if candidate is None or current is None:
        return False
    if np.isnan(candidate) or np.isnan(current):
        return False
    if higher_is_better:
        return candidate >= current - tolerance
    return candidate <= current + tolerance


def _split_by_time(
    prob: np.ndarray,
    y: np.ndarray,
    holdout_frac: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Chronological train/holdout split (arrays are pre-sorted by time)."""
    n = len(prob)
    n_holdout = max(1, int(round(n * holdout_frac)))
    n_train = n - n_holdout
    return (
        prob[:n_train], y[:n_train],
        prob[n_train:], y[n_train:],
    )


def _live_calibration_pairs(rows: list[dict[str, Any]]) -> dict[str, dict[str, np.ndarray]]:
    """Collect time-ordered ``(direction_prob, label_up)`` per horizon.

    Only rows that carry a raw ``direction_prob`` (i.e. ML predictions) are
    usable for calibration; heuristic / text-log rows are skipped.
    """
    buckets: dict[str, list[tuple[Any, float, int]]] = {}
    for row in rows:
        prob = row.get("direction_prob")
        label = row.get("label_up")
        tf = row.get("timeframe")
        if prob is None or label is None or tf is None:
            continue
        buckets.setdefault(tf, []).append(
            (row.get("prediction_timestamp"), float(prob), int(label))
        )

    out: dict[str, dict[str, np.ndarray]] = {}
    for tf, triples in buckets.items():
        triples.sort(key=lambda t: (t[0] is None, t[0]))
        out[tf] = {
            "prob": np.array([t[1] for t in triples], dtype=float),
            "y": np.array([t[2] for t in triples], dtype=int),
        }
    return out


def refit_live_calibrator(
    labeled_rows: list[dict[str, Any]],
    serving_model_dir: Path,
    min_rows: int = MIN_CALIBRATION_ROWS,
    holdout_frac: float = DEFAULT_HOLDOUT_FRAC,
    tolerance: float = DEFAULT_CALIBRATION_TOLERANCE,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Refit the probability calibrator on live outcomes, guarded by promotion.

    For each horizon with enough live data we fit a *candidate* isotonic map on
    the older portion and compare its Brier score to the current serving
    calibrator on the held-out recent portion. Horizons whose candidate is not
    worse are "promoted": their isotonic map (refit on the full live history) is
    overlaid onto the serving calibrator. Horizons we don't have live data for
    keep their existing (offline OOF) calibration untouched.

    Returns a per-horizon report and the overall promotion decision. When
    ``dry_run`` is set, nothing is written to disk.
    """
    serving_model_dir = Path(serving_model_dir)
    current = ProbabilityCalibrator(model_dir=serving_model_dir).load()
    pairs = _live_calibration_pairs(labeled_rows)

    report: dict[str, Any] = {
        "horizons": {},
        "promoted_horizons": [],
        "kept_horizons": [],
        "skipped_horizons": [],
    }
    promoted_isotonic: dict[str, Any] = {}

    for tf, arr in sorted(pairs.items()):
        prob, y = arr["prob"], arr["y"]
        n = len(prob)

        if n < min_rows:
            report["horizons"][tf] = {"status": "skipped", "reason": "insufficient_data", "n": n}
            report["skipped_horizons"].append(tf)
            continue

        prob_tr, y_tr, prob_ho, y_ho = _split_by_time(prob, y, holdout_frac)
        if len(prob_tr) < 50 or len(np.unique(y_tr)) < 2 or len(np.unique(y_ho)) < 1:
            report["horizons"][tf] = {"status": "skipped", "reason": "degenerate_split", "n": n}
            report["skipped_horizons"].append(tf)
            continue

        # Candidate fit on the train slice -> Brier on the held-out recent slice.
        candidate = ProbabilityCalibrator(model_dir=serving_model_dir)
        fit_stats = candidate.fit(prob_tr, y_tr, tf)
        if "error" in fit_stats:
            report["horizons"][tf] = {"status": "skipped", "reason": "fit_failed", "n": n}
            report["skipped_horizons"].append(tf)
            continue

        cand_p_ho = candidate.calibrate_prob_array(prob_ho, tf)
        curr_p_ho = current.calibrate_prob_array(prob_ho, tf)  # identity if unfit
        cand_brier = brier_score(cand_p_ho, y_ho)
        curr_brier = brier_score(curr_p_ho, y_ho)

        promote = is_not_worse(cand_brier, curr_brier, tolerance, higher_is_better=False)
        horizon_report = {
            "status": "promote" if promote else "keep",
            "n": n,
            "n_train": int(len(prob_tr)),
            "n_holdout": int(len(prob_ho)),
            "candidate_brier": round(float(cand_brier), 4),
            "current_brier": round(float(curr_brier), 4),
            "improvement": round(float(curr_brier - cand_brier), 4),
        }
        report["horizons"][tf] = horizon_report

        if promote:
            # Production fit uses the FULL live history for this horizon.
            production = ProbabilityCalibrator(model_dir=serving_model_dir)
            prod_stats = production.fit(prob, y, tf)
            if "error" not in prod_stats:
                promoted_isotonic[tf] = production._isotonic[tf]
                report["promoted_horizons"].append(tf)
            else:  # pragma: no cover - guarded above, defensive
                report["kept_horizons"].append(tf)
                horizon_report["status"] = "keep"
        else:
            report["kept_horizons"].append(tf)

    report["n_promoted"] = len(report["promoted_horizons"])
    report["n_kept"] = len(report["kept_horizons"])
    report["n_skipped"] = len(report["skipped_horizons"])

    if promoted_isotonic and not dry_run:
        # Overlay promoted horizons onto the current serving calibrator so
        # untouched horizons keep their offline calibration.
        for tf, iso in promoted_isotonic.items():
            current._isotonic[tf] = iso
            current._fit_stats[tf] = {"source": "live_closed_loop", "n": len(pairs[tf]["prob"])}
        current.save()
        logger.info(
            "Live calibrator promoted %d horizon(s): %s",
            len(promoted_isotonic), ", ".join(sorted(promoted_isotonic)),
        )
    elif promoted_isotonic:
        logger.info(
            "[dry-run] Would promote %d calibrator horizon(s): %s",
            len(promoted_isotonic), ", ".join(sorted(promoted_isotonic)),
        )
    else:
        logger.info("Live calibrator: no horizons promoted")

    report["written"] = bool(promoted_isotonic) and not dry_run
    return report


def evaluate_direction_accuracy(
    predicted_probs: np.ndarray,
    labels_up: np.ndarray,
) -> float:
    """Directional accuracy of ``P(up) > 0.5`` calls vs realized up/down."""
    if len(predicted_probs) == 0:
        return float("nan")
    pred_up = (np.asarray(predicted_probs, dtype=float) > 0.5).astype(int)
    return float((pred_up == np.asarray(labels_up, dtype=int)).mean())


# ═══════════════════════════════════════════════════════════════════════════
# Model retraining with a live blend (heavy; gated + promotion-guarded)
# ═══════════════════════════════════════════════════════════════════════════


def _promote_dir(staging: Path, serving: Path) -> None:
    """Atomically-ish replace *serving* with *staging* (best effort on Windows)."""
    serving.parent.mkdir(parents=True, exist_ok=True)
    if serving.exists():
        shutil.rmtree(serving)
    shutil.copytree(staging, serving)


def retrain_models_with_live_blend(
    labeled_rows: list[dict[str, Any]],
    serving_model_dir: Path,
    staging_model_dir: Path,
    min_rows_per_horizon: int = MIN_RETRAIN_ROWS,
    holdout_frac: float = DEFAULT_HOLDOUT_FRAC,
    tolerance: float = DEFAULT_ACCURACY_TOLERANCE,
    skip_tft: bool = True,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Blend live labeled rows into model retraining, per horizon, with a guard.

    For every horizon that has accumulated ``>= min_rows_per_horizon`` live
    labeled rows we:

      1. build the historical price-derived training set (the existing path),
      2. append the live ``(features -> realized return)`` rows to it,
      3. train a candidate XGBoost direction/magnitude model into *staging*,
      4. compare candidate vs current serving model by directional accuracy on
         the held-out recent live rows, and
      5. promote the candidate into *serving* only if it is not worse.

    Gracefully returns ``{"status": "skipped", ...}`` when price history is
    unavailable or no horizon has enough live data -- so on a fresh deployment
    this never crashes and simply leaves the historical models in place.
    """
    serving_model_dir = Path(serving_model_dir)
    staging_model_dir = Path(staging_model_dir)

    from src.training.labeled_store import labeled_rows_to_frame

    live_df = labeled_rows_to_frame(labeled_rows)
    if live_df.empty:
        return {"status": "skipped", "reason": "no_live_data", "horizons": {}}

    horizon_counts = live_df["timeframe"].value_counts().to_dict()
    eligible = {tf: int(n) for tf, n in horizon_counts.items() if n >= min_rows_per_horizon}
    if not eligible:
        return {
            "status": "skipped",
            "reason": "insufficient_live_data_per_horizon",
            "max_rows_any_horizon": int(max(horizon_counts.values(), default=0)),
            "min_required": min_rows_per_horizon,
            "horizons": {},
        }

    # Heavy historical-data dependency is imported lazily so the calibrator loop
    # (the common case) never pays for it and missing price data degrades to a
    # clean skip rather than an import-time failure.
    try:
        import pandas as pd

        from src.models.xgboost_model import XGBoostPredictor
        from src.simulation.data_loader import HistoricalDataLoader
        from src.simulation.labeler import ForwardReturnLabeler
        from src.training.feature_builder import TrainingFeatureBuilder
    except Exception as exc:  # pragma: no cover - dependency/import guard
        return {"status": "skipped", "reason": f"imports_unavailable: {exc}", "horizons": {}}

    try:
        loader = HistoricalDataLoader()
        price_df = loader.load_price_data()
        merged_df = loader.get_merged_dataset()
    except FileNotFoundError:
        return {"status": "skipped", "reason": "no_price_history", "horizons": eligible}

    staging_model_dir.mkdir(parents=True, exist_ok=True)
    feature_builder = TrainingFeatureBuilder(model_dir=staging_model_dir)
    labeler = ForwardReturnLabeler()

    hist_features = feature_builder.build_features(merged_df)
    hist_labels = labeler.compute_labels(price_df)
    common_idx = hist_features.index.intersection(hist_labels.index)
    hist_features = hist_features.loc[common_idx]
    hist_labels = hist_labels.loc[common_idx]
    hist_numeric = hist_features.select_dtypes(include=[np.number]).fillna(0)
    feature_cols = hist_numeric.columns.tolist()

    report: dict[str, Any] = {"status": "ran", "horizons": {}, "promoted_horizons": []}

    for tf in sorted(eligible):
        target_col = f"return_{tf}"
        if target_col not in hist_labels.columns:
            report["horizons"][tf] = {"status": "skipped", "reason": "no_historical_label"}
            continue

        tf_live = live_df[live_df["timeframe"] == tf].sort_values("prediction_timestamp")
        n_live = len(tf_live)
        n_holdout = max(1, int(round(n_live * holdout_frac)))
        live_train = tf_live.iloc[: n_live - n_holdout]
        live_holdout = tf_live.iloc[n_live - n_holdout :]

        # Build the blended training matrix: historical + live-train rows.
        hist_valid = hist_labels[target_col].notna()
        X_hist = hist_numeric[hist_valid]
        y_hist = hist_labels.loc[hist_valid, target_col].astype(float)

        X_live = _live_feature_matrix(live_train, feature_cols, pd)
        y_live = live_train["actual_return_pct"].astype(float)

        X_blend = pd.concat([X_hist, X_live], axis=0)
        y_blend = pd.concat([y_hist, y_live], axis=0)
        if len(X_blend) < 500:
            report["horizons"][tf] = {"status": "skipped", "reason": "too_few_blend_rows", "n": len(X_blend)}
            continue

        y_dir_blend = (y_blend > 0).astype(int)

        candidate = XGBoostPredictor(timeframe=tf)
        candidate._model_path = staging_model_dir / f"xgb_{tf}"
        candidate.train(X_blend, y_dir_blend, y_blend)

        # Promotion guard: directional accuracy on the held-out recent live rows.
        X_ho = _live_feature_matrix(live_holdout, feature_cols, pd)
        y_ho = live_holdout["label_up"].astype(int).to_numpy()
        cand_probs = candidate.predict_batch(X_ho)["direction_prob"].to_numpy()
        cand_acc = evaluate_direction_accuracy(cand_probs, y_ho)

        current = XGBoostPredictor(timeframe=tf)
        current._model_path = serving_model_dir / f"xgb_{tf}"
        current.load()
        if current.model_direction is not None:
            curr_probs = current.predict_batch(X_ho)["direction_prob"].to_numpy()
            curr_acc = evaluate_direction_accuracy(curr_probs, y_ho)
        else:
            curr_acc = float("nan")

        # No incumbent -> any candidate is an improvement.
        promote = np.isnan(curr_acc) or is_not_worse(
            cand_acc, curr_acc, tolerance, higher_is_better=True
        )
        report["horizons"][tf] = {
            "status": "promote" if promote else "keep",
            "n_live": n_live,
            "n_blend": int(len(X_blend)),
            "candidate_accuracy": round(float(cand_acc), 4) if not np.isnan(cand_acc) else None,
            "current_accuracy": round(float(curr_acc), 4) if not np.isnan(curr_acc) else None,
        }

        if promote and not dry_run:
            _promote_dir(staging_model_dir / f"xgb_{tf}", serving_model_dir / f"xgb_{tf}")
            report["promoted_horizons"].append(tf)
        elif promote:
            report["promoted_horizons"].append(tf)

    report["n_promoted"] = len(report["promoted_horizons"])
    return report


def _live_feature_matrix(live_subset, feature_cols: list[str], pd) -> Any:
    """Build a DataFrame of expanded live features aligned to *feature_cols*.

    Missing feature columns are filled with 0.0 so the live matrix always lines
    up with the historical training matrix's column order.
    """
    data: dict[str, list[float]] = {}
    for col in feature_cols:
        src_col = f"feat__{col}"
        if src_col in live_subset.columns:
            data[col] = live_subset[src_col].astype(float).fillna(0.0).tolist()
        else:
            data[col] = [0.0] * len(live_subset)
    return pd.DataFrame(data, columns=feature_cols)
