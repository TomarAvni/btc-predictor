"""Probability calibration for direction predictions.

The models output ``P(up)`` for each horizon. Two things were historically
wrong with how that probability became a user-facing "confidence":

1. The pipeline reported ``|P(up) - 0.5| * 2`` (a *margin*) as the confidence.
   That number lives in [0, 1] and collapses toward 0 whenever the model is
   unsure -- which is most of the time for noisy financial data. A genuine
   ``P(up) = 0.66`` was therefore displayed as "32% confidence" even though the
   directional call is ~66% likely to be right. The downstream Kelly sizing
   (``conf/100 * 2 - 1``) and the ``MIN_CONFIDENCE`` gate both *expect*
   confidence to be ``P(correct direction)`` in percent, so the margin form was
   doubly wrong.

2. Gradient-boosted trees are not calibrated out of the box -- their
   ``predict_proba`` is typically over-confident near the extremes.

This module fits a per-horizon **isotonic regression** that maps the raw
``P(up)`` onto an empirically-calibrated ``P(up)``, learned from *out-of-fold*
walk-forward predictions (never in-sample). The directional confidence we
surface is then ``max(p_cal, 1 - p_cal) * 100`` -- exactly the semantics the
trading layer assumes.

It also provides the honest calibration diagnostics: Expected Calibration Error
(ECE), a reliability table, Brier score and ROC-AUC.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Optional

import numpy as np

from src import DATA_DIR
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

MODEL_DIR = DATA_DIR / "models"

# A directional call cannot be less than 50% confident in its own direction,
# and we never claim near-certainty from a noisy model.
CONFIDENCE_FLOOR = 50.0
CONFIDENCE_CAP = 95.0


def directional_confidence(prob_up: np.ndarray) -> np.ndarray:
    """Confidence in the *predicted* direction = max(p, 1 - p)."""
    p = np.asarray(prob_up, dtype=float)
    return np.maximum(p, 1.0 - p)


def expected_calibration_error(
    confidence: np.ndarray,
    correct: np.ndarray,
    n_bins: int = 10,
) -> dict[str, Any]:
    """Expected Calibration Error on directional confidence vs correctness.

    Args:
        confidence: Predicted P(correct direction) in [0.5, 1.0].
        correct: 1 if the directional call was right, else 0.
        n_bins: Number of equal-width bins across [0.5, 1.0].

    Returns:
        Dict with ``ece`` (percentage points), ``n`` and a ``bins`` reliability
        table (each: confidence range, mean confidence, empirical accuracy,
        count, gap in pp).
    """
    confidence = np.asarray(confidence, dtype=float)
    correct = np.asarray(correct, dtype=float)
    n = len(confidence)
    if n == 0:
        return {"ece": 0.0, "n": 0, "bins": []}

    edges = np.linspace(0.5, 1.0, n_bins + 1)
    bins: list[dict[str, Any]] = []
    weighted_gap = 0.0

    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        # Include the right edge in the final bin.
        if i == n_bins - 1:
            mask = (confidence >= lo) & (confidence <= hi)
        else:
            mask = (confidence >= lo) & (confidence < hi)
        count = int(mask.sum())
        if count == 0:
            continue
        mean_conf = float(confidence[mask].mean())
        acc = float(correct[mask].mean())
        gap = abs(mean_conf - acc)
        weighted_gap += gap * count
        bins.append({
            "range": f"{lo * 100:.0f}-{hi * 100:.0f}%",
            "mean_confidence": round(mean_conf * 100, 1),
            "accuracy": round(acc * 100, 1),
            "n": count,
            "gap_pp": round(gap * 100, 1),
        })

    ece = weighted_gap / n * 100  # percentage points
    return {"ece": round(ece, 2), "n": n, "bins": bins}


def brier_score(prob_up: np.ndarray, y_up: np.ndarray) -> float:
    """Brier score of the up-probability (lower is better)."""
    p = np.asarray(prob_up, dtype=float)
    y = np.asarray(y_up, dtype=float)
    if len(p) == 0:
        return float("nan")
    return float(np.mean((p - y) ** 2))


def safe_auc(prob_up: np.ndarray, y_up: np.ndarray) -> Optional[float]:
    """ROC-AUC of P(up) vs realized up/down, guarded against single-class folds."""
    y = np.asarray(y_up, dtype=int)
    if len(y) == 0 or len(np.unique(y)) < 2:
        return None
    try:
        from sklearn.metrics import roc_auc_score
        return float(roc_auc_score(y, np.asarray(prob_up, dtype=float)))
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("AUC computation failed: %s", exc)
        return None


class ProbabilityCalibrator:
    """Per-horizon isotonic calibration of direction probabilities.

    Fit on out-of-fold predictions, then persisted so the live engine can map
    raw ``P(up)`` -> calibrated ``P(up)`` and emit an honest confidence.
    """

    def __init__(self, model_dir: Optional[Path] = None) -> None:
        self.model_dir = Path(model_dir) if model_dir else MODEL_DIR
        self._isotonic: dict[str, Any] = {}
        self._fit_stats: dict[str, dict[str, Any]] = {}

    def fit(
        self,
        prob_up: np.ndarray,
        y_up: np.ndarray,
        horizon_label: str,
    ) -> dict[str, Any]:
        """Fit isotonic regression mapping raw P(up) -> calibrated P(up)."""
        from sklearn.isotonic import IsotonicRegression

        prob_up = np.asarray(prob_up, dtype=float)
        y_up = np.asarray(y_up, dtype=int)

        if len(prob_up) < 50 or len(np.unique(y_up)) < 2:
            logger.warning(
                "Calibration skipped for %s: insufficient/degenerate data (n=%d)",
                horizon_label, len(prob_up),
            )
            return {"error": "insufficient_data", "n": int(len(prob_up))}

        iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        iso.fit(prob_up, y_up)
        self._isotonic[horizon_label] = iso

        cal_p = iso.predict(prob_up)
        stats = {
            "n": int(len(prob_up)),
            "brier_before": round(brier_score(prob_up, y_up), 4),
            "brier_after": round(brier_score(cal_p, y_up), 4),
        }
        self._fit_stats[horizon_label] = stats
        logger.info(
            "Calibrated %s: Brier %.4f -> %.4f (n=%d)",
            horizon_label, stats["brier_before"], stats["brier_after"], stats["n"],
        )
        return stats

    def calibrate_prob(self, prob_up: float, horizon_label: str) -> float:
        """Map a raw P(up) to the calibrated P(up). Identity if not fitted."""
        iso = self._isotonic.get(horizon_label)
        if iso is None:
            return float(prob_up)
        return float(iso.predict([prob_up])[0])

    def calibrate_prob_array(
        self, prob_up: np.ndarray, horizon_label: str
    ) -> np.ndarray:
        """Vectorized version of :meth:`calibrate_prob`."""
        prob_up = np.asarray(prob_up, dtype=float)
        iso = self._isotonic.get(horizon_label)
        if iso is None:
            return prob_up
        return np.asarray(iso.predict(prob_up), dtype=float)

    def confidence(self, prob_up: float, horizon_label: str) -> float:
        """Calibrated confidence (%) in the predicted direction, in [50, 95]."""
        cal_p = self.calibrate_prob(prob_up, horizon_label)
        conf = max(cal_p, 1.0 - cal_p) * 100.0
        return float(max(CONFIDENCE_FLOOR, min(CONFIDENCE_CAP, conf)))

    def has_horizon(self, horizon_label: str) -> bool:
        return horizon_label in self._isotonic

    @property
    def fitted_horizons(self) -> list[str]:
        return list(self._isotonic.keys())

    def save(self) -> None:
        self.model_dir.mkdir(parents=True, exist_ok=True)
        path = self.model_dir / "probability_calibrator.pkl"
        with open(path, "wb") as f:
            pickle.dump(
                {"isotonic": self._isotonic, "fit_stats": self._fit_stats}, f
            )
        logger.info("Probability calibrator saved to %s", path)

    def load(self) -> "ProbabilityCalibrator":
        path = self.model_dir / "probability_calibrator.pkl"
        if not path.exists():
            return self
        try:
            with open(path, "rb") as f:
                state = pickle.load(f)
            self._isotonic = state.get("isotonic", {})
            self._fit_stats = state.get("fit_stats", {})
            logger.info(
                "Probability calibrator loaded (%s)", ", ".join(self._isotonic) or "empty"
            )
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Failed to load probability calibrator: %s", exc)
        return self
