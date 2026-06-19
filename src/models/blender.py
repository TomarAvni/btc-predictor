"""Lightweight prediction blending utilities.

The first blended strategy should not be a new trainable model. It should be a
small deterministic combiner that averages module probabilities and magnitudes,
then a trainable meta-blender can be introduced after enough scored rows exist.
"""

from __future__ import annotations

from typing import Any


def _prob_from_prediction(pred: dict[str, Any]) -> float:
    prob = pred.get("direction_prob")
    if prob is not None:
        return max(0.0, min(1.0, float(prob)))
    direction = str(pred.get("direction", "")).upper()
    confidence = float(pred.get("confidence", 50.0)) / 100.0
    confidence = max(0.0, min(1.0, confidence))
    if direction == "UP":
        return 0.5 + confidence / 2
    if direction == "DOWN":
        return 0.5 - confidence / 2
    return 0.5


def blend_predictions(
    numeric_predictions: list[dict[str, Any]],
    twitter_predictions: list[dict[str, Any]],
    numeric_weight: float = 0.5,
) -> list[dict[str, Any]]:
    """Blend numeric and Twitter prediction lists by matching timeframe."""
    numeric_weight = max(0.0, min(1.0, numeric_weight))
    twitter_weight = 1.0 - numeric_weight
    twitter_by_tf = {p.get("timeframe"): p for p in twitter_predictions}

    blended: list[dict[str, Any]] = []
    for numeric in numeric_predictions:
        tf = numeric.get("timeframe")
        twitter = twitter_by_tf.get(tf)
        if not twitter:
            out = dict(numeric)
            out["model_source"] = "blended_numeric_only"
            out["blend_status"] = "missing_twitter"
            blended.append(out)
            continue

        n_prob = _prob_from_prediction(numeric)
        t_prob = _prob_from_prediction(twitter)
        prob = numeric_weight * n_prob + twitter_weight * t_prob

        n_mag = abs(float(numeric.get("magnitude", 0.0) or 0.0))
        t_mag = abs(float(twitter.get("magnitude", 0.0) or 0.0))
        magnitude = numeric_weight * n_mag + twitter_weight * t_mag
        direction = "UP" if prob >= 0.5 else "DOWN"
        confidence = int(round(50 + abs(prob - 0.5) * 100))

        blended.append({
            "timeframe": tf,
            "direction": direction,
            "direction_prob": prob,
            "magnitude": magnitude,
            "predicted_return": magnitude if direction == "UP" else -magnitude,
            "confidence": confidence,
            "model_source": "blended_50_50",
            "calibrated": False,
            "components": {
                "numeric": numeric,
                "twitter": twitter,
                "numeric_weight": numeric_weight,
                "twitter_weight": twitter_weight,
            },
        })
    return blended
