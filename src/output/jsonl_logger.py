"""Append-only JSONL logger for prediction runs.

Writes one JSON line per prediction cycle to data/predictions/predictions.jsonl,
capturing the full feature vector, raw direction probabilities, and signals
for post-hoc calibration analysis, feature drift analysis, and
prediction→trade→outcome linkage.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.utils.logger import setup_logger

logger = setup_logger(__name__)

PREDICTIONS_JSONL_PATH = Path("data/predictions/predictions.jsonl")


class PredictionJSONLLogger:
    """Appends one JSON line per prediction run to data/predictions/predictions.jsonl.

    Schema per line::

        {
          "run_number": 5,
          "timestamp": "2026-06-15T06:15:00Z",
          "btc_price": 65732.0,
          "used_ml": true,
          "model_source": "ensemble",
          "predictions": [
            {
              "timeframe": "24h",
              "direction": "DOWN",
              "direction_prob": 0.43,
              "magnitude": 0.7,
              "confidence": 38,
              "calibrated": true
            }
          ],
          "features": {"rsi_14": 71.0, "return_24h": 0.02, ...},
          "signals_summary": {"RSI (14)": "71.0 -- Overbought", ...}
        }

    Missing features are tolerated — an empty dict is written rather than raising.
    """

    def __init__(self, output_path: Path | str | None = None) -> None:
        self.output_path = Path(output_path) if output_path else PREDICTIONS_JSONL_PATH

    def log(
        self,
        run_number: int,
        timestamp: datetime,
        btc_price: float,
        used_ml: bool,
        model_source: str,
        predictions: list[dict[str, Any]],
        features: dict[str, float] | None = None,
        signals_summary: dict[str, Any] | None = None,
    ) -> None:
        """Append one JSON record for this prediction run.

        Args:
            run_number: Monotonically increasing counter from PredictionLogger.
            timestamp: UTC datetime of the prediction cycle.
            btc_price: BTC/USD close price at time of prediction.
            used_ml: True if a trained ML model produced the predictions.
            model_source: "ensemble", "xgboost", "baseline", or "heuristic".
            predictions: List of prediction dicts (may include direction_prob,
                calibrated in addition to the standard fields).
            features: Full numeric feature vector (may be partial or empty).
            signals_summary: Human-readable signals dict for quick inspection.
        """
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        record: dict[str, Any] = {
            "run_number": run_number,
            "timestamp": timestamp.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "btc_price": btc_price,
            "used_ml": used_ml,
            "model_source": model_source,
            "predictions": predictions,
            "features": features or {},
            "signals_summary": signals_summary or {},
        }

        try:
            with open(self.output_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(record) + "\n")
            logger.debug(
                "JSONL prediction #%d logged to %s", run_number, self.output_path
            )
        except OSError as exc:
            logger.warning("Failed to write prediction JSONL: %s", exc)
