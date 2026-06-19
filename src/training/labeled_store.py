"""Append-only labeled training store built from realized prediction outcomes.

This is the dataset of "what the model saw -> what actually happened": each row
joins the feature vector + raw ``direction_prob`` the model emitted at
prediction time with the realized outcome (direction label + realized return)
once the horizon has matured and the scorer has graded it.

The store is the substrate for the closed-loop trainer (``autotrain.py``):

  * the probability calibrator is refit on real live ``(direction_prob, label)``
    pairs per horizon, and
  * where enough data has accumulated, the labeled rows are blended into model
    retraining / used to select between candidate models.

Format & location
-----------------
JSONL at ``data/training_data/labeled.jsonl`` -- one JSON object per
``run_number:timeframe``. JSONL (not Parquet) is deliberate: Parquet is
gitignored, while the JSONL store is git-tracked and therefore persists across
stateless cloud (GitHub Actions) runs, exactly like ``prediction_scores.jsonl``.

The store is append-only and deduped by ``run_number:timeframe`` so it
accumulates over time and is never overwritten. It is derived from the enriched
``prediction_scores.jsonl`` produced by the scorer, so it can always be rebuilt
from scratch if deleted.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from src import DATA_DIR
from src.horizons import HORIZON_HOURS
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

TRAINING_DATA_DIR = DATA_DIR / "training_data"
LABELED_STORE_PATH = TRAINING_DATA_DIR / "labeled.jsonl"

# Fields copied verbatim from an enriched score record onto a labeled row.
_PASSTHROUGH_FIELDS = (
    "prediction_timestamp",
    "predicted_direction",
    "direction_prob",
    "predicted_magnitude",
    "confidence",
    "calibrated",
    "model_source",
    "used_ml",
    "actual_return_pct",
    "actual_direction",
    "direction_correct",
    "magnitude_error",
    "start_price",
    "end_price",
)


def _store_key(rec: dict[str, Any]) -> str:
    return f"{rec.get('run_number')}:{rec.get('timeframe')}"


def load_labeled_keys(path: Path | None = None) -> set[str]:
    """Return the set of ``run_number:timeframe`` keys already in the store."""
    store = path or LABELED_STORE_PATH
    if not store.exists():
        return set()
    keys: set[str] = set()
    for line in store.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        keys.add(_store_key(rec))
    return keys


def load_labeled_rows(path: Path | None = None) -> list[dict[str, Any]]:
    """Load every labeled row from the store (oldest first)."""
    store = path or LABELED_STORE_PATH
    if not store.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in store.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def _to_labeled_row(score: dict[str, Any]) -> Optional[dict[str, Any]]:
    """Project an enriched score record onto a labeled-store row.

    Returns ``None`` for rows that carry no learning signal -- i.e. an unknown
    horizon or an empty feature vector (legacy text-log / heuristic rows). Those
    are skipped so the store only holds genuinely trainable examples.
    """
    timeframe = score.get("timeframe")
    if timeframe not in HORIZON_HOURS:
        return None

    features = score.get("features") or {}
    if not isinstance(features, dict) or not features:
        # No feature vector -> nothing the model can learn to map from.
        return None

    actual_direction = score.get("actual_direction")
    if actual_direction not in ("UP", "DOWN"):
        return None

    row: dict[str, Any] = {
        "run_number": score.get("run_number"),
        "timeframe": timeframe,
        "horizon_hours": HORIZON_HOURS[timeframe],
        # Binary realized label (1 = up) -- the calibrator target.
        "label_up": 1 if actual_direction == "UP" else 0,
        "features": features,
    }
    for field in _PASSTHROUGH_FIELDS:
        row[field] = score.get(field)
    return row


def append_labeled_rows(
    scores: list[dict[str, Any]],
    path: Path | None = None,
) -> int:
    """Append new labeled rows derived from enriched score records.

    Deduped by ``run_number:timeframe`` against what is already on disk, so the
    store accumulates monotonically. Returns the number of rows appended.
    """
    store = path or LABELED_STORE_PATH
    store.parent.mkdir(parents=True, exist_ok=True)

    existing = load_labeled_keys(store)
    new_rows: list[dict[str, Any]] = []
    seen: set[str] = set()

    for score in scores:
        row = _to_labeled_row(score)
        if row is None:
            continue
        key = _store_key(row)
        if key in existing or key in seen:
            continue
        seen.add(key)
        new_rows.append(row)

    if new_rows:
        with open(store, "a", encoding="utf-8") as fh:
            for row in new_rows:
                fh.write(json.dumps(row) + "\n")
        logger.info("Labeled store: appended %d rows -> %s", len(new_rows), store)
    else:
        logger.info("Labeled store: no new trainable rows to append")

    return len(new_rows)


def update_labeled_store_from_scores(
    scores_path: Path | None = None,
    store_path: Path | None = None,
) -> int:
    """Refresh the labeled store from the enriched ``prediction_scores.jsonl``.

    Reads every scored record, projects the trainable ones onto labeled rows and
    appends those not already present. Returns the number of rows added.
    """
    from src.engine.scorer import load_prediction_scores

    scores = load_prediction_scores(scores_path)
    if not scores:
        return 0
    return append_labeled_rows(scores, store_path)


def labeled_rows_to_frame(
    rows: list[dict[str, Any]] | None = None,
    path: Path | None = None,
) -> pd.DataFrame:
    """Flatten labeled rows into a tidy DataFrame indexed by prediction time.

    The ``features`` dict is expanded into one column per feature. Rows are
    sorted chronologically so callers can do time-based train/holdout splits.
    Returns an empty frame when the store is empty.
    """
    data = rows if rows is not None else load_labeled_rows(path)
    if not data:
        return pd.DataFrame()

    records: list[dict[str, Any]] = []
    for row in data:
        flat = {k: v for k, v in row.items() if k != "features"}
        feats = row.get("features") or {}
        for fk, fv in feats.items():
            flat[f"feat__{fk}"] = fv
        records.append(flat)

    df = pd.DataFrame(records)
    if "prediction_timestamp" in df.columns:
        df["prediction_timestamp"] = pd.to_datetime(
            df["prediction_timestamp"], utc=True, errors="coerce"
        )
        df = df.sort_values("prediction_timestamp").reset_index(drop=True)
    return df


def feature_columns(df: pd.DataFrame) -> list[str]:
    """Return the expanded feature column names (``feat__*``) of a labeled frame."""
    return [c for c in df.columns if c.startswith("feat__")]


def per_horizon_counts(path: Path | None = None) -> dict[str, int]:
    """Return ``{timeframe: n_rows}`` currently in the store."""
    counts: dict[str, int] = {}
    for row in load_labeled_rows(path):
        tf = row.get("timeframe")
        if tf is not None:
            counts[tf] = counts.get(tf, 0) + 1
    return counts
