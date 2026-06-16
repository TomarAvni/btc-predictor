"""Score mature predictions against actual BTC price moves.

Compares each mature prediction to realized returns from hourly price data
and maintains rolling accuracy statistics.

The scorer prefers the rich JSONL prediction log
(``data/predictions/predictions.jsonl``) because it preserves the full feature
vector, the raw ``direction_prob`` and provenance fields (``model_source``,
``used_ml``, ``calibrated``) that the closed-loop trainer needs to learn from
realized outcomes.  It transparently falls back to the legacy text log
(``predictions.log``) when the JSONL store is absent, in which case the
feature vector and ``direction_prob`` are simply unavailable on those rows.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src import DATA_DIR, PREDICTIONS_LOG
from src.horizons import HORIZON_HOURS, LEGACY_MODEL_ALIASES, TIMEFRAMES
from src.output.jsonl_logger import PREDICTIONS_JSONL_PATH
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

PERFORMANCE_DIR = DATA_DIR / "performance"
SCORES_PATH = PERFORMANCE_DIR / "prediction_scores.jsonl"
ROLLING_PATH = PERFORMANCE_DIR / "rolling_accuracy.json"
CALIBRATION_LIVE_PATH = PERFORMANCE_DIR / "calibration_live.json"
PRICE_PATH = DATA_DIR / "price" / "btc_hourly.parquet"

# Reverse of LEGACY_MODEL_ALIASES: old horizon label -> current canonical label.
# e.g. the legacy "7d" prediction targets the same 168-hour forward return as
# the current "168h" point, so old JSONL/text rows are normalized onto "168h".
_HORIZON_NORMALIZE: dict[str, str] = {
    legacy: current for current, legacy in LEGACY_MODEL_ALIASES.items()
}

_PRED_LINE = re.compile(
    r"^\s*(\S+)\s*\|\s*(UP|DOWN)\s*\|\s*([+\-]?\d+\.?\d*)%\s*\|\s*Confidence:\s*(\d+)%",
    re.IGNORECASE,
)
_RUN_HEADER = re.compile(
    r"^\[(\d{4}-\d{2}-\d{2}\s\d{2}:\d{2}\sUTC)\]\s*--\s*Prediction Run #(\d+)"
)


def _normalize_timeframe(tf: str | None) -> str | None:
    """Map a (possibly legacy) horizon label onto the canonical horizon set.

    Returns the canonical label when the horizon is known (directly or via a
    legacy alias), otherwise ``None`` so unknown/retired horizons (e.g. an old
    "90d" point) are skipped gracefully instead of crashing the scorer.
    """
    if tf is None:
        return None
    if tf in HORIZON_HOURS:
        return tf
    aliased = _HORIZON_NORMALIZE.get(tf)
    if aliased in HORIZON_HOURS:
        return aliased
    return None


def parse_predictions_log(path: Path | None = None) -> list[dict[str, Any]]:
    """Parse predictions.log into structured run dicts."""
    log_path = path or PREDICTIONS_LOG
    if not log_path.exists():
        return []

    runs: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    section = ""

    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        hdr = _RUN_HEADER.search(line)
        if hdr:
            if current:
                runs.append(current)
            current = {
                "timestamp": hdr.group(1),
                "run_number": int(hdr.group(2)),
                "predictions": [],
            }
            section = ""
            continue

        if current is None:
            continue

        stripped = line.strip()
        if stripped.startswith("PREDICTIONS"):
            section = "pred"
            continue
        if stripped.startswith("SIGNAL SUMMARY"):
            section = ""
            continue

        if section == "pred":
            m = _PRED_LINE.match(line)
            if m:
                current["predictions"].append({
                    "timeframe": m.group(1),
                    "direction": m.group(2).upper(),
                    "magnitude": float(m.group(3)),
                    "confidence": int(m.group(4)),
                })

    if current:
        runs.append(current)
    return runs


def parse_predictions_jsonl(path: Path | None = None) -> list[dict[str, Any]]:
    """Parse the rich JSONL prediction log into structured run dicts.

    Each returned run carries the run-level ``features`` vector and provenance
    (``model_source``, ``used_ml``, ``btc_price``) plus a list of per-horizon
    prediction dicts that retain ``direction_prob``, ``confidence`` and
    ``calibrated`` -- the learning signal that the text log drops.

    Malformed lines and rows missing a ``run_number`` are skipped rather than
    aborting the scan, so a single bad append never blocks scoring.
    """
    jsonl_path = path or PREDICTIONS_JSONL_PATH
    if not jsonl_path.exists():
        return []

    runs: list[dict[str, Any]] = []
    for line in jsonl_path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if rec.get("run_number") is None or "timestamp" not in rec:
            continue
        runs.append({
            "run_number": rec["run_number"],
            "timestamp": rec["timestamp"],
            "btc_price": rec.get("btc_price"),
            "used_ml": bool(rec.get("used_ml", False)),
            "model_source": rec.get("model_source"),
            "features": rec.get("features") or {},
            "predictions": rec.get("predictions", []),
        })
    return runs


def _parse_prediction_ts(ts_str: str) -> pd.Timestamp:
    """Parse a prediction timestamp from either the text-log or ISO format.

    The text log uses ``"%Y-%m-%d %H:%M UTC"`` while the JSONL log uses ISO-8601
    (e.g. ``"2026-06-15T11:02:52Z"``). Both normalize to a UTC pandas Timestamp.
    """
    try:
        dt = datetime.strptime(ts_str, "%Y-%m-%d %H:%M UTC").replace(tzinfo=timezone.utc)
        return pd.Timestamp(dt)
    except ValueError:
        pass
    ts = pd.Timestamp(ts_str)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts


def load_price_data(path: Path | None = None) -> pd.DataFrame:
    """Load hourly BTC price data for scoring."""
    price_dir = DATA_DIR / "price"
    candidates = [path] if path else []
    if not candidates:
        preferred = price_dir / "btc_hourly.parquet"
        if preferred.exists():
            candidates.append(preferred)
        candidates.extend(sorted(price_dir.glob("*.parquet")))

    for p in candidates:
        if p is None or not p.exists():
            continue
        try:
            df = pd.read_parquet(p)
            if "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
                df = df.set_index("timestamp")
            elif not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index, utc=True)
            df = df.sort_index()
            if "close" not in df.columns:
                continue
            return df
        except Exception as exc:
            logger.warning("Failed to load price data from %s: %s", p, exc)

    return pd.DataFrame()


def _price_at_or_before(price_df: pd.DataFrame, ts: pd.Timestamp) -> float | None:
    subset = price_df.loc[:ts]
    if subset.empty:
        return None
    return float(subset["close"].iloc[-1])


def _price_at_or_after(price_df: pd.DataFrame, ts: pd.Timestamp) -> float | None:
    subset = price_df.loc[ts:]
    if subset.empty:
        return None
    return float(subset["close"].iloc[-1])


def _score_key(run_number: Any, timeframe: str, model_source: Any) -> str:
    """Dedupe key for a scored prediction.

    Includes ``model_source`` so the parallel tracks (``numbers``, ``llm_direct``,
    ``llm_calibrated``, ``blended``) for the same run+horizon are scored
    independently instead of colliding.
    """
    return f"{run_number}:{timeframe}:{model_source}"


def _load_scored_keys(path: Path) -> set[str]:
    if not path.exists():
        return set()
    keys: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
            keys.add(_score_key(
                rec.get("run_number"), rec.get("timeframe"), rec.get("model_source")
            ))
        except json.JSONDecodeError:
            continue
    return keys


def _is_mature(pred_ts: pd.Timestamp, timeframe: str, now: pd.Timestamp) -> bool:
    hours = HORIZON_HOURS.get(timeframe)
    if hours is None:
        return False
    return (now - pred_ts) >= pd.Timedelta(hours=hours)


def load_prediction_runs(
    jsonl_path: Path | None = None,
    text_path: Path | None = None,
) -> tuple[list[dict[str, Any]], str]:
    """Load prediction runs, preferring the rich JSONL log over the text log.

    Returns ``(runs, source)`` where ``source`` is ``"jsonl"`` or ``"text"``.
    The JSONL log is preferred because it preserves the feature vector and
    ``direction_prob``; the text log is a backward-compatible fallback.
    """
    runs = parse_predictions_jsonl(jsonl_path)
    if runs:
        return runs, "jsonl"
    return parse_predictions_log(text_path), "text"


def score_mature_predictions(
    predictions_path: Path | None = None,
    price_path: Path | None = None,
    scores_path: Path | None = None,
    now: pd.Timestamp | None = None,
    predictions_jsonl_path: Path | None = None,
) -> list[dict[str, Any]]:
    """Score predictions whose horizons have elapsed. Returns newly scored records.

    Reads from the rich JSONL prediction log when available (preserving the
    feature vector and ``direction_prob`` on each scored record) and falls back
    to the legacy text log otherwise. Legacy horizon labels are normalized onto
    the canonical horizon set; unknown horizons are skipped gracefully.
    """
    PERFORMANCE_DIR.mkdir(parents=True, exist_ok=True)
    scores_file = scores_path or SCORES_PATH
    now_ts = now or pd.Timestamp.now(tz="UTC")

    # ``predictions_path`` keeps the historical (text-log) meaning for callers.
    runs, source = load_prediction_runs(
        jsonl_path=predictions_jsonl_path,
        text_path=predictions_path,
    )
    if not runs:
        logger.info("No predictions to score")
        return []
    logger.info("Scoring from %s prediction log (%d runs)", source, len(runs))

    price_df = load_price_data(price_path)
    if price_df.empty:
        logger.warning("No price data available for scoring")
        return []

    already_scored = _load_scored_keys(scores_file)
    new_scores: list[dict[str, Any]] = []

    for run in runs:
        try:
            pred_ts = _parse_prediction_ts(run["timestamp"])
        except (ValueError, TypeError):
            continue

        run_features = run.get("features") or {}
        model_source = run.get("model_source")
        used_ml = bool(run.get("used_ml", False))

        for pred in run.get("predictions", []):
            tf = _normalize_timeframe(pred.get("timeframe"))
            if tf is None:
                continue

            key = _score_key(run["run_number"], tf, model_source)
            if key in already_scored:
                continue
            if not _is_mature(pred_ts, tf, now_ts):
                continue

            horizon_end = pred_ts + pd.Timedelta(hours=HORIZON_HOURS[tf])
            start_price = _price_at_or_before(price_df, pred_ts)
            end_price = _price_at_or_after(price_df, horizon_end)
            if start_price is None or end_price is None or start_price <= 0:
                continue

            actual_return = (end_price / start_price - 1) * 100
            actual_direction = "UP" if actual_return > 0 else "DOWN"
            predicted_direction = str(pred["direction"]).upper()
            direction_correct = actual_direction == predicted_direction
            magnitude = float(pred.get("magnitude", 0.0) or 0.0)

            record = {
                "scored_at": now_ts.isoformat(),
                "prediction_timestamp": pred_ts.isoformat(),
                "run_number": run["run_number"],
                "timeframe": tf,
                "predicted_direction": predicted_direction,
                "predicted_magnitude": magnitude,
                "confidence": pred.get("confidence"),
                # Learning signal preserved from the rich JSONL log (None for
                # text-log rows / heuristic runs that never carried a prob).
                "direction_prob": pred.get("direction_prob"),
                "calibrated": pred.get("calibrated"),
                "model_source": model_source,
                "used_ml": used_ml,
                "actual_return_pct": round(actual_return, 4),
                "actual_direction": actual_direction,
                "direction_correct": direction_correct,
                "magnitude_error": round(abs(magnitude) - abs(actual_return), 4),
                "start_price": round(start_price, 2),
                "end_price": round(end_price, 2),
                "features": run_features,
            }
            new_scores.append(record)
            already_scored.add(key)

    if new_scores:
        with open(scores_file, "a", encoding="utf-8") as f:
            for rec in new_scores:
                f.write(json.dumps(rec) + "\n")
        logger.info("Scored %d mature predictions -> %s", len(new_scores), scores_file)
    else:
        logger.info("No new mature predictions to score")

    return new_scores


def load_prediction_scores(path: Path | None = None) -> list[dict[str, Any]]:
    """Load all scored prediction records from JSONL."""
    scores_file = path or SCORES_PATH
    if not scores_file.exists():
        return []

    records: list[dict[str, Any]] = []
    for line in scores_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return records


def compute_rolling_accuracy(
    scores: list[dict[str, Any]] | None = None,
    now: pd.Timestamp | None = None,
) -> dict[str, Any]:
    """Compute rolling direction accuracy per timeframe."""
    records = scores if scores is not None else load_prediction_scores()
    now_ts = now or pd.Timestamp.now(tz="UTC")

    result: dict[str, Any] = {
        "updated_at": now_ts.isoformat(),
        "timeframes": {},
    }

    if not records:
        for tf in TIMEFRAMES:
            result["timeframes"][tf] = {
                "last_7d": _empty_window_stats(),
                "last_30d": _empty_window_stats(),
                "all_time": _empty_window_stats(),
            }
        return result

    df = pd.DataFrame(records)
    df["prediction_timestamp"] = pd.to_datetime(df["prediction_timestamp"], utc=True)

    windows = {
        "last_7d": now_ts - pd.Timedelta(days=7),
        "last_30d": now_ts - pd.Timedelta(days=30),
        "all_time": None,
    }

    def _accuracy_block(frame: pd.DataFrame) -> dict[str, Any]:
        block: dict[str, Any] = {}
        for tf in TIMEFRAMES:
            tf_df = frame[frame["timeframe"] == tf]
            block[tf] = {}
            for label, cutoff in windows.items():
                sub = tf_df if cutoff is None else tf_df[tf_df["prediction_timestamp"] >= cutoff]
                block[tf][label] = _window_stats(sub)
        return block

    # Aggregate across all tracks (backward compatible top-level view).
    result["timeframes"] = _accuracy_block(df)

    # Per-track breakdown so numbers / llm_direct / llm_calibrated / blended can
    # be compared head-to-head, each with its own calibration error.
    by_model: dict[str, Any] = {}
    if "model_source" in df.columns:
        for model_source, mdf in df.groupby(df["model_source"].fillna("unknown")):
            by_model[str(model_source)] = {
                "timeframes": _accuracy_block(mdf),
                "calibration": compute_calibration_bins(mdf.to_dict("records")),
                "n_scored": int(len(mdf)),
            }
    result["by_model"] = by_model

    return result


def _empty_window_stats() -> dict[str, Any]:
    return {
        "direction_accuracy_pct": None,
        "n_scored": 0,
        "n_correct": 0,
        "mean_magnitude_error": None,
        "mean_actual_return_pct": None,
    }


def _window_stats(df: pd.DataFrame) -> dict[str, Any]:
    if df.empty:
        return _empty_window_stats()

    n = len(df)
    n_correct = int(df["direction_correct"].sum())
    return {
        "direction_accuracy_pct": round(n_correct / n * 100, 1),
        "n_scored": n,
        "n_correct": n_correct,
        "mean_magnitude_error": round(float(df["magnitude_error"].mean()), 2),
        "mean_actual_return_pct": round(float(df["actual_return_pct"].mean()), 2),
    }


def update_rolling_accuracy(path: Path | None = None) -> dict[str, Any]:
    """Recompute and persist rolling accuracy stats."""
    PERFORMANCE_DIR.mkdir(parents=True, exist_ok=True)
    rolling = compute_rolling_accuracy()
    out_path = path or ROLLING_PATH
    out_path.write_text(json.dumps(rolling, indent=2), encoding="utf-8")
    logger.info("Rolling accuracy updated -> %s", out_path)
    return rolling


def compute_calibration_bins(
    scores: list[dict[str, Any]] | None = None,
    n_bins: int = 10,
) -> dict[str, Any]:
    """Compute per-confidence-bucket accuracy for live calibration analysis.

    Buckets scored predictions into *n_bins* equal-width confidence bands and
    calculates direction accuracy per bucket plus an overall ECE (Expected
    Calibration Error).  This lets the dashboard compare live calibration
    against backtest calibration without re-running offline metrics.

    Args:
        scores: Pre-loaded score records; defaults to loading from disk.
        n_bins: Number of equal-width buckets across the 0-100 confidence range.

    Returns:
        Dict with keys ``n_total``, ``ece``, ``bins``, ``updated_at``.
    """
    records = scores if scores is not None else load_prediction_scores()

    bin_width = 100.0 / n_bins
    bins: list[dict[str, Any]] = []
    for i in range(n_bins):
        lo = i * bin_width
        hi = lo + bin_width
        in_bucket = [
            r for r in records
            if r.get("confidence") is not None and lo <= r["confidence"] < hi
        ]
        if in_bucket:
            n = len(in_bucket)
            n_correct = sum(1 for r in in_bucket if r.get("direction_correct"))
            acc = n_correct / n
            mean_conf = sum(r["confidence"] for r in in_bucket) / n
        else:
            n = 0
            n_correct = 0
            acc = None
            mean_conf = (lo + hi) / 2.0
        bins.append({
            "confidence_lo": round(lo, 1),
            "confidence_hi": round(hi, 1),
            "mean_confidence": round(mean_conf, 1),
            "accuracy_pct": round(acc * 100, 1) if acc is not None else None,
            "n": n,
            "n_correct": n_correct,
        })

    total = len(records)
    ece: float | None = None
    if total > 0:
        ece = sum(
            (b["n"] / total)
            * abs(
                (b["accuracy_pct"] or 0.0) / 100.0
                - b["mean_confidence"] / 100.0
            )
            for b in bins
            if b["n"] > 0
        )
        ece = round(ece, 4)

    return {
        "n_total": total,
        "ece": ece,
        "bins": bins,
    }


def update_calibration_live(path: Path | None = None) -> dict[str, Any]:
    """Recompute and persist live calibration bins to *calibration_live.json*."""
    PERFORMANCE_DIR.mkdir(parents=True, exist_ok=True)
    result = compute_calibration_bins()
    result["updated_at"] = pd.Timestamp.now(tz="UTC").isoformat()
    out_path = path or CALIBRATION_LIVE_PATH
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    logger.info("Live calibration bins updated -> %s", out_path)
    return result


def run_scorer() -> dict[str, Any]:
    """Score mature predictions and refresh rolling accuracy and calibration bins.

    Also feeds the append-only labeled training store so the closed-loop
    trainer (``autotrain.py``) accumulates ``(features -> realized outcome)``
    rows over time. The store update is best-effort: a failure there never
    blocks the core scoring/accuracy refresh.
    """
    new_scores = score_mature_predictions()
    rolling = update_rolling_accuracy()
    calibration = update_calibration_live()

    labeled_added = 0
    try:
        from src.training.labeled_store import update_labeled_store_from_scores

        labeled_added = update_labeled_store_from_scores()
    except Exception as exc:  # pragma: no cover - defensive, never block scoring
        logger.warning("Labeled store update skipped: %s", exc)

    return {
        "new_scores": len(new_scores),
        "rolling_accuracy": rolling,
        "calibration_ece": calibration.get("ece"),
        "labeled_rows_added": labeled_added,
    }


def main() -> int:
    result = run_scorer()
    print(f"Scored {result['new_scores']} new prediction(s)")
    for tf, windows in result["rolling_accuracy"].get("timeframes", {}).items():
        all_time = windows.get("all_time", {})
        acc = all_time.get("direction_accuracy_pct")
        n = all_time.get("n_scored", 0)
        if acc is not None:
            print(f"  {tf}: {acc:.1f}% accuracy ({n} scored)")
        else:
            print(f"  {tf}: no scored predictions yet")
    ece = result.get("calibration_ece")
    if ece is not None:
        print(f"Live ECE: {ece:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
