"""Score mature predictions against actual BTC price moves.

Reads predictions.log, compares each mature prediction to realized returns
from hourly price data, and maintains rolling accuracy statistics.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src import DATA_DIR, PREDICTIONS_LOG
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

TIMEFRAMES = ["24h", "7d", "30d", "90d"]
HORIZON_HOURS = {"24h": 24, "7d": 168, "30d": 720, "90d": 2160}

PERFORMANCE_DIR = DATA_DIR / "performance"
SCORES_PATH = PERFORMANCE_DIR / "prediction_scores.jsonl"
ROLLING_PATH = PERFORMANCE_DIR / "rolling_accuracy.json"
PRICE_PATH = DATA_DIR / "price" / "btc_hourly.parquet"

_PRED_LINE = re.compile(
    r"^\s*(\S+)\s*\|\s*(UP|DOWN)\s*\|\s*([+\-]?\d+\.?\d*)%\s*\|\s*Confidence:\s*(\d+)%",
    re.IGNORECASE,
)
_RUN_HEADER = re.compile(
    r"^\[(\d{4}-\d{2}-\d{2}\s\d{2}:\d{2}\sUTC)\]\s*--\s*Prediction Run #(\d+)"
)


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


def _parse_prediction_ts(ts_str: str) -> pd.Timestamp:
    dt = datetime.strptime(ts_str, "%Y-%m-%d %H:%M UTC").replace(tzinfo=timezone.utc)
    return pd.Timestamp(dt)


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
            keys.add(f"{rec.get('run_number')}:{rec.get('timeframe')}")
        except json.JSONDecodeError:
            continue
    return keys


def _is_mature(pred_ts: pd.Timestamp, timeframe: str, now: pd.Timestamp) -> bool:
    hours = HORIZON_HOURS.get(timeframe)
    if hours is None:
        return False
    return (now - pred_ts) >= pd.Timedelta(hours=hours)


def score_mature_predictions(
    predictions_path: Path | None = None,
    price_path: Path | None = None,
    scores_path: Path | None = None,
    now: pd.Timestamp | None = None,
) -> list[dict[str, Any]]:
    """Score predictions whose horizons have elapsed. Returns newly scored records."""
    PERFORMANCE_DIR.mkdir(parents=True, exist_ok=True)
    scores_file = scores_path or SCORES_PATH
    now_ts = now or pd.Timestamp.now(tz="UTC")

    runs = parse_predictions_log(predictions_path)
    if not runs:
        logger.info("No predictions to score")
        return []

    price_df = load_price_data(price_path)
    if price_df.empty:
        logger.warning("No price data available for scoring")
        return []

    already_scored = _load_scored_keys(scores_file)
    new_scores: list[dict[str, Any]] = []

    for run in runs:
        try:
            pred_ts = _parse_prediction_ts(run["timestamp"])
        except ValueError:
            continue

        for pred in run.get("predictions", []):
            tf = pred.get("timeframe")
            if tf not in HORIZON_HOURS:
                continue

            key = f"{run['run_number']}:{tf}"
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
            predicted_direction = pred["direction"].upper()
            direction_correct = actual_direction == predicted_direction

            record = {
                "scored_at": now_ts.isoformat(),
                "prediction_timestamp": pred_ts.isoformat(),
                "run_number": run["run_number"],
                "timeframe": tf,
                "predicted_direction": predicted_direction,
                "predicted_magnitude": pred["magnitude"],
                "confidence": pred["confidence"],
                "actual_return_pct": round(actual_return, 4),
                "actual_direction": actual_direction,
                "direction_correct": direction_correct,
                "magnitude_error": round(abs(pred["magnitude"]) - abs(actual_return), 4),
                "start_price": round(start_price, 2),
                "end_price": round(end_price, 2),
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

    for tf in TIMEFRAMES:
        tf_df = df[df["timeframe"] == tf]
        result["timeframes"][tf] = {}
        for label, cutoff in windows.items():
            sub = tf_df if cutoff is None else tf_df[tf_df["prediction_timestamp"] >= cutoff]
            result["timeframes"][tf][label] = _window_stats(sub)

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


def run_scorer() -> dict[str, Any]:
    """Score mature predictions and refresh rolling accuracy."""
    new_scores = score_mature_predictions()
    rolling = update_rolling_accuracy()
    return {"new_scores": len(new_scores), "rolling_accuracy": rolling}


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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
