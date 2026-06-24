"""Unified data loading for the dashboard.

Parses predictions.log, loads backtest / price / model artefacts from disk,
and generates synthetic demo data when real data is not yet available.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

from dashboard.config import (
    BACKTEST_DIR,
    DATA_DIR,
    MODELS_DIR,
    PERFORMANCE_DIR,
    PREDICTIONS_LOG,
    PRICE_DIR,
    VALIDATION_DIR,
)
from src.horizons import HORIZON_HOURS, TIMEFRAMES
from src.collectors.onchain_flows import LATEST_PATH as ONCHAIN_FLOW_LATEST_PATH
from src.collectors.onchain_flows import HISTORY_PATH as ONCHAIN_FLOW_HISTORY_PATH
from src.output.jsonl_logger import PREDICTIONS_JSONL_PATH
from src.training.closed_loop import MIN_CALIBRATION_ROWS, MIN_RETRAIN_ROWS
from src.training.labeled_store import LABELED_STORE_PATH

# ── Prediction-log parser ──────────────────────────────────────────────────


_PRED_LINE = re.compile(
    r"^\s*(\S+)\s*\|\s*(UP|DOWN)\s*\|\s*([+\-]?\d+\.?\d*)%\s*\|\s*Confidence:\s*(\d+)%",
    re.IGNORECASE,
)
_SIGNAL_LINE = re.compile(r"^\s{2}(\S[\w /&()]+?)\s*:\s*(.+?)(?:\s+--\s+(.+))?$")
_RUN_HEADER = re.compile(
    r"^\[(\d{4}-\d{2}-\d{2}\s\d{2}:\d{2}\s\S+)\]\s*--\s*Prediction Run #(\d+)"
)


def _parse_predictions_log(path: Path) -> list[dict[str, Any]]:
    """Parse the rolling predictions.log into a list of run dicts."""
    if not path.exists():
        return []

    text = path.read_text(encoding="utf-8", errors="replace")
    runs: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    section: str = ""

    for line in text.splitlines():
        hdr = _RUN_HEADER.search(line)
        if hdr:
            if current:
                runs.append(current)
            current = {
                "timestamp": hdr.group(1),
                "run_number": int(hdr.group(2)),
                "predictions": [],
                "signals": {},
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
            section = "signal"
            continue

        if section == "pred":
            m = _PRED_LINE.match(line)
            if m:
                current["predictions"].append(
                    {
                        "timeframe": m.group(1),
                        "direction": m.group(2).upper(),
                        "magnitude": float(m.group(3)),
                        "confidence": int(m.group(4)),
                    }
                )

        elif section == "signal":
            m = _SIGNAL_LINE.match(line)
            if m:
                current["signals"][m.group(1).strip()] = {
                    "value": m.group(2).strip(),
                    "interpretation": (m.group(3) or "").strip(),
                }

    if current:
        runs.append(current)
    return runs


# ── Public loaders (cached) ────────────────────────────────────────────────


@st.cache_data(ttl=60)
def load_prediction_runs() -> list[dict[str, Any]]:
    return _parse_predictions_log(PREDICTIONS_LOG)


@st.cache_data(ttl=60)
def load_latest_prediction() -> dict[str, Any] | None:
    runs = load_prediction_runs()
    return runs[-1] if runs else None


@st.cache_data(ttl=300)
def load_price_data() -> pd.DataFrame:
    """Load OHLCV price data from Parquet."""
    parquet_files = sorted(PRICE_DIR.glob("*.parquet")) if PRICE_DIR.exists() else []
    if not parquet_files:
        return pd.DataFrame()
    try:
        df = pd.read_parquet(parquet_files[-1])
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            df = df.set_index("timestamp").sort_index()
        elif not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index, utc=True)
            df = df.sort_index()
        return df
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=300)
def load_backtest_results() -> pd.DataFrame:
    """Load backtest results from JSON or Parquet."""
    for ext in ("parquet", "json"):
        candidates = sorted(BACKTEST_DIR.glob(f"*.{ext}")) if BACKTEST_DIR.exists() else []
        if not candidates:
            continue
        try:
            if ext == "parquet":
                return pd.read_parquet(candidates[-1])
            with open(candidates[-1], encoding="utf-8") as f:
                return pd.DataFrame(json.load(f))
        except Exception:
            continue
    return pd.DataFrame()


@st.cache_data(ttl=600)
def load_model_metrics() -> dict[str, Any]:
    """Load model training metrics / feature importances."""
    validation = load_validation_results()
    if validation.get("feature_importance"):
        importance_list = validation["feature_importance"]
        if isinstance(importance_list, list):
            importance = {
                item["feature"]: item["importance_pct"]
                for item in importance_list
                if isinstance(item, dict) and "feature" in item
            }
            return {**validation, "feature_importance": importance}

    if not MODELS_DIR.exists():
        return {}
    for p in sorted(MODELS_DIR.glob("*.json")):
        try:
            with open(p, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            continue
    return {}


# ── Synthetic demo data ────────────────────────────────────────────────────
# Used when the real pipeline hasn't produced artefacts yet.


def _demo_price_series(days: int = 365) -> pd.DataFrame:
    """Generate a realistic-looking BTC OHLCV DataFrame for demo purposes."""
    rng = np.random.default_rng(42)
    n = days * 24
    timestamps = pd.date_range(
        end=datetime.now(timezone.utc), periods=n, freq="h", tz="UTC"
    )
    base = 30_000
    log_returns = rng.normal(0.00003, 0.012, n)
    prices = base * np.exp(np.cumsum(log_returns))
    spread = prices * rng.uniform(0.002, 0.008, n)
    volume = rng.lognormal(10, 1.0, n)

    return pd.DataFrame(
        {
            "open": prices - spread / 2,
            "high": prices + spread,
            "low": prices - spread,
            "close": prices + spread / 4,
            "volume": volume,
        },
        index=timestamps,
    )


def _demo_backtest(days: int = 365) -> pd.DataFrame:
    """Synthetic walk-forward backtest results."""
    rng = np.random.default_rng(7)
    periods = days // 30
    starts = pd.date_range(
        end=datetime.now(timezone.utc), periods=periods, freq="30D", tz="UTC"
    )
    regimes = rng.choice(
        ["bull_steady", "bear_steady", "sideways", "bull_volatile", "bear_volatile"],
        periods,
    )
    return pd.DataFrame(
        {
            "period_start": starts,
            "period_end": starts + timedelta(days=30),
            "direction_accuracy": rng.uniform(0.48, 0.72, periods),
            "mae": rng.uniform(0.5, 3.0, periods),
            "regime": regimes,
            "avg_actual_return": rng.normal(0.5, 3.0, periods),
            "avg_predicted_return": rng.normal(0.4, 2.5, periods),
            "n_samples": rng.integers(600, 800, periods),
        }
    )


def _demo_predictions(n: int = 50) -> list[dict[str, Any]]:
    """Generate synthetic historical prediction runs."""
    rng = np.random.default_rng(99)
    runs: list[dict[str, Any]] = []
    base_ts = datetime.now(timezone.utc) - timedelta(days=n)

    for i in range(n):
        ts = base_ts + timedelta(days=i)
        preds = []
        # Coherent demo direction per run, with magnitude/confidence scaled
        # along the full horizon curve (longer horizon -> bigger move, lower
        # confidence) so the curve chart renders sensibly in demo mode.
        run_bias = rng.uniform(0.35, 0.65)
        for tf in TIMEFRAMES:
            hrs = HORIZON_HOURS[tf]
            mag_s = 0.6 + 0.35 * hrs ** 0.6
            conf_b = int(round(max(35, 66 - 3.0 * np.log2(max(hrs, 1) / 6))))
            d = "UP" if rng.uniform(0, 1) < run_bias else "DOWN"
            preds.append(
                {
                    "timeframe": tf,
                    "direction": d,
                    "magnitude": round(rng.uniform(0.3, mag_s), 1),
                    "confidence": int(rng.integers(conf_b - 12, conf_b + 12)),
                }
            )

        signals = {
            "Halving Cycle": {"value": f"Day {700 + i} of ~1460 ({int(48 + i * 0.07)}%)", "interpretation": "historically bullish zone"},
            "Power Law": {"value": f"Price at {rng.integers(55, 80)}% of corridor", "interpretation": "room above"},
            "Fear & Greed": {"value": str(rng.integers(25, 85)), "interpretation": ""},
            "RSI (4h)": {"value": f"{rng.integers(30, 75)}", "interpretation": ""},
            "Funding Rate": {"value": f"{rng.uniform(-0.02, 0.05):.3f}%", "interpretation": ""},
            "BTC Dominance": {"value": f"{rng.uniform(52, 62):.1f}%", "interpretation": ""},
            "ETF Flows": {"value": f"+${rng.integers(-200, 500)}M (7d)", "interpretation": ""},
            "MACD (daily)": {"value": rng.choice(["Bullish crossover", "Bearish crossover"]), "interpretation": ""},
            "Volume": {"value": f"{rng.integers(-15, 30)}% vs 20d avg", "interpretation": ""},
            "DXY": {"value": f"{rng.choice(['+', '-'])}{rng.uniform(0.1, 0.8):.1f}% this week", "interpretation": ""},
        }

        runs.append(
            {
                "timestamp": ts.strftime("%Y-%m-%d %H:%M UTC"),
                "run_number": i + 1,
                "predictions": preds,
                "signals": signals,
            }
        )
    return runs


def get_price_data() -> pd.DataFrame:
    """Return price data, falling back to demo data."""
    df = load_price_data()
    if df.empty:
        df = _demo_price_series()
    return df


def get_prediction_history() -> list[dict[str, Any]]:
    """Return prediction runs, falling back to demo data."""
    runs = load_prediction_runs()
    if not runs:
        runs = _demo_predictions()
    return runs


def get_backtest_results() -> pd.DataFrame:
    """Return backtest results, falling back to demo data."""
    df = load_backtest_results()
    if df.empty:
        df = _demo_backtest()
    return df


def has_real_data() -> bool:
    """Check whether any real pipeline data exists on disk."""
    return (
        PREDICTIONS_LOG.exists()
        or (PRICE_DIR.exists() and any(PRICE_DIR.glob("*.parquet")))
        or (BACKTEST_DIR.exists() and any(BACKTEST_DIR.iterdir()))
        or has_live_performance()
    )


# ── Validation results loader ──────────────────────────────────────────────


@st.cache_data(ttl=300)
def load_validation_results() -> dict[str, Any]:
    """Load validation results from data/validation/results.json."""
    results_path = VALIDATION_DIR / "results.json"
    if not results_path.exists():
        return {}
    try:
        with open(results_path, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


@st.cache_data(ttl=300)
def load_validation_equity_curve() -> list[dict[str, Any]]:
    """Load validation equity curve from data/validation/equity_curve.json."""
    path = VALIDATION_DIR / "equity_curve.json"
    if not path.exists():
        return []
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return []


def get_validation_results() -> dict[str, Any]:
    """Return validation results if available."""
    return load_validation_results()


# ── Live performance tracking ─────────────────────────────────────────────


@st.cache_data(ttl=60)
def load_prediction_scores() -> list[dict[str, Any]]:
    """Load scored live predictions from prediction_scores.jsonl."""
    path = PERFORMANCE_DIR / "prediction_scores.jsonl"
    if not path.exists():
        return []

    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return records


@st.cache_data(ttl=60)
def load_rolling_accuracy() -> dict[str, Any]:
    """Load rolling accuracy stats from rolling_accuracy.json.

    When the JSON snapshot is empty but scored rows exist on disk (for example
    after a branch merge dropped ``prediction_scores.jsonl`` but left a zeroed
    rolling file), recompute from the score log so the dashboard stays fresh.
    """
    path = PERFORMANCE_DIR / "rolling_accuracy.json"
    scores = load_prediction_scores()
    if path.exists():
        try:
            with open(path, encoding="utf-8") as f:
                rolling = json.load(f)
        except (json.JSONDecodeError, OSError):
            rolling = {}
    else:
        rolling = {}

    if scores and _rolling_accuracy_is_empty(rolling):
        from src.engine.scorer import compute_rolling_accuracy

        return compute_rolling_accuracy(scores)
    return rolling


def _rolling_accuracy_is_empty(rolling: dict[str, Any]) -> bool:
    """True when rolling stats contain no scored predictions."""
    timeframes = rolling.get("timeframes")
    if not isinstance(timeframes, dict):
        return True
    for windows in timeframes.values():
        if not isinstance(windows, dict):
            continue
        all_time = windows.get("all_time") or {}
        if all_time.get("n_scored", 0):
            return False
    return True


def get_live_performance_scores() -> list[dict[str, Any]]:
    """Return live prediction scores if available."""
    return load_prediction_scores()


def has_live_performance() -> bool:
    """Whether live prediction scoring data exists."""
    return (PERFORMANCE_DIR / "prediction_scores.jsonl").exists()


# ── Trading / training status loaders ──────────────────────────────────────

TRADING_DIR = DATA_DIR / "trading"
PORTFOLIO_PATH = TRADING_DIR / "portfolio.json"
TRADES_PATH = TRADING_DIR / "trades.json"
JOURNAL_PATH = TRADING_DIR / "journal.json"
BACKTEST_TRADING_PATH = TRADING_DIR / "backtest_results.json"


def _load_json(path: Path, fallback: Any) -> Any:
    if not path.exists():
        return fallback
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return fallback


@st.cache_data(ttl=60)
def load_portfolio_state() -> dict[str, Any] | None:
    data = _load_json(PORTFOLIO_PATH, None)
    return data if isinstance(data, dict) else None


@st.cache_data(ttl=60)
def load_trades() -> list[dict[str, Any]]:
    data = _load_json(TRADES_PATH, [])
    return data if isinstance(data, list) else []


@st.cache_data(ttl=60)
def load_trading_journal() -> list[dict[str, Any]]:
    data = _load_json(JOURNAL_PATH, [])
    return data if isinstance(data, list) else []


@st.cache_data(ttl=60)
def load_trading_backtest() -> dict[str, Any] | None:
    data = _load_json(BACKTEST_TRADING_PATH, None)
    return data if isinstance(data, dict) else None


@st.cache_data(ttl=60)
def load_prediction_jsonl_runs() -> list[dict[str, Any]]:
    """Load rich prediction records with feature vectors, if available."""
    path = PREDICTIONS_JSONL_PATH
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return records


@st.cache_data(ttl=60)
def load_labeled_training_rows() -> list[dict[str, Any]]:
    if not LABELED_STORE_PATH.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in LABELED_STORE_PATH.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def _latest_timestamp(records: list[dict[str, Any]], keys: tuple[str, ...]) -> str | None:
    values: list[str] = []
    for rec in records:
        for key in keys:
            value = rec.get(key)
            if value:
                values.append(str(value))
                break
    return max(values) if values else None


def _count_scored_by_horizon(scores: list[dict[str, Any]]) -> dict[str, int]:
    counts = {tf: 0 for tf in TIMEFRAMES}
    for rec in scores:
        tf = rec.get("timeframe")
        if tf in counts:
            counts[tf] += 1
    return counts


def get_training_status() -> dict[str, Any]:
    """Return prediction -> score -> labeled-store progress for the dashboard."""
    prediction_runs = load_prediction_jsonl_runs()
    scores = load_prediction_scores()
    labeled_rows = load_labeled_training_rows()
    labeled_counts = {tf: 0 for tf in TIMEFRAMES}
    for row in labeled_rows:
        tf = row.get("timeframe")
        if tf in labeled_counts:
            labeled_counts[tf] += 1

    return {
        "prediction_jsonl_rows": len(prediction_runs),
        "scored_rows": len(scores),
        "labeled_rows": len(labeled_rows),
        "latest_prediction_jsonl": _latest_timestamp(prediction_runs, ("timestamp",)),
        "latest_score": _latest_timestamp(scores, ("scored_at", "prediction_timestamp")),
        "scored_by_horizon": _count_scored_by_horizon(scores),
        "labeled_by_horizon": labeled_counts,
        "calibration_min_rows": MIN_CALIBRATION_ROWS,
        "retrain_min_rows": MIN_RETRAIN_ROWS,
        "calibration_ready": {
            tf: n >= MIN_CALIBRATION_ROWS for tf, n in labeled_counts.items()
        },
        "retrain_ready": {
            tf: n >= MIN_RETRAIN_ROWS for tf, n in labeled_counts.items()
        },
    }


def get_data_health() -> dict[str, Any]:
    """Summarize artifact freshness and obvious sync problems."""
    runs = load_prediction_runs()
    scores = load_prediction_scores()
    portfolio = load_portfolio_state()
    trades = load_trades()
    journal = load_trading_journal()

    latest_prediction = runs[-1] if runs else None
    latest_trade_exit = _latest_timestamp(trades, ("exit_time",))
    latest_journal = journal[-1] if journal else None
    portfolio_updated = portfolio.get("updated_at") if portfolio else None
    latest_score = _latest_timestamp(scores, ("scored_at", "prediction_timestamp"))

    warnings: list[str] = []
    if not scores:
        warnings.append("No scored predictions yet; performance charts should not be treated as live accuracy.")
    elif _rolling_accuracy_is_empty(load_rolling_accuracy()) and scores:
        warnings.append(
            "Rolling accuracy snapshot is empty but scored rows exist; "
            "charts recompute from prediction_scores.jsonl."
        )
    if latest_journal and portfolio_updated and str(latest_journal.get("timestamp")) > str(portfolio_updated):
        warnings.append("Trading journal is newer than portfolio state; portfolio view may be stale.")
    if latest_journal and latest_trade_exit and str(latest_journal.get("timestamp")) > str(latest_trade_exit):
        action = latest_journal.get("action")
        if action != "SKIP":
            warnings.append("Latest journal action is newer than closed-trade history.")

    return {
        "latest_prediction_run": latest_prediction.get("run_number") if latest_prediction else None,
        "latest_prediction_time": latest_prediction.get("timestamp") if latest_prediction else None,
        "prediction_runs": len(runs),
        "scored_predictions": len(scores),
        "latest_score_time": latest_score,
        "portfolio_updated_at": portfolio_updated,
        "closed_trades": len(trades),
        "latest_trade_exit": latest_trade_exit,
        "journal_entries": len(journal),
        "latest_journal_action": latest_journal.get("action") if latest_journal else None,
        "latest_journal_time": latest_journal.get("timestamp") if latest_journal else None,
        "warnings": warnings,
    }


@st.cache_data(ttl=300)
def load_onchain_flow_latest() -> dict[str, Any]:
    """Load latest on-chain flow snapshot for dashboard display."""
    data = _load_json(ONCHAIN_FLOW_LATEST_PATH, {})
    return data if isinstance(data, dict) else {}


@st.cache_data(ttl=300)
def load_onchain_flow_history() -> pd.DataFrame:
    """Load persisted on-chain flow history from the collector."""
    if not ONCHAIN_FLOW_HISTORY_PATH.exists():
        return pd.DataFrame()
    try:
        df = pd.read_parquet(ONCHAIN_FLOW_HISTORY_PATH)
        if not isinstance(df.index, pd.DatetimeIndex):
            if "timestamp" in df.columns:
                df = df.set_index("timestamp")
            df.index = pd.to_datetime(df.index, utc=True)
        return df.sort_index()
    except Exception:
        return pd.DataFrame()

