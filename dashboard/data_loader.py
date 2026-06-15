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
    """Load rolling accuracy stats from rolling_accuracy.json."""
    path = PERFORMANCE_DIR / "rolling_accuracy.json"
    if not path.exists():
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def get_live_performance_scores() -> list[dict[str, Any]]:
    """Return live prediction scores if available."""
    return load_prediction_scores()


def has_live_performance() -> bool:
    """Whether live prediction scoring data exists."""
    return (PERFORMANCE_DIR / "prediction_scores.jsonl").exists()

