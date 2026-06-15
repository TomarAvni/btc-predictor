"""Page 2 — Performance Tracking.

Shows how well the model has been performing: accuracy, calibration,
simulated P&L, and regime-based breakdowns.
"""

from __future__ import annotations

import _bootstrap  # noqa: F401 — repo root on sys.path for Streamlit Cloud

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Performance", page_icon="📊", layout="wide")

from dashboard.components.charts import (
    create_bar_chart,
    create_calibration_curve,
    create_equity_curve,
    create_line_chart,
)
from dashboard.components.metrics_cards import render_performance_card
from dashboard.components.mobile_nav import render_mobile_nav
from dashboard.data_loader import (
    get_backtest_results,
    get_live_performance_scores,
    get_prediction_history,
    has_live_performance,
    has_real_data,
    load_rolling_accuracy,
)
from dashboard.styles import BLUE, GREEN, RED, YELLOW, inject_css, layout_marker

inject_css()
render_mobile_nav()

st.markdown("# 📊 Performance Tracking")
st.caption("How well the predictions have actually held up over time.")

with st.expander("ℹ️ How to read this page"):
    st.markdown(
        """
        - **Direction accuracy** — how often the UP/DOWN call was right. 50% is a
          coin flip; the model aims for a small, consistent edge (~52–54%).
        - **Confidence calibration** — whether confidence means what it says. If
          the model is "70% confident", those calls should be right ~70% of the
          time. Points near the diagonal line = well calibrated.
        - **Cumulative P&L (simulated)** — a what-if of following every signal; it
          illustrates the strategy, it is not the live demo portfolio (see the
          **Trading** page for that).
        - **Maturing data:** predictions are only scored once their horizon
          completes (a 90-day call takes 90 days). Until enough have matured,
          some sections show **simulated/demo** numbers, clearly labelled.
        """
    )

live_scores = get_live_performance_scores()
rolling = load_rolling_accuracy()

if has_live_performance():
    st.success("Showing live prediction accuracy from scored predictions.")
elif not has_real_data():
    st.info(
        "Showing simulated backtest data. "
        "Once the predictor has run for a while, this page will blend live results."
    )

# ── Helpers ────────────────────────────────────────────────────────────────

def _simulate_performance(runs: list[dict], horizon: str, rng: np.random.Generator) -> pd.DataFrame:
    """Create a per-run accuracy DataFrame using synthetic outcome data."""
    records = []
    for run in runs:
        pred = next((p for p in run.get("predictions", []) if p["timeframe"] == horizon), None)
        if not pred:
            continue
        correct = rng.random() < (pred["confidence"] / 100 * 0.85 + 0.1)
        records.append({
            "timestamp": run["timestamp"],
            "direction": pred["direction"],
            "magnitude": pred["magnitude"],
            "confidence": pred["confidence"],
            "correct": correct,
        })
    return pd.DataFrame(records)


# ── Data ───────────────────────────────────────────────────────────────────

runs = get_prediction_history()
bt = get_backtest_results()
rng = np.random.default_rng(42)

horizons = ["6h", "12h", "24h", "7d", "30d", "90d"]
perf: dict[str, pd.DataFrame] = {}

if live_scores:
    live_df = pd.DataFrame(live_scores)
    live_df["timestamp"] = pd.to_datetime(live_df["prediction_timestamp"], utc=True, errors="coerce")
    for h in horizons:
        sub = live_df[live_df["timeframe"] == h].copy()
        if sub.empty:
            perf[h] = pd.DataFrame()
            continue
        perf[h] = pd.DataFrame({
            "timestamp": sub["timestamp"].dt.strftime("%Y-%m-%d %H:%M UTC"),
            "direction": sub["predicted_direction"],
            "magnitude": sub["predicted_magnitude"],
            "confidence": sub["confidence"],
            "correct": sub["direction_correct"],
            "actual_return": sub["actual_return_pct"],
        })
else:
    for h in horizons:
        perf[h] = _simulate_performance(runs, h, rng)

# ── Live prediction accuracy ──────────────────────────────────────────────

if rolling.get("timeframes"):
    st.markdown("### Live Prediction Accuracy")
    st.caption("Direction accuracy from mature predictions scored against actual BTC moves.")

    layout_marker("stack")
    roll_cols = st.columns(len(horizons), gap="small")
    for col, h in zip(roll_cols, horizons):
        tf_stats = rolling["timeframes"].get(h, {})
        with col:
            st.markdown(f"**{h}**")
            for label, key in [("7d", "last_7d"), ("30d", "last_30d"), ("All", "all_time")]:
                window = tf_stats.get(key, {})
                acc = window.get("direction_accuracy_pct")
                n = window.get("n_scored", 0)
                if acc is not None:
                    st.metric(f"{label}", f"{acc:.1f}%", help=f"{n} scored predictions")
                else:
                    st.metric(f"{label}", "—", help="No scored predictions yet")

# ── Summary cards ─────────────────────────────────────────────────────────

st.markdown("### Summary")

def _summary_for(df: pd.DataFrame, days: int | None = None, label: str = "All Time"):
    if df.empty:
        return label, 0.0, 0, 0, "", ""
    sub = df
    if days:
        try:
            cutoff = datetime.now(timezone.utc) - timedelta(days=days)
            sub = df[pd.to_datetime(df["timestamp"]) >= cutoff]
        except Exception:
            sub = df.tail(days)
    if sub.empty:
        return label, 0.0, 0, 0, "", ""
    acc = sub["correct"].mean() * 100
    correct = int(sub["correct"].sum())
    total = len(sub)
    best_idx = sub["confidence"].idxmax()
    worst_idx = sub["confidence"].idxmin()
    best = f'{sub.loc[best_idx, "direction"]} {sub.loc[best_idx, "magnitude"]:+.1f}%' if best_idx is not None else ""
    worst = f'{sub.loc[worst_idx, "direction"]} {sub.loc[worst_idx, "magnitude"]:+.1f}%' if worst_idx is not None else ""
    return label, acc, correct, total, best, worst

tab_horizon = st.selectbox("Horizon for summary", horizons, index=0)
df_h = perf.get(tab_horizon, pd.DataFrame())

periods = [(7, "Last 7 Days"), (30, "Last 30 Days"), (365, "Last 365 Days"), (None, "All Time")]
layout_marker("grid-2")
cols = st.columns(4, gap="small")
for col, (days, label) in zip(cols, periods):
    with col:
        lbl, acc, cor, tot, best, worst = _summary_for(df_h, days, label)
        render_performance_card(lbl, acc, cor, tot, best, worst)

# ── Rolling accuracy chart ────────────────────────────────────────────────

st.markdown("### Prediction Accuracy Over Time")

rolling_data: dict[str, list[float]] = {}
for h in horizons:
    df = perf[h]
    if df.empty:
        continue
    rolling = df["correct"].rolling(window=10, min_periods=3).mean() * 100
    rolling_data[h] = rolling.tolist()

if rolling_data:
    max_len = max(len(v) for v in rolling_data.values())
    roll_df = pd.DataFrame({k: v + [np.nan] * (max_len - len(v)) for k, v in rolling_data.items()})
    roll_df.index = pd.RangeIndex(max_len)
    st.plotly_chart(
        create_line_chart(roll_df, title="Rolling 10-Run Accuracy", colors=[BLUE, GREEN, RED, YELLOW]),
        width="stretch",
    )

# ── Calibration curve ────────────────────────────────────────────────────

st.markdown("### Confidence Calibration")

all_preds = pd.concat(perf.values(), ignore_index=True)
if not all_preds.empty:
    bins = list(range(20, 81, 10))
    predicted: list[float] = []
    actual: list[float] = []
    for lo, hi in zip(bins, bins[1:] + [100]):
        mask = (all_preds["confidence"] >= lo) & (all_preds["confidence"] < hi)
        bucket = all_preds[mask]
        if len(bucket) > 2:
            predicted.append((lo + hi) / 2)
            actual.append(bucket["correct"].mean() * 100)
    if predicted:
        st.plotly_chart(create_calibration_curve(predicted, actual), width="stretch")
    else:
        st.caption("Not enough data for calibration curve.")

# ── Cumulative P&L simulation ────────────────────────────────────────────

st.markdown("### Cumulative P&L Simulation")
st.caption("Simulated returns from following every prediction with $1,000.")

equity: dict[str, list[float]] = {}
for h in horizons:
    df = perf[h]
    if df.empty:
        continue
    portfolio = [1000.0]
    for _, row in df.iterrows():
        ret = row["magnitude"] / 100
        if not row["correct"]:
            ret = -ret
        portfolio.append(portfolio[-1] * (1 + ret))
    equity[h] = portfolio

if equity:
    st.plotly_chart(create_equity_curve(equity), width="stretch")

# ── Performance by regime ────────────────────────────────────────────────

st.markdown("### Performance by Market Regime")

if not bt.empty and "regime" in bt.columns and "direction_accuracy" in bt.columns:
    regime_acc = bt.groupby("regime")["direction_accuracy"].mean() * 100
    st.plotly_chart(
        create_bar_chart(
            regime_acc.index.tolist(),
            regime_acc.values.tolist(),
            title="Average Accuracy by Regime",
            color=BLUE,
        ),
        width="stretch",
    )

# ── Monthly performance table ─────────────────────────────────────────────

st.markdown("### Monthly Performance")

if not all_preds.empty:
    try:
        all_preds["month"] = pd.to_datetime(all_preds["timestamp"]).dt.to_period("M").astype(str)
    except Exception:
        all_preds["month"] = "unknown"

    monthly = (
        all_preds.groupby("month")
        .agg(
            accuracy=("correct", lambda x: f"{x.mean() * 100:.1f}%"),
            predictions=("correct", "count"),
            avg_confidence=("confidence", lambda x: f"{x.mean():.0f}%"),
        )
        .reset_index()
    )
    st.dataframe(monthly, width="stretch", hide_index=True)

# ── Win / loss streak ────────────────────────────────────────────────────

st.markdown("### Win / Loss Streaks")

if not all_preds.empty:
    streaks: list[int] = []
    current_streak = 0
    for val in all_preds["correct"]:
        if val:
            current_streak = max(current_streak + 1, 1) if current_streak >= 0 else 1
        else:
            current_streak = min(current_streak - 1, -1) if current_streak <= 0 else -1
        streaks.append(current_streak)

    max_win = max(streaks) if streaks else 0
    max_loss = abs(min(streaks)) if streaks else 0
    curr = streaks[-1] if streaks else 0

    layout_marker("stack")
    c1, c2, c3 = st.columns(3, gap="small")
    with c1:
        st.metric("Current Streak", f"{'W' if curr > 0 else 'L'}{abs(curr)}" if curr != 0 else "—")
    with c2:
        st.metric("Longest Win Streak", f"W{max_win}")
    with c3:
        st.metric("Longest Loss Streak", f"L{max_loss}")
