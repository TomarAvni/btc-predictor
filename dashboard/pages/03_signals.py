"""Page 3 — Signal Deep Dive.

Explore individual signals: history, correlation with price, effectiveness,
and feature importance.
"""

from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


import re

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Signals", page_icon="📡", layout="wide")

from dashboard.components.charts import (
    create_bar_chart,
    create_line_chart,
    create_scatter_chart,
)
from dashboard.data_loader import (
    get_prediction_history,
    get_price_data,
    load_model_metrics,
    load_validation_results,
)
from dashboard.components.mobile_nav import render_mobile_nav
from dashboard.styles import BLUE, GREEN, RED, inject_css

inject_css()
render_mobile_nav()

st.markdown("# 📡 Signal Deep Dive")

# ── Gather available signal names ─────────────────────────────────────────

runs = get_prediction_history()
price_df = get_price_data()

all_signals: set[str] = set()
for run in runs:
    all_signals.update(run.get("signals", {}).keys())

if not all_signals:
    st.warning("No signal data available yet. Run predictions first.")
    st.stop()

signal_name = st.selectbox("Select a signal", sorted(all_signals))

# ── Extract time series for the chosen signal ────────────────────────────

_NUM_RE = re.compile(r"[+\-]?\d+\.?\d*")


def _extract_numeric(val: str) -> float | None:
    m = _NUM_RE.search(val)
    return float(m.group()) if m else None


records: list[dict] = []
for run in runs:
    sig = run.get("signals", {}).get(signal_name)
    if not sig:
        continue
    num = _extract_numeric(sig.get("value", ""))
    records.append({
        "timestamp": run["timestamp"],
        "value": num,
        "raw": sig.get("value", ""),
        "interpretation": sig.get("interpretation", ""),
    })

sig_df = pd.DataFrame(records)
if not sig_df.empty:
    sig_df["timestamp"] = pd.to_datetime(sig_df["timestamp"], utc=True, errors="coerce")
    sig_df = sig_df.set_index("timestamp").sort_index()

# ── Signal history chart ─────────────────────────────────────────────────

st.markdown(f"### {signal_name} — History")

if sig_df.empty or sig_df["value"].isna().all():
    st.caption(f"No numeric data extractable for *{signal_name}*.")
else:
    chart_df = sig_df[["value"]].dropna()
    chart_df.columns = [signal_name]
    st.plotly_chart(
        create_line_chart(chart_df, title=f"{signal_name} over time", colors=[BLUE]),
        use_container_width=True,
    )

# ── Correlation with price ───────────────────────────────────────────────

st.markdown("### Correlation with BTC Price")

if not sig_df.empty and not price_df.empty and not sig_df["value"].isna().all():
    merged = sig_df[["value"]].dropna().join(price_df[["close"]], how="inner")
    if len(merged) > 5:
        st.plotly_chart(
            create_scatter_chart(
                merged["value"].tolist(),
                merged["close"].tolist(),
                title=f"{signal_name} vs BTC Price",
                x_label=signal_name,
                y_label="BTC Price ($)",
            ),
            use_container_width=True,
        )
        corr = merged["value"].corr(merged["close"])
        st.metric("Pearson Correlation", f"{corr:.3f}")
    else:
        st.caption("Not enough overlapping data points.")
else:
    st.caption("Insufficient data for correlation analysis.")

# ── Signal effectiveness ─────────────────────────────────────────────────

st.markdown("### Signal Effectiveness")

if not sig_df.empty and not price_df.empty:
    from dashboard.components.signal_badges import infer_sentiment

    effectiveness_records: list[dict] = []
    for idx, row in sig_df.iterrows():
        sentiment = infer_sentiment(row.get("raw", ""), row.get("interpretation", ""))
        ts = idx
        try:
            future_prices = price_df["close"].loc[ts:]
            if len(future_prices) > 24:
                ret_24h = (future_prices.iloc[24] - future_prices.iloc[0]) / future_prices.iloc[0] * 100
                effectiveness_records.append({
                    "sentiment": sentiment,
                    "price_went_up": ret_24h > 0,
                    "return_24h": ret_24h,
                })
        except (IndexError, KeyError):
            continue

    if effectiveness_records:
        eff_df = pd.DataFrame(effectiveness_records)
        for sent in ["bullish", "bearish", "neutral"]:
            sub = eff_df[eff_df["sentiment"] == sent]
            if sub.empty:
                continue
            correct = sub["price_went_up"].sum() if sent == "bullish" else (~sub["price_went_up"]).sum() if sent == "bearish" else len(sub)
            total = len(sub)
            avg_ret = sub["return_24h"].mean()
            st.markdown(f"**{sent.title()}** — {correct}/{total} correct ({correct/total*100:.1f}%), avg 24h return: {avg_ret:+.2f}%")
    else:
        st.caption("Not enough data to compute effectiveness.")
else:
    st.caption("Awaiting data for effectiveness analysis.")

# ── Feature importance ───────────────────────────────────────────────────

st.markdown("### Feature Importance (from model)")

metrics = load_model_metrics()
importance = metrics.get("feature_importance", {})

if not importance:
    validation = load_validation_results()
    raw = validation.get("feature_importance", [])
    if isinstance(raw, list):
        importance = {
            item["feature"]: item["importance_pct"]
            for item in raw
            if isinstance(item, dict) and "feature" in item
        }

if not importance:
    rng = np.random.default_rng(55)
    importance = {s: round(rng.uniform(0.01, 0.15), 3) for s in sorted(all_signals)}
    st.caption("Using placeholder importances — run validation to see real model features.")

sorted_imp = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True)[:15])
st.plotly_chart(
    create_bar_chart(
        list(sorted_imp.keys()),
        list(sorted_imp.values()),
        title="Top 15 Feature Importances",
        color=BLUE,
        horizontal=True,
        height=450,
    ),
    use_container_width=True,
)
