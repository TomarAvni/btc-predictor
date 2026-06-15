"""Page 1 — Live Predictions (extended view).

Provides an expanded signal breakdown and prediction details beyond
what the home page shows.
"""

from __future__ import annotations

import _bootstrap  # noqa: F401 — repo root on sys.path for Streamlit Cloud

import streamlit as st

st.set_page_config(page_title="Live Predictions", page_icon="₿", layout="wide")

from dashboard.components.charts import create_gauge_chart, create_horizon_curve_chart
from dashboard.components.metrics_cards import render_metric_card, render_prediction_cards
from dashboard.components.mobile_nav import render_mobile_nav
from dashboard.components.prediction_table import render_prediction_table
from dashboard.components.signal_badges import infer_sentiment, render_signal_grid
from dashboard.config import AUTO_REFRESH_INTERVAL_MS, SIGNAL_CATEGORIES, SUMMARY_HORIZONS
from dashboard.data_loader import get_prediction_history, get_price_data, has_real_data
from dashboard.styles import inject_css, layout_marker
from src.utils.timez import utc_str_to_israel

inject_css()
render_mobile_nav()

# ── Auto-refresh ───────────────────────────────────────────────────────────

with st.sidebar:
    auto_refresh = st.toggle("Auto-refresh", value=False, key="live_ar")
    if auto_refresh:
        try:
            from streamlit_autorefresh import st_autorefresh
            st_autorefresh(interval=AUTO_REFRESH_INTERVAL_MS, key="live_refresh")
        except ImportError:
            pass

# ── Data ───────────────────────────────────────────────────────────────────

runs = get_prediction_history()
latest = runs[-1] if runs else None
price_df = get_price_data()

st.markdown("# 📡 Live Predictions")
st.caption(
    "A detailed view of the latest prediction run: the UP/DOWN call and "
    "confidence for each horizon, plus the market signals behind it. "
    "**Confidence** is the model's conviction (not a probability of profit), "
    "and **magnitude** is the size of the expected move."
)

if not latest:
    st.warning("No predictions available yet. Run `python main.py --predict` to get started.")
    st.stop()

# ── Price header ───────────────────────────────────────────────────────────

if not price_df.empty:
    last_close = price_df["close"].iloc[-1]
    prev_close = price_df["close"].iloc[-2] if len(price_df) > 1 else last_close
    change = (last_close - prev_close) / prev_close * 100

    layout_marker("stack")
    c1, c2, c3, c4 = st.columns(4, gap="small")
    with c1:
        render_metric_card("BTC Price", f"${last_close:,.2f}", f"{change:+.2f}%", "green" if change >= 0 else "red")
    with c2:
        render_metric_card("24h High", f"${price_df['high'].iloc[-24:].max():,.2f}")
    with c3:
        render_metric_card("24h Low", f"${price_df['low'].iloc[-24:].min():,.2f}")
    with c4:
        render_metric_card("Last Update", utc_str_to_israel(latest["timestamp"]))

# ── Prediction cards ──────────────────────────────────────────────────────

st.markdown("### Direction Forecasts")
preds = latest.get("predictions", [])
render_prediction_cards(preds)

# ── Full horizon curve ────────────────────────────────────────────────────

st.markdown("### Prediction Curve")
st.caption(
    "Expected move (bars) and confidence (line) across every horizon from 6h "
    "to 168h (7d) in 6-hour steps, plus the long-range 30d point."
)
st.plotly_chart(create_horizon_curve_chart(preds), width="stretch")

# ── Confidence gauges per headline horizon ────────────────────────────────

st.markdown("### Confidence by Horizon")
st.caption("How sure the model is for the headline horizons. Green ≥60% (strong), amber 40–60%, red <40% (weak).")
layout_marker("stack")
gcols = st.columns(len(SUMMARY_HORIZONS), gap="small")
for i, tf in enumerate(SUMMARY_HORIZONS):
    match = next((p for p in preds if p["timeframe"] == tf), None)
    with gcols[i]:
        val = match["confidence"] if match else 0
        st.plotly_chart(create_gauge_chart(val, title=tf), width="stretch")

# ── Signal breakdown ──────────────────────────────────────────────────────

signals = latest.get("signals", {})
if signals:
    st.markdown("### Signal Dashboard")
    render_signal_grid(signals, SIGNAL_CATEGORIES)

    st.markdown("### Signal Sentiment Summary")
    bullish = sum(1 for s in signals.values() if infer_sentiment(s.get("value", ""), s.get("interpretation", "")) == "bullish")
    bearish = sum(1 for s in signals.values() if infer_sentiment(s.get("value", ""), s.get("interpretation", "")) == "bearish")
    neutral = len(signals) - bullish - bearish

    layout_marker("stack")
    sc1, sc2, sc3 = st.columns(3, gap="small")
    with sc1:
        render_metric_card("Bullish Signals", str(bullish), delta_color="green")
    with sc2:
        render_metric_card("Bearish Signals", str(bearish), delta_color="red")
    with sc3:
        render_metric_card("Neutral Signals", str(neutral))

# ── Recent history ────────────────────────────────────────────────────────

st.markdown("### Prediction History")
render_prediction_table(runs, max_rows=20)
