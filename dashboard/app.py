"""BTC Prediction Dashboard — main entry point.

Launch:
    streamlit run dashboard/app.py
"""

from __future__ import annotations

import _bootstrap  # noqa: F401 — repo root on sys.path for Streamlit Cloud

import streamlit as st

st.set_page_config(
    page_title="BTC Predictor",
    page_icon="₿",
    layout="wide",
    initial_sidebar_state="auto",
)

from dashboard.components.charts import create_gauge_chart
from dashboard.components.metrics_cards import render_metric_card, render_prediction_cards
from dashboard.components.mobile_nav import render_mobile_nav
from dashboard.components.prediction_table import render_prediction_table
from dashboard.components.signal_badges import render_signal_grid
from dashboard.config import AUTO_REFRESH_INTERVAL_MS, SIGNAL_CATEGORIES
from dashboard.data_loader import get_prediction_history, get_price_data, has_real_data
from dashboard.styles import inject_css, layout_marker

inject_css()
render_mobile_nav(show_sidebar_hint=True)

# ── Sidebar ────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ₿ BTC Predictor")
    st.caption("ML-powered price movement predictions")
    st.divider()

    auto_refresh = st.toggle("Auto-refresh (60 s)", value=False)
    if auto_refresh:
        try:
            from streamlit_autorefresh import st_autorefresh
            st_autorefresh(interval=AUTO_REFRESH_INTERVAL_MS, key="home_refresh")
        except ImportError:
            st.warning("Install `streamlit-autorefresh` for auto-refresh.")

    st.divider()
    if not has_real_data():
        st.info(
            "📊 **Using demo data**\n\n"
            "Run the predictor to see live data:\n"
            "```\npython main.py --predict\n```"
        )
    else:
        st.success("Connected to live data")


# ── Data ───────────────────────────────────────────────────────────────────

runs = get_prediction_history()
latest = runs[-1] if runs else None
price_df = get_price_data()

# ── Header: current price + last prediction time ──────────────────────────

st.markdown("# Live Predictions")

if not price_df.empty:
    last_close = price_df["close"].iloc[-1]
    prev_close = price_df["close"].iloc[-2] if len(price_df) > 1 else last_close
    change_pct = (last_close - prev_close) / prev_close * 100
    change_color = "green" if change_pct >= 0 else "red"

    layout_marker("stack")
    h1, h2, h3 = st.columns([2, 1, 1], gap="small")
    with h1:
        render_metric_card("BTC Price", f"${last_close:,.2f}", f"{change_pct:+.2f}% (24h)", change_color)
    with h2:
        render_metric_card("Last Prediction", latest["timestamp"] if latest else "—")
    with h3:
        render_metric_card("Total Runs", str(len(runs)))

# ── Prediction cards ──────────────────────────────────────────────────────

if latest:
    st.markdown("### Current Predictions")
    preds = latest.get("predictions", [])
    render_prediction_cards(preds)

    # ── Confidence gauge ──────────────────────────────────────────────────
    avg_conf = sum(p["confidence"] for p in preds) / max(len(preds), 1) if preds else 0
    st.markdown("### Overall Confidence")
    st.plotly_chart(
        create_gauge_chart(avg_conf, title="Average Model Confidence"),
        use_container_width=True,
    )

    # ── Signal dashboard ──────────────────────────────────────────────────
    signals = latest.get("signals", {})
    if signals:
        st.markdown("### Signal Dashboard")
        render_signal_grid(signals, SIGNAL_CATEGORIES)

else:
    st.warning(
        "No predictions available yet. Run the predictor first:\n"
        "```\npython main.py --predict\n```"
    )

# ── Recent history ────────────────────────────────────────────────────────

st.markdown("### Recent Prediction History")
render_prediction_table(runs, max_rows=10)
