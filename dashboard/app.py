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
from dashboard.components.price_monitor import render_price_monitor
from dashboard.components.signal_badges import render_signal_grid
from dashboard.config import AUTO_REFRESH_INTERVAL_MS, SIGNAL_CATEGORIES
from dashboard.data_loader import get_prediction_history, get_price_data, has_real_data
from dashboard.styles import inject_css, layout_marker
from src.utils.timez import now_israel_str, utc_str_to_israel

inject_css()
render_mobile_nav(show_sidebar_hint=True)

# ── Sidebar ────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ₿ BTC Predictor")
    st.caption("ML-powered price movement predictions")
    st.caption(f"🕐 {now_israel_str()} (Israel time)")
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

st.markdown("# ₿ BTC Predictor")
st.caption(
    "Experimental machine-learning forecasts for Bitcoin, with a demo "
    "(paper-money) trading agent. Not financial advice."
)

# ── How this works ─────────────────────────────────────────────────────────

with st.expander("ℹ️ How this works (start here)", expanded=False):
    st.markdown(
        """
        **The pipeline, in plain English:**

        1. **Predict** — every ~30 minutes the model looks at price action and
           ~30 market signals (on-chain, sentiment, macro, technicals, derivatives)
           and outputs a **direction** (UP/DOWN), an expected **magnitude** (%),
           and a **confidence** (%) for several horizons (6h, 12h, 24h, 7d, 30d, 90d).
        2. **Demo-trade** — if confidence clears a threshold, a **paper-trading**
           agent opens a simulated position in a **$2,000 virtual portfolio**.
           If confidence is too low it returns **SKIP** (no trade).
        3. **Score** — once a prediction's horizon matures, it's graded against
           the actual BTC move so accuracy can be tracked over time.

        **Honest caveats**
        - This is an **experiment**, *not financial advice*. No real money is involved.
        - Backtested direction accuracy is roughly **52–54%** — a small edge over a
          coin flip, not a crystal ball.
        - The **Performance** and **Signals** pages need a few weeks of live runs
          before their accuracy numbers become meaningful; until then some views
          show simulated/demo data (clearly labelled).
        """
    )

# ── Live price monitor ─────────────────────────────────────────────────────

render_price_monitor()
st.divider()

if not price_df.empty:
    last_close = price_df["close"].iloc[-1]
    # price_df is hourly; 24 rows back ≈ 24h ago (true 24h change, not 1h).
    prev_close = price_df["close"].iloc[-25] if len(price_df) > 24 else price_df["close"].iloc[0]
    change_pct = (last_close - prev_close) / prev_close * 100 if prev_close else 0.0
    change_color = "green" if change_pct >= 0 else "red"

    layout_marker("stack")
    h1, h2, h3 = st.columns([2, 1, 1], gap="small")
    with h1:
        render_metric_card("Last Candle Close", f"${last_close:,.2f}", f"{change_pct:+.2f}% (24h)", change_color)
    with h2:
        render_metric_card(
            "Last Prediction",
            utc_str_to_israel(latest["timestamp"]) if latest else "—",
        )
    with h3:
        render_metric_card("Total Runs", str(len(runs)))

# ── Prediction cards ──────────────────────────────────────────────────────

if latest:
    st.markdown("### Current Predictions")
    st.caption(
        "Each card is one **horizon**. **Direction** = the model's UP/DOWN call, "
        "**magnitude** = how big a move it expects, and **confidence** = how sure "
        "it is (higher = stronger conviction, not a guarantee)."
    )
    preds = latest.get("predictions", [])
    render_prediction_cards(preds)

    # ── Confidence gauge ──────────────────────────────────────────────────
    avg_conf = sum(p["confidence"] for p in preds) / max(len(preds), 1) if preds else 0
    st.markdown("### Overall Confidence")
    st.caption(
        "Average confidence across all horizons. Roughly: under 40% is weak "
        "(red), 40–60% is moderate (amber), above 60% is strong (green)."
    )
    st.plotly_chart(
        create_gauge_chart(avg_conf, title="Average Model Confidence"),
        width="stretch",
    )

    # ── Signal dashboard ──────────────────────────────────────────────────
    signals = latest.get("signals", {})
    if signals:
        st.markdown("### Signal Dashboard")
        st.caption(
            "The market inputs behind the forecast, grouped by theme. A green dot "
            "leans bullish, red leans bearish, amber is neutral. Hover a badge for detail."
        )
        render_signal_grid(signals, SIGNAL_CATEGORIES)

else:
    st.warning(
        "No predictions available yet. Run the predictor first:\n"
        "```\npython main.py --predict\n```"
    )

# ── Recent history ────────────────────────────────────────────────────────

st.markdown("### Recent Prediction History")
st.caption(
    "The last few prediction runs. Each row shows the UP/DOWN call, expected "
    "move and confidence per horizon, with times in Israel local time."
)
render_prediction_table(runs, max_rows=10)
