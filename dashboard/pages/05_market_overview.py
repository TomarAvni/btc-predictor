"""Page 5 — Market Overview.

Interactive BTC price chart with indicators, halving cycle overlay,
and power law corridor.
"""

from __future__ import annotations

import _bootstrap  # noqa: F401 — repo root on sys.path for Streamlit Cloud

from datetime import timedelta

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Market Overview", page_icon="📈", layout="wide")

from dashboard.components.charts import (
    CHART_HEIGHT_CANDLESTICK,
    create_candlestick_chart,
    create_line_chart,
)
from dashboard.components.mobile_nav import render_mobile_nav
from dashboard.data_loader import get_price_data
from dashboard.styles import BLUE, GREEN, RED, TEXT_DIM, YELLOW, inject_css, layout_marker

inject_css()
render_mobile_nav()

st.markdown("# 📈 Market Overview")

price_df = get_price_data()

if price_df.empty:
    st.warning(
        "No price data available. Download history first:\n"
        "```\npython main.py --download\n```"
    )
    st.stop()

# ── Timeframe selector ───────────────────────────────────────────────────

tf_map = {"1D": 24, "1W": 168, "1M": 720, "3M": 2160, "1Y": 8760, "All": len(price_df)}
tf = st.radio("Timeframe", list(tf_map.keys()), index=3, horizontal=True)
n = min(tf_map[tf], len(price_df))
df = price_df.iloc[-n:].copy()

# ── Compute indicators on the fly ────────────────────────────────────────

for span in [9, 21, 50, 200]:
    col = f"ema_{span}"
    if col not in df.columns:
        df[col] = df["close"].ewm(span=span, adjust=False).mean()

if "bb_upper" not in df.columns:
    sma20 = df["close"].rolling(20).mean()
    std20 = df["close"].rolling(20).std()
    df["bb_upper"] = sma20 + 2 * std20
    df["bb_lower"] = sma20 - 2 * std20

if "rsi_14" not in df.columns:
    delta = df["close"].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    df["rsi_14"] = 100 - 100 / (1 + rs)

if "macd" not in df.columns:
    ema12 = df["close"].ewm(span=12, adjust=False).mean()
    ema26 = df["close"].ewm(span=26, adjust=False).mean()
    df["macd"] = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()

# ── Overlay toggles ─────────────────────────────────────────────────────

st.markdown("### BTC Price Chart")
layout_marker("stack")
ov1, ov2, ov3 = st.columns(3, gap="small")
ov4, ov5, _ = st.columns(3, gap="small")
overlays: list[str] = []
with ov1:
    if st.checkbox("EMA 9/21", value=False):
        overlays.extend(["ema_9", "ema_21"])
with ov2:
    if st.checkbox("EMA 50/200", value=True):
        overlays.extend(["ema_50", "ema_200"])
with ov3:
    if st.checkbox("Bollinger Bands", value=False):
        overlays.extend(["bb_upper", "bb_lower"])
with ov4:
    show_rsi = st.checkbox("RSI", value=True)
with ov5:
    show_macd = st.checkbox("MACD", value=False)

show_volume = st.checkbox("Volume", value=True)

st.plotly_chart(
    create_candlestick_chart(
        df,
        overlays=overlays,
        show_volume=show_volume,
        show_rsi=show_rsi,
        show_macd=show_macd,
        height=CHART_HEIGHT_CANDLESTICK,
    ),
    use_container_width=True,
)

# ── Halving cycle overlay ───────────────────────────────────────────────

st.markdown("### Halving Cycle Position")

HALVING_DATES = [
    pd.Timestamp("2012-11-28", tz="UTC"),
    pd.Timestamp("2016-07-09", tz="UTC"),
    pd.Timestamp("2020-05-11", tz="UTC"),
    pd.Timestamp("2024-04-19", tz="UTC"),
]

latest_halving = HALVING_DATES[-1]
days_since = (pd.Timestamp.now(tz="UTC") - latest_halving).days
cycle_pct = days_since / 1460 * 100

layout_marker("stack")
c1, c2, c3 = st.columns(3, gap="small")
with c1:
    st.metric("Days Since Halving", f"{days_since}")
with c2:
    st.metric("Cycle Progress", f"{cycle_pct:.1f}%")
with c3:
    st.metric("Estimated Next Halving", "~April 2028")

cycle_data: dict[str, list[float]] = {}
for i, hd in enumerate(HALVING_DATES):
    cycle_end = HALVING_DATES[i + 1] if i + 1 < len(HALVING_DATES) else pd.Timestamp.now(tz="UTC")
    mask = (price_df.index >= hd) & (price_df.index < cycle_end)
    cycle_prices = price_df.loc[mask, "close"]
    if len(cycle_prices) > 0:
        normalised = (cycle_prices / cycle_prices.iloc[0]).values.tolist()
        label = f"Cycle {i + 1} ({hd.year})"
        cycle_data[label] = normalised

if cycle_data:
    max_len = max(len(v) for v in cycle_data.values())
    cycle_df = pd.DataFrame(
        {k: v + [np.nan] * (max_len - len(v)) for k, v in cycle_data.items()}
    )
    st.plotly_chart(
        create_line_chart(
            cycle_df,
            title="Normalised Price by Halving Cycle (1x = halving price)",
            colors=[BLUE, GREEN, RED, YELLOW],
        ),
        use_container_width=True,
    )

# ── Power law chart ──────────────────────────────────────────────────────

st.markdown("### Power Law Corridor")

genesis = pd.Timestamp("2009-01-03", tz="UTC")
days_arr = np.array([(t - genesis).days for t in price_df.index])
valid = days_arr > 0

if valid.sum() > 100:
    log_days = np.log10(days_arr[valid].astype(float))
    log_price = np.log10(price_df["close"].values[valid].astype(float))

    coeffs = np.polyfit(log_days, log_price, 1)
    fitted = 10 ** (coeffs[0] * log_days + coeffs[1])
    upper = fitted * 3.5
    lower = fitted / 3.5

    pl_df = pd.DataFrame(
        {"Price": price_df["close"].values[valid], "Power Law": fitted, "Upper": upper, "Lower": lower},
        index=price_df.index[valid],
    )
    st.plotly_chart(
        create_line_chart(
            pl_df,
            title="BTC Price vs Power Law Corridor",
            colors=[BLUE, YELLOW, GREEN, RED],
            height=450,
        ),
        use_container_width=True,
    )
else:
    st.caption("Not enough data for power law analysis.")
