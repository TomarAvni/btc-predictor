"""Page 7 - Strategy Comparison.

Diagnostic views for comparing the recorded strategy against inverse, random,
and future blended/Twitter strategy options.
"""

from __future__ import annotations

import _bootstrap  # noqa: F401 - repo root on sys.path for Streamlit Cloud

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Strategy Comparison", page_icon="BTC", layout="wide")

from dashboard.components.metrics_cards import render_metric_card
from dashboard.components.mobile_nav import render_mobile_nav
from dashboard.data_loader import get_training_status, load_trades, load_trading_journal
from dashboard.evaluation import (
    StrategySeries,
    build_strategy_series,
    trade_notional_usd,
    trade_size_btc,
)
from dashboard.styles import BLUE, GREEN, RED, YELLOW, inject_css, layout_marker

inject_css()
render_mobile_nav()


def _comparison_chart(series_map: dict[str, StrategySeries]) -> go.Figure:
    fig = go.Figure()
    colors = {
        "normal": BLUE,
        "inverse": RED,
        "random": YELLOW,
        "buy_hold": GREEN,
        "blended": "#A855F7",
    }
    max_windows = max(len(series.cumulative_pnl) for series in series_map.values())
    trade_windows = list(range(max_windows))
    ticktext = ["Start"] + [str(i) for i in trade_windows[1:]]
    for key, series in series_map.items():
        fig.add_trace(
            go.Scatter(
                x=list(range(len(series.cumulative_pnl))),
                y=series.cumulative_pnl,
                mode="lines+markers",
                name=series.name,
                line=dict(color=colors.get(key, BLUE), width=2),
                marker=dict(size=4),
            )
        )
    fig.add_hline(y=0, line_dash="dash", line_color="#8B949E", opacity=0.5)
    fig.update_layout(
        title="Cumulative P&L by Diagnostic Strategy",
        xaxis_title="Trade window (0 = start)",
        yaxis_title="P&L ($)",
        template="plotly_dark",
        height=420,
    )
    fig.update_xaxes(tickmode="array", tickvals=trade_windows, ticktext=ticktext)
    return fig


def _metrics_rows(series_map: dict[str, StrategySeries]) -> list[dict]:
    rows = []
    for key, series in series_map.items():
        m = series.metrics
        rows.append({
            "Mode": series.name,
            "Trades": m["total_trades"],
            "Total P&L": f"${m['total_pnl']:+.2f}",
            "Return": f"{m['return_pct']:+.2f}%",
            "Win Rate": f"{m['win_rate_pct']:.1f}%",
            "Profit Factor": "-" if m["profit_factor"] is None else f"{m['profit_factor']:.2f}",
            "Expectancy": f"${m['expectancy']:+.2f}",
            "Max DD": f"${m['max_drawdown_usd']:.2f}",
            "Note": series.note,
        })
    return rows


def _breakdown_frame(metric_dict: dict[str, float | int], value_name: str) -> pd.DataFrame:
    return pd.DataFrame(
        [{"Bucket": key, value_name: value} for key, value in metric_dict.items()]
    )


st.markdown("# Strategy Comparison")
st.caption(
    "Compare the recorded paper-trading decisions against diagnostics. "
    "Inverse and random modes are what-if baselines, not live trading modes."
)

trades = load_trades()
journal = load_trading_journal()
training_status = get_training_status()

if not trades:
    st.info("No closed trades available yet. Run the trading agent or a backtest first.")
    st.stop()

series_map = build_strategy_series(trades, journal)

if len(trades) < 50:
    st.warning(
        f"Only {len(trades)} closed trades are available. Treat strategy rankings "
        "as diagnostics, not proof of edge."
    )
if training_status["scored_rows"] < 100:
    st.warning(
        f"Only {training_status['scored_rows']} scored prediction rows are available. "
        "Do not tune live strategy rules from this sample yet."
    )

layout_marker("stack")
headline = st.columns(5, gap="small")
for col, key in zip(headline, ["normal", "inverse", "random", "buy_hold", "blended"]):
    series = series_map[key]
    with col:
        total = series.metrics["total_pnl"]
        render_metric_card(
            series.name,
            f"${total:+.2f}",
            f"{series.metrics['return_pct']:+.2f}%",
            "green" if total >= 0 else "red",
        )

st.plotly_chart(_comparison_chart(series_map), width="stretch")

st.markdown("### Metric Summary")
st.dataframe(pd.DataFrame(_metrics_rows(series_map)), width="stretch", hide_index=True)

tabs = st.tabs(["Main", "Inverse", "Random", "Always Long", "Twitter", "50/50 Blend"])

tab_keys = ["normal", "inverse", "random", "buy_hold"]
for tab, key in zip(tabs[:4], tab_keys):
    series = series_map[key]
    with tab:
        st.markdown(f"### {series.name}")
        if series.note:
            st.info(series.note)
        m = series.metrics
        layout_marker("stack")
        c1, c2, c3, c4 = st.columns(4, gap="small")
        with c1:
            st.metric("Total P&L", f"${m['total_pnl']:+.2f}")
        with c2:
            st.metric("Win Rate", f"{m['win_rate_pct']:.1f}%")
        with c3:
            st.metric("Profit Factor", "-" if m["profit_factor"] is None else f"{m['profit_factor']:.2f}")
        with c4:
            st.metric("Expectancy", f"${m['expectancy']:+.2f}")

        b1, b2 = st.columns(2)
        with b1:
            st.markdown("#### P&L by Side")
            st.dataframe(_breakdown_frame(m["by_side"], "P&L"), width="stretch", hide_index=True)
            st.markdown("#### P&L by Timeframe")
            st.dataframe(_breakdown_frame(m["by_timeframe"], "P&L"), width="stretch", hide_index=True)
        with b2:
            st.markdown("#### Exit Reasons")
            st.dataframe(_breakdown_frame(m["by_exit_reason"], "Count"), width="stretch", hide_index=True)
            st.markdown("#### P&L by Confidence Bucket")
            st.dataframe(_breakdown_frame(m["by_confidence"], "P&L"), width="stretch", hide_index=True)

        rows = []
        for row in reversed(series.rows[-50:]):
            entry_price = row.get("entry_price")
            rows.append({
                "Exit Time": row.get("exit_time", "-"),
                "Side": row.get("side", "-"),
                "Original Side": row.get("original_side", row.get("side", "-")),
                "Timeframe": row.get("timeframe", "-"),
                "Confidence": row.get("confidence", "-"),
                "Size (BTC)": f"{trade_size_btc(row):.6f}",
                "Notional ($)": f"${trade_notional_usd(row):.2f}",
                "Entry ($)": "-" if not entry_price else f"${float(entry_price):,.2f}",
                "P&L": f"${row.get('pnl_usd', 0):+.2f}",
                "Reason": row.get("exit_reason", "-"),
            })
        st.markdown("#### Recent Trade Windows")
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

with tabs[4]:
    st.markdown("### Twitter Module")
    st.info(
        "Placeholder for the upcoming Twitter/X sentiment predictor. Once it emits "
        "per-horizon direction, magnitude, confidence, and raw probability, this "
        "tab can show its standalone accuracy and trading replay."
    )
    st.markdown(
        """
        Expected fields:
        - `model_source`: `twitter`
        - `predictions`: same horizon schema as the numeric model
        - `features`: sentiment and event features
        - scored outcomes through the same scorer/labeled-store path
        """
    )

with tabs[5]:
    series = series_map["blended"]
    st.markdown("### 50/50 Blend")
    st.warning(
        "The real 50/50 blend should wait until the Twitter module exists. This "
        "placeholder only proves the dashboard contract and keeps the mode visible."
    )
    st.dataframe(pd.DataFrame([_metrics_rows({"blended": series})[0]]), width="stretch", hide_index=True)
    st.markdown(
        """
        Initial implementation recommendation:
        - no full new model yet
        - use a lightweight blender that averages numeric and Twitter probabilities
        - train a real meta-blender only after both modules have enough scored rows
        """
    )
