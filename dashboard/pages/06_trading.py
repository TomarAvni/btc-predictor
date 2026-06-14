"""Page 6 — Trading Agent.

Displays portfolio status, trade history, P&L chart, and open positions
from the demo trading agent.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import json
from datetime import datetime, timezone
from pathlib import Path

import streamlit as st

st.set_page_config(page_title="Trading", page_icon="💹", layout="wide")

import numpy as np
import pandas as pd

from dashboard.styles import inject_css, layout_marker
from dashboard.components.charts import create_cumulative_pnl_chart
from dashboard.components.metrics_cards import render_metric_card
from dashboard.components.mobile_nav import render_mobile_nav

inject_css()
render_mobile_nav()

TRADING_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "trading"
PORTFOLIO_PATH = TRADING_DIR / "portfolio.json"
TRADES_PATH = TRADING_DIR / "trades.json"
BACKTEST_PATH = TRADING_DIR / "backtest_results.json"


def load_portfolio() -> dict | None:
    if not PORTFOLIO_PATH.exists():
        return None
    try:
        return json.loads(PORTFOLIO_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def load_trades() -> list[dict]:
    if not TRADES_PATH.exists():
        return []
    try:
        data = json.loads(TRADES_PATH.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except (json.JSONDecodeError, OSError):
        return []


def load_backtest() -> dict | None:
    if not BACKTEST_PATH.exists():
        return None
    try:
        return json.loads(BACKTEST_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


# ── Load data ─────────────────────────────────────────────────────────────

portfolio = load_portfolio()
trades = load_trades()
backtest = load_backtest()

st.markdown("# 💹 Trading Agent")

if portfolio is None and not trades and backtest is None:
    st.info(
        "**No trading data available yet.**\n\n"
        "Run the trading agent to generate data:\n"
        "```\npython trade.py --backtest\n```\n\n"
        "Or for a single live tick:\n"
        "```\npython trade.py --live-tick\n```"
    )
    st.stop()

# ── Portfolio Overview ────────────────────────────────────────────────────

st.markdown("### Portfolio Overview")

if portfolio:
    cash = portfolio.get("cash", 2000)
    last_price = portfolio.get("last_price", 0)
    btc = portfolio.get("btc_holdings", 0)
    long_value = btc * last_price
    short_value = 0.0
    for p in portfolio.get("positions", []):
        if p.get("side", "LONG") == "SHORT":
            entry_price = p.get("entry_price", 0)
            amount_btc = p.get("amount_btc", 0)
            collateral = p.get("amount_usd", 0)
            unrealized = amount_btc * (entry_price - last_price)
            short_value += collateral + unrealized
    total_value = cash + long_value + short_value
    peak = portfolio.get("peak_value", 2000)
    pnl = total_value - 2000
    pnl_pct = (pnl / 2000) * 100
    drawdown = (peak - total_value) / peak * 100 if peak > 0 else 0

    layout_marker("stack")
    c1, c2, c3, c4 = st.columns(4, gap="small")
    with c1:
        render_metric_card(
            "Portfolio Value",
            f"${total_value:,.2f}",
            f"{pnl_pct:+.2f}%",
            "green" if pnl >= 0 else "red",
        )
    with c2:
        render_metric_card("Cash", f"${cash:,.2f}")
    with c3:
        btc = portfolio.get("btc_holdings", 0)
        render_metric_card("BTC Holdings", f"{btc:.6f}")
    with c4:
        render_metric_card(
            "Max Drawdown",
            f"{drawdown:.2f}%",
            delta_color="red" if drawdown > 5 else "green",
        )

# ── Open Positions ────────────────────────────────────────────────────────

if portfolio and portfolio.get("positions"):
    st.markdown("### Open Positions")
    positions = portfolio["positions"]
    pos_data = []
    for p in positions:
        entry_price = p.get("entry_price", 0)
        current_price = portfolio.get("last_price", entry_price)
        amount_btc = p.get("amount_btc", 0)
        side = p.get("side", "LONG")
        if side == "SHORT":
            unrealized_pnl = (entry_price - current_price) * amount_btc
            unrealized_pnl_pct = ((entry_price - current_price) / entry_price * 100) if entry_price > 0 else 0
        else:
            unrealized_pnl = (current_price - entry_price) * amount_btc
            unrealized_pnl_pct = ((current_price - entry_price) / entry_price * 100) if entry_price > 0 else 0

        pos_data.append({
            "Side": side,
            "Timeframe": p.get("timeframe", "—"),
            "Entry Price": f"${entry_price:,.2f}",
            "Amount (USD)": f"${p.get('amount_usd', 0):.2f}",
            "Unrealized P&L": f"${unrealized_pnl:.2f} ({unrealized_pnl_pct:+.1f}%)",
            "Stop Loss": f"${p.get('stop_loss', 0):,.2f}",
            "Take Profit": f"${p.get('take_profit', 0):,.2f}",
        })

    st.dataframe(pd.DataFrame(pos_data), use_container_width=True, hide_index=True)

# ── P&L Chart ─────────────────────────────────────────────────────────────

if trades:
    st.markdown("### Cumulative P&L")

    sorted_trades = sorted(trades, key=lambda t: t.get("exit_time", ""))
    cumulative_pnl = []
    running = 0.0

    for t in sorted_trades:
        running += t.get("pnl_usd", 0)
        cumulative_pnl.append(running)

    st.plotly_chart(create_cumulative_pnl_chart(cumulative_pnl), use_container_width=True)

# ── Trade History ─────────────────────────────────────────────────────────

if trades:
    st.markdown("### Trade History")

    trade_rows = []
    for t in reversed(trades[-50:]):
        pnl_usd = t.get("pnl_usd", 0)
        pnl_pct_val = t.get("pnl_pct", 0)
        trade_rows.append({
            "Exit Time": t.get("exit_time", "—")[:19] if isinstance(t.get("exit_time"), str) else "—",
            "Side": t.get("side", "LONG"),
            "Timeframe": t.get("timeframe", "—"),
            "Entry": f"${t.get('entry_price', 0):,.2f}",
            "Exit": f"${t.get('exit_price', 0):,.2f}",
            "Amount": f"${t.get('amount_usd', 0):.2f}",
            "P&L": f"${pnl_usd:+.2f} ({pnl_pct_val:+.1f}%)",
            "Reason": t.get("exit_reason", "—"),
        })

    st.dataframe(pd.DataFrame(trade_rows), use_container_width=True, hide_index=True)

    # Summary stats
    st.markdown("### Performance Summary")
    winners = sum(1 for t in trades if t.get("pnl_usd", 0) > 0)
    losers = sum(1 for t in trades if t.get("pnl_usd", 0) <= 0)
    win_rate = (winners / len(trades) * 100) if trades else 0
    total_pnl = sum(t.get("pnl_usd", 0) for t in trades)
    avg_win = np.mean([t["pnl_usd"] for t in trades if t.get("pnl_usd", 0) > 0]) if winners else 0
    avg_loss = np.mean([t["pnl_usd"] for t in trades if t.get("pnl_usd", 0) <= 0]) if losers else 0

    layout_marker("stack")
    s1, s2, s3, s4, s5 = st.columns(5, gap="small")
    with s1:
        render_metric_card("Total Trades", str(len(trades)))
    with s2:
        render_metric_card("Win Rate", f"{win_rate:.1f}%", delta_color="green" if win_rate > 50 else "red")
    with s3:
        render_metric_card("Total P&L", f"${total_pnl:+.2f}", delta_color="green" if total_pnl >= 0 else "red")
    with s4:
        render_metric_card("Avg Win", f"${avg_win:+.2f}", delta_color="green")
    with s5:
        render_metric_card("Avg Loss", f"${avg_loss:.2f}", delta_color="red")

# ── Backtest Results ──────────────────────────────────────────────────────

if backtest and isinstance(backtest, dict):
    st.markdown("### Latest Backtest Results")
    layout_marker("stack")
    bc1, bc2, bc3 = st.columns(3, gap="small")
    with bc1:
        render_metric_card("Final Value", f"${backtest.get('final_value', 0):,.2f}")
    with bc2:
        render_metric_card("Total Return", f"{backtest.get('total_return_pct', 0):+.2f}%")
    with bc3:
        render_metric_card("Sharpe Ratio", f"{backtest.get('sharpe_ratio', 0):.2f}")

# ── Coming Soon: Live Exchange ────────────────────────────────────────────

st.divider()
st.markdown(
    """
    <div style="text-align: center; padding: 1rem; opacity: 0.6;">
        <h4>🔗 Live Exchange Connection</h4>
        <p>Coming soon — connect to a real exchange for live paper trading.</p>
    </div>
    """,
    unsafe_allow_html=True,
)
