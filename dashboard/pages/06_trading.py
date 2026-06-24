"""Page 6 — Trading Agent.

Displays portfolio status, trade history, P&L chart, and open positions
from the demo trading agent.
"""

from __future__ import annotations

import _bootstrap  # noqa: F401 — repo root on sys.path for Streamlit Cloud

import streamlit as st

st.set_page_config(page_title="Trading", page_icon="💹", layout="wide")

import numpy as np
import pandas as pd

from dashboard.styles import inject_css, layout_marker
from dashboard.components.charts import create_cumulative_pnl_chart
from dashboard.components.metrics_cards import render_metric_card
from dashboard.components.mobile_nav import render_mobile_nav
from dashboard.data_loader import (
    get_data_health,
    get_trading_activity_summary,
    load_portfolio_state,
    load_trades,
    load_trades_for_analytics,
    load_trading_backtest,
    load_trading_journal,
)
from src.utils.timez import utc_str_to_israel
from src.trading.paper_profile import PROFILE_LABEL, get_paper_profile_summary

inject_css()
render_mobile_nav()

# ── Load data ─────────────────────────────────────────────────────────────

portfolio = load_portfolio_state()
all_trades = load_trades()
analytics_bundle = load_trades_for_analytics()
analytics_trades = analytics_bundle["trades"]
excluded_trades = analytics_bundle["excluded_trades"]
backtest = load_trading_backtest()
journal = load_trading_journal()
health = get_data_health()
activity = get_trading_activity_summary(trades=all_trades, journal=journal, portfolio=portfolio)

st.markdown("# 💹 Trading Agent")
profile = get_paper_profile_summary()
st.info(
    f"**{PROFILE_LABEL}** — simulated trades only, no real funds. "
    f"Entry ≥ **{profile['min_confidence_pct']:.0f}%** confidence, "
    f"up to **{profile['max_position_pct']:.0f}%** per trade, "
    f"**{profile['max_open_positions']}** max open positions, "
    f"**{profile['max_exposure_pct']:.0f}%** max exposure."
)
st.caption(
    "A **demo (paper-money)** trading agent that acts on the predictions — "
    "**no real funds are involved**. It starts from a **$2,000 virtual** balance."
)

with st.expander("ℹ️ How the demo agent works"):
    st.markdown(
        """
        - When a prediction is confident enough, the agent opens a simulated
          position; if confidence is too low it **SKIPs** (no trade).
        - **Portfolio Value** = cash + the current value of any open positions.
          **P&L** is profit/loss vs. the $2,000 starting balance, and
          **Max Drawdown** is the largest drop from a peak.
        - **Open Positions** shows live trades with their **unrealized P&L**
          (paper gains/losses that move with price until the position closes).
        - **Win Rate** is the share of closed trades that finished in profit.
        - Everything here is **simulated** to test the strategy safely — it is
          **not financial advice**.
        """
    )

if portfolio is None and not all_trades and backtest is None:
    st.info(
        "**No trading data available yet.**\n\n"
        "Run the trading agent to generate data:\n"
        "```\npython trade.py --backtest\n```\n\n"
        "Or for a single live tick:\n"
        "```\npython trade.py --live-tick\n```"
    )
    st.stop()

# ── Data Health ───────────────────────────────────────────────────────────

st.markdown("### Trading Data Health")
st.caption(
    "Closed trades for metrics exclude the earliest completed trade by default "
    "(audit history remains on disk). Journal entries include every decision — "
    "entries, exits, and SKIPs."
)
if activity.get("excluded_trades_count", 0) > 0:
    excluded = activity.get("excluded_trades") or []
    first = excluded[0] if excluded else {}
    st.info(
        f"**Analytics exclude {activity['excluded_trades_count']} earliest closed trade(s).** "
        f"Raw history: {activity.get('raw_closed_trades', 0)} closed trades "
        f"(${activity.get('raw_total_closed_pnl', 0):+.2f} P&L). "
        f"Earliest excluded: {first.get('id', '—')} "
        f"({first.get('side', '—')}, ${float(first.get('pnl_usd') or 0):+.2f})."
    )
layout_marker("stack")
h1, h2, h3, h4, h5, h6 = st.columns(6, gap="small")
with h1:
    render_metric_card("Closed Trades", str(activity.get("closed_trades", 0)))
with h2:
    render_metric_card("Open Positions", str(activity.get("open_positions", 0)))
with h3:
    render_metric_card("Journal Decisions", str(activity.get("journal_entries", 0)))
with h4:
    render_metric_card("Entries (BUY/SHORT)", str(activity.get("journal_entries_count", 0)))
with h5:
    render_metric_card("Exits (CLOSE)", str(activity.get("journal_exits_count", 0)))
with h6:
    render_metric_card("Skips", str(activity.get("journal_skips_count", 0)))

layout_marker("stack")
h7, h8 = st.columns(2, gap="small")
with h7:
    render_metric_card(
        "Portfolio Updated",
        utc_str_to_israel(health.get("portfolio_updated_at"), fallback="—"),
    )
with h8:
    latest_action = health.get("latest_journal_action") or "—"
    render_metric_card("Latest Journal Action", str(latest_action))

for warning in health.get("warnings", []):
    st.warning(warning)

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
            "Entry Time (Israel)": utc_str_to_israel(p.get("entry_time"), fallback="—"),
            "Entry Price": f"${entry_price:,.2f}",
            "Amount (USD)": f"${p.get('amount_usd', 0):.2f}",
            "Unrealized P&L": f"${unrealized_pnl:.2f} ({unrealized_pnl_pct:+.1f}%)",
            "Stop Loss": f"${p.get('stop_loss', 0):,.2f}",
            "Take Profit": f"${p.get('take_profit', 0):,.2f}",
        })

    st.dataframe(pd.DataFrame(pos_data), width="stretch", hide_index=True)

# ── P&L Chart ─────────────────────────────────────────────────────────────

if analytics_trades:
    st.markdown("### Cumulative P&L")
    st.caption("Chart uses analytics trades (earliest closed trade excluded).")

    sorted_trades = sorted(analytics_trades, key=lambda t: t.get("exit_time", ""))
    cumulative_pnl = []
    running = 0.0

    for t in sorted_trades:
        running += t.get("pnl_usd", 0)
        cumulative_pnl.append(running)

    st.plotly_chart(create_cumulative_pnl_chart(cumulative_pnl), width="stretch")

# ── Trade History ─────────────────────────────────────────────────────────

if all_trades:
    st.markdown("### Trade History")
    st.caption("Full audit history from trades.json (includes excluded trades).")

    excluded_ids = {str(t.get("id")) for t in excluded_trades if t.get("id")}
    trade_rows = []
    for t in reversed(all_trades[-50:]):
        pnl_usd = t.get("pnl_usd", 0)
        pnl_pct_val = t.get("pnl_pct", 0)
        trade_id = str(t.get("id", ""))
        trade_rows.append({
            "Exit Time (Israel)": utc_str_to_israel(t.get("exit_time"), fallback="—"),
            "Side": t.get("side", "LONG"),
            "Timeframe": t.get("timeframe", "—"),
            "Entry": f"${t.get('entry_price', 0):,.2f}",
            "Exit": f"${t.get('exit_price', 0):,.2f}",
            "Amount": f"${t.get('amount_usd', 0):.2f}",
            "P&L": f"${pnl_usd:+.2f} ({pnl_pct_val:+.1f}%)",
            "Reason": t.get("exit_reason", "—"),
            "Analytics": "Excluded" if trade_id in excluded_ids else "Included",
        })

    st.dataframe(pd.DataFrame(trade_rows), width="stretch", hide_index=True)

    # Summary stats
    st.markdown("### Performance Summary")
    st.caption("Metrics below exclude the earliest closed trade.")
    winners = sum(1 for t in analytics_trades if t.get("pnl_usd", 0) > 0)
    losers = sum(1 for t in analytics_trades if t.get("pnl_usd", 0) <= 0)
    win_rate = (winners / len(analytics_trades) * 100) if analytics_trades else 0
    total_pnl = sum(t.get("pnl_usd", 0) for t in analytics_trades)
    avg_win = np.mean([t["pnl_usd"] for t in analytics_trades if t.get("pnl_usd", 0) > 0]) if winners else 0
    avg_loss = np.mean([t["pnl_usd"] for t in analytics_trades if t.get("pnl_usd", 0) <= 0]) if losers else 0

    layout_marker("stack")
    s1, s2, s3, s4, s5, s6 = st.columns(6, gap="small")
    with s1:
        render_metric_card("Closed Trades", str(len(analytics_trades)))
    with s2:
        render_metric_card("Win Rate", f"{win_rate:.1f}%", delta_color="green" if win_rate > 50 else "red")
    with s3:
        render_metric_card("Closed P&L", f"${total_pnl:+.2f}", delta_color="green" if total_pnl >= 0 else "red")
    with s4:
        render_metric_card("Avg Win", f"${avg_win:+.2f}", delta_color="green")
    with s5:
        render_metric_card("Avg Loss", f"${avg_loss:.2f}", delta_color="red")
    with s6:
        render_metric_card("Journal Skips", str(activity.get("journal_skips_count", 0)))

# ── Decision Journal ──────────────────────────────────────────────────────

if journal:
    st.markdown("### Recent Trader Decisions")
    st.caption(
        "Full decision trail from the trading journal, including SKIP decisions "
        "that do not appear in closed-trade history."
    )
    decision_rows = []
    for entry in reversed(journal[-25:]):
        action = entry.get("action", "—")
        preds = entry.get("predictions_summary") or []
        reason = entry.get("reason") or "; ".join(entry.get("reasons", [])) or entry.get("exit_reason", "—")
        decision_rows.append({
            "Time (Israel)": utc_str_to_israel(entry.get("timestamp"), fallback="—"),
            "Action": action,
            "Run": entry.get("run_number", "—"),
            "Side": entry.get("position_side") or entry.get("side", "—"),
            "Timeframe": entry.get("timeframe", "—"),
            "Confidence": entry.get("confidence", "—"),
            "Amount": f"${entry.get('amount_usd', 0):.2f}" if entry.get("amount_usd") else "—",
            "Reason": reason,
            "Predictions": " | ".join(preds[:4]) if preds else "—",
        })
    st.dataframe(pd.DataFrame(decision_rows), width="stretch", hide_index=True)

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
