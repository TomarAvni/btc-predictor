"""Live BTC price monitor — current price, 24h change, and intraday sparkline.

Data is fetched from public REST APIs that work from Streamlit Cloud without
authentication or geo-restrictions:

  * Primary  — Bitstamp public ticker + OHLC (``api.bitstamp.net``)
  * Fallback — CoinGecko simple price + market chart

Results are cached with a short TTL so the widget refreshes roughly once a
minute without hammering the API, and every network call degrades gracefully:
on failure we reuse the last-known-good snapshot (kept in ``st.session_state``)
and surface a small "stale" notice instead of crashing the page.
"""

from __future__ import annotations

from typing import Any

import plotly.graph_objects as go
import requests
import streamlit as st

from dashboard.styles import BG_DARK, GREEN, RED, TEXT_DIM
from src.utils.timez import now_israel_str

_BITSTAMP_TICKER = "https://www.bitstamp.net/api/v2/ticker/btcusd/"
_BITSTAMP_OHLC = "https://www.bitstamp.net/api/v2/ohlc/btcusd/"
_COINGECKO_PRICE = "https://api.coingecko.com/api/v3/simple/price"
_COINGECKO_CHART = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart"

_TIMEOUT = 8
_LAST_GOOD_KEY = "_btc_monitor_last_good"


# ── Fetchers ────────────────────────────────────────────────────────────────


def _fetch_bitstamp() -> dict[str, Any]:
    """Current price + 24h stats + intraday closes from Bitstamp."""
    ticker = requests.get(_BITSTAMP_TICKER, timeout=_TIMEOUT)
    ticker.raise_for_status()
    t = ticker.json()

    # 15-minute candles for the last ~24h (96 steps).
    ohlc = requests.get(
        _BITSTAMP_OHLC,
        params={"step": 900, "limit": 96},
        timeout=_TIMEOUT,
    )
    ohlc.raise_for_status()
    candles = ohlc.json().get("data", {}).get("ohlc", [])

    spark = [float(c["close"]) for c in candles if c.get("close")]

    return {
        "price": float(t["last"]),
        "change_pct": float(t.get("percent_change_24", 0.0)),
        "high": float(t.get("high", 0.0)),
        "low": float(t.get("low", 0.0)),
        "volume": float(t.get("volume", 0.0)),
        "spark": spark,
        "source": "Bitstamp",
    }


def _fetch_coingecko() -> dict[str, Any]:
    """Fallback: current price + 24h change + intraday closes from CoinGecko."""
    price = requests.get(
        _COINGECKO_PRICE,
        params={
            "ids": "bitcoin",
            "vs_currencies": "usd",
            "include_24hr_change": "true",
            "include_24hr_vol": "true",
        },
        timeout=_TIMEOUT,
    )
    price.raise_for_status()
    p = price.json()["bitcoin"]

    chart = requests.get(
        _COINGECKO_CHART,
        params={"vs_currency": "usd", "days": 1},
        timeout=_TIMEOUT,
    )
    chart.raise_for_status()
    prices = chart.json().get("prices", [])
    spark = [float(pt[1]) for pt in prices]

    return {
        "price": float(p["usd"]),
        "change_pct": float(p.get("usd_24h_change", 0.0)),
        "high": max(spark) if spark else 0.0,
        "low": min(spark) if spark else 0.0,
        "volume": float(p.get("usd_24h_vol", 0.0)),
        "spark": spark,
        "source": "CoinGecko",
    }


@st.cache_data(ttl=45, show_spinner=False)
def _load_live_btc() -> dict[str, Any]:
    """Return a live BTC snapshot, trying Bitstamp then CoinGecko."""
    errors: list[str] = []
    for fetch in (_fetch_bitstamp, _fetch_coingecko):
        try:
            data = fetch()
            if data.get("price"):
                data["ok"] = True
                data["fetched_at"] = now_israel_str()
                return data
        except Exception as exc:  # noqa: BLE001 — any network/parse error
            errors.append(f"{fetch.__name__}: {exc}")
    return {"ok": False, "error": " | ".join(errors) or "unknown error"}


# ── Sparkline ───────────────────────────────────────────────────────────────


def _sparkline(values: list[float], rising: bool) -> go.Figure:
    color = GREEN if rising else RED
    fill = "rgba(0, 210, 106, 0.12)" if rising else "rgba(255, 75, 75, 0.12)"
    fig = go.Figure(
        go.Scatter(
            y=values,
            mode="lines",
            line=dict(color=color, width=2),
            fill="tozeroy",
            fillcolor=fill,
            hovertemplate="$%{y:,.0f}<extra></extra>",
        )
    )
    lo, hi = (min(values), max(values)) if values else (0, 1)
    pad = (hi - lo) * 0.15 or 1
    fig.update_layout(
        height=110,
        margin=dict(l=0, r=0, t=6, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        xaxis=dict(visible=False, fixedrange=True),
        yaxis=dict(visible=False, fixedrange=True, range=[lo - pad, hi + pad]),
    )
    return fig


# ── Public renderer ───────────────────────────────────────────────────────


def render_price_monitor(*, title: str = "Live BTC Price") -> None:
    """Render the live price monitor: current price, 24h change, sparkline."""
    data = _load_live_btc()

    if data.get("ok"):
        st.session_state[_LAST_GOOD_KEY] = data
        stale = False
    else:
        data = st.session_state.get(_LAST_GOOD_KEY, {})
        stale = True

    if not data:
        st.warning("⚠️ Live price unavailable right now — couldn't reach the price feed.")
        st.caption("This is a display-only feed; predictions and trades are unaffected.")
        return

    price = data["price"]
    change = data.get("change_pct", 0.0)
    rising = change >= 0
    spark = data.get("spark", [])

    st.markdown(f"#### {title}")
    col_price, col_chart = st.columns([1, 1.7], gap="medium")
    with col_price:
        st.metric(
            label="BTC / USD",
            value=f"${price:,.0f}",
            delta=f"{change:+.2f}% (24h)",
            help="Live spot price. The 24h change compares now vs. 24 hours ago.",
        )
        if data.get("high") and data.get("low"):
            st.caption(f"24h range  ${data['low']:,.0f} – ${data['high']:,.0f}")
    with col_chart:
        if spark:
            st.plotly_chart(
                _sparkline(spark, rising),
                width="stretch",
                config={"displayModeBar": False},
            )
        else:
            st.caption("Intraday chart unavailable.")

    if stale:
        st.caption("⚠️ Showing last-known price — live feed is temporarily unreachable.")
    else:
        st.caption(
            f"Source: {data.get('source', '—')} · updated {data.get('fetched_at', '—')} "
            "· refreshes ~every 45s · display only, not a trading feed."
        )
