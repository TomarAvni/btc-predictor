"""Signal-strength badges for the signal dashboard."""

from __future__ import annotations

import streamlit as st

from dashboard.styles import GREEN, RED, YELLOW, TEXT_DIM

_SENTIMENT_COLORS = {
    "bullish": GREEN,
    "bearish": RED,
    "neutral": YELLOW,
}

_KEYWORD_MAP = {
    "bullish": "bullish",
    "accumulating": "bullish",
    "buying": "bullish",
    "inflow": "bullish",
    "golden cross": "bullish",
    "outflow": "bullish",
    "up": "bullish",
    "positive": "bullish",
    "tailwind": "bullish",
    "rising": "bullish",
    "bearish": "bearish",
    "capitulation": "bearish",
    "selling": "bearish",
    "death cross": "bearish",
    "overbought": "bearish",
    "down": "bearish",
    "overheated": "neutral",
    "neutral": "neutral",
    "caution": "neutral",
    "normal": "neutral",
    "moderate": "neutral",
    "sideways": "neutral",
}


def infer_sentiment(value: str, interpretation: str = "") -> str:
    """Infer bullish / bearish / neutral from free-text value + interpretation."""
    combined = f"{value} {interpretation}".lower()
    for keyword, sentiment in _KEYWORD_MAP.items():
        if keyword in combined:
            return sentiment
    return "neutral"


def render_signal_badge(
    name: str,
    value: str,
    interpretation: str = "",
    sentiment: str | None = None,
) -> None:
    """Render a coloured signal badge."""
    if sentiment is None:
        sentiment = infer_sentiment(value, interpretation)

    dot_color = _SENTIMENT_COLORS.get(sentiment, YELLOW)
    tip = interpretation if interpretation else sentiment.title()

    st.markdown(
        f"""
        <div class="signal-badge" title="{tip}">
            <span class="dot" style="background:{dot_color};"></span>
            <span class="name">{name}</span>
            <span class="val">{value}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_signal_grid(signals: dict[str, dict], categories: dict[str, list[str]]) -> None:
    """Render all signals grouped by category in a responsive grid."""
    for category, signal_names in categories.items():
        matching = {k: v for k, v in signals.items() if k in signal_names}
        if not matching:
            continue
        st.markdown(f"**{category}**")
        cols = st.columns(min(len(matching), 3))
        for i, (name, info) in enumerate(matching.items()):
            with cols[i % len(cols)]:
                render_signal_badge(
                    name,
                    info.get("value", "N/A"),
                    info.get("interpretation", ""),
                )

    uncategorised = {
        k: v
        for k, v in signals.items()
        if not any(k in names for names in categories.values())
    }
    if uncategorised:
        st.markdown("**Other**")
        cols = st.columns(min(len(uncategorised), 3))
        for i, (name, info) in enumerate(uncategorised.items()):
            with cols[i % len(cols)]:
                render_signal_badge(name, info.get("value", "N/A"), info.get("interpretation", ""))
