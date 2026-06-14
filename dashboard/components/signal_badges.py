"""Signal-strength badges for the signal dashboard."""

from __future__ import annotations

import html

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


def _badge_html(name: str, value: str, interpretation: str = "", sentiment: str | None = None) -> str:
    if sentiment is None:
        sentiment = infer_sentiment(value, interpretation)
    dot_color = _SENTIMENT_COLORS.get(sentiment, YELLOW)
    tip = html.escape(interpretation if interpretation else sentiment.title())
    safe_name = html.escape(name)
    safe_value = html.escape(value)
    return f"""
        <div class="signal-badge" title="{tip}">
            <span class="dot" style="background:{dot_color};"></span>
            <span class="name">{safe_name}</span>
            <span class="val">{safe_value}</span>
        </div>
    """


def render_signal_badge(
    name: str,
    value: str,
    interpretation: str = "",
    sentiment: str | None = None,
) -> None:
    """Render a coloured signal badge."""
    st.markdown(_badge_html(name, value, interpretation, sentiment), unsafe_allow_html=True)


def render_signal_grid(signals: dict[str, dict], categories: dict[str, list[str]]) -> None:
    """Render all signals grouped by category in a responsive CSS grid."""
    sections: list[str] = []

    for category, signal_names in categories.items():
        matching = {k: v for k, v in signals.items() if k in signal_names}
        if not matching:
            continue
        badges = "".join(
            _badge_html(name, info.get("value", "N/A"), info.get("interpretation", ""))
            for name, info in matching.items()
        )
        sections.append(
            f'<p class="signal-grid-category">{html.escape(category)}</p>'
            f'<div class="signal-grid">{badges}</div>'
        )

    uncategorised = {
        k: v
        for k, v in signals.items()
        if not any(k in names for names in categories.values())
    }
    if uncategorised:
        badges = "".join(
            _badge_html(name, info.get("value", "N/A"), info.get("interpretation", ""))
            for name, info in uncategorised.items()
        )
        sections.append(
            '<p class="signal-grid-category">Other</p>'
            f'<div class="signal-grid">{badges}</div>'
        )

    if sections:
        st.markdown("".join(sections), unsafe_allow_html=True)
