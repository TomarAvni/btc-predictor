"""KPI / metric card components rendered as styled HTML."""

from __future__ import annotations

from typing import Any

import streamlit as st

from dashboard.config import PREDICTION_HORIZONS
from dashboard.styles import GREEN, RED, YELLOW, layout_marker


def render_metric_card(
    title: str,
    value: str,
    delta: str | None = None,
    delta_color: str | None = None,
) -> None:
    """Render a single KPI metric card."""
    delta_html = ""
    if delta:
        css = "delta-up" if delta_color == "green" else "delta-down" if delta_color == "red" else ""
        delta_html = f'<p class="delta {css}">{delta}</p>'

    st.markdown(
        f"""
        <div class="metric-card">
            <h4>{title}</h4>
            <p class="value">{value}</p>
            {delta_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_prediction_card(
    timeframe: str,
    direction: str,
    magnitude: float,
    confidence: int,
) -> None:
    """Render a prediction card with colour-coded direction and confidence bar."""
    is_up = direction.upper() == "UP"
    color = GREEN if is_up else RED
    arrow = "&#9650;" if is_up else "&#9660;"
    sign = "+" if is_up else "-"
    opacity = max(0.3, confidence / 100)

    st.markdown(
        f"""
        <div class="pred-card" style="background: {color}15; border-color: {color}40;">
            <div class="timeframe">{timeframe}</div>
            <div class="direction" style="color:{color};">{arrow} {direction.upper()}</div>
            <div class="magnitude" style="color:{color};">{sign}{abs(magnitude):.1f}%</div>
            <div class="confidence">Confidence: {confidence}%</div>
            <div class="conf-bar">
                <div class="conf-fill" style="width:{confidence}%; background:{color}; opacity:{opacity};"></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_prediction_cards(
    predictions: list[dict[str, Any]],
    timeframes: tuple[str, ...] = tuple(PREDICTION_HORIZONS),
    *,
    mobile_stack: bool = True,
) -> None:
    """Render horizon prediction cards in a responsive row."""
    if mobile_stack:
        layout_marker("stack")
    cols = st.columns(len(timeframes), gap="small")
    for i, tf in enumerate(timeframes):
        match = next((p for p in predictions if p["timeframe"] == tf), None)
        with cols[i]:
            if match:
                render_prediction_card(
                    tf, match["direction"], match["magnitude"], match["confidence"]
                )
            else:
                render_metric_card(tf, "—")


def render_performance_card(
    title: str,
    accuracy: float,
    correct: int,
    total: int,
    best: str = "",
    worst: str = "",
) -> None:
    """Render a performance summary card."""
    color = GREEN if accuracy >= 55 else YELLOW if accuracy >= 45 else RED
    st.markdown(
        f"""
        <div class="metric-card">
            <h4>{title}</h4>
            <p class="value" style="color:{color};">{accuracy:.1f}%</p>
            <p class="delta" style="color:{color};">{correct}/{total} correct</p>
            {"<p class='delta' style='font-size:0.75rem;color:#8B949E;'>Best: " + best + " · Worst: " + worst + "</p>" if best else ""}
        </div>
        """,
        unsafe_allow_html=True,
    )
