"""Reusable Plotly chart builders with consistent dark theming."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from dashboard.styles import (
    BG_DARK,
    BLUE,
    BORDER,
    CARD_BG,
    GREEN,
    PLOTLY_TEMPLATE,
    RED,
    TEXT,
    TEXT_DIM,
    YELLOW,
)

_BASE_LAYOUT: dict[str, Any] = PLOTLY_TEMPLATE["layout"]


def _apply_theme(fig: go.Figure) -> go.Figure:
    fig.update_layout(**_BASE_LAYOUT)
    return fig


# ── Candlestick / OHLCV ───────────────────────────────────────────────────


def create_candlestick_chart(
    df,
    overlays: list[str] | None = None,
    show_volume: bool = True,
    show_rsi: bool = False,
    show_macd: bool = False,
    height: int = 600,
) -> go.Figure:
    """Interactive candlestick chart with optional overlays and sub-plots."""
    subplot_rows = 1 + int(show_volume) + int(show_rsi) + int(show_macd)
    row_heights = [0.55]
    if show_volume:
        row_heights.append(0.15)
    if show_rsi:
        row_heights.append(0.15)
    if show_macd:
        row_heights.append(0.15)

    fig = make_subplots(
        rows=subplot_rows,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=row_heights,
    )

    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            increasing_line_color=GREEN,
            decreasing_line_color=RED,
            name="BTC",
        ),
        row=1,
        col=1,
    )

    overlay_colors = {"ema_9": "#F59E0B", "ema_21": BLUE, "ema_50": "#A855F7", "ema_200": RED}
    for ov in overlays or []:
        if ov in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df[ov],
                    mode="lines",
                    name=ov.upper().replace("_", " "),
                    line=dict(width=1, color=overlay_colors.get(ov, YELLOW)),
                ),
                row=1,
                col=1,
            )

    next_row = 2
    if show_volume and "volume" in df.columns:
        colors = [GREEN if c >= o else RED for o, c in zip(df["open"], df["close"])]
        fig.add_trace(
            go.Bar(x=df.index, y=df["volume"], marker_color=colors, name="Volume", opacity=0.6),
            row=next_row,
            col=1,
        )
        next_row += 1

    if show_rsi and "rsi_14" in df.columns:
        fig.add_trace(
            go.Scatter(x=df.index, y=df["rsi_14"], mode="lines", name="RSI", line=dict(color=BLUE, width=1.5)),
            row=next_row,
            col=1,
        )
        fig.add_hline(y=70, line_dash="dash", line_color=RED, row=next_row, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color=GREEN, row=next_row, col=1)
        next_row += 1

    if show_macd and "macd" in df.columns:
        fig.add_trace(
            go.Scatter(x=df.index, y=df["macd"], mode="lines", name="MACD", line=dict(color=BLUE, width=1.5)),
            row=next_row,
            col=1,
        )
        if "macd_signal" in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df.index, y=df["macd_signal"], mode="lines", name="Signal",
                    line=dict(color=RED, width=1, dash="dot"),
                ),
                row=next_row,
                col=1,
            )

    fig.update_layout(
        height=height,
        xaxis_rangeslider_visible=False,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1),
    )
    _apply_theme(fig)
    return fig


# ── Line chart ─────────────────────────────────────────────────────────────


def create_line_chart(
    df,
    y_cols: list[str] | None = None,
    title: str = "",
    colors: list[str] | None = None,
    height: int = 400,
) -> go.Figure:
    fig = go.Figure()
    cols = y_cols or [c for c in df.columns if c != "timestamp"]
    palette = colors or [BLUE, GREEN, RED, YELLOW, "#A855F7", "#F97316"]

    for i, col in enumerate(cols):
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df[col],
                mode="lines",
                name=col,
                line=dict(color=palette[i % len(palette)], width=2),
            )
        )

    fig.update_layout(title=title, height=height)
    _apply_theme(fig)
    return fig


# ── Gauge (confidence meter) ──────────────────────────────────────────────


def create_gauge_chart(
    value: float,
    min_val: float = 0,
    max_val: float = 100,
    title: str = "",
    height: int = 250,
) -> go.Figure:
    if value < 40:
        bar_color = RED
    elif value < 60:
        bar_color = YELLOW
    else:
        bar_color = GREEN

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=value,
            title=dict(text=title, font=dict(size=14, color=TEXT_DIM)),
            number=dict(suffix="%", font=dict(size=28, color=TEXT)),
            gauge=dict(
                axis=dict(range=[min_val, max_val], tickfont=dict(color=TEXT_DIM)),
                bar=dict(color=bar_color, thickness=0.75),
                bgcolor=CARD_BG,
                borderwidth=0,
                steps=[
                    dict(range=[0, 40], color="#1a1a2e"),
                    dict(range=[40, 60], color="#1e1e30"),
                    dict(range=[60, 100], color="#1a2a1a"),
                ],
            ),
        )
    )
    fig.update_layout(height=height)
    _apply_theme(fig)
    return fig


# ── Calibration curve ─────────────────────────────────────────────────────


def create_calibration_curve(
    predicted_conf: Sequence[float],
    actual_accuracy: Sequence[float],
    height: int = 400,
) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=list(predicted_conf),
            y=list(actual_accuracy),
            mode="lines+markers",
            name="Model",
            line=dict(color=BLUE, width=2),
            marker=dict(size=8),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[0, 100],
            y=[0, 100],
            mode="lines",
            name="Perfect",
            line=dict(color=TEXT_DIM, width=1, dash="dash"),
        )
    )
    fig.update_layout(
        title="Confidence Calibration",
        xaxis_title="Predicted Confidence (%)",
        yaxis_title="Actual Accuracy (%)",
        height=height,
    )
    _apply_theme(fig)
    return fig


# ── Equity curve ──────────────────────────────────────────────────────────


def create_equity_curve(
    equity_series: dict[str, Sequence[float]],
    timestamps: Sequence | None = None,
    height: int = 400,
) -> go.Figure:
    fig = go.Figure()
    palette = [BLUE, GREEN, RED, YELLOW]
    for i, (label, values) in enumerate(equity_series.items()):
        x = list(timestamps) if timestamps is not None else list(range(len(values)))
        fig.add_trace(
            go.Scatter(x=x, y=list(values), mode="lines", name=label, line=dict(color=palette[i % len(palette)], width=2))
        )
    fig.update_layout(title="Cumulative P&L Simulation", yaxis_title="Portfolio Value ($)", height=height)
    _apply_theme(fig)
    return fig


# ── Heatmap ───────────────────────────────────────────────────────────────


def create_heatmap(
    data,
    x_labels: list[str] | None = None,
    y_labels: list[str] | None = None,
    title: str = "",
    height: int = 400,
) -> go.Figure:
    fig = go.Figure(
        go.Heatmap(
            z=data,
            x=x_labels,
            y=y_labels,
            colorscale=[[0, "#1a1a2e"], [0.5, BLUE], [1, GREEN]],
            texttemplate="%{z:.0f}",
            textfont=dict(size=12),
        )
    )
    fig.update_layout(title=title, height=height)
    _apply_theme(fig)
    return fig


# ── Bar chart ─────────────────────────────────────────────────────────────


def create_bar_chart(
    categories: list[str],
    values: list[float],
    title: str = "",
    color: str | None = None,
    height: int = 400,
    horizontal: bool = False,
) -> go.Figure:
    colors = [GREEN if v >= 0 else RED for v in values] if color is None else [color] * len(values)
    if horizontal:
        fig = go.Figure(go.Bar(y=categories, x=values, orientation="h", marker_color=colors))
    else:
        fig = go.Figure(go.Bar(x=categories, y=values, marker_color=colors))
    fig.update_layout(title=title, height=height)
    _apply_theme(fig)
    return fig


# ── Scatter (predicted vs actual) ─────────────────────────────────────────


def create_scatter_chart(
    x: Sequence[float],
    y: Sequence[float],
    title: str = "",
    x_label: str = "",
    y_label: str = "",
    height: int = 400,
) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=list(x), y=list(y), mode="markers", marker=dict(color=BLUE, size=6, opacity=0.6), name="Predictions")
    )
    min_v = min(min(x), min(y))
    max_v = max(max(x), max(y))
    fig.add_trace(
        go.Scatter(x=[min_v, max_v], y=[min_v, max_v], mode="lines", line=dict(color=TEXT_DIM, dash="dash"), name="Perfect")
    )
    fig.update_layout(title=title, xaxis_title=x_label, yaxis_title=y_label, height=height)
    _apply_theme(fig)
    return fig
