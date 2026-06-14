"""Compact prediction-history table component."""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from dashboard.styles import GREEN, RED, YELLOW


def _direction_badge(d: str) -> str:
    color = GREEN if d == "UP" else RED
    arrow = "▲" if d == "UP" else "▼"
    return f'<span style="color:{color};font-weight:600;">{arrow} {d}</span>'


def _confidence_color(c: int) -> str:
    if c >= 60:
        return GREEN
    if c >= 45:
        return YELLOW
    return RED


def render_prediction_table(runs: list[dict[str, Any]], max_rows: int = 10) -> None:
    """Render recent prediction runs as a compact HTML table."""
    if not runs:
        st.info("No prediction history available yet.")
        return

    recent = runs[-max_rows:][::-1]
    rows_html: list[str] = []

    for run in recent:
        ts = run.get("timestamp", "?")
        run_num = run.get("run_number", "?")
        preds = run.get("predictions", [])

        cells = [f"<td style='white-space:nowrap;'>#{run_num}</td>", f"<td>{ts}</td>"]
        for tf in ("24h", "7d", "30d", "90d"):
            match = next((p for p in preds if p["timeframe"] == tf), None)
            if match:
                badge = _direction_badge(match["direction"])
                mag = f"{'+' if match['direction']=='UP' else '-'}{abs(match['magnitude']):.1f}%"
                conf_col = _confidence_color(match["confidence"])
                cell = f'{badge} {mag} <span style="color:{conf_col};font-size:0.8em;">({match["confidence"]}%)</span>'
            else:
                cell = '<span style="color:#555;">—</span>'
            cells.append(f"<td>{cell}</td>")

        rows_html.append(f"<tr>{''.join(cells)}</tr>")

    st.markdown(
        f"""
        <div class="pred-table-wrap">
        <table style="width:100%;border-collapse:collapse;font-size:0.88rem;">
        <thead>
            <tr style="border-bottom:1px solid #30363D;color:#8B949E;text-align:left;">
                <th style="padding:0.5rem;">Run</th>
                <th style="padding:0.5rem;">Time</th>
                <th style="padding:0.5rem;">24h</th>
                <th style="padding:0.5rem;">7d</th>
                <th style="padding:0.5rem;">30d</th>
                <th style="padding:0.5rem;">90d</th>
            </tr>
        </thead>
        <tbody>{''.join(rows_html)}</tbody>
        </table>
        </div>
        """,
        unsafe_allow_html=True,
    )
