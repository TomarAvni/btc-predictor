"""Floating calibration / retrain readiness cards for the main dashboard."""

from __future__ import annotations

from typing import Any

import streamlit as st

from dashboard.styles import BLUE, BORDER, CARD_BG, GREEN, TEXT, TEXT_DIM, YELLOW
from src.horizons import TIMEFRAMES


def _closest_horizon_status(
    labeled_by_horizon: dict[str, int],
    min_rows: int,
) -> dict[str, Any]:
    """Pick the horizon closest to the threshold (fewest rows remaining)."""
    best: dict[str, Any] | None = None
    ready_count = 0

    for tf in TIMEFRAMES:
        labeled = int(labeled_by_horizon.get(tf, 0))
        if labeled >= min_rows:
            ready_count += 1
            continue
        remaining = min_rows - labeled
        progress = min(100.0, labeled / min_rows * 100.0) if min_rows else 100.0
        candidate = {
            "horizon": tf,
            "labeled": labeled,
            "min_rows": min_rows,
            "remaining": remaining,
            "progress_pct": progress,
            "ready": False,
        }
        if best is None or remaining < best["remaining"]:
            best = candidate
        elif remaining == best["remaining"]:
            # Prefer shorter horizons when tied.
            if TIMEFRAMES.index(tf) < TIMEFRAMES.index(best["horizon"]):
                best = candidate

    total = len(TIMEFRAMES)
    if ready_count == total:
        return {
            "horizon": "all",
            "labeled": min_rows,
            "min_rows": min_rows,
            "remaining": 0,
            "progress_pct": 100.0,
            "ready": True,
            "ready_count": ready_count,
            "total_count": total,
        }

    if best is None:
        # All ready except we already handled total; fallback for empty TIMEFRAMES.
        best = {
            "horizon": TIMEFRAMES[0] if TIMEFRAMES else "—",
            "labeled": 0,
            "min_rows": min_rows,
            "remaining": min_rows,
            "progress_pct": 0.0,
            "ready": False,
        }

    return {**best, "ready_count": ready_count, "total_count": total}


def compute_readiness_snapshot(status: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Derive calibration and retrain card payloads from get_training_status()."""
    labeled = status.get("labeled_by_horizon") or {}
    return {
        "calibration": _closest_horizon_status(
            labeled, int(status.get("calibration_min_rows", 100))
        ),
        "retrain": _closest_horizon_status(
            labeled, int(status.get("retrain_min_rows", 400))
        ),
    }


def _render_card(title: str, icon: str, snap: dict[str, Any]) -> str:
    ready = snap["ready"]
    status_cls = "ready" if ready else "pending"
    status_label = "Ready" if ready else "Not ready"
    accent = GREEN if ready else YELLOW

    if snap["horizon"] == "all":
        horizon_line = f"All {snap['total_count']} horizons"
        detail = f"{snap['min_rows']}+ labeled rows each"
    elif snap.get("ready_count", 0) > 0:
        horizon_line = f"Next: {snap['horizon']}"
        detail = (
            f"{snap['ready_count']}/{snap['total_count']} horizons ready · "
            f"{snap['remaining']} rows to go"
        )
    else:
        horizon_line = f"Closest: {snap['horizon']}"
        detail = f"{snap['remaining']} rows to go"

    progress = snap["progress_pct"]
    bar_color = GREEN if ready else BLUE

    return f"""
    <div class="training-readiness-card {status_cls}">
        <div class="tr-header">
            <span class="tr-icon">{icon}</span>
            <span class="tr-title">{title}</span>
            <span class="tr-badge" style="color:{accent}; border-color:{accent}40;
                  background:{accent}18;">{status_label}</span>
        </div>
        <div class="tr-horizon">{horizon_line}</div>
        <div class="tr-count">{snap['labeled']:,} / {snap['min_rows']:,} labeled</div>
        <div class="tr-progress">
            <div class="tr-progress-fill" style="width:{progress:.1f}%; background:{bar_color};"></div>
        </div>
        <div class="tr-detail">{detail}</div>
    </div>
    """


def render_training_readiness_dock(status: dict[str, Any]) -> None:
    """Render sticky floating cards for calibration and retrain readiness."""
    snap = compute_readiness_snapshot(status)
    cards_html = (
        _render_card("Calibration", "◎", snap["calibration"])
        + _render_card("Retrain", "↻", snap["retrain"])
    )
    st.markdown(
        f"""
        <div class="training-readiness-dock">
            {cards_html}
        </div>
        """,
        unsafe_allow_html=True,
    )
