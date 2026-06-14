"""Mobile navigation fallback when the Streamlit sidebar is collapsed."""

from __future__ import annotations

import streamlit as st

NAV_PAGES: list[tuple[str, str]] = [
    ("app.py", "Home"),
    ("pages/01_live_predictions.py", "Predictions"),
    ("pages/02_performance.py", "Performance"),
    ("pages/03_signals.py", "Signals"),
    ("pages/04_backtest.py", "Backtest"),
    ("pages/05_market_overview.py", "Market"),
    ("pages/06_trading.py", "Trading"),
]


def render_mobile_nav(*, show_sidebar_hint: bool = False) -> None:
    """Render a horizontal page-link bar visible only on small screens."""
    st.markdown(
        '<div class="mobile-nav"><span class="mobile-nav-label">Menu</span></div>',
        unsafe_allow_html=True,
    )
    cols = st.columns(len(NAV_PAGES))
    for col, (path, label) in zip(cols, NAV_PAGES):
        with col:
            st.page_link(path, label=label)

    if show_sidebar_hint:
        st.markdown(
            '<p class="mobile-sidebar-hint">'
            "Tap <strong>&gt;</strong> (top-left) to open the sidebar menu"
            "</p>",
            unsafe_allow_html=True,
        )
