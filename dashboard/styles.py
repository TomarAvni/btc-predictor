"""Custom CSS and Plotly theming for a dark trading-terminal aesthetic."""

import streamlit as st

# ── Colour palette ──────────────────────────────────────────────────────────
BG_DARK = "#0E1117"
CARD_BG = "#1E2128"
GREEN = "#00D26A"
RED = "#FF4B4B"
YELLOW = "#FFD700"
BLUE = "#4DA6FF"
TEXT = "#FAFAFA"
TEXT_DIM = "#8B949E"
BORDER = "#30363D"

PLOTLY_TEMPLATE = dict(
    layout=dict(
        paper_bgcolor=BG_DARK,
        plot_bgcolor="#161B22",
        font=dict(color=TEXT, family="Inter, sans-serif"),
        xaxis=dict(gridcolor=BORDER, zerolinecolor=BORDER),
        yaxis=dict(gridcolor=BORDER, zerolinecolor=BORDER),
        margin=dict(l=40, r=20, t=50, b=40),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        colorway=[BLUE, GREEN, RED, YELLOW, "#A855F7", "#F97316"],
    )
)


# Breakpoints (documented for maintenance):
#   ≤480px  — phone: single-column stacks, smallest typography
#   ≤768px  — tablet / mobile: reduced padding, stacked layouts, shorter charts
#   >768px  — desktop: unchanged default layout


def inject_css() -> None:
    """Inject custom CSS into the Streamlit app."""
    st.markdown(
        f"""
        <style>
        /* ── Global ─────────────────────────────── */
        .stApp {{
            background-color: {BG_DARK};
            overflow-x: hidden;
        }}
        .main .block-container {{
            padding-top: 1.5rem;
            padding-bottom: 2rem;
            max-width: 100%;
        }}
        section[data-testid="stSidebar"] {{
            background-color: #161B22;
        }}
        section[data-testid="stSidebar"] .stMarkdown,
        section[data-testid="stSidebar"] label {{
            word-wrap: break-word;
            overflow-wrap: break-word;
        }}
        section[data-testid="stSidebar"] [data-testid="stToggle"] {{
            min-height: 44px;
        }}
        section[data-testid="stSidebar"] [data-testid="stToggle"] label {{
            font-size: 0.95rem;
        }}

        /* ── Metric cards ───────────────────────── */
        .metric-card {{
            background: {CARD_BG};
            border: 1px solid {BORDER};
            border-radius: 12px;
            padding: 1.2rem 1.4rem;
            margin-bottom: 0.6rem;
        }}
        .metric-card h4 {{
            color: {TEXT_DIM};
            font-size: 0.82rem;
            font-weight: 500;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin: 0 0 0.3rem 0;
        }}
        .metric-card .value {{
            font-size: 1.9rem;
            font-weight: 700;
            color: {TEXT};
            margin: 0;
            line-height: 1.2;
        }}
        .metric-card .delta {{
            font-size: 0.9rem;
            font-weight: 500;
            margin-top: 0.2rem;
        }}
        .delta-up   {{ color: {GREEN}; }}
        .delta-down {{ color: {RED}; }}

        /* ── Prediction card ────────────────────── */
        .pred-card {{
            border-radius: 12px;
            padding: 1.2rem 1rem;
            text-align: center;
            border: 1px solid {BORDER};
        }}
        .pred-card .timeframe {{
            font-size: 0.8rem;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            color: {TEXT_DIM};
            margin-bottom: 0.3rem;
        }}
        .pred-card .direction {{
            font-size: 1.6rem;
            font-weight: 700;
        }}
        .pred-card .magnitude {{
            font-size: 1.1rem;
            font-weight: 600;
        }}
        .pred-card .confidence {{
            font-size: 0.8rem;
            color: {TEXT_DIM};
            margin-top: 0.4rem;
        }}
        .pred-card .conf-bar {{
            height: 4px;
            border-radius: 2px;
            margin-top: 0.5rem;
            background: {BORDER};
        }}
        .pred-card .conf-fill {{
            height: 100%;
            border-radius: 2px;
        }}

        /* ── Signal badge ───────────────────────── */
        .signal-badge {{
            display: inline-flex;
            align-items: center;
            gap: 0.4rem;
            background: {CARD_BG};
            border: 1px solid {BORDER};
            border-radius: 8px;
            padding: 0.55rem 0.85rem;
            margin: 0.25rem 0;
            width: 100%;
        }}
        .signal-badge .dot {{
            width: 8px;
            height: 8px;
            border-radius: 50%;
            flex-shrink: 0;
        }}
        .signal-badge .name {{
            font-size: 0.78rem;
            color: {TEXT_DIM};
            white-space: nowrap;
        }}
        .signal-badge .val {{
            font-size: 0.85rem;
            font-weight: 600;
            color: {TEXT};
            margin-left: auto;
            text-align: right;
        }}

        /* ── Training readiness dock (sticky) ─────── */
        .training-readiness-dock {{
            position: fixed;
            bottom: 1rem;
            right: 1rem;
            z-index: 999;
            display: flex;
            flex-direction: column;
            gap: 0.5rem;
            width: min(260px, calc(100vw - 2rem));
            pointer-events: none;
        }}
        .training-readiness-card {{
            background: {CARD_BG};
            border: 1px solid {BORDER};
            border-radius: 10px;
            padding: 0.75rem 0.85rem;
            box-shadow: 0 4px 18px rgba(0, 0, 0, 0.35);
            pointer-events: auto;
        }}
        .training-readiness-card.ready {{
            border-color: {GREEN}55;
        }}
        .training-readiness-card.pending {{
            border-color: {BORDER};
        }}
        .training-readiness-card .tr-header {{
            display: flex;
            align-items: center;
            gap: 0.35rem;
            margin-bottom: 0.35rem;
        }}
        .training-readiness-card .tr-icon {{
            font-size: 0.85rem;
            color: {TEXT_DIM};
        }}
        .training-readiness-card .tr-title {{
            font-size: 0.72rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: {TEXT_DIM};
            flex: 1;
        }}
        .training-readiness-card .tr-badge {{
            font-size: 0.65rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.04em;
            padding: 0.12rem 0.4rem;
            border-radius: 999px;
            border: 1px solid;
        }}
        .training-readiness-card .tr-horizon {{
            font-size: 0.95rem;
            font-weight: 700;
            color: {TEXT};
            margin-bottom: 0.15rem;
        }}
        .training-readiness-card .tr-count {{
            font-size: 0.78rem;
            color: {TEXT_DIM};
            margin-bottom: 0.4rem;
        }}
        .training-readiness-card .tr-progress {{
            height: 4px;
            border-radius: 2px;
            background: {BORDER};
            overflow: hidden;
            margin-bottom: 0.35rem;
        }}
        .training-readiness-card .tr-progress-fill {{
            height: 100%;
            border-radius: 2px;
            transition: width 0.3s ease;
        }}
        .training-readiness-card .tr-detail {{
            font-size: 0.72rem;
            color: {TEXT_DIM};
        }}
        @media (max-width: 768px) {{
            .training-readiness-dock {{
                bottom: 0.5rem;
                right: 0.5rem;
                left: 0.5rem;
                width: auto;
            }}
            .training-readiness-dock {{
                flex-direction: row;
            }}
            .training-readiness-card {{
                flex: 1;
                min-width: 0;
            }}
            .training-readiness-card .tr-horizon {{
                font-size: 0.82rem;
            }}
        }}

        /* ── Coming-soon placeholder ────────────── */
        .coming-soon {{
            background: {CARD_BG};
            border: 1px dashed {BORDER};
            border-radius: 12px;
            padding: 3rem 2rem;
            text-align: center;
            color: {TEXT_DIM};
        }}
        .coming-soon h3 {{
            color: {TEXT};
            margin-bottom: 0.6rem;
        }}

        /* ── Hide Streamlit boilerplate (desktop) ─ */
        @media (min-width: 769px) {{
            #MainMenu {{visibility: hidden;}}
            header {{visibility: hidden;}}
            footer {{visibility: hidden;}}
        }}
        @media (max-width: 768px) {{
            footer {{visibility: hidden;}}
            header {{
                visibility: visible !important;
            }}
            [data-testid="collapsedControl"] {{
                display: block !important;
                visibility: visible !important;
            }}
        }}

        /* ── Mobile nav fallback ─────────────────── */
        .mobile-nav {{
            display: none;
        }}
        .mobile-nav + [data-testid="stHorizontalBlock"] {{
            display: none;
        }}
        .mobile-sidebar-hint {{
            display: none;
        }}
        .mobile-nav-label {{
            font-size: 0.82rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: {TEXT_DIM};
            margin-bottom: 0.35rem;
        }}
        @media (max-width: 768px) {{
            .mobile-nav {{
                display: block;
                margin-bottom: 0.25rem;
            }}
            .mobile-nav + [data-testid="stHorizontalBlock"] {{
                display: flex !important;
                flex-wrap: nowrap !important;
                overflow-x: auto;
                -webkit-overflow-scrolling: touch;
                gap: 0.35rem !important;
                padding-bottom: 0.35rem;
                margin-bottom: 0.75rem;
            }}
            .mobile-nav + [data-testid="stHorizontalBlock"] [data-testid="column"] {{
                flex: 0 0 auto !important;
                min-width: unset !important;
                width: auto !important;
            }}
            .mobile-nav + [data-testid="stHorizontalBlock"] a {{
                white-space: nowrap;
                font-size: 0.82rem;
                padding: 0.35rem 0.65rem;
                border-radius: 999px;
                border: 1px solid {BORDER};
                background: {CARD_BG};
            }}
            .mobile-sidebar-hint {{
                display: block;
                font-size: 0.85rem;
                color: {TEXT_DIM};
                background: {CARD_BG};
                border: 1px solid {BORDER};
                border-radius: 8px;
                padding: 0.55rem 0.75rem;
                margin: 0 0 1rem 0;
            }}
        }}

        /* ── Tabs styling ───────────────────────── */
        .stTabs [data-baseweb="tab-list"] {{
            gap: 1rem;
        }}
        .stTabs [data-baseweb="tab"] {{
            color: {TEXT_DIM};
        }}
        .stTabs [aria-selected="true"] {{
            color: {BLUE};
        }}

        /* ── Signal grid (CSS grid, not st.columns) ─ */
        .signal-grid {{
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 0.5rem;
            margin-bottom: 1rem;
        }}
        .signal-grid-category {{
            margin-top: 0.75rem;
            margin-bottom: 0.35rem;
            font-weight: 600;
        }}

        /* ── Prediction table scroll wrapper ─────── */
        .pred-table-wrap {{
            overflow-x: auto;
            -webkit-overflow-scrolling: touch;
        }}
        .pred-table-wrap table {{
            min-width: 520px;
        }}

        /* ── Plotly chart containers ─────────────── */
        [data-testid="stPlotlyChart"] {{
            width: 100% !important;
            overflow: hidden;
        }}
        [data-testid="stPlotlyChart"] .js-plotly-plot,
        [data-testid="stPlotlyChart"] .plot-container {{
            width: 100% !important;
        }}

        /* ── Touch-friendly controls ─────────────── */
        button[kind="secondary"],
        button[kind="primary"],
        .stCheckbox label,
        .stRadio label,
        [data-testid="stSelectbox"] {{
            min-height: 44px;
        }}

        /* ── Mobile / tablet (≤768px) ────────────── */
        @media (max-width: 768px) {{
            .main .block-container {{
                padding-left: 0.75rem;
                padding-right: 0.75rem;
                padding-top: 1rem;
            }}
            h1 {{
                font-size: 1.45rem !important;
            }}
            h2, h3 {{
                font-size: 1.1rem !important;
            }}
            .metric-card {{
                padding: 0.85rem 1rem;
                margin-bottom: 0.5rem;
            }}
            .metric-card .value {{
                font-size: 1.45rem;
            }}
            .metric-card h4 {{
                font-size: 0.75rem;
            }}
            .pred-card {{
                width: 100%;
                box-sizing: border-box;
                padding: 1rem 0.85rem;
                margin-bottom: 0.5rem;
            }}
            .pred-card .direction {{
                font-size: 1.35rem;
            }}
            .pred-card .magnitude {{
                font-size: 1rem;
            }}
            .signal-grid {{
                grid-template-columns: repeat(2, 1fr);
            }}
            .signal-badge {{
                min-height: 44px;
                padding: 0.65rem 0.75rem;
            }}
            .signal-badge .name {{
                white-space: normal;
                font-size: 0.72rem;
            }}
            .pred-table-wrap table {{
                font-size: 0.78rem;
            }}
            .pred-table-wrap th,
            .pred-table-wrap td {{
                padding: 0.35rem !important;
            }}
            .coming-soon {{
                padding: 1.5rem 1rem;
            }}
            [data-testid="stPlotlyChart"] {{
                min-height: 250px;
            }}
            [data-testid="stPlotlyChart"] .js-plotly-plot {{
                max-height: 280px;
            }}
            .stTabs [data-baseweb="tab-list"] {{
                gap: 0.5rem;
                flex-wrap: wrap;
            }}
            [data-testid="stHorizontalBlock"] {{
                flex-wrap: wrap !important;
                gap: 0.5rem !important;
            }}
            [data-testid="column"] {{
                min-width: 0 !important;
            }}
            .layout-marker[data-layout="stack"] + [data-testid="stHorizontalBlock"] [data-testid="column"] {{
                flex: 1 1 100% !important;
                width: 100% !important;
            }}
            .layout-marker[data-layout="grid-2"] + [data-testid="stHorizontalBlock"] [data-testid="column"] {{
                flex: 1 1 calc(50% - 0.5rem) !important;
                min-width: calc(50% - 0.5rem) !important;
            }}
            [data-testid="stDataFrame"] {{
                overflow-x: auto;
            }}
            [data-testid="stDataFrame"] div[data-testid="stTable"] {{
                font-size: 0.78rem;
            }}
        }}

        /* ── Phone (≤480px) ───────────────────────── */
        @media (max-width: 480px) {{
            .main .block-container {{
                padding-left: 0.5rem;
                padding-right: 0.5rem;
            }}
            h1 {{
                font-size: 1.25rem !important;
            }}
            .metric-card .value {{
                font-size: 1.25rem;
            }}
            .signal-grid {{
                grid-template-columns: 1fr;
            }}
            .layout-marker[data-layout="grid-2"] + [data-testid="stHorizontalBlock"] [data-testid="column"] {{
                flex: 1 1 100% !important;
                min-width: 100% !important;
            }}
            [data-testid="stPlotlyChart"] .js-plotly-plot {{
                max-height: 250px;
            }}
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def layout_marker(layout: str = "stack") -> None:
    """Insert a CSS marker before the next st.columns row for responsive layout.

    layout: "stack" (single column on mobile) or "grid-2" (2×2 tablet, 1 col phone).
    """
    st.markdown(
        f'<div class="layout-marker" data-layout="{layout}"></div>',
        unsafe_allow_html=True,
    )
