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


def inject_css() -> None:
    """Inject custom CSS into the Streamlit app."""
    st.markdown(
        f"""
        <style>
        /* ── Global ─────────────────────────────── */
        .stApp {{
            background-color: {BG_DARK};
        }}
        section[data-testid="stSidebar"] {{
            background-color: #161B22;
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

        /* ── Hide Streamlit boiler plate ────────── */
        #MainMenu {{visibility: hidden;}}
        header {{visibility: hidden;}}
        footer {{visibility: hidden;}}

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
        </style>
        """,
        unsafe_allow_html=True,
    )
