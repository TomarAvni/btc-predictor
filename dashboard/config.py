"""Dashboard configuration constants."""

from pathlib import Path

from src.horizons import KEY_HORIZONS, TIMEFRAMES

PROJECT_ROOT = Path(__file__).resolve().parent.parent

PREDICTIONS_LOG = PROJECT_ROOT / "predictions.log"
DATA_DIR = PROJECT_ROOT / "data"
PRICE_DIR = DATA_DIR / "price"
MODELS_DIR = DATA_DIR / "models"
BACKTEST_DIR = DATA_DIR / "backtest"
VALIDATION_DIR = DATA_DIR / "validation"
PERFORMANCE_DIR = DATA_DIR / "performance"
HISTORY_DIR = DATA_DIR / "history"
CONFIG_PATH = PROJECT_ROOT / "config" / "settings.yaml"

# Full continuous horizon curve (single source of truth: src/horizons.py).
PREDICTION_HORIZONS = list(TIMEFRAMES)
# Compact "headline" horizons for summary cards / gauges / table columns.
SUMMARY_HORIZONS = list(KEY_HORIZONS)

SIGNAL_CATEGORIES = {
    "Cycle": ["Halving Cycle", "Cycle Phase"],
    "On-chain": [
        "LTH/STH Supply", "Exchange Flow", "Whale Activity",
        "Miner Health", "BTC Dominance",
    ],
    "Sentiment": ["Fear & Greed", "Google Trends", "Coinbase Premium"],
    "Macro": ["M2/Net Liquidity", "DXY"],
    "Technical": [
        "RSI (4h)", "MACD (daily)", "Volume", "Power Law",
        "CME Gap", "EMA 50/200", "ADX", "Volume Ratio",
    ],
    "Derivatives": [
        "Funding Rate", "Options Max Pain", "Liquidations",
        "ETF Flows",
    ],
}

AUTO_REFRESH_INTERVAL_MS = 60_000
