"""Dashboard configuration constants."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

PREDICTIONS_LOG = PROJECT_ROOT / "predictions.log"
DATA_DIR = PROJECT_ROOT / "data"
PRICE_DIR = DATA_DIR / "price"
MODELS_DIR = DATA_DIR / "models"
BACKTEST_DIR = DATA_DIR / "backtest"
VALIDATION_DIR = DATA_DIR / "validation"
HISTORY_DIR = DATA_DIR / "history"
CONFIG_PATH = PROJECT_ROOT / "config" / "settings.yaml"

PREDICTION_HORIZONS = ["24h", "7d", "30d", "90d"]

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
