# BTC Price Movement Predictor

An ML-powered Bitcoin price movement prediction system that combines multiple signal sources across different timeframes to generate probabilistic forecasts, with a simulated trading agent and live dashboard.

## What It Does

Outputs a continuous prediction curve — every 6 hours from 6h to 168h (7 days),
plus a long-range 30d point (see `src/horizons.py`, the single source of truth):
```
[2026-06-15 00:30 UTC] -- Prediction Run #42

PREDICTIONS:
  6h    | UP    | +0.4%   | Confidence: 66%
  24h   | UP    | +1.8%   | Confidence: 68%
  72h   | UP    | +3.1%   | Confidence: 58%
  168h  | UP    | +4.2%   | Confidence: 55%
  30d   | DOWN  | -6.5%   | Confidence: 42%
  ...   (every 6h step in between)
```

And manages a virtual $2,000 trading portfolio based on model predictions.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run everything end-to-end (download, train, backtest, dashboard)
python run_demo.py

# Or step by step:

# Download full hourly price history (2013-present)
python main.py --download

# Train models with walk-forward validation
python train.py

# Run a single prediction
python main.py --predict

# Run trading agent backtest
python trade.py --backtest --start 2024-01-01 --end 2025-01-01

# Run a single trading tick (for cron/CI)
python trade.py --live-tick

# Launch the dashboard
streamlit run dashboard/app.py
```

## Commands

| Command | Description |
|---------|-------------|
| `python main.py --download` | Download/resume full hourly BTC price history |
| `python main.py --predict` | Run one prediction cycle and log results |
| `python main.py --status` | Show current data status |
| `python score_predictions.py` | Score mature predictions against actual price moves |
| `python train.py` | Full training pipeline with walk-forward validation |
| `python train.py --backtest` | Run backtest on trained model |
| `python trade.py --backtest` | Run trading agent over historical data |
| `python trade.py --live` | Live demo mode (continuous paper trading) |
| `python trade.py --live-tick` | Single prediction+trade cycle then exit |
| `python trade.py --status` | Show current portfolio status |
| `python run_demo.py` | One command to see everything working |
| `streamlit run dashboard/app.py` | Launch the Streamlit dashboard |

## Signal Sources

- **Price**: Full hourly BTC history (2013-present), technical indicators (RSI, MACD, BB, EMAs, ATR). Post-2017 candles use an exchange fallback chain (Binance → Bitstamp → Kraken); GitHub Actions sets `BTC_PRICE_PRIMARY_EXCHANGE=bitstamp` because Binance returns HTTP 451 on US runners.
- **Halving Cycle**: Position in 4-year cycle, historical comparison, power law corridor
- **Derivatives**: Funding rates, open interest, long/short ratio, options put/call ratio, max pain
- **On-Chain**: Active addresses, hash rate, exchange reserves, mempool, MVRV
- **Whale Activity**: Large transaction tracking, accumulation vs distribution scoring
- **Miner Health**: Hash price, miner revenue, capitulation detection
- **Institutional**: ETF flow estimates, Coinbase premium, Korean premium
- **Sentiment**: Fear & Greed Index, Google Trends
- **Macro**: DXY, S&P 500, Gold, VIX, Treasury yields, global M2 liquidity
- **Market Structure**: BTC dominance, stablecoin supply, CME gaps, liquidation levels

## Models

- **XGBoost**: Gradient boosting on tabular features (cycle position, sentiment, TA, macro correlations)
- **LSTM/TFT**: Sequential pattern recognition on hourly price/volume windows
- **Ensemble**: Stacking meta-learner with per-timeframe weights calibrated by walk-forward backtest

## Project Structure

```
btc-predictor/
├── main.py                  # Prediction engine entry point
├── train.py                 # Model training entry point
├── trade.py                 # Trading agent entry point
├── run_demo.py              # One-command demo script
├── requirements.txt         # Python dependencies
├── config/
│   └── settings.yaml        # App configuration
├── src/
│   ├── collectors/          # Data collectors (price, technical, cycle, macro, etc.)
│   ├── features/            # Feature engineering (engineer, scaler, temporal)
│   ├── models/              # ML models (xgboost, tft, ensemble, confidence)
│   ├── training/            # Training pipeline (trainer, walk_forward, metrics)
│   ├── simulation/          # Market replay & labeling (for training)
│   ├── engine/              # Prediction engine & backtest orchestration
│   ├── trading/             # Trading agent (portfolio, strategy, risk, simulator)
│   ├── output/              # Logging & formatting
│   └── utils/               # Logger, cache utilities
├── dashboard/
│   ├── app.py               # Streamlit main page
│   ├── pages/               # Multi-page dashboard (predictions, signals, backtest, trading)
│   ├── components/          # Reusable UI components
│   ├── data_loader.py       # Dashboard data access layer
│   ├── config.py            # Dashboard configuration
│   └── styles.py            # CSS & Plotly theming
├── data/                    # Runtime data (price, models, trading state)
├── .github/workflows/       # GitHub Actions (Download → Train → Predict)
└── .streamlit/config.toml   # Streamlit Cloud theme config
```

## Dashboard

A Streamlit-based dashboard provides a visual interface for predictions, performance tracking, signal analysis, backtest results, market overview, and trading agent status.

```bash
streamlit run dashboard/app.py
```

| Page | Description |
|------|-------------|
| Home | Current predictions, signal badges, confidence gauge |
| Live Predictions | Extended signal breakdown and sentiment summary |
| Performance | Accuracy tracking, calibration curve, simulated P&L |
| Signals | Deep dive into individual signals and feature importance |
| Backtest | Walk-forward equity curve, drawdown, regime analysis |
| Market Overview | Interactive candlestick chart with indicators, halving cycles |
| Trading | Portfolio value, open positions, trade history, P&L chart |

The dashboard works with demo data out of the box — run the predictor to populate it with real data.

## Deployment

### GitHub Actions (Automated Pipeline)

Three workflows run a hands-off pipeline: **Download → Train → Predict**.

| Workflow | Trigger | What it does |
|----------|---------|--------------|
| **Download** (`download.yml`) | Manual (`workflow_dispatch`) | Downloads full hourly BTC price history (Bitstamp on CI; Binance fallback locally) and commits `data/price/` |
| **Train** (`train.yml`) | Auto after Download succeeds, or manual | Runs 80/20 validation (`validate.py --split 0.8`), trains models, backtests the trading agent, commits `data/validation/` |
| **Predict** (`predict.yml`) | Every 30 minutes (cron) or manual | Runs one prediction cycle + live demo trading tick + scores mature predictions, commits results |
| **Predict Watchdog** (`predict-watchdog.yml`) | Hourly cron or manual | Checks `predictions.log` and dispatches Predict if no prediction has landed for over 3 hours |
| **Retrain** (`retrain.yml`) | Weekly Sunday 3am UTC or manual | Incremental price update, score predictions, retrain models, commit `data/validation/` + `data/performance/` |

**Setup (one time):** In GitHub Actions, run **Download** manually. When it finishes, **Train** starts automatically. After models are committed, **Predict** runs every 30 minutes on the schedule.

Until Train has run at least once, Predict logs a warning and uses TA heuristics instead of ML models.

All workflow commits use `[skip ci]` in the message to avoid infinite re-runs.

### Continuous Learning Loop

The system improves over time through a closed feedback loop:

1. **Predict** (`predict.yml`, every 30 min) — runs ML models on latest data, logs predictions to `predictions.log`, executes a paper-trade tick, then scores mature predictions.
2. **Score** (`score_predictions.py`) — for each prediction whose horizon has elapsed (any 6h-step point up to 168h, plus 30d), compares predicted direction and magnitude to the actual BTC move from `data/price/btc_hourly.parquet`. Results append to `data/performance/prediction_scores.jsonl`; rolling stats land in `data/performance/rolling_accuracy.json`.
3. **Retrain** (`retrain.yml`, weekly Sunday 3am UTC) — downloads fresh candles, scores any newly mature predictions, retrains models via `validate.py`, and commits updated models + performance data.
4. **Predict Watchdog** (`predict-watchdog.yml`, hourly) — checks the committed prediction log and dispatches Predict when the latest run is older than 3 hours and no Predict run is already queued or running.
5. **Dashboard** — the Performance page shows live rolling accuracy; the Signals page shows feature importance from the latest validation run.

```
Download → Train (initial) → Predict (every 30m) → Score → Retrain (weekly) → Predict uses new models
```

Live predict loads trained models from `data/validation/models/` (configured in `config/settings.yaml`). If models are missing, it falls back to TA heuristics and logs a warning.

### Streamlit Cloud (Dashboard)

1. Push the repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Point it to `dashboard/app.py`
4. The `.streamlit/config.toml` theme is applied automatically

## Configuration

Edit `config/settings.yaml` to adjust:
- Update intervals per tier
- Prediction timeframes
- Model parameters
- API endpoints

## Limitations

- Cannot predict black swan events or sudden regulatory actions
- Confidence decreases significantly for longer timeframes
- Some data sources require paid API access for full history
- Past performance does not guarantee future results
- This is an exploration/research tool, not financial advice
