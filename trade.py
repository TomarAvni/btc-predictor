"""BTC Demo Trading Agent -- entry point.

Usage:
    python trade.py --backtest --start 2024-01-01 --end 2025-01-01
    python trade.py --live
    python trade.py --status
    python trade.py --report
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.trading.agent import TradingAgent
from src.trading.performance import PerformanceTracker
from src.trading.portfolio import Portfolio
from src.trading.position_sizer import PositionSizer
from src.trading.risk_manager import RiskManager
from src.trading.simulator import OrderSimulator
from src.trading.strategy import TradingStrategy
from src.trading.trade_journal import TradeJournal
from src.output.jsonl_logger import PREDICTIONS_JSONL_PATH
from src.utils.timez import now_israel_str


def create_agent() -> TradingAgent:
    """Create a fully configured trading agent."""
    return TradingAgent(
        portfolio=Portfolio(load_existing=True),
        strategy=TradingStrategy(),
        risk_manager=RiskManager(),
        position_sizer=PositionSizer(),
        simulator=OrderSimulator(),
        performance=PerformanceTracker(),
        journal=TradeJournal(),
    )


def load_price_history(start: str, end: str) -> pd.DataFrame:
    """Load historical price data for backtesting.

    Tries to load from local parquet/csv, falls back to generating
    synthetic data for demo purposes.
    """
    price_path = Path("data/price")

    # Try loading existing price data
    parquet_files = list(price_path.glob("*.parquet")) if price_path.exists() else []
    csv_files = list(price_path.glob("*.csv")) if price_path.exists() else []

    if parquet_files:
        df = pd.read_parquet(parquet_files[0])
    elif csv_files:
        df = pd.read_csv(csv_files[0], parse_dates=["timestamp"], index_col="timestamp")
    else:
        print("[INFO] No historical price data found. Generating synthetic data for demo...")
        df = _generate_synthetic_prices(start, end)

    # Filter to requested date range
    start_dt = pd.Timestamp(start)
    end_dt = pd.Timestamp(end)

    if isinstance(df.index, pd.DatetimeIndex):
        df = df[(df.index >= start_dt) & (df.index <= end_dt)]
    elif "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df[(df["timestamp"] >= start_dt) & (df["timestamp"] <= end_dt)]
        df = df.set_index("timestamp")

    return df


def _generate_synthetic_prices(start: str, end: str) -> pd.DataFrame:
    """Generate synthetic BTC price data for demo/testing.

    Uses geometric Brownian motion to create realistic-looking price series.
    """
    start_dt = pd.Timestamp(start)
    end_dt = pd.Timestamp(end)
    hours = int((end_dt - start_dt).total_seconds() / 3600)

    np.random.seed(42)

    initial_price = 42000.0
    mu = 0.0001  # slight upward drift
    sigma = 0.015  # hourly volatility

    returns = np.random.normal(mu, sigma, hours)
    prices = initial_price * np.cumprod(1 + returns)

    timestamps = pd.date_range(start=start_dt, periods=hours, freq="h")

    # Generate OHLCV from close prices
    highs = prices * (1 + np.abs(np.random.normal(0, 0.005, hours)))
    lows = prices * (1 - np.abs(np.random.normal(0, 0.005, hours)))
    opens = np.roll(prices, 1)
    opens[0] = initial_price
    volumes = np.random.lognormal(mean=10, sigma=1, size=hours)

    df = pd.DataFrame(
        {
            "open": opens,
            "high": highs,
            "low": lows,
            "close": prices,
            "volume": volumes,
        },
        index=timestamps,
    )
    df.index.name = "timestamp"
    return df


def generate_synthetic_predictions(
    price: float, prev_price: float, timestamp: datetime
) -> list[dict]:
    """Generate synthetic predictions based on price movement.

    In live mode, these come from the ML engine. For backtesting
    without a trained model, we simulate predictions with noise.
    """
    np.random.seed(int(timestamp.timestamp()) % (2**31))

    actual_move = (price - prev_price) / prev_price if prev_price > 0 else 0

    # Add noise to simulate imperfect predictions
    noise = np.random.normal(0, 0.3)
    signal = np.sign(actual_move + noise)
    direction = "UP" if signal >= 0 else "DOWN"

    predictions = []
    for timeframe, base_conf, mag_scale in [
        ("24h", 60, 1.5),
        ("168h", 55, 4.0),
        ("30d", 48, 10.0),
    ]:
        conf_noise = np.random.uniform(-10, 15)
        confidence = max(30, min(92, base_conf + conf_noise))
        magnitude = abs(actual_move * 100 * mag_scale) + np.random.uniform(0.5, 3.0)

        predictions.append({
            "timeframe": timeframe,
            "direction": direction,
            "magnitude": round(magnitude, 1),
            "confidence": round(confidence, 1),
        })

    return predictions


def run_backtest(start: str, end: str) -> None:
    """Run a backtest over historical data."""
    print(f"\n{'='*70}")
    print(f"  BTC TRADING AGENT -- BACKTEST")
    print(f"  Period: {start} to {end}")
    print(f"  Starting Balance: $2,000")
    print(f"{'='*70}\n")

    # Load price data
    df = load_price_history(start, end)
    if df.empty:
        print("[ERROR] No price data available for the specified period.")
        sys.exit(1)

    print(f"[INFO] Loaded {len(df):,} hourly candles")
    print(f"[INFO] Price range: ${df['close'].min():,.0f} - ${df['close'].max():,.0f}")
    print()

    # Create agent with fresh state
    agent = TradingAgent(
        portfolio=Portfolio(load_existing=False),
        strategy=TradingStrategy(),
        risk_manager=RiskManager(),
        position_sizer=PositionSizer(),
        simulator=OrderSimulator(),
        performance=PerformanceTracker(),
        journal=TradeJournal(),
    )
    agent.reset()

    # Run simulation: check predictions every 4 hours, price updates every hour
    prediction_interval = 4
    total_candles = len(df)
    prev_price = float(df["close"].iloc[0])

    trades_made = 0
    for i, (ts, row) in enumerate(df.iterrows()):
        price = float(row["close"])
        high = float(row.get("high", price))
        low = float(row.get("low", price))

        # Price update every candle (check SL/TP)
        exits = agent.on_price_update(
            price=price, high=high, low=low, timestamp=ts
        )
        trades_made += len(exits)

        # Generate predictions every N hours
        if i % prediction_interval == 0 and i > 0:
            predictions = generate_synthetic_predictions(price, prev_price, ts)
            result = agent.on_new_prediction(predictions, price, timestamp=ts)
            if result.get("actions"):
                for action in result["actions"]:
                    trades_made += 1

        prev_price = price

        # Progress reporting
        if i % (total_candles // 10) == 0 and i > 0:
            pct = i / total_candles * 100
            val = agent.portfolio.total_value_usd
            print(
                f"  [{pct:5.1f}%] {ts} | "
                f"Portfolio: ${val:,.2f} | "
                f"P&L: {agent.portfolio.total_pnl_pct:+.2f}% | "
                f"Trades: {len(agent.portfolio.closed_trades)}"
            )

    # Final report
    print(f"\n{'='*70}")
    print(agent.get_performance_report())

    # Save final state
    summary = agent.get_performance_summary()
    results_path = Path("data/trading/backtest_results.json")
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(f"\n[INFO] Results saved to {results_path}")


async def run_live() -> None:
    """Run in live demo mode, connected to the prediction engine."""
    print(f"\n{'='*70}")
    print(f"  BTC TRADING AGENT -- LIVE DEMO MODE")
    print(f"  Starting Balance: $2,000")
    print(f"  WARNING: This is a DEMO. No real money is involved.")
    print(f"{'='*70}\n")

    agent = create_agent()

    # Try to connect to the prediction engine
    try:
        from src.engine.predictor import PredictionEngine
        engine = PredictionEngine()
    except ImportError:
        engine = None
        print("[WARN] Prediction engine not available. Using synthetic predictions.")

    print("[INFO] Trading agent started. Press Ctrl+C to stop.\n")

    # Simulated live loop -- in production this would subscribe to real-time data
    try:
        while True:
            # Get current price (synthetic for demo)
            current_price = _get_current_price()

            if engine:
                engine_result = await engine.run_prediction()
                predictions = engine_result.get("predictions", [])
                used_ml = bool(engine_result.get("using_ml", False))
                run_number = int(engine_result.get("run_number", 0))
            else:
                prev_price = current_price * (1 + np.random.normal(0, 0.002))
                predictions = generate_synthetic_predictions(
                    current_price, prev_price, datetime.now(timezone.utc)
                )
                used_ml = False
                run_number = 0

            # Process prediction
            result = agent.on_new_prediction(
                predictions, current_price, used_ml=used_ml, run_number=run_number
            )

            # Display status
            status = agent.get_status()
            _print_live_status(status, predictions)

            # Wait before next cycle (every 5 minutes for demo)
            await asyncio.sleep(300)

    except KeyboardInterrupt:
        print("\n\n[INFO] Shutting down...")
        print(agent.get_performance_report())


def _get_current_price() -> float:
    """Get current BTC price. Falls back to synthetic for demo."""
    try:
        price_path = Path("data/price")
        parquet_files = list(price_path.glob("*.parquet")) if price_path.exists() else []
        if parquet_files:
            df = pd.read_parquet(parquet_files[0])
            return float(df["close"].iloc[-1])
    except Exception:
        pass

    # Synthetic price for demo
    return 104000.0 + np.random.normal(0, 500)


def _print_live_status(status: dict, predictions: list[dict]) -> None:
    """Print live trading status to console."""
    portfolio = status["portfolio"]
    print(f"  [{now_israel_str(fmt='%H:%M:%S')}] ", end="")
    print(
        f"Value: ${portfolio['total_value_usd']:,.2f} | "
        f"P&L: {portfolio['total_pnl_pct']:+.2f}% | "
        f"Positions: {portfolio['open_positions']} | "
        f"Trades: {portfolio['total_trades']}"
    )

    if predictions:
        best = max(predictions, key=lambda p: p["confidence"])
        print(
            f"           Best signal: {best['timeframe']} {best['direction']} "
            f"+{best['magnitude']}% @ {best['confidence']}% confidence"
        )
    print()


def load_latest_logged_prediction() -> dict | None:
    """Load the latest numeric prediction run written by ``main.py --predict``."""
    if not PREDICTIONS_JSONL_PATH.exists():
        return None

    latest: dict | None = None
    for line in PREDICTIONS_JSONL_PATH.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        # Sentiment tracks are logged into the same JSONL file. The paper trader
        # should consume the main numeric/ensemble prediction until the blended
        # strategy is explicitly enabled.
        source = str(record.get("model_source", "")).lower()
        if source in {"llm_direct", "llm_calibrated", "twitter", "blended_50_50"}:
            continue
        if record.get("predictions"):
            latest = record
    return latest


async def run_live_tick() -> None:
    """Run a single trade cycle from the latest logged prediction then exit."""
    agent = create_agent()
    current_price = _get_current_price()

    used_ml = False
    run_number = 0
    latest = load_latest_logged_prediction()
    if latest:
        predictions = latest.get("predictions", [])
        current_price = float(latest.get("btc_price") or current_price)
        used_ml = bool(latest.get("used_ml", False))
        run_number = int(latest.get("run_number", 0) or 0)
    else:
        print("[WARN] No logged prediction found. Falling back to synthetic prediction.")
        prev_price = current_price * (1 + np.random.normal(0, 0.002))
        predictions = generate_synthetic_predictions(
            current_price, prev_price, datetime.now(timezone.utc)
        )
        used_ml = False

    result = agent.on_new_prediction(
        predictions, current_price, used_ml=used_ml, run_number=run_number
    )

    if not used_ml:
        print("[GATE] No real ML model used for this prediction — "
              "trades are blocked by the pre-trade safety gate.")

    print(f"[{now_israel_str()}] Live tick complete")
    print(f"  Price: ${current_price:,.2f}")
    print(f"  Portfolio: ${result['portfolio_value']:,.2f}")
    print(f"  Actions: {len(result.get('actions', []))}")
    for action in result.get("actions", []):
        print(f"    {action['action']} ${action.get('amount_usd', 0):.2f}")

    # Persist PerformanceTracker snapshot so Sharpe/drawdown survive restarts.
    try:
        perf = agent.get_performance_summary()
        perf["snapshot_at"] = datetime.now(timezone.utc).isoformat()
        snap_path = Path("data/trading/performance_snapshot.json")
        snap_path.parent.mkdir(parents=True, exist_ok=True)
        snap_path.write_text(json.dumps(perf, indent=2, default=str), encoding="utf-8")
    except Exception as exc:  # noqa: BLE001
        print(f"[WARN] Could not save performance snapshot: {exc}")


def show_status() -> None:
    """Display current portfolio status."""
    agent = create_agent()
    status = agent.get_status()
    portfolio = status["portfolio"]

    print(f"\n{'='*70}")
    print(f"  BTC TRADING AGENT -- STATUS")
    print(f"{'='*70}")
    print(f"  Portfolio Value:  ${portfolio['total_value_usd']:,.2f}")
    print(f"  Cash:             ${portfolio['cash']:,.2f}")
    print(f"  BTC Holdings:     {portfolio['btc_holdings']:.8f} BTC")
    print(f"  Total P&L:        {portfolio['total_pnl_pct']:+.2f}% (${portfolio['total_pnl']:+.2f})")
    print(f"  Max Drawdown:     {portfolio['max_drawdown']:.2f}%")
    print(f"  Win Rate:         {portfolio['win_rate']:.1f}%")
    print(f"  Open Positions:   {portfolio['open_positions']}")
    print(f"  Total Trades:     {portfolio['total_trades']}")
    print(f"  7d P&L:           ${portfolio['last_7d_pnl']:+.2f}")
    print(f"  30d P&L:          ${portfolio['last_30d_pnl']:+.2f}")

    if status["open_positions"]:
        print(f"\n  OPEN POSITIONS:")
        for pos in status["open_positions"]:
            print(
                f"    {pos['side']} {pos['timeframe']} | "
                f"Entry: ${pos['entry_price']:,.2f} | "
                f"Size: ${pos['amount_usd']:.2f} | "
                f"P&L: ${pos['unrealized_pnl']:.2f} ({pos['unrealized_pnl_pct']:.2f}%)"
            )

    print(f"{'='*70}\n")


def show_report() -> None:
    """Display performance report."""
    agent = create_agent()
    print(agent.get_performance_report())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="BTC Demo Trading Agent"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--backtest",
        action="store_true",
        help="Run backtest over historical data",
    )
    group.add_argument(
        "--live",
        action="store_true",
        help="Run in live demo mode (paper trading)",
    )
    group.add_argument(
        "--live-tick",
        action="store_true",
        help="Run a single live prediction+trade cycle then exit (for CI/cron)",
    )
    group.add_argument(
        "--status",
        action="store_true",
        help="Show current portfolio status",
    )
    group.add_argument(
        "--report",
        action="store_true",
        help="Show performance report",
    )

    parser.add_argument("--start", default="2024-01-01", help="Backtest start date")
    parser.add_argument("--end", default="2025-01-01", help="Backtest end date")

    args = parser.parse_args()

    if args.backtest:
        run_backtest(args.start, args.end)
    elif args.live:
        asyncio.run(run_live())
    elif getattr(args, "live_tick", False):
        asyncio.run(run_live_tick())
    elif args.status:
        show_status()
    elif args.report:
        show_report()


if __name__ == "__main__":
    main()
