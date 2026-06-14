"""Training entry point for the BTC Price Movement Prediction Engine.

Usage:
    python train.py --data-dir data/price --output-dir data/models --cycles 2,3,4
    python train.py --backtest --start 2023-01-01 --end 2024-01-01

Commands:
    (default)    Run full training pipeline with walk-forward validation
    --backtest   Run backtest on trained model over specified date range
    --report     Print performance report from last training run
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src import DATA_DIR
from src.utils.logger import setup_logger

logger = setup_logger("btc_trainer")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="BTC Price Prediction - Model Training Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--data-dir",
        type=str,
        default=str(DATA_DIR / "price"),
        help="Directory containing Parquet price files (default: data/price/)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DATA_DIR / "models"),
        help="Directory to save trained models (default: data/models/)",
    )
    parser.add_argument(
        "--cycles",
        type=str,
        default="2,3,4",
        help="Comma-separated cycle numbers to include (default: 2,3,4)",
    )

    # Mode selection
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--backtest",
        action="store_true",
        help="Run backtest instead of training",
    )
    mode_group.add_argument(
        "--report",
        action="store_true",
        help="Print report from last training run",
    )

    # Backtest options
    parser.add_argument(
        "--start",
        type=str,
        default="2023-01-01",
        help="Backtest start date (default: 2023-01-01)",
    )
    parser.add_argument(
        "--end",
        type=str,
        default="2024-12-31",
        help="Backtest end date (default: 2024-12-31)",
    )
    parser.add_argument(
        "--step-hours",
        type=int,
        default=4,
        help="Hours between backtest predictions (default: 4)",
    )

    return parser.parse_args()


def run_training(args: argparse.Namespace) -> None:
    """Execute the full training pipeline."""
    from src.training.trainer import TrainingOrchestrator

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    cycles = [int(c.strip()) for c in args.cycles.split(",")]

    logger.info("Starting training pipeline...")
    logger.info("  Data directory: %s", data_dir)
    logger.info("  Output directory: %s", output_dir)
    logger.info("  Cycles: %s", cycles)

    try:
        orchestrator = TrainingOrchestrator(
            data_dir=data_dir,
            output_dir=output_dir,
            cycles=cycles,
        )
        results = orchestrator.run()

        # Print summary
        print("\n" + results.get("report", "No report generated."))

        if results.get("walk_forward"):
            print("\nWalk-forward summary:")
            for tf, metrics in results["walk_forward"].items():
                if isinstance(metrics, dict) and "mean_accuracy" in metrics:
                    print(f"  {tf}: {metrics['mean_accuracy']:.1%} ± {metrics['std_accuracy']:.1%}")

    except FileNotFoundError as e:
        logger.error("Data not found: %s", e)
        print(f"\nError: {e}")
        print("Make sure to download price data first: python main.py --download")
        sys.exit(1)
    except Exception as e:
        logger.error("Training failed: %s", e, exc_info=True)
        sys.exit(1)


def run_backtest(args: argparse.Namespace) -> None:
    """Run backtest on a trained model."""
    from src.engine.backtest import Backtester
    from src.models.xgboost_model import XGBoostPredictor

    logger.info("Running backtest: %s to %s", args.start, args.end)

    # Load trained XGBoost model for backtesting
    model = XGBoostPredictor(timeframe="24h")
    model.load()

    if model.model_direction is None:
        print("Error: No trained model found. Run training first: python train.py")
        sys.exit(1)

    try:
        backtester = Backtester(
            data_dir=Path(args.data_dir),
            model_dir=Path(args.output_dir),
        )
        report = backtester.run(
            model=model,
            start_date=args.start,
            end_date=args.end,
            step_hours=args.step_hours,
        )
        print(report.summary_text)

    except FileNotFoundError as e:
        logger.error("Data not found: %s", e)
        print(f"\nError: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error("Backtest failed: %s", e, exc_info=True)
        sys.exit(1)


def show_report(args: argparse.Namespace) -> None:
    """Show the training report from the last run."""
    output_dir = Path(args.output_dir)
    report_path = output_dir / "training_report.txt"

    if not report_path.exists():
        print("No training report found. Run training first: python train.py")
        sys.exit(1)

    print(report_path.read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()

    if args.backtest:
        run_backtest(args)
    elif args.report:
        show_report(args)
    else:
        run_training(args)


if __name__ == "__main__":
    main()
