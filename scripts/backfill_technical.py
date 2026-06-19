"""Backfill persisted technical indicators for offline training.

Reads existing hourly BTC OHLCV parquet files from ``data/price/``, computes the
same indicators used by the live ``TechnicalCollector``, and writes the signal
history to ``data/history/technical.parquet`` so training and validation can
merge real TA features instead of relying on live-only computation.

Usage:
    python scripts/backfill_technical.py
    python scripts/backfill_technical.py --dry-run
    python scripts/backfill_technical.py --price-dir data/price --output data/history/technical.parquet
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import cast

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import DATA_DIR  # noqa: E402
from src.collectors.price import PriceCollector  # noqa: E402
from src.collectors.technical import TechnicalCollector  # noqa: E402
from src.simulation.data_loader import HistoricalDataLoader  # noqa: E402
from src.utils.logger import setup_logger  # noqa: E402

logger = setup_logger("backfill_technical")

DEFAULT_PRICE_DIR = DATA_DIR / "price"
DEFAULT_OUTPUT_PATH = DATA_DIR / "history" / "technical.parquet"
PRICE_COLUMNS = {"open", "high", "low", "close", "volume"}


def load_price_history(price_dir: Path = DEFAULT_PRICE_DIR) -> pd.DataFrame:
    """Load and normalize hourly OHLCV history using the simulation loader."""
    return HistoricalDataLoader(price_dir=price_dir).load_price_data()


def build_technical_history(price_df: pd.DataFrame) -> pd.DataFrame:
    """Compute persisted TA signal columns from hourly OHLCV history."""
    if price_df.empty:
        raise ValueError("Price history is empty")

    collector = TechnicalCollector(price_collector=cast(PriceCollector, None))
    technical = collector.compute_indicators(price_df)
    indicator_cols = [c for c in technical.columns if c not in PRICE_COLUMNS]
    if not indicator_cols:
        raise ValueError(
            "No technical indicators were computed. "
            "At least 200 hourly candles are required."
        )

    history = technical[indicator_cols].copy()
    history.sort_index(inplace=True)
    history.index.name = "timestamp"
    return history


def backfill_technical_history(
    price_dir: Path = DEFAULT_PRICE_DIR,
    output_path: Path = DEFAULT_OUTPUT_PATH,
    dry_run: bool = False,
) -> pd.DataFrame:
    """Compute and optionally persist technical history."""
    price_df = load_price_history(price_dir)
    history = build_technical_history(price_df)

    logger.info(
        "Computed %d technical rows × %d columns from %s to %s",
        len(history),
        len(history.columns),
        history.index[0],
        history.index[-1],
    )

    if dry_run:
        logger.info("Dry run: not writing %s", output_path)
        return history

    output_path.parent.mkdir(parents=True, exist_ok=True)
    history.to_parquet(output_path, engine="pyarrow", compression="snappy")
    logger.info("Wrote technical history to %s", output_path)
    return history


def main() -> int:
    parser = argparse.ArgumentParser(description="Backfill technical indicator history")
    parser.add_argument(
        "--price-dir",
        type=Path,
        default=DEFAULT_PRICE_DIR,
        help="Directory containing hourly BTC price parquet files",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Path for the technical signal parquet output",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute indicators and print a summary without writing parquet",
    )
    args = parser.parse_args()

    history = backfill_technical_history(
        price_dir=args.price_dir,
        output_path=args.output,
        dry_run=args.dry_run,
    )
    action = "Computed" if args.dry_run else "Wrote"
    print(f"{action} {len(history)} hourly technical rows -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
