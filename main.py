"""BTC Price Movement Prediction Engine -- entry point.

Usage:
    python main.py --download   Download / resume full hourly price history
    python main.py --predict    Run a single prediction cycle
    python main.py --status     Show current data status
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

import yaml

from src import CONFIG_PATH
from src.engine.predictor import PredictionEngine
from src.utils.logger import setup_logger

logger = setup_logger("btc_predictor")


def load_config() -> dict:
    """Load settings.yaml from the config directory."""
    try:
        with open(CONFIG_PATH, encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except FileNotFoundError:
        logger.warning("Config file not found at %s; using defaults", CONFIG_PATH)
        return {}


async def cmd_download(engine: PredictionEngine) -> None:
    """Download (or resume) the full hourly price history."""
    logger.info("Starting full price history download ...")
    df = await engine.price.download_full_history()
    if df.empty:
        logger.error("Download failed -- no data retrieved")
        sys.exit(1)
    logger.info(
        "Download complete: %s candles (%s -> %s)",
        f"{len(df):,}",
        df.index[0],
        df.index[-1],
    )


async def cmd_predict(engine: PredictionEngine) -> None:
    """Run a single prediction cycle."""
    result = await engine.run_prediction()
    if not result:
        logger.error("Prediction cycle produced no results")
        sys.exit(1)


def cmd_status(engine: PredictionEngine) -> None:
    """Print current data status."""
    status = engine.get_status()
    from src.output.formatter import PredictionFormatter

    print(PredictionFormatter().format_status(status))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="BTC Price Movement Prediction Engine"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--download",
        action="store_true",
        help="Download / resume full hourly price history",
    )
    group.add_argument(
        "--predict",
        action="store_true",
        help="Run a single prediction cycle",
    )
    group.add_argument(
        "--status",
        action="store_true",
        help="Show current data status",
    )
    args = parser.parse_args()

    config = load_config()
    engine = PredictionEngine(config)

    if args.download:
        asyncio.run(cmd_download(engine))
    elif args.predict:
        asyncio.run(cmd_predict(engine))
    elif args.status:
        cmd_status(engine)


if __name__ == "__main__":
    main()
