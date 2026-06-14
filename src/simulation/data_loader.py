"""Historical data manager for the market simulation.

Loads hourly price data from Parquet files, aligns all signals to the same
timeline, forward-fills slower signals, and provides efficient point-in-time
slicing so the simulator never exposes future data.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src import DATA_DIR
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

PRICE_DIR = DATA_DIR / "price"
HISTORY_DIR = DATA_DIR / "history"


class HistoricalDataLoader:
    """Loads and manages all historical data for market simulation.

    Responsibilities:
    - Load hourly OHLCV from Parquet files
    - Load cached signal history (TA, on-chain, sentiment, etc.)
    - Align everything to a unified hourly DatetimeIndex
    - Forward-fill slower signals (e.g. daily macro → repeated per hour)
    - Provide efficient slicing: all data available up to time T
    """

    def __init__(
        self,
        price_dir: Optional[Path] = None,
        history_dir: Optional[Path] = None,
    ) -> None:
        self.price_dir = price_dir or PRICE_DIR
        self.history_dir = history_dir or HISTORY_DIR
        self._price_df: Optional[pd.DataFrame] = None
        self._signals: dict[str, pd.DataFrame] = {}
        self._merged: Optional[pd.DataFrame] = None

    def load_price_data(self) -> pd.DataFrame:
        """Load hourly OHLCV data from Parquet files in data/price/.

        Supports both a single combined file and multiple per-year files.
        Returns DataFrame with DatetimeIndex and columns: open, high, low, close, volume.
        """
        if self._price_df is not None:
            return self._price_df

        self.price_dir.mkdir(parents=True, exist_ok=True)

        parquet_files = sorted(self.price_dir.glob("*.parquet"))
        if not parquet_files:
            raise FileNotFoundError(
                f"No Parquet files found in {self.price_dir}. "
                "Run the price download first: python main.py --download"
            )

        frames: list[pd.DataFrame] = []
        for pf in parquet_files:
            df = pd.read_parquet(pf)
            frames.append(df)
            logger.info("Loaded %s: %d rows", pf.name, len(df))

        combined = pd.concat(frames, axis=0)
        combined.sort_index(inplace=True)
        combined = combined[~combined.index.duplicated(keep="first")]

        required_cols = {"open", "high", "low", "close", "volume"}
        available_cols = set(combined.columns.str.lower())
        if not required_cols.issubset(available_cols):
            combined.columns = combined.columns.str.lower()

        if not required_cols.issubset(set(combined.columns)):
            missing = required_cols - set(combined.columns)
            raise ValueError(f"Price data missing columns: {missing}")

        if not isinstance(combined.index, pd.DatetimeIndex):
            if "timestamp" in combined.columns:
                combined.set_index("timestamp", inplace=True)
                combined.index = pd.to_datetime(combined.index, utc=True)
            else:
                combined.index = pd.to_datetime(combined.index, utc=True)

        combined.index.name = "timestamp"
        self._price_df = combined[["open", "high", "low", "close", "volume"]]
        logger.info(
            "Price data loaded: %d candles from %s to %s",
            len(self._price_df),
            self._price_df.index[0],
            self._price_df.index[-1],
        )
        return self._price_df

    def load_signal_history(self, signal_name: str) -> Optional[pd.DataFrame]:
        """Load a cached signal history from data/history/<signal_name>.parquet.

        Returns None if the file doesn't exist (signal not yet available).
        """
        if signal_name in self._signals:
            return self._signals[signal_name]

        self.history_dir.mkdir(parents=True, exist_ok=True)
        path = self.history_dir / f"{signal_name}.parquet"
        if not path.exists():
            logger.debug("Signal history not found: %s", path)
            return None

        df = pd.read_parquet(path)
        if not isinstance(df.index, pd.DatetimeIndex):
            if "timestamp" in df.columns:
                df.set_index("timestamp", inplace=True)
            df.index = pd.to_datetime(df.index, utc=True)

        self._signals[signal_name] = df
        logger.info("Loaded signal '%s': %d rows", signal_name, len(df))
        return df

    def get_merged_dataset(self, reload: bool = False) -> pd.DataFrame:
        """Get the fully merged dataset with price + all available signals.

        All signals are aligned to the hourly price timeline via forward-fill
        for slower (daily/weekly) signals.
        """
        if self._merged is not None and not reload:
            return self._merged

        price_df = self.load_price_data()
        merged = price_df.copy()

        signal_names = [
            "technical", "cycle", "macro", "sentiment",
            "onchain", "institutional", "options",
        ]

        for name in signal_names:
            signal_df = self.load_signal_history(name)
            if signal_df is None:
                continue

            new_cols = [c for c in signal_df.columns if c not in merged.columns]
            if not new_cols:
                continue

            signal_subset = signal_df[new_cols]
            signal_reindexed = signal_subset.reindex(merged.index, method="ffill")
            merged = merged.join(signal_reindexed, how="left")
            logger.info("Merged signal '%s': %d columns", name, len(new_cols))

        self._merged = merged
        logger.info(
            "Merged dataset: %d rows × %d columns",
            len(merged), len(merged.columns),
        )
        return self._merged

    def get_data_up_to(self, timestamp: pd.Timestamp) -> pd.DataFrame:
        """Get all data available up to (and including) a given timestamp.

        This is the core method for preventing lookahead bias: the simulation
        only ever calls this to get the model's visible data at time T.
        """
        merged = self.get_merged_dataset()
        return merged.loc[:timestamp]

    def get_slice(
        self,
        start: pd.Timestamp,
        end: pd.Timestamp,
    ) -> pd.DataFrame:
        """Get data for a specific time range (inclusive on both ends)."""
        merged = self.get_merged_dataset()
        return merged.loc[start:end]

    def get_date_range(self) -> tuple[pd.Timestamp, pd.Timestamp]:
        """Return the first and last timestamps in the price data."""
        price_df = self.load_price_data()
        return price_df.index[0], price_df.index[-1]

    @property
    def hourly_index(self) -> pd.DatetimeIndex:
        """The full hourly timeline from the price data."""
        return self.load_price_data().index


DataLoader = HistoricalDataLoader
