"""Full BTC hourly price history collector.

Downloads and maintains complete hourly OHLCV data:
  - Bitstamp  2013-09 to 2017-08  (BTC/USD)
  - Binance   2017-08 to present  (BTC/USDT)

Data is stored as a single Parquet file.  Subsequent runs only fetch
candles newer than the last stored timestamp (incremental update).
The bulk download is resumable -- if interrupted it picks up from the
last checkpoint.
"""

from __future__ import annotations

import asyncio
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import ccxt
import pandas as pd

from src.collectors.base import BaseCollector
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

BINANCE_START = "2017-08-17T00:00:00Z"
BITSTAMP_START = "2013-09-01T00:00:00Z"
HOUR_MS = 3_600_000
MAX_RETRIES = 5
CHECKPOINT_EVERY = 50  # batches between checkpoint saves


class PriceCollector(BaseCollector):
    """Collects and maintains the full BTC hourly price history."""

    name = "price"
    update_interval_seconds = 900  # 15 min (fast tier)

    def __init__(self, storage_path: str = "data/price") -> None:
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.parquet_file = self.storage_path / "btc_hourly.parquet"
        self._df_cache: pd.DataFrame | None = None

    # ------------------------------------------------------------------
    # Exchange helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _create_exchange(exchange_id: str) -> ccxt.Exchange:
        cls = getattr(ccxt, exchange_id)
        return cls({"enableRateLimit": True})

    async def _fetch_ohlcv_batch(
        self,
        exchange: ccxt.Exchange,
        symbol: str,
        since_ms: int,
        limit: int = 1000,
    ) -> list:
        """Fetch one page of hourly candles with exponential-backoff retries."""
        for attempt in range(MAX_RETRIES):
            try:
                return exchange.fetch_ohlcv(
                    symbol, "1h", since=since_ms, limit=limit
                )
            except ccxt.RateLimitExceeded:
                wait = (2**attempt) * 5
                logger.warning(
                    "%s: rate-limited, backing off %ds…", exchange.id, wait
                )
                await asyncio.sleep(wait)
            except ccxt.NetworkError as exc:
                wait = (2**attempt) * 2
                logger.warning(
                    "%s: network error (attempt %d/%d): %s",
                    exchange.id,
                    attempt + 1,
                    MAX_RETRIES,
                    exc,
                )
                await asyncio.sleep(wait)
            except Exception as exc:
                logger.error(
                    "%s: unexpected error at %s: %s",
                    exchange.id,
                    since_ms,
                    exc,
                )
                if attempt < MAX_RETRIES - 1:
                    await asyncio.sleep(2**attempt)
                else:
                    return []
        return []

    # ------------------------------------------------------------------
    # Full history download (resumable)
    # ------------------------------------------------------------------

    async def download_full_history(
        self, progress_callback: Any = None
    ) -> pd.DataFrame:
        """Download the complete hourly BTC history.

        If a partial file already exists the download resumes from the
        last stored timestamp rather than starting from scratch.
        """
        existing = self.load_history()
        if not existing.empty:
            binance_start_ms = int(
                pd.Timestamp(BINANCE_START).timestamp() * 1000
            )
            last_ms = int(existing.index[-1].timestamp() * 1000)
            if last_ms >= binance_start_ms:
                logger.info(
                    "Resuming download from %s …", existing.index[-1]
                )
                return await self._resume_download(existing, progress_callback)

        frames: list[pd.DataFrame] = []

        # --- Bitstamp 2013-2017 ---
        logger.info("Phase 1/2: Bitstamp history (2013-2017) …")
        bitstamp = self._create_exchange("bitstamp")
        try:
            bdf = await self._download_exchange_range(
                bitstamp, "BTC/USD", BITSTAMP_START, BINANCE_START,
                label="Bitstamp", progress_callback=progress_callback,
            )
            if not bdf.empty:
                frames.append(bdf)
        finally:
            await bitstamp.close()

        # --- Binance 2017-present ---
        logger.info("Phase 2/2: Binance history (2017-present) …")
        binance = self._create_exchange("binance")
        try:
            bndf = await self._download_exchange_range(
                binance, "BTC/USDT", BINANCE_START, None,
                label="Binance", progress_callback=progress_callback,
            )
            if not bndf.empty:
                frames.append(bndf)
        finally:
            await binance.close()

        if not frames:
            logger.error("No price data downloaded from any exchange")
            return pd.DataFrame()

        combined = self._merge_frames(frames)
        self._save_parquet(combined)
        self._df_cache = combined
        logger.info(
            "Full history saved: %s candles, %s → %s",
            f"{len(combined):,}",
            combined.index[0],
            combined.index[-1],
        )
        return combined

    async def _resume_download(
        self,
        existing: pd.DataFrame,
        progress_callback: Any = None,
    ) -> pd.DataFrame:
        """Continue a partial download from the last stored candle."""
        since_ms = int(existing.index[-1].timestamp() * 1000) + HOUR_MS
        since_iso = pd.Timestamp(since_ms, unit="ms", tz="UTC").isoformat()

        binance = self._create_exchange("binance")
        try:
            new = await self._download_exchange_range(
                binance, "BTC/USDT", since_iso, None,
                label="Binance (resume)", progress_callback=progress_callback,
            )
        finally:
            await binance.close()

        if new.empty:
            logger.info("Price history already up to date")
            return existing

        combined = self._merge_frames([existing, new])
        self._save_parquet(combined)
        self._df_cache = combined
        logger.info(
            "Resumed download: +%s candles, total %s",
            f"{len(new):,}",
            f"{len(combined):,}",
        )
        return combined

    # ------------------------------------------------------------------
    # Core range downloader
    # ------------------------------------------------------------------

    async def _download_exchange_range(
        self,
        exchange: ccxt.Exchange,
        symbol: str,
        start_iso: str,
        end_iso: str | None,
        *,
        label: str = "",
        progress_callback: Any = None,
    ) -> pd.DataFrame:
        """Download hourly candles for a time range with progress logging."""
        since_ms = int(pd.Timestamp(start_iso).timestamp() * 1000)
        end_ms = (
            int(pd.Timestamp(end_iso).timestamp() * 1000)
            if end_iso
            else int(time.time() * 1000)
        )
        total_hours = (end_ms - since_ms) // HOUR_MS

        all_candles: list[list] = []
        batch_count = 0
        t0 = time.time()

        while since_ms < end_ms:
            candles = await self._fetch_ohlcv_batch(exchange, symbol, since_ms)
            if not candles:
                logger.warning(
                    "%s: no data at %s, stopping",
                    label,
                    pd.Timestamp(since_ms, unit="ms", tz="UTC"),
                )
                break

            all_candles.extend(candles)
            since_ms = candles[-1][0] + HOUR_MS
            batch_count += 1

            # Progress reporting
            if batch_count % 10 == 0:
                self._log_progress(
                    label, len(all_candles), total_hours, t0, progress_callback
                )

            # Periodic checkpoint saves for resilience
            if batch_count % CHECKPOINT_EVERY == 0 and all_candles:
                self._save_checkpoint(all_candles, label)

            await asyncio.sleep(exchange.rateLimit / 1000)

        if not all_candles:
            return pd.DataFrame()

        df = self._candles_to_df(all_candles)
        if end_iso:
            df = df[df.index < pd.Timestamp(end_iso, tz="UTC")]
        self._cleanup_checkpoints(label)

        logger.info("%s: finished — %s candles", label, f"{len(df):,}")
        return df

    # ------------------------------------------------------------------
    # Incremental update
    # ------------------------------------------------------------------

    async def incremental_update(self) -> pd.DataFrame:
        """Fetch only new candles since the last stored timestamp."""
        existing = self.load_history()
        if existing.empty:
            return await self.download_full_history()

        since_ms = int(existing.index[-1].timestamp() * 1000) + HOUR_MS

        binance = self._create_exchange("binance")
        new_candles: list[list] = []
        try:
            while True:
                batch = await self._fetch_ohlcv_batch(
                    binance, "BTC/USDT", since_ms
                )
                if not batch:
                    break
                new_candles.extend(batch)
                since_ms = batch[-1][0] + HOUR_MS
                if len(batch) < 1000:
                    break
                await asyncio.sleep(binance.rateLimit / 1000)
        finally:
            await binance.close()

        if not new_candles:
            logger.info("Price history already up to date")
            return existing

        new_df = self._candles_to_df(new_candles)
        combined = self._merge_frames([existing, new_df])
        self._save_parquet(combined)
        self._df_cache = combined
        logger.info(
            "Incremental update: +%s candles, total %s",
            f"{len(new_df):,}",
            f"{len(combined):,}",
        )
        return combined

    # ------------------------------------------------------------------
    # Storage helpers
    # ------------------------------------------------------------------

    def load_history(self) -> pd.DataFrame:
        """Load the stored hourly history from Parquet."""
        if self._df_cache is not None:
            return self._df_cache.copy()

        if not self.parquet_file.exists():
            return pd.DataFrame()

        df = pd.read_parquet(self.parquet_file)
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index, utc=True)
        self._df_cache = df
        return df.copy()

    def _save_parquet(self, df: pd.DataFrame) -> None:
        df.to_parquet(self.parquet_file, engine="pyarrow", compression="snappy")

    def _save_checkpoint(self, candles: list[list], label: str) -> None:
        tag = label.lower().replace(" ", "_").replace("(", "").replace(")", "")
        path = self.storage_path / f"_checkpoint_{tag}.parquet"
        df = self._candles_to_df(candles)
        df.to_parquet(path, engine="pyarrow", compression="snappy")
        logger.debug("Checkpoint: %s candles → %s", f"{len(df):,}", path.name)

    def _cleanup_checkpoints(self, label: str) -> None:
        tag = label.lower().replace(" ", "_").replace("(", "").replace(")", "")
        for p in self.storage_path.glob(f"_checkpoint_{tag}*"):
            p.unlink(missing_ok=True)

    # ------------------------------------------------------------------
    # Timeframe aggregation
    # ------------------------------------------------------------------

    def get_timeframe(self, timeframe: str) -> pd.DataFrame:
        """Resample hourly candles to a higher timeframe.

        Args:
            timeframe: '4h', '1d', '1w', or '1M'.
        """
        hourly = self.load_history()
        if hourly.empty:
            return hourly

        rule_map = {"4h": "4h", "1d": "1D", "1w": "1W", "1M": "1ME"}
        rule = rule_map.get(timeframe)
        if rule is None:
            raise ValueError(
                f"Unsupported timeframe {timeframe!r}; choose from {list(rule_map)}"
            )

        return (
            hourly.resample(rule)
            .agg(
                {"open": "first", "high": "max", "low": "min",
                 "close": "last", "volume": "sum"}
            )
            .dropna()
        )

    # ------------------------------------------------------------------
    # BaseCollector interface
    # ------------------------------------------------------------------

    async def _collect(self) -> pd.DataFrame:
        return await self.incremental_update()

    async def _get_latest(self) -> dict[str, Any]:
        df = self.load_history()
        if df.empty:
            return {}
        row = df.iloc[-1]
        return {
            "timestamp": str(df.index[-1]),
            "open": float(row["open"]),
            "high": float(row["high"]),
            "low": float(row["low"]),
            "close": float(row["close"]),
            "volume": float(row["volume"]),
        }

    async def _get_historical(
        self, start: str, end: str | None = None
    ) -> pd.DataFrame:
        df = self.load_history()
        if df.empty:
            df = await self.download_full_history()
        mask = df.index >= pd.Timestamp(start, tz="UTC")
        if end:
            mask &= df.index <= pd.Timestamp(end, tz="UTC")
        return df[mask]

    # ------------------------------------------------------------------
    # Internal utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _candles_to_df(candles: list[list]) -> pd.DataFrame:
        df = pd.DataFrame(
            candles,
            columns=["timestamp", "open", "high", "low", "close", "volume"],
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df.set_index("timestamp", inplace=True)
        return df

    @staticmethod
    def _merge_frames(frames: list[pd.DataFrame]) -> pd.DataFrame:
        combined = pd.concat(frames)
        combined = combined[~combined.index.duplicated(keep="last")]
        combined.sort_index(inplace=True)
        return combined

    @staticmethod
    def _log_progress(
        label: str,
        fetched: int,
        total: int,
        t0: float,
        callback: Any = None,
    ) -> None:
        pct = min(fetched / max(total, 1) * 100, 100)
        elapsed = time.time() - t0
        rate = fetched / max(elapsed, 1)
        eta_min = (total - fetched) / max(rate, 0.1) / 60
        logger.info(
            "%s: %s/%s candles (%.1f%%) — ETA %.0f min",
            label,
            f"{fetched:,}",
            f"{total:,}",
            pct,
            eta_min,
        )
        if callback:
            callback(label, fetched, total)
