"""Market replay engine -- the heart of the training system.

Replays historical market conditions step by step, reconstructing the complete
market state visible at each moment without any lookahead bias. The model
experiences the market as if living through it in real-time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Generator, Optional, Protocol

import numpy as np
import pandas as pd

from src.collectors.cycle import HALVING_DATES, CycleCollector
from src.simulation.data_loader import HistoricalDataLoader
from src.simulation.labeler import ForwardReturnLabeler
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class PredictionModel(Protocol):
    """Protocol for any model that can receive market state and predict."""

    def predict(self, features: pd.DataFrame) -> dict[str, Any]: ...


@dataclass
class MarketState:
    """Complete market state visible at a single point in time."""

    timestamp: pd.Timestamp
    price_history: pd.DataFrame
    current_price: float
    features: dict[str, float]
    cycle_position: dict[str, Any]
    market_regime: str
    available_signals: list[str]


@dataclass
class SimulationStep:
    """One step of the simulation: state + model prediction + actual outcome."""

    timestamp: pd.Timestamp
    market_state: MarketState
    prediction: Optional[dict[str, Any]] = None
    actual_returns: Optional[dict[str, float]] = None
    regime: str = "unknown"


@dataclass
class SimulationResult:
    """Complete results from a simulation run."""

    steps: list[SimulationStep] = field(default_factory=list)
    start_date: Optional[pd.Timestamp] = None
    end_date: Optional[pd.Timestamp] = None
    total_steps: int = 0
    predictions_made: int = 0

    def to_dataframe(self) -> pd.DataFrame:
        """Convert results to a DataFrame for analysis."""
        records = []
        for step in self.steps:
            record = {
                "timestamp": step.timestamp,
                "regime": step.regime,
                "current_price": step.market_state.current_price,
            }
            if step.prediction:
                for k, v in step.prediction.items():
                    record[f"pred_{k}"] = v
            if step.actual_returns:
                for k, v in step.actual_returns.items():
                    record[f"actual_{k}"] = v
            records.append(record)
        return pd.DataFrame(records).set_index("timestamp") if records else pd.DataFrame()


class MarketReplay:
    """Core market simulation engine.

    Takes a date range and replays historical market conditions hourly,
    feeding each state to a model and recording predictions vs actuals.

    Key guarantees:
    - NO lookahead: at time T, only data from [0, T] is visible
    - Handles missing signals gracefully (NaN for unavailable data)
    - Tags each period with market regime for later analysis
    """

    def __init__(
        self,
        data_loader: Optional[HistoricalDataLoader] = None,
        labeler: Optional[ForwardReturnLabeler] = None,
        lookback_hours: int = 720,
    ) -> None:
        self.data_loader = data_loader or HistoricalDataLoader()
        self.labeler = labeler or ForwardReturnLabeler()
        self.lookback_hours = lookback_hours
        self._cycle_collector = CycleCollector()
        self._price_df: Optional[pd.DataFrame] = None
        self._labels_df: Optional[pd.DataFrame] = None
        self._ema_200: Optional[pd.Series] = None

    def run(
        self,
        start_date: str | pd.Timestamp,
        end_date: str | pd.Timestamp,
        model: Optional[PredictionModel] = None,
        step_hours: int = 1,
        progress_interval: int = 1000,
    ) -> SimulationResult:
        """Run a full simulation over the given date range.

        Args:
            start_date: Simulation start (must have sufficient lookback before this).
            end_date: Simulation end.
            model: Optional model to generate predictions at each step.
            step_hours: Hours between simulation steps (default 1 = hourly).
            progress_interval: Log progress every N steps.

        Returns:
            SimulationResult with all steps, predictions, and actuals.
        """
        start = pd.Timestamp(start_date, tz="UTC")
        end = pd.Timestamp(end_date, tz="UTC")

        self._prepare_data(start, end)

        result = SimulationResult(start_date=start, end_date=end)

        timestamps = self._get_simulation_timestamps(start, end, step_hours)
        result.total_steps = len(timestamps)

        logger.info(
            "Starting simulation: %s -> %s (%d steps, %dh interval)",
            start.date(), end.date(), len(timestamps), step_hours,
        )

        for i, ts in enumerate(timestamps):
            step = self._simulate_step(ts, model)
            result.steps.append(step)

            if step.prediction is not None:
                result.predictions_made += 1

            if (i + 1) % progress_interval == 0:
                logger.info("Simulation progress: %d/%d steps", i + 1, len(timestamps))

        logger.info(
            "Simulation complete: %d steps, %d predictions",
            result.total_steps, result.predictions_made,
        )
        return result

    def iterate(
        self,
        start_date: str | pd.Timestamp,
        end_date: str | pd.Timestamp,
        step_hours: int = 1,
    ) -> Generator[MarketState, None, None]:
        """Iterate over market states without running a model.

        Useful for feature extraction and label generation during training.
        Yields MarketState objects one at a time for memory efficiency.
        """
        start = pd.Timestamp(start_date, tz="UTC")
        end = pd.Timestamp(end_date, tz="UTC")

        self._prepare_data(start, end)
        timestamps = self._get_simulation_timestamps(start, end, step_hours)

        for ts in timestamps:
            yield self._build_market_state(ts)

    def get_features_and_labels(
        self,
        start_date: str | pd.Timestamp,
        end_date: str | pd.Timestamp,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Get the full feature matrix and labels for a date range.

        Efficient batch method for training -- computes all features and labels
        at once rather than iterating step-by-step.

        Returns:
            (features_df, labels_df) aligned on the same index.
        """
        start = pd.Timestamp(start_date, tz="UTC")
        end = pd.Timestamp(end_date, tz="UTC")

        merged = self.data_loader.get_merged_dataset()
        price_df = self.data_loader.load_price_data()

        data_slice = merged.loc[start:end]
        if data_slice.empty:
            raise ValueError(f"No data available for range {start} to {end}")

        labels = self.labeler.compute_labels(price_df.loc[start:end])

        aligned_labels = labels.reindex(data_slice.index)

        return data_slice, aligned_labels

    def _prepare_data(self, start: pd.Timestamp, end: pd.Timestamp) -> None:
        """Load and prepare all data needed for the simulation."""
        self._price_df = self.data_loader.load_price_data()
        self._labels_df = self.labeler.compute_labels(self._price_df)

        close = self._price_df["close"]
        self._ema_200 = close.ewm(span=200 * 24, adjust=False).mean()

        data_start = self._price_df.index[0]
        required_start = start - pd.Timedelta(hours=self.lookback_hours)
        if required_start < data_start:
            logger.warning(
                "Requested lookback extends before data start (%s). "
                "Simulation will have limited history at early timestamps.",
                data_start,
            )

    def _get_simulation_timestamps(
        self,
        start: pd.Timestamp,
        end: pd.Timestamp,
        step_hours: int,
    ) -> list[pd.Timestamp]:
        """Get the list of timestamps to simulate, filtered to available data."""
        assert self._price_df is not None

        all_timestamps = self._price_df.index
        mask = (all_timestamps >= start) & (all_timestamps <= end)
        available = all_timestamps[mask]

        if step_hours > 1:
            available = available[::step_hours]

        return list(available)

    def _simulate_step(
        self,
        timestamp: pd.Timestamp,
        model: Optional[PredictionModel],
    ) -> SimulationStep:
        """Execute one simulation step at the given timestamp."""
        state = self._build_market_state(timestamp)

        prediction = None
        if model is not None:
            try:
                feature_row = pd.DataFrame([state.features], index=[timestamp])
                prediction = model.predict(feature_row)
            except Exception as e:
                logger.debug("Model prediction failed at %s: %s", timestamp, e)

        actual_returns = self._get_actual_returns(timestamp)

        return SimulationStep(
            timestamp=timestamp,
            market_state=state,
            prediction=prediction,
            actual_returns=actual_returns,
            regime=state.market_regime,
        )

    def _build_market_state(self, timestamp: pd.Timestamp) -> MarketState:
        """Reconstruct the complete market state visible at a given moment."""
        assert self._price_df is not None

        lookback_start = timestamp - pd.Timedelta(hours=self.lookback_hours)
        price_history = self._price_df.loc[lookback_start:timestamp]

        current_price = float(self._price_df.loc[:timestamp, "close"].iloc[-1])

        features = self._extract_features_at(timestamp, price_history)

        cycle_pos = self._cycle_collector.get_cycle_position(
            timestamp.to_pydatetime().replace(tzinfo=timezone.utc)
        )

        regime = self._classify_market_regime(timestamp)

        merged = self.data_loader.get_merged_dataset()
        row = merged.loc[:timestamp].iloc[-1] if timestamp in merged.index or len(merged.loc[:timestamp]) > 0 else pd.Series()
        available_signals = [col for col in row.index if pd.notna(row[col])] if not row.empty else []

        return MarketState(
            timestamp=timestamp,
            price_history=price_history,
            current_price=current_price,
            features=features,
            cycle_position=cycle_pos,
            market_regime=regime,
            available_signals=available_signals,
        )

    def _extract_features_at(
        self,
        timestamp: pd.Timestamp,
        price_history: pd.DataFrame,
    ) -> dict[str, float]:
        """Extract all features visible at a given timestamp.

        Only uses data available up to and including the timestamp.
        """
        features: dict[str, float] = {}

        if price_history.empty:
            return features

        close = price_history["close"]
        current = float(close.iloc[-1])

        # Price returns over various lookback windows
        for hours, label in [(1, "1h"), (4, "4h"), (24, "24h"), (168, "7d"), (720, "30d")]:
            if len(close) > hours:
                past_price = float(close.iloc[-hours - 1])
                features[f"return_{label}"] = (current / past_price - 1) * 100
            else:
                features[f"return_{label}"] = np.nan

        # Volatility (std of hourly returns over 24h and 7d)
        if len(close) > 24:
            returns = close.pct_change().dropna()
            features["volatility_24h"] = float(returns.tail(24).std()) * 100
            if len(returns) > 168:
                features["volatility_7d"] = float(returns.tail(168).std()) * 100
            else:
                features["volatility_7d"] = np.nan
        else:
            features["volatility_24h"] = np.nan
            features["volatility_7d"] = np.nan

        # Volume ratio (current vs 20-period SMA)
        if "volume" in price_history.columns and len(price_history) > 20:
            vol = price_history["volume"]
            vol_sma = vol.rolling(20).mean()
            if pd.notna(vol_sma.iloc[-1]) and vol_sma.iloc[-1] > 0:
                features["volume_ratio"] = float(vol.iloc[-1] / vol_sma.iloc[-1])
            else:
                features["volume_ratio"] = np.nan
        else:
            features["volume_ratio"] = np.nan

        # Cycle features from collector
        dt = timestamp.to_pydatetime().replace(tzinfo=timezone.utc)
        cycle = self._cycle_collector.get_cycle_position(dt)
        features["days_since_halving"] = float(cycle["days_since_halving"])
        features["pct_through_cycle"] = float(cycle["pct_through_cycle"])

        # Temporal features
        features["hour_sin"] = float(np.sin(2 * np.pi * timestamp.hour / 24))
        features["hour_cos"] = float(np.cos(2 * np.pi * timestamp.hour / 24))
        features["day_sin"] = float(np.sin(2 * np.pi * timestamp.dayofweek / 7))
        features["day_cos"] = float(np.cos(2 * np.pi * timestamp.dayofweek / 7))

        # Merge any pre-computed signals available at this timestamp
        merged = self.data_loader.get_merged_dataset()
        if timestamp in merged.index:
            row = merged.loc[timestamp]
            signal_cols = [
                c for c in row.index
                if c not in ("open", "high", "low", "close", "volume")
            ]
            for col in signal_cols:
                val = row[col]
                if pd.notna(val) and isinstance(val, (int, float, np.floating, np.integer)):
                    features[col] = float(val)

        return features

    def _get_actual_returns(self, timestamp: pd.Timestamp) -> dict[str, float]:
        """Get the actual forward returns for a timestamp (the 'answer key')."""
        if self._labels_df is None:
            return {}

        if timestamp not in self._labels_df.index:
            return {}

        row = self._labels_df.loc[timestamp]
        returns = {}
        for name in self.labeler.horizons:
            col = f"return_{name}"
            if col in row.index and pd.notna(row[col]):
                returns[col] = float(row[col])
        return returns

    def _classify_market_regime(self, timestamp: pd.Timestamp) -> str:
        """Classify the market regime at a given timestamp using 200-day EMA trend.

        Uses only data available up to the timestamp (no lookahead).
        """
        if self._ema_200 is None or self._price_df is None:
            return "unknown"

        available_ema = self._ema_200.loc[:timestamp]
        if len(available_ema) < 200 * 24:
            return "unknown"

        current_price = float(self._price_df.loc[:timestamp, "close"].iloc[-1])
        current_ema = float(available_ema.iloc[-1])

        # Trend direction: compare current EMA to 30-day-ago EMA
        past_idx = max(0, len(available_ema) - 720)
        past_ema = float(available_ema.iloc[past_idx])
        ema_trend = (current_ema - past_ema) / past_ema if past_ema > 0 else 0

        price_vs_ema = (current_price - current_ema) / current_ema

        if price_vs_ema > 0.1 and ema_trend > 0.02:
            return "bull"
        elif price_vs_ema < -0.1 and ema_trend < -0.02:
            return "bear"
        else:
            return "sideways"
