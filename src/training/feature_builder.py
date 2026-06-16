"""Training feature engineering.

Builds the complete feature matrix for model training from raw market data.
Computes price-based, technical, cycle, temporal, and placeholder features,
then applies proper scaling for model consumption.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.features.scaler import FeatureScaler
from src.simulation.labeler import ForwardReturnLabeler
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

PRICE_RETURN_WINDOWS = {
    "return_1h": 1,
    "return_4h": 4,
    "return_24h": 24,
    "return_7d": 168,
    "return_30d": 720,
}

TA_COLUMNS = [
    "rsi_14", "rsi_7", "macd", "macd_signal", "macd_histogram",
    "bb_upper", "bb_lower", "bb_width",
    "ema_9", "ema_21", "ema_50", "ema_200",
    "ema_9_21_cross", "ema_50_200_cross",
    "atr_14", "volume_ratio", "volume_sma_20",
    "stoch_rsi_k", "stoch_rsi_d", "adx", "obv",
    "price_position_24h", "price_position_168h",
]

PLACEHOLDER_COLUMNS = [
    "exchange_netflow", "mvrv_zscore", "nupl",
    "sopr", "funding_rate_avg", "open_interest_change",
    "fear_greed_index",
    "etf_net_flow", "grayscale_premium",
    "put_call_ratio", "max_pain_distance",
    "dxy_change", "sp500_correlation",
]

# Tweet-derived feature columns (supersede the old "social_volume" placeholder).
# Sourced from the merged dataset when data/history/twitter_llm_signal.parquet
# exists; otherwise added as NaN placeholders so the schema stays stable.
from src.features.tweet_aggregator import SIGNAL_COLUMNS as TWEET_FEATURE_COLUMNS


class TrainingFeatureBuilder:
    """Builds the model-ready feature matrix from raw market data.

    Computes:
    - Price-based: returns over lookback windows, volatility, volume ratio
    - TA indicators: RSI, MACD, BB %B, EMA ratios (from pre-computed data)
    - Cycle features: days since halving, cycle %, phase encoding
    - Temporal: hour/day/month (sin/cos encoded)
    - Power law: deviation from regression (if available)
    - Placeholders: columns for future signals (NaN until data arrives)
    """

    def __init__(self, model_dir: Optional[Path] = None) -> None:
        self.model_dir = Path(model_dir) if model_dir else Path("data/models")
        self.scaler = FeatureScaler(str(self.model_dir))
        self._feature_names: list[str] = []
        self._is_fitted = False

    def build_features(
        self,
        price_df: pd.DataFrame,
        include_placeholders: bool = True,
        include_tweets: bool = False,
    ) -> pd.DataFrame:
        """Build the full feature matrix from price + available signals.

        Args:
            price_df: DataFrame with OHLCV columns and DatetimeIndex.
                      May also contain pre-computed TA columns from the
                      TechnicalCollector or merged dataset.
            include_placeholders: Whether to add NaN columns for future signals.
            include_tweets: Whether to add the X/Twitter sentiment feature
                columns. Default False keeps the ``numbers`` model pure; the
                ``llm_calibrated`` / ``blended`` tracks opt in by passing True.

        Returns:
            DataFrame with one row per timestamp and all feature columns.
        """
        if price_df.empty:
            logger.warning("Empty price data passed to feature builder")
            return pd.DataFrame()

        features = pd.DataFrame(index=price_df.index)

        features = self._add_price_features(features, price_df)
        features = self._add_volatility_features(features, price_df)
        features = self._add_ta_features(features, price_df)
        features = self._add_cycle_features(features)
        features = self._add_temporal_features(features)
        if include_tweets:
            features = self._add_tweet_features(features, price_df)

        if include_placeholders:
            features = self._add_placeholder_columns(features, price_df)

        features.replace([np.inf, -np.inf], np.nan, inplace=True)

        self._feature_names = features.columns.tolist()
        logger.info(
            "Feature matrix built: %d rows × %d features",
            len(features), len(features.columns),
        )
        return features

    def fit_scaler(self, features_df: pd.DataFrame) -> None:
        """Fit the feature scaler on training data."""
        numeric = features_df.select_dtypes(include=[np.number])
        self.scaler.fit(numeric)
        self.scaler.save("training_scaler")
        self._is_fitted = True
        logger.info("Scaler fitted on %d features", len(numeric.columns))

    def transform(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Apply the fitted scaler to transform features."""
        if not self._is_fitted:
            self.scaler.load("training_scaler")
            self._is_fitted = True
        return self.scaler.transform(features_df)

    def fit_transform(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Fit scaler and transform in one step."""
        self.fit_scaler(features_df)
        return self.transform(features_df)

    @property
    def feature_names(self) -> list[str]:
        return self._feature_names

    def _add_price_features(
        self, features: pd.DataFrame, price_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Add price return features over various lookback windows."""
        close = price_df["close"]

        for name, hours in PRICE_RETURN_WINDOWS.items():
            features[name] = (close / close.shift(hours) - 1) * 100

        # High-low range as percentage of close
        if "high" in price_df.columns and "low" in price_df.columns:
            features["hl_range_pct"] = (
                (price_df["high"] - price_df["low"]) / close * 100
            )

        # Close position within the candle
        if all(c in price_df.columns for c in ("open", "high", "low")):
            hl_range = price_df["high"] - price_df["low"]
            features["candle_body_pct"] = np.where(
                hl_range > 0,
                (close - price_df["open"]) / hl_range,
                0,
            )

        return features

    def _add_volatility_features(
        self, features: pd.DataFrame, price_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Add volatility measures."""
        returns = price_df["close"].pct_change()

        features["volatility_24h"] = returns.rolling(24).std() * 100
        features["volatility_7d"] = returns.rolling(168).std() * 100
        features["volatility_30d"] = returns.rolling(720).std() * 100

        # Volatility ratio (short-term vs long-term)
        features["vol_ratio_24h_7d"] = np.where(
            features["volatility_7d"] > 0,
            features["volatility_24h"] / features["volatility_7d"],
            np.nan,
        )

        return features

    def _add_ta_features(
        self, features: pd.DataFrame, price_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Add technical analysis features from pre-computed columns or compute key ones."""
        close = price_df["close"]

        for col in TA_COLUMNS:
            if col in price_df.columns:
                features[col] = price_df[col]

        # Compute derived TA ratios if EMAs are available
        if "ema_50" in features.columns and "ema_200" in features.columns:
            features["ema_50_200_ratio"] = np.where(
                features["ema_200"] > 0,
                features["ema_50"] / features["ema_200"],
                np.nan,
            )

        # BB %B (position within Bollinger Bands)
        if "bb_upper" in features.columns and "bb_lower" in features.columns:
            bb_range = features["bb_upper"] - features["bb_lower"]
            features["bb_pct_b"] = np.where(
                bb_range > 0,
                (close - features["bb_lower"]) / bb_range,
                0.5,
            )

        # RSI is already computed if available; add RSI derivative
        if "rsi_14" in features.columns:
            features["rsi_14_change"] = features["rsi_14"].diff(periods=4)

        return features

    def _add_cycle_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Add halving cycle position features."""
        from src.collectors.cycle import HALVING_DATES, AVG_CYCLE_DAYS

        timestamps = features.index

        days_since = pd.Series(index=timestamps, dtype=float)
        cycle_pct = pd.Series(index=timestamps, dtype=float)
        cycle_number = pd.Series(index=timestamps, dtype=float)

        for ts in timestamps:
            dt = ts.to_pydatetime()
            if dt.tzinfo is None:
                from datetime import timezone as tz
                dt = dt.replace(tzinfo=tz.utc)

            past_halvings = [h for h in HALVING_DATES if h <= dt]
            if past_halvings:
                last = past_halvings[-1]
                d = (dt - last).days
                days_since[ts] = d
                cycle_pct[ts] = d / AVG_CYCLE_DAYS
                cycle_number[ts] = len(past_halvings)
            else:
                days_since[ts] = 0
                cycle_pct[ts] = 0
                cycle_number[ts] = 0

        features["days_since_halving"] = days_since
        features["cycle_pct"] = cycle_pct
        features["cycle_number"] = cycle_number

        # Encode cycle phase as one-hot
        features["phase_accumulation"] = (cycle_pct < 0.15).astype(int)
        features["phase_markup"] = ((cycle_pct >= 0.15) & (cycle_pct < 0.50)).astype(int)
        features["phase_distribution"] = ((cycle_pct >= 0.50) & (cycle_pct < 0.75)).astype(int)
        features["phase_markdown"] = (cycle_pct >= 0.75).astype(int)

        return features

    def _add_temporal_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Add time-based cyclical features."""
        if not isinstance(features.index, pd.DatetimeIndex):
            return features

        features["hour_sin"] = np.sin(2 * np.pi * features.index.hour / 24)
        features["hour_cos"] = np.cos(2 * np.pi * features.index.hour / 24)
        features["day_sin"] = np.sin(2 * np.pi * features.index.dayofweek / 7)
        features["day_cos"] = np.cos(2 * np.pi * features.index.dayofweek / 7)
        features["month_sin"] = np.sin(2 * np.pi * features.index.month / 12)
        features["month_cos"] = np.cos(2 * np.pi * features.index.month / 12)

        return features

    def _add_tweet_features(
        self, features: pd.DataFrame, price_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Copy tweet-sentiment columns from the merged dataset when present.

        When the tweet signal history is absent the columns are added as NaN so
        the feature schema is identical with and without the Twitter module
        (placeholders are filled with 0 at scale/transform time).
        """
        for col in TWEET_FEATURE_COLUMNS:
            if col in price_df.columns:
                features[col] = price_df[col]
            elif col not in features.columns:
                features[col] = np.nan
        return features

    def _add_placeholder_columns(
        self, features: pd.DataFrame, price_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Add NaN placeholder columns for signals not yet available.

        These get populated when the corresponding collectors are built.
        """
        for col in PLACEHOLDER_COLUMNS:
            if col not in features.columns and col not in price_df.columns:
                features[col] = np.nan

        return features


FeatureBuilder = TrainingFeatureBuilder
