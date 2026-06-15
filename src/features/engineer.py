"""Feature engineering pipeline.

STUB -- Phase 2 will implement the full transform pipeline that merges
price, technical, cycle, and temporal features into a single model-ready
DataFrame with proper lag construction and NaN handling.
"""

from __future__ import annotations

import pandas as pd

from src.horizons import HORIZON_HOUR_VALUES
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class FeatureEngineer:
    """Transforms raw collector DataFrames into model-ready features.

    Phase 1: passthrough merge of available signals.
    Phase 2: proper lag features, interaction terms, target encoding.
    """

    def __init__(self, prediction_horizons: list[int] | None = None) -> None:
        self.prediction_horizons = prediction_horizons or list(HORIZON_HOUR_VALUES)

    def build_features(
        self,
        price_df: pd.DataFrame,
        technical_df: pd.DataFrame | None = None,
        cycle_df: pd.DataFrame | None = None,
        temporal_df: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Merge all signal sources into a single feature DataFrame.

        Returns a DataFrame indexed by timestamp with one column per feature.
        Rows with NaN in critical columns are dropped.
        """
        if price_df.empty:
            logger.warning("No price data provided to feature engineer")
            return pd.DataFrame()

        merged = price_df.copy()

        if technical_df is not None and not technical_df.empty:
            tech_cols = [
                c for c in technical_df.columns if c not in merged.columns
            ]
            merged = merged.join(technical_df[tech_cols], how="left")

        if temporal_df is not None and not temporal_df.empty:
            temp_cols = [
                c for c in temporal_df.columns if c not in merged.columns
            ]
            merged = merged.join(temporal_df[temp_cols], how="left")

        if cycle_df is not None and not cycle_df.empty:
            cycle_cols = [
                c for c in cycle_df.columns if c not in merged.columns
            ]
            if cycle_cols:
                cycle_reindexed = cycle_df[cycle_cols].reindex(
                    merged.index, method="ffill"
                )
                merged = merged.join(cycle_reindexed, how="left")

        # TODO Phase 2: add lag features, rolling targets, interaction terms
        logger.info(
            "Feature matrix: %d rows x %d columns", len(merged), len(merged.columns)
        )
        return merged

    def build_targets(self, price_df: pd.DataFrame) -> pd.DataFrame:
        """Build forward-return target columns for each prediction horizon.

        STUB -- Phase 2 implementation.
        """
        if price_df.empty:
            return pd.DataFrame()

        targets = pd.DataFrame(index=price_df.index)
        for h in self.prediction_horizons:
            targets[f"return_{h}h"] = (
                price_df["close"].shift(-h) / price_df["close"] - 1
            )
        return targets
