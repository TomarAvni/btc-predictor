"""Forward return labeler for training data.

Computes the actual future returns at multiple horizons for each timestamp,
producing the training targets (labels) that the model learns to predict.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from src.horizons import HORIZON_HOURS
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

# Forward-return horizons keyed by label -> window length in hours
# (single source of truth: src/horizons.py).
HORIZONS = dict(HORIZON_HOURS)

MAGNITUDE_BUCKETS = {
    "small": (0, 2),
    "medium": (2, 5),
    "large": (5, 10),
    "extreme": (10, float("inf")),
}


class ForwardReturnLabeler:
    """Creates training labels from historical price data.

    For each hourly timestamp, computes:
    - Forward returns at every 6h step from 6h to 168h (7d), plus 30d
    - Direction (UP/DOWN) for each horizon
    - Magnitude bucket (small/medium/large/extreme) for each horizon
    """

    def __init__(self, horizons: Optional[dict[str, int]] = None) -> None:
        self.horizons = horizons or HORIZONS

    def compute_labels(self, price_df: pd.DataFrame) -> pd.DataFrame:
        """Compute all forward return labels from an OHLCV DataFrame.

        Args:
            price_df: DataFrame with DatetimeIndex and 'close' column.

        Returns:
            DataFrame with the same index as price_df containing:
            - return_{horizon}: percentage forward return
            - direction_{horizon}: 1 (UP) or 0 (DOWN)
            - magnitude_{horizon}: categorical bucket name
            Rows near the end that lack sufficient forward data will have NaN.
        """
        if price_df.empty or "close" not in price_df.columns:
            logger.warning("Cannot compute labels: empty or missing 'close' column")
            return pd.DataFrame(index=price_df.index)

        close = price_df["close"].astype(float)
        labels = pd.DataFrame(index=price_df.index)

        for name, hours in self.horizons.items():
            future_close = close.shift(-hours)
            pct_return = (future_close / close - 1) * 100

            labels[f"return_{name}"] = pct_return
            labels[f"direction_{name}"] = (pct_return > 0).astype(int)
            labels[f"magnitude_{name}"] = pct_return.abs().apply(self._classify_magnitude)

        n_valid = labels[f"return_{list(self.horizons.keys())[0]}"].notna().sum()
        n_total = len(labels)
        logger.info(
            "Labels computed: %d/%d timestamps have full forward returns",
            n_valid, n_total,
        )

        return labels

    def compute_returns_only(self, price_df: pd.DataFrame) -> pd.DataFrame:
        """Compute only the numeric forward returns (no direction/magnitude).

        Lighter version for quick use in walk-forward validation.
        """
        close = price_df["close"].astype(float)
        returns = pd.DataFrame(index=price_df.index)

        for name, hours in self.horizons.items():
            future_close = close.shift(-hours)
            returns[f"return_{name}"] = (future_close / close - 1) * 100

        return returns

    def get_valid_range(self, price_df: pd.DataFrame) -> tuple[pd.Timestamp, pd.Timestamp]:
        """Return the timestamp range where all forward returns are available.

        The end is cut short by the longest horizon.
        """
        max_horizon = max(self.horizons.values())
        start = price_df.index[0]
        end = price_df.index[-max_horizon] if len(price_df) > max_horizon else price_df.index[-1]
        return start, end

    @staticmethod
    def _classify_magnitude(abs_pct: float) -> str:
        """Classify absolute percentage change into magnitude bucket."""
        if pd.isna(abs_pct):
            return "unknown"
        for bucket, (low, high) in MAGNITUDE_BUCKETS.items():
            if low <= abs_pct < high:
                return bucket
        return "extreme"

    def get_label_statistics(self, labels_df: pd.DataFrame) -> dict:
        """Compute summary statistics of the labels for diagnostics."""
        stats: dict = {}

        for name in self.horizons:
            ret_col = f"return_{name}"
            dir_col = f"direction_{name}"
            mag_col = f"magnitude_{name}"

            if ret_col not in labels_df.columns:
                continue

            valid = labels_df[ret_col].dropna()
            stats[name] = {
                "count": len(valid),
                "mean_return": float(valid.mean()),
                "std_return": float(valid.std()),
                "median_return": float(valid.median()),
                "pct_up": float(labels_df[dir_col].dropna().mean() * 100),
                "magnitude_distribution": (
                    labels_df[mag_col].value_counts(normalize=True).to_dict()
                    if mag_col in labels_df.columns else {}
                ),
            }

        return stats


Labeler = ForwardReturnLabeler
