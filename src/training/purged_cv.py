"""Purged, embargoed walk-forward cross-validation for time-series ML.

This is the methodologically-correct way to validate a forecasting model on
overlapping-label financial data. Two leakage sources are eliminated:

1. **Look-ahead across the train/test boundary.** A training sample at time ``t``
   carries a forward-return label that "sees" prices up to ``t + horizon``. If
   ``t + horizon`` falls inside the test window, that label leaks future
   (test-period) information into training. We therefore *purge* every training
   sample whose label window overlaps the test block by dropping samples with
   ``t > test_start - horizon`` (a per-horizon embargo on the train side).

2. **Serial-correlation bleed at the seam.** An additional fixed ``embargo`` can
   be applied so the train set ends a little before the purge boundary.

The splitter produces contiguous, chronologically-ordered test blocks over the
tail of the series (so every test prediction is strictly out-of-sample relative
to its training data). Test-block boundaries are *horizon-independent*, which
lets us pool out-of-fold predictions across horizons for calibration while still
purging correctly per horizon.

References: López de Prado, *Advances in Financial Machine Learning* (2018),
ch. 7 (purged k-fold / embargoing).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Optional

import numpy as np
import pandas as pd

from src.utils.logger import setup_logger

logger = setup_logger(__name__)


@dataclass
class CVFold:
    """One walk-forward fold (integer positions into the supplied index)."""

    fold_number: int
    train_positions: np.ndarray
    test_positions: np.ndarray
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    horizon_hours: int
    embargo_hours: int

    @property
    def n_train(self) -> int:
        return len(self.train_positions)

    @property
    def n_test(self) -> int:
        return len(self.test_positions)


class PurgedWalkForwardCV:
    """Expanding- or rolling-window walk-forward CV with purge + embargo.

    Args:
        n_splits: Number of contiguous test blocks carved from the tail.
        embargo_hours: Extra gap (in hours) between the purge boundary and the
            train set, on top of the per-horizon purge. Guards against
            serial-correlation bleed. Default 0 (purge alone is usually enough).
        min_train_frac: Minimum fraction of the series that must precede the
            first test block, so fold 1 still has a substantial training set.
        mode: ``"expanding"`` (train always starts at the series start) or
            ``"rolling"`` (fixed-length training window).
        rolling_window_hours: Training-window length for ``mode="rolling"``.
    """

    def __init__(
        self,
        n_splits: int = 5,
        embargo_hours: int = 0,
        min_train_frac: float = 0.4,
        mode: str = "expanding",
        rolling_window_hours: Optional[int] = None,
    ) -> None:
        if n_splits < 1:
            raise ValueError("n_splits must be >= 1")
        if mode not in ("expanding", "rolling"):
            raise ValueError("mode must be 'expanding' or 'rolling'")
        self.n_splits = n_splits
        self.embargo_hours = max(0, int(embargo_hours))
        self.min_train_frac = min_train_frac
        self.mode = mode
        self.rolling_window_hours = rolling_window_hours

    def split(
        self,
        index: pd.DatetimeIndex,
        horizon_hours: int,
        label_valid: Optional[pd.Series] = None,
    ) -> Iterator[CVFold]:
        """Yield purged walk-forward folds for a given label horizon.

        Args:
            index: The (sorted, unique) DatetimeIndex of the feature matrix.
            horizon_hours: Forward-return horizon of the label, in hours. Used
                for purging: training samples whose label window reaches into
                the test block are dropped.
            label_valid: Optional boolean Series aligned to ``index`` marking
                rows whose label is non-NaN. Rows with invalid labels are
                excluded from both train and test.

        Yields:
            ``CVFold`` objects with integer positions into ``index``.
        """
        n = len(index)
        if n < 2:
            return

        if not index.is_monotonic_increasing:
            raise ValueError("index must be sorted ascending for walk-forward CV")

        valid_mask = (
            label_valid.to_numpy(dtype=bool)
            if label_valid is not None
            else np.ones(n, dtype=bool)
        )

        # First test block starts after min_train_frac of the series.
        first_test_pos = int(n * self.min_train_frac)
        first_test_pos = max(first_test_pos, 1)
        if first_test_pos >= n:
            logger.warning("min_train_frac leaves no room for test blocks")
            return

        # Carve the tail [first_test_pos, n) into n_splits contiguous blocks.
        block_bounds = np.linspace(first_test_pos, n, self.n_splits + 1, dtype=int)
        purge_gap = pd.Timedelta(hours=horizon_hours + self.embargo_hours)

        fold_num = 0
        for i in range(self.n_splits):
            test_lo = int(block_bounds[i])
            test_hi = int(block_bounds[i + 1])
            if test_hi <= test_lo:
                continue

            test_start_ts = index[test_lo]
            # Train must end before the test block by the full purge gap.
            train_cutoff_ts = test_start_ts - purge_gap

            train_upper_mask = index < train_cutoff_ts
            if self.mode == "rolling" and self.rolling_window_hours:
                lower_ts = train_cutoff_ts - pd.Timedelta(hours=self.rolling_window_hours)
                train_window_mask = index >= lower_ts
            else:
                train_window_mask = np.ones(n, dtype=bool)

            train_sel = train_upper_mask & train_window_mask & valid_mask

            test_block_mask = np.zeros(n, dtype=bool)
            test_block_mask[test_lo:test_hi] = True
            test_sel = test_block_mask & valid_mask

            train_positions = np.flatnonzero(train_sel)
            test_positions = np.flatnonzero(test_sel)

            if len(train_positions) == 0 or len(test_positions) == 0:
                logger.warning(
                    "Fold %d skipped (train=%d, test=%d) after purge",
                    i + 1, len(train_positions), len(test_positions),
                )
                continue

            fold_num += 1
            yield CVFold(
                fold_number=fold_num,
                train_positions=train_positions,
                test_positions=test_positions,
                train_start=index[train_positions[0]],
                train_end=index[train_positions[-1]],
                test_start=index[test_positions[0]],
                test_end=index[test_positions[-1]],
                horizon_hours=horizon_hours,
                embargo_hours=self.embargo_hours,
            )
