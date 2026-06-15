"""Single source of truth for prediction horizons.

Every component (prediction engine, model training, ensemble, dashboard,
trading and validation) imports its horizon definitions from this module so
the horizon set can never drift between layers.

The horizon set is a CONTINUOUS 6-hour-step curve from 6h to 168h (= 7 days),
plus a single long-range 30-day point:

    6h, 12h, 18h, 24h, ..., 162h, 168h, 30d

Labelling rules:
    * Everything from 6h through 168h uses consistent HOUR-based labels
      ("6h", "12h", ... "168h").  Note 168h == 7 days, so the old "7d"
      horizon is now the 168h point.
    * The long-range point keeps its calendar label "30d" (== 720 hours).
"""

from __future__ import annotations

# ── Curve definition ────────────────────────────────────────────────────────
HOURLY_STEP = 6
HOURLY_MAX = 168  # 7 days

# Continuous intraday/weekly curve: "6h" .. "168h" (28 points).
HOURLY_HORIZONS: list[str] = [
    f"{h}h" for h in range(HOURLY_STEP, HOURLY_MAX + 1, HOURLY_STEP)
]

# Long-range calendar point(s) kept on their own scale.
LONG_HORIZONS: list[str] = ["30d"]

# Canonical, ordered horizon list used everywhere (29 points).
TIMEFRAMES: list[str] = HOURLY_HORIZONS + LONG_HORIZONS

# Horizon label -> forward window length in hours.
HORIZON_HOURS: dict[str, int] = {
    f"{h}h": h for h in range(HOURLY_STEP, HOURLY_MAX + 1, HOURLY_STEP)
}
HORIZON_HOURS["30d"] = 720

# Integer hour windows used by feature builders / TFT multi-horizon heads.
HORIZON_HOUR_VALUES: list[int] = [HORIZON_HOURS[tf] for tf in TIMEFRAMES]

# Legacy model-directory aliases: reuse an already-trained artifact whose
# horizon still exists under a new label.  The old "7d" model predicts the
# 168-hour forward return, i.e. exactly the new "168h" point, so the runtime
# falls back to xgb_7d / baseline_7d when xgb_168h / baseline_168h is absent.
LEGACY_MODEL_ALIASES: dict[str, str] = {"168h": "7d"}

# Compact set of "headline" horizons for summary widgets (cards / gauges /
# table columns) so the UI stays readable despite the full 29-point curve.
KEY_HORIZONS: list[str] = ["24h", "72h", "168h", "30d"]


def horizon_hours(label: str) -> int | None:
    """Return the forward window length in hours for *label* (or ``None``)."""
    return HORIZON_HOURS.get(label)
