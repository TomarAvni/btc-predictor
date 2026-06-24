"""Aggressive paper-trading profile metadata.

All trades route through ``OrderSimulator`` only — there is no exchange API or
real-money execution path. Thresholds are relaxed for paper experimentation
while ML and cost-aware gates remain in place.
"""

from __future__ import annotations

from src.trading.position_sizer import PositionSizer
from src.trading.risk_manager import RiskManager
from src.trading.strategy import TradingStrategy


PAPER_ONLY: bool = True
PROFILE_LABEL: str = "Aggressive Paper (simulated only)"


def get_paper_profile_summary() -> dict[str, str | float | int]:
    """Human-readable snapshot of active paper-trading thresholds."""
    return {
        "mode": PROFILE_LABEL,
        "paper_only": PAPER_ONLY,
        "min_confidence_pct": TradingStrategy.MIN_CONFIDENCE,
        "min_move_pct": TradingStrategy.MIN_ACTIONABLE_MOVE_PCT,
        "max_same_direction": TradingStrategy.MAX_SAME_DIRECTION_POSITIONS,
        "min_evidence_samples": TradingStrategy.MIN_EVIDENCE_SAMPLES,
        "max_open_positions": RiskManager.MAX_OPEN_POSITIONS,
        "max_exposure_pct": RiskManager.MAX_EXPOSURE_PCT,
        "max_position_pct": PositionSizer.MAX_POSITION_PCT * 100,
        "kelly_fraction": PositionSizer.KELLY_FRACTION,
    }
