"""Position sizing logic using a modified Kelly Criterion.

Determines how much capital to allocate to each trade based on
model confidence, multi-timeframe alignment, and portfolio state.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class SizingResult:
    """Result of a position sizing calculation."""

    should_trade: bool
    amount_usd: float
    position_pct: float  # % of portfolio
    tier: str  # e.g. "Small", "Medium", "Large", "Maximum"
    kelly_fraction: float
    adjustments: list[str]  # Human-readable adjustment reasons


class PositionSizer:
    """Confidence-based position sizing with Kelly Criterion."""

    MIN_CONFIDENCE: float = 55.0
    MIN_POSITION_USD: float = 50.0
    MAX_POSITION_PCT: float = 0.40  # 40% of portfolio
    KELLY_FRACTION: float = 0.25  # Use 25% of Kelly suggestion

    TIERS = [
        (55.0, 65.0, 0.05, 0.10, "Small"),
        (65.0, 75.0, 0.10, 0.20, "Medium"),
        (75.0, 85.0, 0.20, 0.30, "Large"),
        (85.0, 100.1, 0.30, 0.40, "Maximum"),
    ]

    def calculate(
        self,
        confidence: float,
        portfolio_value: float,
        alignment_score: float = 1.0,
        current_drawdown_pct: float = 0.0,
        open_position_count: int = 0,
    ) -> SizingResult:
        """Calculate position size for a trade.

        Args:
            confidence: Model confidence (0-100).
            portfolio_value: Current total portfolio value in USD.
            alignment_score: Multi-timeframe alignment score.
            current_drawdown_pct: Current portfolio drawdown (positive = drawdown).
            open_position_count: Number of currently open positions.

        Returns:
            SizingResult with trade decision and amount.
        """
        adjustments: list[str] = []

        if confidence < self.MIN_CONFIDENCE:
            return SizingResult(
                should_trade=False,
                amount_usd=0.0,
                position_pct=0.0,
                tier="No Trade",
                kelly_fraction=0.0,
                adjustments=["Confidence below minimum threshold (55%)"],
            )

        # Kelly fraction: (confidence * 2 - 1) for binary outcomes, capped
        kelly_raw = (confidence / 100.0 * 2 - 1)
        kelly_sized = kelly_raw * self.KELLY_FRACTION
        kelly_amount = kelly_sized * portfolio_value

        # Tier-based sizing
        tier_name = "Small"
        tier_min_pct = 0.05
        tier_max_pct = 0.10
        for low, high, t_min, t_max, name in self.TIERS:
            if low <= confidence < high:
                tier_name = name
                tier_min_pct = t_min
                tier_max_pct = t_max
                break

        # Use midpoint of tier range as base, blend with Kelly
        tier_mid_pct = (tier_min_pct + tier_max_pct) / 2
        base_amount = tier_mid_pct * portfolio_value

        # Take the more conservative of Kelly and tier midpoint
        position_amount = min(kelly_amount, base_amount)
        # But ensure at least tier minimum
        position_amount = max(position_amount, tier_min_pct * portfolio_value)

        # --- Adjustments ---

        # Multi-timeframe alignment boost/reduction
        if alignment_score > 1.5:
            position_amount *= 1.20
            adjustments.append(f"Strong alignment (score {alignment_score:.2f}): +20%")
        elif alignment_score < 0.5:
            position_amount *= 0.50
            adjustments.append(f"Weak/conflicting signals (score {alignment_score:.2f}): -50%")

        # Drawdown defensive mode
        if current_drawdown_pct > 10.0:
            position_amount *= 0.50
            adjustments.append(f"Defensive mode (drawdown {current_drawdown_pct:.1f}%): -50%")

        # Reduce if too many open positions
        if open_position_count >= 3:
            reduction = 0.25 * (open_position_count - 2)
            reduction = min(reduction, 0.75)
            position_amount *= (1.0 - reduction)
            adjustments.append(
                f"{open_position_count} open positions: -{reduction*100:.0f}%"
            )

        # Enforce absolute limits
        max_allowed = self.MAX_POSITION_PCT * portfolio_value
        if position_amount > max_allowed:
            position_amount = max_allowed
            adjustments.append(f"Capped at max {self.MAX_POSITION_PCT*100:.0f}% of portfolio")

        if position_amount < self.MIN_POSITION_USD:
            return SizingResult(
                should_trade=False,
                amount_usd=0.0,
                position_pct=0.0,
                tier=tier_name,
                kelly_fraction=kelly_sized,
                adjustments=adjustments + [
                    f"Position too small (${position_amount:.0f} < ${self.MIN_POSITION_USD:.0f})"
                ],
            )

        position_pct = position_amount / portfolio_value if portfolio_value > 0 else 0.0

        return SizingResult(
            should_trade=True,
            amount_usd=round(position_amount, 2),
            position_pct=round(position_pct, 4),
            tier=tier_name,
            kelly_fraction=round(kelly_sized, 4),
            adjustments=adjustments,
        )
