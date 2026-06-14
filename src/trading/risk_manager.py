"""Risk management module for the trading agent.

Protects virtual capital through stop losses, position limits,
drawdown circuit breakers, and daily loss limits.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional

from src.trading.order import Position


@dataclass
class RiskCheck:
    """Result of a risk assessment."""

    approved: bool
    reason: str
    max_position_usd: float = 0.0
    suggested_stop_loss: float = 0.0
    suggested_take_profit: float = 0.0


class RiskManager:
    """Portfolio-level risk management and position protection."""

    MAX_OPEN_POSITIONS: int = 5
    MAX_EXPOSURE_PCT: float = 80.0  # Always keep 20% cash reserve
    MAX_DRAWDOWN_CIRCUIT_BREAKER: float = 25.0  # Stop trading if down 25% from peak
    DAILY_LOSS_LIMIT_PCT: float = 5.0  # No new trades if down 5% today
    MAX_SINGLE_TRADE_RISK_PCT: float = 5.0  # Position * SL distance < 5% of portfolio
    MIN_TIME_BETWEEN_TRADES: timedelta = timedelta(hours=1)
    CIRCUIT_BREAKER_COOLDOWN: timedelta = timedelta(hours=48)

    MIN_STOP_LOSS_PCT: float = 1.0  # Minimum SL to avoid noise exits
    TRAILING_STOP_ACTIVATION_PCT: float = 50.0  # Activate trailing after 50% of target

    def __init__(self) -> None:
        self._circuit_breaker_until: Optional[datetime] = None
        self._last_trade_time: Optional[datetime] = None

    def check_new_trade(
        self,
        portfolio_value: float,
        cash_available: float,
        current_drawdown_pct: float,
        daily_pnl_pct: float,
        open_position_count: int,
        current_exposure_pct: float,
        proposed_amount_usd: float,
        timestamp: Optional[datetime] = None,
    ) -> RiskCheck:
        """Evaluate whether a new trade is allowed under risk constraints.

        Returns a RiskCheck indicating approval or rejection with reason.
        """
        now = timestamp or datetime.now(timezone.utc)

        # Circuit breaker
        if self._circuit_breaker_until and now < self._circuit_breaker_until:
            remaining = (self._circuit_breaker_until - now).total_seconds() / 3600
            return RiskCheck(
                approved=False,
                reason=f"Circuit breaker active ({remaining:.1f}h remaining)",
            )

        if current_drawdown_pct >= self.MAX_DRAWDOWN_CIRCUIT_BREAKER:
            self._circuit_breaker_until = now + self.CIRCUIT_BREAKER_COOLDOWN
            return RiskCheck(
                approved=False,
                reason=f"Circuit breaker triggered: drawdown {current_drawdown_pct:.1f}% >= {self.MAX_DRAWDOWN_CIRCUIT_BREAKER}%",
            )

        # Daily loss limit
        if daily_pnl_pct <= -self.DAILY_LOSS_LIMIT_PCT:
            return RiskCheck(
                approved=False,
                reason=f"Daily loss limit hit: {daily_pnl_pct:.1f}% (limit: -{self.DAILY_LOSS_LIMIT_PCT}%)",
            )

        # Position count
        if open_position_count >= self.MAX_OPEN_POSITIONS:
            return RiskCheck(
                approved=False,
                reason=f"Max open positions reached ({self.MAX_OPEN_POSITIONS})",
            )

        # Exposure limit
        new_exposure = current_exposure_pct + (proposed_amount_usd / portfolio_value * 100)
        if new_exposure > self.MAX_EXPOSURE_PCT:
            max_allowed = (self.MAX_EXPOSURE_PCT - current_exposure_pct) / 100 * portfolio_value
            if max_allowed < 50:
                return RiskCheck(
                    approved=False,
                    reason=f"Exposure limit: would be {new_exposure:.1f}% (max {self.MAX_EXPOSURE_PCT}%)",
                )
            return RiskCheck(
                approved=True,
                reason=f"Approved (reduced to fit exposure limit)",
                max_position_usd=max(max_allowed, 0),
            )

        # Minimum time between trades
        if self._last_trade_time:
            elapsed = now - self._last_trade_time
            if elapsed < self.MIN_TIME_BETWEEN_TRADES:
                remaining_min = (self.MIN_TIME_BETWEEN_TRADES - elapsed).total_seconds() / 60
                return RiskCheck(
                    approved=False,
                    reason=f"Min time between trades: {remaining_min:.0f}min remaining",
                )

        # Cash check
        if proposed_amount_usd > cash_available:
            if cash_available < 50:
                return RiskCheck(
                    approved=False,
                    reason=f"Insufficient cash: ${cash_available:.2f} available",
                )
            return RiskCheck(
                approved=True,
                reason="Approved (reduced to available cash)",
                max_position_usd=cash_available,
            )

        return RiskCheck(
            approved=True,
            reason="All risk checks passed",
            max_position_usd=proposed_amount_usd,
        )

    def calculate_stop_loss(
        self,
        entry_price: float,
        predicted_magnitude_pct: float,
        side: str = "LONG",
    ) -> float:
        """Calculate stop loss price for a position.

        Default: 2x predicted magnitude in the opposite direction.
        Minimum: 1% to avoid noise stops.
        """
        sl_pct = max(predicted_magnitude_pct * 2, self.MIN_STOP_LOSS_PCT)

        if side == "LONG":
            return entry_price * (1 - sl_pct / 100)
        return entry_price * (1 + sl_pct / 100)

    def calculate_take_profit(
        self,
        entry_price: float,
        predicted_magnitude_pct: float,
        side: str = "LONG",
    ) -> float:
        """Calculate take profit at predicted magnitude."""
        if side == "LONG":
            return entry_price * (1 + predicted_magnitude_pct / 100)
        return entry_price * (1 - predicted_magnitude_pct / 100)

    def check_single_trade_risk(
        self,
        position_amount_usd: float,
        stop_loss_distance_pct: float,
        portfolio_value: float,
    ) -> bool:
        """Verify that position_size * SL_distance < max single trade risk."""
        max_loss = position_amount_usd * (stop_loss_distance_pct / 100)
        max_allowed = portfolio_value * (self.MAX_SINGLE_TRADE_RISK_PCT / 100)
        return max_loss <= max_allowed

    def should_update_trailing_stop(
        self,
        position: Position,
        current_price: float,
    ) -> Optional[float]:
        """Check if trailing stop should be activated/updated.

        Returns new stop loss price if it should be updated, None otherwise.
        """
        pnl_pct = position.unrealized_pnl_pct(current_price)
        target_pct = abs(
            (position.take_profit - position.entry_price) / position.entry_price * 100
        )

        # Activate trailing stop once 50% of target is reached
        if pnl_pct >= target_pct * (self.TRAILING_STOP_ACTIVATION_PCT / 100):
            # Move stop to breakeven (entry price)
            if position.side == "LONG":
                new_stop = max(position.entry_price, position.stop_loss)
            else:
                new_stop = min(position.entry_price, position.stop_loss)
            if new_stop != position.stop_loss:
                return new_stop

        return None

    def check_stop_loss(self, position: Position, current_price: float) -> bool:
        """Check if position's stop loss has been triggered."""
        if position.side == "LONG":
            return current_price <= position.stop_loss
        return current_price >= position.stop_loss

    def check_take_profit(self, position: Position, current_price: float) -> bool:
        """Check if position's take profit has been triggered."""
        if position.side == "LONG":
            return current_price >= position.take_profit
        return current_price <= position.take_profit

    def record_trade(self, timestamp: Optional[datetime] = None) -> None:
        """Record that a trade was made (for time-between-trades check)."""
        self._last_trade_time = timestamp or datetime.now(timezone.utc)

    def reset(self) -> None:
        """Reset risk manager state (for backtesting)."""
        self._circuit_breaker_until = None
        self._last_trade_time = None
