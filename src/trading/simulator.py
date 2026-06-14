"""Order execution simulator.

Simulates filling orders against real price data with realistic
slippage and fee modeling. Also handles stop-loss and take-profit
triggers on price updates.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from src.trading.order import Order, Position, Trade


class OrderSimulator:
    """Simulates trade execution with slippage and fees."""

    SLIPPAGE_PCT: float = 0.05  # 0.05% slippage per trade
    TAKER_FEE_PCT: float = 0.10  # 0.10% taker fee (Binance-realistic)

    def execute_buy(
        self,
        amount_usd: float,
        current_price: float,
        prediction_id: str,
        timeframe: str,
        confidence: float,
        stop_loss: float,
        take_profit: float,
        reason: str,
        timestamp: Optional[datetime] = None,
    ) -> tuple[Order, Position]:
        """Execute a simulated buy order.

        Applies slippage (price moves against us) and fees.
        Returns the Order and resulting Position.
        """
        ts = timestamp or datetime.utcnow()

        # Slippage: buy at slightly higher price
        execution_price = current_price * (1 + self.SLIPPAGE_PCT / 100)

        # Fee deducted from the amount
        fee = amount_usd * (self.TAKER_FEE_PCT / 100)
        net_amount_usd = amount_usd - fee
        amount_btc = net_amount_usd / execution_price

        order = Order(
            side="BUY",
            amount_usd=amount_usd,
            amount_btc=amount_btc,
            price=execution_price,
            order_type="MARKET",
            reason=reason,
            prediction_id=prediction_id,
            timeframe=timeframe,
            confidence=confidence,
            stop_loss=stop_loss,
            take_profit=take_profit,
            timestamp=ts,
        )

        position = Position(
            id=order.id,
            entry_time=ts,
            entry_price=execution_price,
            amount_usd=net_amount_usd,
            amount_btc=amount_btc,
            side="LONG",
            timeframe=timeframe,
            stop_loss=stop_loss,
            take_profit=take_profit,
            prediction_id=prediction_id,
            confidence=confidence,
            reason=reason,
        )

        return order, position

    def execute_sell(
        self,
        position: Position,
        current_price: float,
        reason: str,
        order_type: str = "MARKET",
        timestamp: Optional[datetime] = None,
    ) -> tuple[Order, Trade]:
        """Execute a simulated sell order to close a position.

        Applies slippage (price moves against us) and fees.
        Returns the Order and resulting Trade record.
        """
        ts = timestamp or datetime.utcnow()

        # Slippage: sell at slightly lower price
        execution_price = current_price * (1 - self.SLIPPAGE_PCT / 100)

        # Calculate proceeds after fee
        gross_proceeds = position.amount_btc * execution_price
        fee = gross_proceeds * (self.TAKER_FEE_PCT / 100)
        net_proceeds = gross_proceeds - fee

        # P&L calculation
        pnl_usd = net_proceeds - position.amount_usd
        pnl_pct = (pnl_usd / position.amount_usd * 100) if position.amount_usd > 0 else 0.0

        order = Order(
            side="SELL",
            amount_usd=gross_proceeds,
            amount_btc=position.amount_btc,
            price=execution_price,
            order_type=order_type,
            reason=reason,
            prediction_id=position.prediction_id,
            timeframe=position.timeframe,
            confidence=position.confidence,
            stop_loss=position.stop_loss,
            take_profit=position.take_profit,
            timestamp=ts,
        )

        trade = Trade(
            id=position.id,
            entry_time=position.entry_time,
            exit_time=ts,
            entry_price=position.entry_price,
            exit_price=execution_price,
            amount_usd=position.amount_usd,
            amount_btc=position.amount_btc,
            side=position.side,
            timeframe=position.timeframe,
            pnl_usd=pnl_usd,
            pnl_pct=pnl_pct,
            exit_reason=reason,
            prediction_id=position.prediction_id,
            fees_paid=fee + (position.amount_usd * self.TAKER_FEE_PCT / 100),
        )

        return order, trade

    def check_triggers(
        self,
        position: Position,
        high_price: float,
        low_price: float,
    ) -> Optional[str]:
        """Check if a position's SL or TP was triggered by a price candle.

        Uses high/low to determine if either level was hit.
        Returns 'stop_loss', 'take_profit', or None.
        """
        if position.side == "LONG":
            if low_price <= position.stop_loss:
                return "stop_loss"
            if high_price >= position.take_profit:
                return "take_profit"
        else:
            if high_price >= position.stop_loss:
                return "stop_loss"
            if low_price <= position.take_profit:
                return "take_profit"

        return None

    def get_trigger_price(self, position: Position, trigger_type: str) -> float:
        """Get the execution price for a triggered order."""
        if trigger_type == "stop_loss":
            return position.stop_loss
        elif trigger_type == "take_profit":
            return position.take_profit
        return 0.0
