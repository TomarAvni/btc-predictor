"""Main Trading Agent -- the decision engine orchestrator.

Receives predictions from the ML engine, manages the portfolio,
and executes simulated trades with intelligent position sizing
and risk management.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from src.trading.order import Order, Position, Trade
from src.trading.performance import PerformanceTracker
from src.trading.portfolio import Portfolio
from src.trading.position_sizer import PositionSizer
from src.trading.risk_manager import RiskManager
from src.trading.simulator import OrderSimulator
from src.trading.strategy import TradingStrategy
from src.trading.trade_journal import TradeJournal


class TradingAgent:
    """Orchestrates the full trading decision pipeline.

    Flow on each prediction:
      1. Update portfolio with current price
      2. Review open positions (check SL/TP, signal flips)
      3. Evaluate new entry opportunities via strategy
      4. Size the position via Kelly-based sizer
      5. Run risk checks
      6. Execute via simulator
      7. Log everything to the journal
      8. Persist state
    """

    # --- Pre-trade safety gate (Change 2) ---------------------------------
    # No trade (long or short) may open unless a REAL ML model produced the
    # prediction AND the predicted magnitude is meaningfully non-zero.
    # magnitude is expressed in percent, so 0.05 == 0.05%.
    MIN_MAGNITUDE_PCT: float = 0.05

    def __init__(
        self,
        portfolio: Optional[Portfolio] = None,
        strategy: Optional[TradingStrategy] = None,
        risk_manager: Optional[RiskManager] = None,
        position_sizer: Optional[PositionSizer] = None,
        simulator: Optional[OrderSimulator] = None,
        performance: Optional[PerformanceTracker] = None,
        journal: Optional[TradeJournal] = None,
    ) -> None:
        self.portfolio = portfolio or Portfolio()
        self.strategy = strategy or TradingStrategy()
        self.risk_manager = risk_manager or RiskManager()
        self.position_sizer = position_sizer or PositionSizer()
        self.simulator = simulator or OrderSimulator()
        self.performance = performance or PerformanceTracker()
        self.journal = journal or TradeJournal()
        self._last_predictions: list[dict] = []

    def on_new_prediction(
        self,
        predictions: list[dict],
        current_price: float,
        timestamp: Optional[datetime] = None,
        used_ml: bool = True,
        run_number: int = 0,
    ) -> dict[str, Any]:
        """Process a new set of predictions from the ML engine.

        Args:
            predictions: List of prediction dicts, each containing:
                - timeframe: horizon label (e.g. "6h", "24h", "168h", "30d";
                  see src/horizons.py for the full set)
                - direction: "UP" or "DOWN"
                - magnitude: expected % move
                - confidence: 0-100
            current_price: Current BTC/USD price.
            timestamp: Override timestamp (for backtesting).
            used_ml: Whether a real ML model produced these predictions
                (False = heuristic/synthetic fallback). When False, the
                pre-trade safety gate blocks any new entry. Defaults to
                True so backtests with synthetic predictions still trade.
            run_number: Prediction run counter from PredictionEngine, used to
                link journal entries back to the originating prediction run.

        Returns:
            Dict summarizing actions taken.
        """
        ts = timestamp or datetime.now(timezone.utc)
        actions_taken: list[dict] = []

        # 1. Update price
        self.portfolio.update_price(current_price, ts)
        self.performance.record_daily_value(ts, self.portfolio.total_value_usd)
        self.performance.record_btc_price(ts, current_price)

        # 2. Review open positions
        exit_actions = self._review_positions(predictions, current_price, ts)
        actions_taken.extend(exit_actions)

        # 3. Evaluate new entry
        entry_action = self._evaluate_entry(
            predictions, current_price, ts, used_ml, run_number
        )
        if entry_action:
            actions_taken.append(entry_action)

        self._last_predictions = predictions

        return {
            "timestamp": ts.isoformat(),
            "price": current_price,
            "actions": actions_taken,
            "portfolio_value": self.portfolio.total_value_usd,
            "open_positions": self.portfolio.open_position_count,
        }

    def on_price_update(
        self,
        price: float,
        high: Optional[float] = None,
        low: Optional[float] = None,
        timestamp: Optional[datetime] = None,
    ) -> list[dict]:
        """Process a price candle update -- check SL/TP triggers.

        Args:
            price: Close price of the candle.
            high: High price (defaults to close).
            low: Low price (defaults to close).
            timestamp: Candle timestamp.

        Returns:
            List of actions taken (triggered exits).
        """
        ts = timestamp or datetime.now(timezone.utc)
        candle_high = high or price
        candle_low = low or price

        self.portfolio.update_price(price, ts)
        self.performance.record_daily_value(ts, self.portfolio.total_value_usd)
        self.performance.record_btc_price(ts, price)

        actions = []
        for position in list(self.portfolio.positions):
            # Check trailing stop update
            new_stop = self.risk_manager.should_update_trailing_stop(position, price)
            if new_stop is not None:
                position.stop_loss = new_stop
                position.trailing_stop_active = True

            # Check SL/TP triggers
            trigger = self.simulator.check_triggers(position, candle_high, candle_low)
            if trigger:
                action = self._close_position(position, trigger, price, ts)
                actions.append(action)

        return actions

    def get_status(self) -> dict[str, Any]:
        """Return current agent state for dashboard display."""
        return {
            "portfolio": self.portfolio.get_summary(),
            "open_positions": [
                {
                    "id": p.id,
                    "side": p.side,
                    "entry_price": p.entry_price,
                    "amount_usd": p.amount_usd,
                    "timeframe": p.timeframe,
                    "stop_loss": p.stop_loss,
                    "take_profit": p.take_profit,
                    "unrealized_pnl": p.unrealized_pnl_at(self.portfolio._last_price),
                    "unrealized_pnl_pct": p.unrealized_pnl_pct(self.portfolio._last_price),
                    "entry_time": p.entry_time.isoformat(),
                    "confidence": p.confidence,
                }
                for p in self.portfolio.positions
            ],
            "last_predictions": self._last_predictions,
            "recent_journal": self.journal.get_recent_entries(5),
        }

    def get_performance_summary(self) -> dict[str, Any]:
        """Return comprehensive performance metrics."""
        return self.performance.calculate_metrics(
            closed_trades=self.portfolio.closed_trades,
            current_value=self.portfolio.total_value_usd,
            starting_balance=self.portfolio.STARTING_BALANCE,
        )

    def get_performance_report(self) -> str:
        """Generate a human-readable performance report."""
        metrics = self.get_performance_summary()
        return self.performance.generate_report(metrics)

    def reset(self) -> None:
        """Reset all state for a fresh backtest run."""
        self.portfolio.reset()
        self.risk_manager.reset()
        self.performance.reset()
        self.journal.reset()
        self._last_predictions = []

    # ------------------------------------------------------------------
    # Private methods
    # ------------------------------------------------------------------

    def _review_positions(
        self,
        predictions: list[dict],
        current_price: float,
        timestamp: datetime,
    ) -> list[dict]:
        """Review all open positions against new predictions."""
        actions = []

        for position in list(self.portfolio.positions):
            # Check trailing stop
            new_stop = self.risk_manager.should_update_trailing_stop(
                position, current_price
            )
            if new_stop is not None:
                position.stop_loss = new_stop
                position.trailing_stop_active = True

            # Check SL/TP at current price
            if self.risk_manager.check_stop_loss(position, current_price):
                action = self._close_position(
                    position, "stop_loss", current_price, timestamp
                )
                actions.append(action)
                continue

            if self.risk_manager.check_take_profit(position, current_price):
                action = self._close_position(
                    position, "take_profit", current_price, timestamp
                )
                actions.append(action)
                continue

            # Check strategy exit conditions
            exit_signal = self.strategy.evaluate_exit(
                position, current_price, predictions, timestamp
            )
            if exit_signal.should_exit:
                action = self._close_position(
                    position, exit_signal.exit_type, current_price, timestamp
                )
                actions.append(action)

        return actions

    def _pretrade_gate_reason(self, signal, used_ml: bool) -> Optional[str]:
        """Return a skip reason if the pre-trade safety gate blocks a trade.

        Both conditions must hold to allow a trade:
          (a) a real ML model produced the prediction (``used_ml`` is True), and
          (b) the predicted magnitude is meaningfully non-zero.

        Returns None when the trade is allowed.
        """
        if not used_ml:
            return (
                "Safety gate: no ML model loaded (heuristic/fallback "
                "prediction) — trading disabled until models are ready"
            )
        if abs(signal.magnitude) <= self.MIN_MAGNITUDE_PCT:
            return (
                f"Safety gate: predicted magnitude {signal.magnitude:.4f}% "
                f"<= minimum {self.MIN_MAGNITUDE_PCT}% (no actionable move)"
            )
        return None

    def _evaluate_entry(
        self,
        predictions: list[dict],
        current_price: float,
        timestamp: datetime,
        used_ml: bool = True,
        run_number: int = 0,
    ) -> Optional[dict]:
        """Evaluate whether to open a new position."""
        signal = self.strategy.evaluate_entry(
            predictions=predictions,
            current_price=current_price,
            open_positions=self.portfolio.positions,
            timestamp=timestamp,
        )

        if not signal.should_enter:
            self.journal.log_skip(
                reason=signal.reasons[0] if signal.reasons else "No entry signal",
                predictions=predictions,
                timestamp=timestamp,
                run_number=run_number,
            )
            return None

        # Pre-trade safety gate (Change 2): block trades that were not
        # produced by a real ML model, or whose magnitude is ~zero. This
        # prevents the overnight failure mode where heuristic / 0.0%-magnitude
        # predictions fired losing trades before ML models were ready.
        gate_reason = self._pretrade_gate_reason(signal, used_ml)
        if gate_reason:
            self.journal.log_skip(
                reason=gate_reason,
                predictions=predictions,
                timestamp=timestamp,
                run_number=run_number,
            )
            return None

        # Position sizing
        sizing = self.position_sizer.calculate(
            confidence=signal.confidence,
            portfolio_value=self.portfolio.total_value_usd,
            alignment_score=signal.alignment_score,
            current_drawdown_pct=self.portfolio.max_drawdown,
            open_position_count=self.portfolio.open_position_count,
            expected_move_pct=signal.magnitude,
            round_trip_cost_pct=self.simulator.ROUND_TRIP_COST_PCT,
        )

        if not sizing.should_trade:
            self.journal.log_skip(
                reason=f"Position sizing rejected: {sizing.adjustments}",
                predictions=predictions,
                timestamp=timestamp,
                run_number=run_number,
            )
            return None

        # Risk check
        risk_check = self.risk_manager.check_new_trade(
            portfolio_value=self.portfolio.total_value_usd,
            cash_available=self.portfolio.cash,
            current_drawdown_pct=self.portfolio.max_drawdown,
            daily_pnl_pct=self.portfolio.daily_pnl_pct,
            open_position_count=self.portfolio.open_position_count,
            current_exposure_pct=self.portfolio.exposure_pct,
            proposed_amount_usd=sizing.amount_usd,
            timestamp=timestamp,
        )

        if not risk_check.approved:
            self.journal.log_skip(
                reason=f"Risk rejected: {risk_check.reason}",
                predictions=predictions,
                timestamp=timestamp,
                run_number=run_number,
            )
            return None

        # Adjust amount if risk manager suggested lower
        amount_usd = min(sizing.amount_usd, risk_check.max_position_usd)
        if amount_usd < self.position_sizer.MIN_POSITION_USD:
            self.journal.log_skip(
                reason=(
                    f"position_too_small: adjusted amount ${amount_usd:.2f} "
                    f"< MIN ${self.position_sizer.MIN_POSITION_USD:.2f} "
                    f"after risk cap"
                ),
                predictions=predictions,
                timestamp=timestamp,
                run_number=run_number,
            )
            return None

        # Calculate SL and TP from the slippage-adjusted fill price so the
        # levels match the price the order actually executes at (the simulator
        # applies entry slippage), not the pre-slippage quote.
        fill_price = self.simulator.entry_fill_price(current_price, signal.direction)
        stop_loss = self.risk_manager.calculate_stop_loss(
            entry_price=fill_price,
            predicted_magnitude_pct=signal.magnitude,
            side=signal.direction,
        )
        take_profit = self.risk_manager.calculate_take_profit(
            entry_price=fill_price,
            predicted_magnitude_pct=signal.magnitude,
            side=signal.direction,
        )

        # Verify single-trade risk
        sl_distance_pct = abs(fill_price - stop_loss) / fill_price * 100
        if not self.risk_manager.check_single_trade_risk(
            amount_usd, sl_distance_pct, self.portfolio.total_value_usd
        ):
            self.journal.log_skip(
                reason=f"Single trade risk too high ({sl_distance_pct:.1f}% SL distance)",
                predictions=predictions,
                timestamp=timestamp,
            )
            return None

        # Execute the trade
        prediction_id = uuid.uuid4().hex[:12]
        reason = "; ".join(signal.reasons)

        if signal.direction == "SHORT":
            order, position = self.simulator.execute_short(
                amount_usd=amount_usd,
                current_price=current_price,
                prediction_id=prediction_id,
                timeframe=signal.timeframe,
                confidence=signal.confidence,
                stop_loss=stop_loss,
                take_profit=take_profit,
                reason=reason,
                timestamp=timestamp,
            )
            action_name = "SHORT"
        else:
            order, position = self.simulator.execute_buy(
                amount_usd=amount_usd,
                current_price=current_price,
                prediction_id=prediction_id,
                timeframe=signal.timeframe,
                confidence=signal.confidence,
                stop_loss=stop_loss,
                take_profit=take_profit,
                reason=reason,
                timestamp=timestamp,
            )
            action_name = "BUY"

        # Stamp used_ml on the position so it propagates to the closed Trade record.
        position.used_ml = used_ml

        # Update portfolio
        self.portfolio.open_position(position)
        self.risk_manager.record_trade(timestamp)

        # Log to journal
        self.journal.log_entry(
            action=action_name,
            position=position,
            order=order,
            strategy_reasons=signal.reasons,
            sizing_tier=sizing.tier,
            alignment_score=signal.alignment_score,
            portfolio_cash=self.portfolio.cash,
            portfolio_total=self.portfolio.total_value_usd,
            open_position_count=self.portfolio.open_position_count,
            exposure_pct=self.portfolio.exposure_pct,
            timestamp=timestamp,
            run_number=run_number,
            used_ml=used_ml,
        )

        return {
            "action": action_name,
            "side": position.side,
            "amount_usd": order.amount_usd,
            "amount_btc": order.amount_btc,
            "price": order.price,
            "timeframe": signal.timeframe,
            "confidence": signal.confidence,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "tier": sizing.tier,
            "alignment": signal.alignment_label,
        }

    def _close_position(
        self,
        position: Position,
        reason: str,
        current_price: float,
        timestamp: datetime,
    ) -> dict:
        """Close a position and log the exit."""
        trigger_price = current_price
        order_type = "MARKET"

        if reason == "stop_loss":
            trigger_price = self.simulator.get_trigger_price(position, "stop_loss")
            order_type = "STOP_LOSS"
        elif reason == "take_profit":
            trigger_price = self.simulator.get_trigger_price(position, "take_profit")
            order_type = "TAKE_PROFIT"

        order, trade = self.simulator.execute_close(
            position=position,
            current_price=trigger_price,
            reason=reason,
            order_type=order_type,
            timestamp=timestamp,
        )

        # Update portfolio (close the position). Pass the simulator's
        # fee-accurate net P&L and fees so the persisted Trade in trades.json
        # matches the journal/simulator instead of recomputing gross P&L.
        self.portfolio.close_position(
            position_id=position.id,
            exit_price=trade.exit_price,
            exit_reason=reason,
            timestamp=timestamp,
            pnl_usd=trade.pnl_usd,
            pnl_pct=trade.pnl_pct,
            fees_paid=trade.fees_paid,
        )

        # Log exit
        self.journal.log_exit(
            trade=trade,
            reason=reason,
            portfolio_cash=self.portfolio.cash,
            portfolio_total=self.portfolio.total_value_usd,
            timestamp=timestamp,
        )

        close_action = "COVER" if position.side == "SHORT" else "SELL"

        return {
            "action": close_action,
            "side": position.side,
            "reason": reason,
            "amount_usd": trade.amount_usd,
            "exit_price": trade.exit_price,
            "pnl_usd": trade.pnl_usd,
            "pnl_pct": trade.pnl_pct,
            "holding_hours": (trade.exit_time - trade.entry_time).total_seconds() / 3600,
        }
