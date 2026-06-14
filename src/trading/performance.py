"""Performance tracking and metrics calculation.

Computes comprehensive portfolio statistics: returns, risk ratios,
trade analytics, and comparison benchmarks.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta
from typing import Optional

import numpy as np

from src.trading.order import Trade


class PerformanceTracker:
    """Calculates and reports portfolio performance metrics."""

    RISK_FREE_RATE: float = 0.05  # 5% annual risk-free rate for Sharpe

    def __init__(self) -> None:
        self._daily_values: list[tuple[datetime, float]] = []
        self._btc_prices: list[tuple[datetime, float]] = []

    def record_daily_value(self, timestamp: datetime, value: float) -> None:
        """Record end-of-day portfolio value for return calculations."""
        self._daily_values.append((timestamp, value))

    def record_btc_price(self, timestamp: datetime, price: float) -> None:
        """Record BTC price for buy-and-hold comparison."""
        self._btc_prices.append((timestamp, price))

    def calculate_metrics(
        self,
        closed_trades: list[Trade],
        current_value: float,
        starting_balance: float = 2000.0,
    ) -> dict:
        """Calculate comprehensive performance metrics.

        Returns a dict with all relevant performance statistics.
        """
        metrics: dict = {}

        # --- Returns ---
        total_return = (current_value - starting_balance) / starting_balance * 100
        metrics["total_return_pct"] = round(total_return, 2)
        metrics["total_return_usd"] = round(current_value - starting_balance, 2)
        metrics["current_value"] = round(current_value, 2)
        metrics["starting_balance"] = starting_balance

        # Daily returns for risk metrics
        daily_returns = self._compute_daily_returns()
        metrics["daily_returns_count"] = len(daily_returns)

        if len(daily_returns) >= 2:
            metrics["sharpe_ratio"] = round(self._sharpe_ratio(daily_returns), 2)
            metrics["sortino_ratio"] = round(self._sortino_ratio(daily_returns), 2)
            metrics["max_drawdown_pct"] = round(self._max_drawdown(daily_returns), 2)
            metrics["volatility_annual_pct"] = round(
                float(np.std(daily_returns) * math.sqrt(365) * 100), 2
            )
        else:
            metrics["sharpe_ratio"] = 0.0
            metrics["sortino_ratio"] = 0.0
            metrics["max_drawdown_pct"] = 0.0
            metrics["volatility_annual_pct"] = 0.0

        # --- Trade Statistics ---
        if closed_trades:
            winners = [t for t in closed_trades if t.pnl_usd > 0]
            losers = [t for t in closed_trades if t.pnl_usd <= 0]

            metrics["total_trades"] = len(closed_trades)
            metrics["winning_trades"] = len(winners)
            metrics["losing_trades"] = len(losers)
            metrics["win_rate_pct"] = round(len(winners) / len(closed_trades) * 100, 1)

            avg_win = np.mean([t.pnl_usd for t in winners]) if winners else 0.0
            avg_loss = abs(np.mean([t.pnl_usd for t in losers])) if losers else 0.0
            metrics["avg_win_usd"] = round(float(avg_win), 2)
            metrics["avg_loss_usd"] = round(float(avg_loss), 2)

            metrics["largest_win_usd"] = round(
                max(t.pnl_usd for t in closed_trades), 2
            )
            metrics["largest_loss_usd"] = round(
                min(t.pnl_usd for t in closed_trades), 2
            )

            # Profit factor: gross profits / gross losses
            gross_profit = sum(t.pnl_usd for t in winners)
            gross_loss = abs(sum(t.pnl_usd for t in losers))
            metrics["profit_factor"] = (
                round(gross_profit / gross_loss, 2) if gross_loss > 0 else float("inf")
            )

            # Expectancy: average P&L per trade
            metrics["expectancy_usd"] = round(
                float(np.mean([t.pnl_usd for t in closed_trades])), 2
            )

            # Average holding time
            holding_hours = [
                (t.exit_time - t.entry_time).total_seconds() / 3600
                for t in closed_trades
            ]
            metrics["avg_holding_hours"] = round(float(np.mean(holding_hours)), 1)

            # Total fees paid
            metrics["total_fees_paid"] = round(
                sum(t.fees_paid for t in closed_trades), 2
            )

            # Exit reason breakdown
            exit_reasons: dict[str, int] = {}
            for t in closed_trades:
                reason = t.exit_reason.split(":")[0].strip() if t.exit_reason else "unknown"
                exit_reasons[reason] = exit_reasons.get(reason, 0) + 1
            metrics["exit_reasons"] = exit_reasons

            # Best/worst by timeframe
            by_timeframe: dict[str, list[float]] = {}
            for t in closed_trades:
                by_timeframe.setdefault(t.timeframe, []).append(t.pnl_usd)
            metrics["pnl_by_timeframe"] = {
                tf: round(sum(pnls), 2) for tf, pnls in by_timeframe.items()
            }
        else:
            metrics["total_trades"] = 0
            metrics["winning_trades"] = 0
            metrics["losing_trades"] = 0
            metrics["win_rate_pct"] = 0.0
            metrics["avg_win_usd"] = 0.0
            metrics["avg_loss_usd"] = 0.0
            metrics["largest_win_usd"] = 0.0
            metrics["largest_loss_usd"] = 0.0
            metrics["profit_factor"] = 0.0
            metrics["expectancy_usd"] = 0.0
            metrics["avg_holding_hours"] = 0.0
            metrics["total_fees_paid"] = 0.0
            metrics["exit_reasons"] = {}
            metrics["pnl_by_timeframe"] = {}

        # --- Buy and Hold Comparison ---
        metrics["buy_and_hold_return_pct"] = round(self._buy_and_hold_return(), 2)

        return metrics

    def generate_report(self, metrics: dict) -> str:
        """Generate a human-readable performance report."""
        lines = [
            "=" * 70,
            "  TRADING AGENT PERFORMANCE REPORT",
            "=" * 70,
            "",
            "  RETURNS",
            f"    Total Return:      {metrics['total_return_pct']:+.2f}% (${metrics['total_return_usd']:+.2f})",
            f"    Portfolio Value:    ${metrics['current_value']:,.2f}",
            f"    Starting Balance:  ${metrics['starting_balance']:,.2f}",
            f"    Buy & Hold BTC:    {metrics['buy_and_hold_return_pct']:+.2f}%",
            "",
            "  RISK METRICS",
            f"    Sharpe Ratio:      {metrics['sharpe_ratio']:.2f}",
            f"    Sortino Ratio:     {metrics['sortino_ratio']:.2f}",
            f"    Max Drawdown:      {metrics['max_drawdown_pct']:.2f}%",
            f"    Annual Volatility: {metrics['volatility_annual_pct']:.2f}%",
            "",
            "  TRADE STATISTICS",
            f"    Total Trades:      {metrics['total_trades']}",
            f"    Win Rate:          {metrics['win_rate_pct']:.1f}%",
            f"    Avg Win:           ${metrics['avg_win_usd']:.2f}",
            f"    Avg Loss:          ${metrics['avg_loss_usd']:.2f}",
            f"    Profit Factor:     {metrics['profit_factor']:.2f}",
            f"    Expectancy:        ${metrics['expectancy_usd']:+.2f} per trade",
            f"    Avg Holding Time:  {metrics['avg_holding_hours']:.1f} hours",
            f"    Total Fees Paid:   ${metrics['total_fees_paid']:.2f}",
            "",
        ]

        if metrics.get("pnl_by_timeframe"):
            lines.append("  P&L BY TIMEFRAME")
            for tf, pnl in metrics["pnl_by_timeframe"].items():
                lines.append(f"    {tf:>5s}: ${pnl:+.2f}")
            lines.append("")

        if metrics.get("exit_reasons"):
            lines.append("  EXIT REASONS")
            for reason, count in sorted(
                metrics["exit_reasons"].items(), key=lambda x: -x[1]
            ):
                lines.append(f"    {reason}: {count}")
            lines.append("")

        lines.append("=" * 70)
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Private calculation methods
    # ------------------------------------------------------------------

    def _compute_daily_returns(self) -> list[float]:
        """Compute daily percentage returns from value history."""
        if len(self._daily_values) < 2:
            return []

        returns = []
        for i in range(1, len(self._daily_values)):
            prev_val = self._daily_values[i - 1][1]
            curr_val = self._daily_values[i][1]
            if prev_val > 0:
                returns.append((curr_val - prev_val) / prev_val)
        return returns

    def _sharpe_ratio(self, daily_returns: list[float]) -> float:
        """Annualized Sharpe ratio."""
        if not daily_returns:
            return 0.0
        arr = np.array(daily_returns)
        excess = arr - (self.RISK_FREE_RATE / 365)
        std = float(np.std(arr))
        if std == 0:
            return 0.0
        return float(np.mean(excess) / std * math.sqrt(365))

    def _sortino_ratio(self, daily_returns: list[float]) -> float:
        """Annualized Sortino ratio (penalizes only downside volatility)."""
        if not daily_returns:
            return 0.0
        arr = np.array(daily_returns)
        excess = arr - (self.RISK_FREE_RATE / 365)
        downside = arr[arr < 0]
        if len(downside) == 0:
            return float("inf") if float(np.mean(excess)) > 0 else 0.0
        downside_std = float(np.std(downside))
        if downside_std == 0:
            return 0.0
        return float(np.mean(excess) / downside_std * math.sqrt(365))

    def _max_drawdown(self, daily_returns: list[float]) -> float:
        """Maximum drawdown as a percentage."""
        if not daily_returns:
            return 0.0

        cumulative = [1.0]
        for r in daily_returns:
            cumulative.append(cumulative[-1] * (1 + r))

        peak = cumulative[0]
        max_dd = 0.0
        for val in cumulative:
            if val > peak:
                peak = val
            dd = (peak - val) / peak * 100
            if dd > max_dd:
                max_dd = dd
        return max_dd

    def _buy_and_hold_return(self) -> float:
        """Calculate BTC buy-and-hold return over the same period."""
        if len(self._btc_prices) < 2:
            return 0.0
        start_price = self._btc_prices[0][1]
        end_price = self._btc_prices[-1][1]
        if start_price == 0:
            return 0.0
        return (end_price - start_price) / start_price * 100

    def reset(self) -> None:
        """Reset tracker (for backtesting)."""
        self._daily_values = []
        self._btc_prices = []
