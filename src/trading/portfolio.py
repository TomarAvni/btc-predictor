"""Portfolio state tracker for the demo trading agent.

Tracks virtual cash, BTC holdings, open positions, and closed trades.
Persists state to disk after every mutation.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from src.trading.order import Position, Trade, _ensure_utc

DATA_DIR = Path("data/trading")
PORTFOLIO_PATH = DATA_DIR / "portfolio.json"
TRADES_PATH = DATA_DIR / "trades.json"


class Portfolio:
    """Virtual portfolio with $2,000 starting balance."""

    STARTING_BALANCE: float = 2000.0

    def __init__(self, load_existing: bool = True) -> None:
        self.cash: float = self.STARTING_BALANCE
        self.btc_holdings: float = 0.0
        self.positions: list[Position] = []
        self.closed_trades: list[Trade] = []
        self.peak_value: float = self.STARTING_BALANCE
        self._last_price: float = 0.0
        self._daily_start_value: float = self.STARTING_BALANCE
        self._daily_start_date: Optional[datetime] = None

        if load_existing:
            self._load_state()

    # ------------------------------------------------------------------
    # Key metrics
    # ------------------------------------------------------------------

    @property
    def total_value_usd(self) -> float:
        """Total portfolio value: cash + long BTC value + short collateral + unrealized P&L."""
        long_value = self.btc_holdings * self._last_price
        short_value = sum(
            p.amount_usd + p.unrealized_pnl_at(self._last_price)
            for p in self.positions
            if p.side == "SHORT"
        )
        return self.cash + long_value + short_value

    @property
    def total_pnl(self) -> float:
        return self.total_value_usd - self.STARTING_BALANCE

    @property
    def total_pnl_pct(self) -> float:
        if self.STARTING_BALANCE == 0:
            return 0.0
        return (self.total_pnl / self.STARTING_BALANCE) * 100

    @property
    def max_drawdown(self) -> float:
        """Worst peak-to-trough decline as percentage."""
        if self.peak_value == 0:
            return 0.0
        current = self.total_value_usd
        drawdown = (self.peak_value - current) / self.peak_value * 100
        return max(drawdown, 0.0)

    @property
    def win_rate(self) -> float:
        """Percentage of profitable closed trades."""
        if not self.closed_trades:
            return 0.0
        winners = sum(1 for t in self.closed_trades if t.is_winner)
        return (winners / len(self.closed_trades)) * 100

    @property
    def exposure_pct(self) -> float:
        """Percentage of portfolio currently in positions (long + short notional)."""
        if self.total_value_usd == 0:
            return 0.0
        invested = sum(
            p.amount_btc * self._last_price if p.side == "LONG" else p.amount_usd
            for p in self.positions
        )
        return (invested / self.total_value_usd) * 100

    @property
    def open_position_count(self) -> int:
        return len(self.positions)

    # ------------------------------------------------------------------
    # Time-windowed P&L
    # ------------------------------------------------------------------

    def pnl_since(self, days: int) -> float:
        """Sum P&L of trades closed within the last N days."""
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        return sum(
            t.pnl_usd for t in self.closed_trades if t.exit_time >= cutoff
        )

    @property
    def last_7d_pnl(self) -> float:
        return self.pnl_since(7)

    @property
    def last_30d_pnl(self) -> float:
        return self.pnl_since(30)

    @property
    def last_365d_pnl(self) -> float:
        return self.pnl_since(365)

    @property
    def daily_pnl(self) -> float:
        """P&L since the start of the current trading day."""
        return self.total_value_usd - self._daily_start_value

    @property
    def daily_pnl_pct(self) -> float:
        if self._daily_start_value == 0:
            return 0.0
        return (self.daily_pnl / self._daily_start_value) * 100

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def update_price(self, price: float) -> None:
        """Update the last known BTC price and track peak value."""
        self._last_price = price
        current_value = self.total_value_usd
        if current_value > self.peak_value:
            self.peak_value = current_value

        now = datetime.now(timezone.utc)
        if self._daily_start_date is None or now.date() > self._daily_start_date.date():
            self._daily_start_value = current_value
            self._daily_start_date = now

    def open_position(self, position: Position) -> None:
        """Add a new position and deduct cash (collateral for shorts)."""
        self.cash -= position.amount_usd
        if position.side == "LONG":
            self.btc_holdings += position.amount_btc
        self.positions.append(position)
        self._save_state()

    def close_position(
        self,
        position_id: str,
        exit_price: float,
        exit_reason: str,
        timestamp: Optional[datetime] = None,
        pnl_usd: Optional[float] = None,
        pnl_pct: Optional[float] = None,
        fees_paid: float = 0.0,
    ) -> Optional[Trade]:
        """Close a position and record the trade.

        ``pnl_usd`` / ``pnl_pct`` / ``fees_paid`` should be supplied by the
        caller (the ``OrderSimulator``) so the persisted record reflects the
        fee- and slippage-accurate economics. When omitted they fall back to
        the gross (pre-fee) P&L derived from ``exit_price``, preserving the
        legacy behaviour for direct callers.
        """
        pos = next((p for p in self.positions if p.id == position_id), None)
        if pos is None:
            return None

        exit_time = timestamp or datetime.now(timezone.utc)
        if pnl_usd is None:
            pnl_usd = pos.unrealized_pnl_at(exit_price)
        if pnl_pct is None:
            pnl_pct = pos.unrealized_pnl_pct(exit_price)

        # Cash returns the original collateral/cost basis plus the net P&L.
        # ``pnl_usd`` already nets out the exit fee (and the entry fee was
        # deducted from ``amount_usd`` when the position opened), so this is
        # fee-consistent and never double-counts fees.
        self.cash += pos.amount_usd + pnl_usd
        if pos.side == "LONG":
            self.btc_holdings -= pos.amount_btc
        self.positions = [p for p in self.positions if p.id != position_id]

        trade = Trade(
            id=pos.id,
            entry_time=pos.entry_time,
            exit_time=exit_time,
            entry_price=pos.entry_price,
            exit_price=exit_price,
            amount_usd=pos.amount_usd,
            amount_btc=pos.amount_btc,
            side=pos.side,
            timeframe=pos.timeframe,
            pnl_usd=pnl_usd,
            pnl_pct=pnl_pct,
            exit_reason=exit_reason,
            prediction_id=pos.prediction_id,
            fees_paid=fees_paid,
            used_ml=pos.used_ml,
        )
        self.closed_trades.append(trade)
        self._save_state()
        return trade

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def _save_state(self) -> None:
        """Persist portfolio state to disk."""
        DATA_DIR.mkdir(parents=True, exist_ok=True)

        state = {
            "cash": self.cash,
            "btc_holdings": self.btc_holdings,
            "peak_value": self.peak_value,
            "last_price": self._last_price,
            "daily_start_value": self._daily_start_value,
            "daily_start_date": self._daily_start_date.isoformat() if self._daily_start_date else None,
            "positions": [p.to_dict() for p in self.positions],
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        PORTFOLIO_PATH.write_text(json.dumps(state, indent=2), encoding="utf-8")

        trades_data = [t.to_dict() for t in self.closed_trades]
        TRADES_PATH.write_text(json.dumps(trades_data, indent=2), encoding="utf-8")

    def _load_state(self) -> None:
        """Load portfolio state from disk if available."""
        if PORTFOLIO_PATH.exists():
            try:
                data = json.loads(PORTFOLIO_PATH.read_text(encoding="utf-8"))
                self.cash = data.get("cash", self.STARTING_BALANCE)
                self.btc_holdings = data.get("btc_holdings", 0.0)
                self.peak_value = data.get("peak_value", self.STARTING_BALANCE)
                self._last_price = data.get("last_price", 0.0)
                self._daily_start_value = data.get("daily_start_value", self.STARTING_BALANCE)
                ds = data.get("daily_start_date")
                self._daily_start_date = _ensure_utc(datetime.fromisoformat(ds)) if ds else None
                self.positions = [
                    Position.from_dict(p) for p in data.get("positions", [])
                ]
            except (json.JSONDecodeError, KeyError):
                pass

        if TRADES_PATH.exists():
            try:
                trades_data = json.loads(TRADES_PATH.read_text(encoding="utf-8"))
                self.closed_trades = [Trade.from_dict(t) for t in trades_data]
            except (json.JSONDecodeError, KeyError):
                pass

    def reset(self) -> None:
        """Reset portfolio to initial state (for testing/backtesting)."""
        self.cash = self.STARTING_BALANCE
        self.btc_holdings = 0.0
        self.positions = []
        self.closed_trades = []
        self.peak_value = self.STARTING_BALANCE
        self._last_price = 0.0
        self._daily_start_value = self.STARTING_BALANCE
        self._daily_start_date = None

    def get_summary(self) -> dict:
        """Return a summary dict suitable for dashboard display."""
        return {
            "cash": round(self.cash, 2),
            "btc_holdings": self.btc_holdings,
            "last_price": self._last_price,
            "total_value_usd": round(self.total_value_usd, 2),
            "total_pnl": round(self.total_pnl, 2),
            "total_pnl_pct": round(self.total_pnl_pct, 2),
            "max_drawdown": round(self.max_drawdown, 2),
            "win_rate": round(self.win_rate, 1),
            "exposure_pct": round(self.exposure_pct, 1),
            "open_positions": self.open_position_count,
            "total_trades": len(self.closed_trades),
            "last_7d_pnl": round(self.last_7d_pnl, 2),
            "last_30d_pnl": round(self.last_30d_pnl, 2),
        }
