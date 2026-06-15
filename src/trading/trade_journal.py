"""Trade journal -- logs every decision with full reasoning.

Produces both human-readable (.log) and machine-readable (.json) logs
for full audit trail of all trading activity.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from src.trading.order import Order, Position, Trade

DATA_DIR = Path("data/trading")
JOURNAL_LOG_PATH = DATA_DIR / "journal.log"
JOURNAL_JSON_PATH = DATA_DIR / "journal.json"


class TradeJournal:
    """Logs every trade decision with full reasoning."""

    def __init__(self) -> None:
        self._entries: list[dict] = []
        self._trade_number: int = 0
        self._load_existing()

    def _load_existing(self) -> None:
        """Load existing journal entries to continue numbering."""
        if JOURNAL_JSON_PATH.exists():
            try:
                data = json.loads(JOURNAL_JSON_PATH.read_text(encoding="utf-8"))
                self._entries = data
                self._trade_number = len(data)
            except (json.JSONDecodeError, KeyError):
                pass

    def log_entry(
        self,
        action: str,
        position: Position,
        order: Order,
        strategy_reasons: list[str],
        sizing_tier: str,
        alignment_score: float,
        portfolio_cash: float,
        portfolio_total: float,
        open_position_count: int,
        exposure_pct: float,
        timestamp: Optional[datetime] = None,
        run_number: Optional[int] = None,
        used_ml: Optional[bool] = None,
    ) -> None:
        """Log a new trade entry."""
        self._trade_number += 1
        ts = timestamp or datetime.now(timezone.utc)

        entry = {
            "trade_number": self._trade_number,
            "timestamp": ts.isoformat(),
            "action": action,
            "position_side": position.side,
            "side": order.side,
            "amount_usd": order.amount_usd,
            "amount_btc": order.amount_btc,
            "price": order.price,
            "timeframe": order.timeframe,
            "confidence": order.confidence,
            "reasons": strategy_reasons,
            "sizing_tier": sizing_tier,
            "alignment_score": alignment_score,
            "stop_loss": order.stop_loss,
            "take_profit": order.take_profit,
            "max_loss_usd": order.amount_usd * abs(order.price - order.stop_loss) / order.price,
            "max_loss_pct_portfolio": (
                order.amount_usd * abs(order.price - order.stop_loss) / order.price
            ) / portfolio_total * 100 if portfolio_total > 0 else 0,
            "portfolio_cash": portfolio_cash,
            "portfolio_total": portfolio_total,
            "open_positions": open_position_count,
            "exposure_pct": exposure_pct,
            "prediction_id": order.prediction_id,
            "run_number": run_number,
            "used_ml": used_ml,
        }

        self._entries.append(entry)
        self._write_log_entry(entry)
        self._save_json()

    def log_exit(
        self,
        trade: Trade,
        reason: str,
        portfolio_cash: float,
        portfolio_total: float,
        timestamp: Optional[datetime] = None,
    ) -> None:
        """Log a position exit/close."""
        self._trade_number += 1
        ts = timestamp or datetime.now(timezone.utc)

        entry = {
            "trade_number": self._trade_number,
            "timestamp": ts.isoformat(),
            "action": "CLOSE",
            "position_side": trade.side,
            "side": "SELL" if trade.side == "LONG" else "BUY",
            "amount_usd": trade.amount_usd,
            "amount_btc": trade.amount_btc,
            "entry_price": trade.entry_price,
            "exit_price": trade.exit_price,
            "pnl_usd": trade.pnl_usd,
            "pnl_pct": trade.pnl_pct,
            "exit_reason": reason,
            "timeframe": trade.timeframe,
            "holding_time_hours": (trade.exit_time - trade.entry_time).total_seconds() / 3600,
            "portfolio_cash": portfolio_cash,
            "portfolio_total": portfolio_total,
            "fees_paid": trade.fees_paid,
        }

        self._entries.append(entry)
        self._write_log_entry(entry)
        self._save_json()

    def log_skip(
        self,
        reason: str,
        predictions: list[dict],
        timestamp: Optional[datetime] = None,
        run_number: Optional[int] = None,
    ) -> None:
        """Log a decision NOT to trade (for audit trail)."""
        ts = timestamp or datetime.now(timezone.utc)

        entry = {
            "trade_number": None,
            "timestamp": ts.isoformat(),
            "action": "SKIP",
            "reason": reason,
            "run_number": run_number,
            "predictions_summary": [
                f"{p['timeframe']}: {p['direction']} {p['magnitude']}% @ {p['confidence']}%"
                for p in predictions
            ],
        }

        self._entries.append(entry)
        self._write_log_entry(entry)
        self._save_json()

    def _write_log_entry(self, entry: dict) -> None:
        """Append a human-readable log entry."""
        DATA_DIR.mkdir(parents=True, exist_ok=True)

        lines = [
            "=" * 80,
            f"[{entry['timestamp']}] -- {'Trade #' + str(entry['trade_number']) if entry.get('trade_number') else 'Decision'}",
            "=" * 80,
            f"ACTION: {entry['action']}",
        ]

        position_side = entry.get("position_side")
        if position_side:
            lines.append(f"POSITION: {position_side}")

        if entry["action"] == "SKIP":
            lines.append(f"REASON: {entry.get('reason', 'N/A')}")
            preds = entry.get("predictions_summary", [])
            if preds:
                lines.append("PREDICTIONS:")
                for p in preds:
                    lines.append(f"  - {p}")
        elif entry["action"] == "CLOSE":
            lines.append(
                f"AMOUNT: ${entry['amount_usd']:.2f} / {entry['amount_btc']:.8f} BTC"
            )
            lines.append(
                f"PRICES: Entry ${entry['entry_price']:,.2f} -> Exit ${entry['exit_price']:,.2f}"
            )
            lines.append(f"P&L: ${entry['pnl_usd']:.2f} ({entry['pnl_pct']:.2f}%)")
            lines.append(f"EXIT REASON: {entry['exit_reason']}")
            if position_side == "SHORT":
                lines.append("COVER: Simulated short closed (buy-back)")
            lines.append(
                f"HOLDING TIME: {entry.get('holding_time_hours', 0):.1f} hours"
            )
            lines.append(f"FEES: ${entry.get('fees_paid', 0):.2f}")
        else:
            side_label = entry.get("position_side", entry.get("action", ""))
            lines.append(
                f"AMOUNT: ${entry['amount_usd']:.2f} ({entry.get('sizing_tier', '')} {side_label} position) "
                f"/ {entry['amount_btc']:.8f} BTC @ ${entry['price']:,.2f}"
            )
            if entry.get("action") == "SHORT":
                lines.append("STRATEGY: Simulated short (USD collateral, bet on price decline)")
            lines.append(f"TIMEFRAME: {entry.get('timeframe', 'N/A')}")
            lines.append("REASON:")
            for r in entry.get("reasons", []):
                lines.append(f"  - {r}")
            lines.append("RISK PARAMETERS:")
            lines.append(f"  - Stop Loss: ${entry.get('stop_loss', 0):,.2f}")
            lines.append(f"  - Take Profit: ${entry.get('take_profit', 0):,.2f}")
            lines.append(
                f"  - Max Loss: ${entry.get('max_loss_usd', 0):.2f} "
                f"({entry.get('max_loss_pct_portfolio', 0):.2f}% of portfolio)"
            )
            lines.append("PORTFOLIO STATE:")
            lines.append(
                f"  - Cash: ${entry.get('portfolio_cash', 0):,.2f} / "
                f"Total: ${entry.get('portfolio_total', 0):,.2f}"
            )
            lines.append(
                f"  - Open Positions: {entry.get('open_positions', 0)}"
            )
            lines.append(f"  - Exposure: {entry.get('exposure_pct', 0):.1f}%")

        lines.append("=" * 80)
        lines.append("")

        with open(JOURNAL_LOG_PATH, "a", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

    def _save_json(self) -> None:
        """Save all entries to JSON."""
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        JOURNAL_JSON_PATH.write_text(
            json.dumps(self._entries, indent=2), encoding="utf-8"
        )

    def get_recent_entries(self, count: int = 10) -> list[dict]:
        """Return the most recent journal entries."""
        return self._entries[-count:]

    def reset(self) -> None:
        """Reset journal (for backtesting)."""
        self._entries = []
        self._trade_number = 0
        if JOURNAL_LOG_PATH.exists():
            JOURNAL_LOG_PATH.unlink()
        if JOURNAL_JSON_PATH.exists():
            JOURNAL_JSON_PATH.unlink()
