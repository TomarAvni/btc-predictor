"""Order model for the trading agent."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Literal


def _ensure_utc(dt: datetime) -> datetime:
    """Ensure a datetime is timezone-aware (assume UTC if naive)."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt


@dataclass
class Order:
    """Represents a single trade order (buy or sell)."""

    side: Literal["BUY", "SELL"]
    amount_usd: float
    amount_btc: float
    price: float
    order_type: Literal["MARKET", "STOP_LOSS", "TAKE_PROFIT"]
    reason: str
    prediction_id: str
    timeframe: str
    confidence: float
    stop_loss: float
    take_profit: float
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict:
        """Serialize to dictionary for JSON storage."""
        d = asdict(self)
        d["timestamp"] = self.timestamp.isoformat()
        return d

    @classmethod
    def from_dict(cls, data: dict) -> Order:
        """Deserialize from dictionary."""
        data = data.copy()
        if isinstance(data.get("timestamp"), str):
            data["timestamp"] = _ensure_utc(datetime.fromisoformat(data["timestamp"]))
        return cls(**data)


@dataclass
class Position:
    """An open trading position."""

    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    entry_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    entry_price: float = 0.0
    amount_usd: float = 0.0
    amount_btc: float = 0.0
    side: Literal["LONG", "SHORT"] = "LONG"
    timeframe: str = "24h"
    stop_loss: float = 0.0
    take_profit: float = 0.0
    trailing_stop_active: bool = False
    prediction_id: str = ""
    confidence: float = 0.0
    reason: str = ""

    @property
    def unrealized_pnl(self) -> float:
        """Calculate unrealized P&L given current price (set externally)."""
        return 0.0  # Calculated by portfolio with current price

    def unrealized_pnl_at(self, current_price: float) -> float:
        """Calculate unrealized P&L at a given price."""
        if self.side == "LONG":
            return self.amount_btc * (current_price - self.entry_price)
        return self.amount_btc * (self.entry_price - current_price)

    def unrealized_pnl_pct(self, current_price: float) -> float:
        """P&L as percentage of entry value."""
        if self.entry_price == 0:
            return 0.0
        if self.side == "LONG":
            return (current_price - self.entry_price) / self.entry_price * 100
        return (self.entry_price - current_price) / self.entry_price * 100

    def to_dict(self) -> dict:
        d = {
            "id": self.id,
            "entry_time": self.entry_time.isoformat(),
            "entry_price": self.entry_price,
            "amount_usd": self.amount_usd,
            "amount_btc": self.amount_btc,
            "side": self.side,
            "timeframe": self.timeframe,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "trailing_stop_active": self.trailing_stop_active,
            "prediction_id": self.prediction_id,
            "confidence": self.confidence,
            "reason": self.reason,
        }
        return d

    @classmethod
    def from_dict(cls, data: dict) -> Position:
        data = data.copy()
        data.setdefault("side", "LONG")
        if isinstance(data.get("entry_time"), str):
            data["entry_time"] = _ensure_utc(datetime.fromisoformat(data["entry_time"]))
        return cls(**data)


@dataclass
class Trade:
    """A closed (completed) trade."""

    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    entry_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    exit_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    entry_price: float = 0.0
    exit_price: float = 0.0
    amount_usd: float = 0.0
    amount_btc: float = 0.0
    side: Literal["LONG", "SHORT"] = "LONG"
    timeframe: str = "24h"
    pnl_usd: float = 0.0
    pnl_pct: float = 0.0
    exit_reason: str = ""
    prediction_id: str = ""
    fees_paid: float = 0.0

    @property
    def is_winner(self) -> bool:
        return self.pnl_usd > 0

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "entry_time": self.entry_time.isoformat(),
            "exit_time": self.exit_time.isoformat(),
            "entry_price": self.entry_price,
            "exit_price": self.exit_price,
            "amount_usd": self.amount_usd,
            "amount_btc": self.amount_btc,
            "side": self.side,
            "timeframe": self.timeframe,
            "pnl_usd": self.pnl_usd,
            "pnl_pct": self.pnl_pct,
            "exit_reason": self.exit_reason,
            "prediction_id": self.prediction_id,
            "fees_paid": self.fees_paid,
        }

    @classmethod
    def from_dict(cls, data: dict) -> Trade:
        data = data.copy()
        data.setdefault("side", "LONG")
        for k in ("entry_time", "exit_time"):
            if isinstance(data.get(k), str):
                data[k] = _ensure_utc(datetime.fromisoformat(data[k]))
        return cls(**data)
