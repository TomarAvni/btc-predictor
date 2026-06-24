"""Tests for trading activity summary helpers."""

from __future__ import annotations

from dashboard.data_loader import get_trading_activity_summary


def test_trading_activity_summary_counts_actions():
    trades = [{"id": "a1", "pnl_usd": 1.0}, {"id": "a2", "pnl_usd": -2.0}]
    journal = [
        {"action": "SKIP", "timestamp": "2026-01-01T00:00:00+00:00"},
        {"action": "BUY", "timestamp": "2026-01-01T01:00:00+00:00"},
        {"action": "SHORT", "timestamp": "2026-01-01T02:00:00+00:00"},
        {"action": "CLOSE", "timestamp": "2026-01-01T03:00:00+00:00"},
        {"action": "CLOSE", "timestamp": "2026-01-01T04:00:00+00:00"},
    ]
    portfolio = {"positions": [{"id": "open1"}]}

    summary = get_trading_activity_summary(trades=trades, journal=journal, portfolio=portfolio)

    assert summary["closed_trades"] == 2
    assert summary["open_positions"] == 1
    assert summary["journal_entries"] == 5
    assert summary["journal_entries_count"] == 2
    assert summary["journal_exits_count"] == 2
    assert summary["journal_skips_count"] == 1
    assert summary["total_closed_pnl"] == -1.0
    assert summary["duplicate_trade_ids"] == 0
