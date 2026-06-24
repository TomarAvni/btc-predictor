"""Tests for trading activity summary helpers."""

from __future__ import annotations

from dashboard.data_loader import (
    filter_trades_for_analytics,
    get_trading_activity_summary,
)


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

    assert summary["raw_closed_trades"] == 2
    assert summary["closed_trades"] == 1
    assert summary["excluded_trades_count"] == 1
    assert summary["open_positions"] == 1
    assert summary["journal_entries"] == 5
    assert summary["journal_entries_count"] == 2
    assert summary["journal_exits_count"] == 2
    assert summary["journal_skips_count"] == 1
    assert summary["total_closed_pnl"] == -2.0
    assert summary["raw_total_closed_pnl"] == -1.0
    assert summary["duplicate_trade_ids"] == 0


def test_filter_trades_for_analytics_excludes_earliest_by_entry_time():
    trades = [
        {
            "id": "later",
            "entry_time": "2026-02-01T00:00:00+00:00",
            "exit_time": "2026-02-02T00:00:00+00:00",
            "pnl_usd": 5.0,
        },
        {
            "id": "earliest",
            "entry_time": "2026-01-01T00:00:00+00:00",
            "exit_time": "2026-01-02T00:00:00+00:00",
            "pnl_usd": -10.0,
        },
    ]

    result = filter_trades_for_analytics(trades)

    assert result["raw_count"] == 2
    assert result["analytics_count"] == 1
    assert result["excluded_count"] == 1
    assert result["excluded_trades"][0]["id"] == "earliest"
    assert [t["id"] for t in result["trades"]] == ["later"]


def test_filter_trades_for_analytics_tiebreaks_on_exit_time():
    trades = [
        {
            "id": "second",
            "entry_time": "2026-01-01T00:00:00+00:00",
            "exit_time": "2026-01-03T00:00:00+00:00",
            "pnl_usd": 1.0,
        },
        {
            "id": "first",
            "entry_time": "2026-01-01T00:00:00+00:00",
            "exit_time": "2026-01-02T00:00:00+00:00",
            "pnl_usd": -1.0,
        },
    ]

    result = filter_trades_for_analytics(trades)

    assert result["excluded_trades"][0]["id"] == "first"
    assert result["trades"][0]["id"] == "second"


def test_filter_trades_for_analytics_can_disable_exclusion():
    trades = [{"id": "only", "entry_time": "2026-01-01T00:00:00+00:00", "pnl_usd": 3.0}]

    result = filter_trades_for_analytics(trades, exclude_count=0)

    assert result["excluded_count"] == 0
    assert result["analytics_count"] == 1
    assert result["trades"][0]["id"] == "only"
