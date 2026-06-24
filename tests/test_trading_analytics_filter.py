"""Tests for dashboard trading analytics filtering."""

from __future__ import annotations

from dashboard.evaluation import build_strategy_series


def test_build_strategy_series_uses_filtered_trade_count():
    trades = [
        {
            "id": "keep",
            "entry_time": "2026-02-01T00:00:00+00:00",
            "exit_time": "2026-02-02T00:00:00+00:00",
            "entry_price": 100.0,
            "exit_price": 110.0,
            "amount_usd": 100.0,
            "side": "LONG",
            "pnl_usd": 10.0,
        }
    ]

    series_map = build_strategy_series(trades)

    assert series_map["normal"].metrics["total_trades"] == 1
    assert series_map["normal"].metrics["total_pnl"] == 10.0
