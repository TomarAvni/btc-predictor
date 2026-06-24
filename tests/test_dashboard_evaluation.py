"""Tests for dashboard strategy evaluation helpers."""

from dashboard.evaluation import (
    build_strategy_series,
    cumulative_pnl_with_baseline,
    trade_metrics,
    trade_notional_usd,
    trade_size_btc,
    with_zero_baseline,
)


def _sample_trade(
    *,
    side: str = "LONG",
    entry_price: float = 100_000.0,
    exit_price: float = 102_000.0,
    amount_usd: float = 300.0,
    amount_btc: float = 0.003,
    pnl_usd: float = 6.0,
    exit_time: str = "2025-01-02T00:00:00+00:00",
) -> dict:
    return {
        "id": "t1",
        "entry_time": "2025-01-01T00:00:00+00:00",
        "exit_time": exit_time,
        "entry_price": entry_price,
        "exit_price": exit_price,
        "amount_usd": amount_usd,
        "amount_btc": amount_btc,
        "side": side,
        "timeframe": "24h",
        "pnl_usd": pnl_usd,
        "exit_reason": "take_profit",
        "prediction_id": "pred-1",
    }


def test_cumulative_pnl_series_starts_at_zero():
    series = cumulative_pnl_with_baseline([10.0, -4.0, 2.5])
    assert series == [0.0, 10.0, 6.0, 8.5]


def test_with_zero_baseline_prepends_start_point():
    assert with_zero_baseline([5.0, 8.0]) == [0.0, 5.0, 8.0]
    assert with_zero_baseline([0.0, 5.0]) == [0.0, 5.0]
    assert with_zero_baseline([]) == [0.0]


def test_build_strategy_series_prepends_zero_baseline():
    trades = [
        _sample_trade(pnl_usd=10.0, exit_time="2025-01-01T00:00:00+00:00"),
        _sample_trade(pnl_usd=-3.0, exit_time="2025-01-02T00:00:00+00:00"),
    ]
    trades[1]["id"] = "t2"

    series_map = build_strategy_series(trades)
    assert series_map["normal"].cumulative_pnl == [0.0, 10.0, 7.0]
    assert series_map["normal"].metrics["total_pnl"] == 7.0
    assert series_map["normal"].metrics["total_trades"] == 2


def test_trade_metrics_use_actual_pnl_rows():
    rows = [
        {"pnl_usd": 12.0, "side": "LONG", "timeframe": "24h", "exit_reason": "tp"},
        {"pnl_usd": -5.0, "side": "SHORT", "timeframe": "24h", "exit_reason": "sl"},
    ]
    metrics = trade_metrics(rows)
    assert metrics["total_pnl"] == 7.0
    assert metrics["total_trades"] == 2


def test_trade_notional_usd_uses_collateral_not_entry_price():
    trade = _sample_trade(amount_usd=100_000.0, amount_btc=0.003, entry_price=100_000.0)
    assert trade_notional_usd(trade) == 300.0
    assert trade_size_btc(trade) == 0.003


def test_inverse_pnl_uses_btc_size_not_misstored_notional():
    trade = _sample_trade(
        side="LONG",
        entry_price=100_000.0,
        exit_price=102_000.0,
        amount_usd=100_000.0,
        amount_btc=0.003,
        pnl_usd=6.0,
    )
    series_map = build_strategy_series([trade])
    inverse_pnl = series_map["inverse"].rows[0]["pnl_usd"]
    assert inverse_pnl == -6.0
