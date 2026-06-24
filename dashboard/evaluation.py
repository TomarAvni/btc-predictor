"""Dashboard evaluation helpers for strategy diagnostics.

These helpers intentionally separate recorded performance from diagnostic
"what-if" views such as inverted or random decisions. The what-if modes are
for comparison only; they are not trading recommendations.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import random
from typing import Any, Iterable


STARTING_BALANCE = 2000.0


@dataclass(frozen=True)
class StrategySeries:
    """A comparable equity/P&L series for one strategy view."""

    name: str
    rows: list[dict[str, Any]]
    metrics: dict[str, Any]
    cumulative_pnl: list[float]
    note: str = ""


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _opposite_side(side: str) -> str:
    return "SHORT" if str(side).upper() == "LONG" else "LONG"


def _trade_size_btc(trade: dict[str, Any]) -> float:
    """Return position size in BTC, inferring from notional when needed."""
    amount_btc = _safe_float(trade.get("amount_btc"))
    if amount_btc > 0:
        return amount_btc
    entry = _safe_float(trade.get("entry_price"))
    amount_usd = _safe_float(trade.get("amount_usd"))
    if entry > 0 and amount_usd > 0:
        return amount_usd / entry
    return 0.0


def trade_notional_usd(trade: dict[str, Any]) -> float:
    """Return USD collateral/notional rather than BTC spot entry price."""
    amount_usd = _safe_float(trade.get("amount_usd"))
    entry = _safe_float(trade.get("entry_price"))
    amount_btc = _trade_size_btc(trade)
    if amount_btc > 0 and entry > 0:
        implied = amount_btc * entry
        if amount_usd >= entry * 0.9:
            return implied
    return amount_usd


def trade_size_btc(trade: dict[str, Any]) -> float:
    """Return position size in BTC for display and diagnostics."""
    return _trade_size_btc(trade)


def _pnl_for_side(trade: dict[str, Any], side: str) -> float:
    """Approximate gross P&L for a trade window with a chosen direction."""
    entry = _safe_float(trade.get("entry_price"))
    exit_price = _safe_float(trade.get("exit_price"))
    amount_btc = _trade_size_btc(trade)
    if entry <= 0 or exit_price <= 0 or amount_btc <= 0:
        recorded = _safe_float(trade.get("pnl_usd"))
        original = str(trade.get("side", "LONG")).upper()
        if side.upper() == original:
            return recorded
        return -recorded

    if side == "SHORT":
        return amount_btc * (entry - exit_price)
    return amount_btc * (exit_price - entry)


def cumulative_pnl_with_baseline(pnls: Iterable[float]) -> list[float]:
    """Build a cumulative P&L series that starts at zero before the first trade."""
    cumulative: list[float] = [0.0]
    running = 0.0
    for pnl in pnls:
        running += _safe_float(pnl)
        cumulative.append(running)
    return cumulative


def with_zero_baseline(values: Iterable[float]) -> list[float]:
    """Prepend a zero baseline to an already-cumulative P&L series."""
    series = list(values)
    if not series:
        return [0.0]
    if series[0] == 0.0:
        return series
    return [0.0, *series]


def _confidence_bucket(confidence: Any) -> str:
    conf = _safe_float(confidence, -1)
    if conf < 0:
        return "unknown"
    if conf < 55:
        return "<55"
    if conf < 65:
        return "55-65"
    if conf < 75:
        return "65-75"
    if conf < 85:
        return "75-85"
    return "85+"


def _journal_by_prediction_id(journal: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for entry in journal:
        pred_id = entry.get("prediction_id")
        if pred_id:
            out[str(pred_id)] = entry
    return out


def _enrich_trade(trade: dict[str, Any], journal_by_id: dict[str, dict[str, Any]]) -> dict[str, Any]:
    row = dict(trade)
    entry = journal_by_id.get(str(trade.get("prediction_id")))
    if entry:
        row.setdefault("confidence", entry.get("confidence"))
        row.setdefault("alignment_score", entry.get("alignment_score"))
        row.setdefault("sizing_tier", entry.get("sizing_tier"))
        row.setdefault("run_number", entry.get("run_number"))
    return row


def _series_from_rows(name: str, rows: list[dict[str, Any]], note: str = "") -> StrategySeries:
    cumulative = cumulative_pnl_with_baseline(_safe_float(row.get("pnl_usd")) for row in rows)
    return StrategySeries(name=name, rows=rows, metrics=trade_metrics(rows), cumulative_pnl=cumulative, note=note)


def trade_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute comparable trading metrics from rows with ``pnl_usd``."""
    pnls = [_safe_float(r.get("pnl_usd")) for r in rows]
    winners = [p for p in pnls if p > 0]
    losers = [p for p in pnls if p <= 0]
    total_pnl = sum(pnls)

    by_side: dict[str, float] = defaultdict(float)
    by_timeframe: dict[str, float] = defaultdict(float)
    by_exit_reason: dict[str, int] = defaultdict(int)
    by_confidence: dict[str, float] = defaultdict(float)
    for row in rows:
        pnl = _safe_float(row.get("pnl_usd"))
        by_side[str(row.get("side", "unknown"))] += pnl
        by_timeframe[str(row.get("timeframe", "unknown"))] += pnl
        by_exit_reason[str(row.get("exit_reason", "unknown"))] += 1
        by_confidence[_confidence_bucket(row.get("confidence"))] += pnl

    running = 0.0
    peak = 0.0
    max_drawdown = 0.0
    for pnl in pnls:
        running += pnl
        peak = max(peak, running)
        max_drawdown = min(max_drawdown, running - peak)

    gross_profit = sum(winners)
    gross_loss = abs(sum(losers))
    return {
        "total_trades": len(rows),
        "total_pnl": round(total_pnl, 2),
        "return_pct": round(total_pnl / STARTING_BALANCE * 100, 2),
        "win_rate_pct": round(len(winners) / len(rows) * 100, 1) if rows else 0.0,
        "profit_factor": round(gross_profit / gross_loss, 2) if gross_loss else None,
        "expectancy": round(total_pnl / len(rows), 2) if rows else 0.0,
        "avg_win": round(sum(winners) / len(winners), 2) if winners else 0.0,
        "avg_loss": round(sum(losers) / len(losers), 2) if losers else 0.0,
        "max_drawdown_usd": round(max_drawdown, 2),
        "by_side": dict(sorted(by_side.items())),
        "by_timeframe": dict(sorted(by_timeframe.items())),
        "by_exit_reason": dict(sorted(by_exit_reason.items())),
        "by_confidence": dict(sorted(by_confidence.items())),
    }


def build_strategy_series(
    trades: list[dict[str, Any]],
    journal: list[dict[str, Any]] | None = None,
    random_seed: int = 42,
) -> dict[str, StrategySeries]:
    """Build normal, inverse, and random diagnostic strategy series."""
    journal_by_id = _journal_by_prediction_id(journal or [])
    enriched = [_enrich_trade(t, journal_by_id) for t in sorted(trades, key=lambda t: t.get("exit_time", ""))]

    normal_rows = [dict(t, mode="normal") for t in enriched]

    inverse_rows: list[dict[str, Any]] = []
    for trade in enriched:
        side = _opposite_side(str(trade.get("side", "LONG")).upper())
        inverse_rows.append(
            {
                **trade,
                "side": side,
                "original_side": trade.get("side"),
                "pnl_usd": round(_pnl_for_side(trade, side), 6),
                "mode": "inverse",
            }
        )

    rng = random.Random(random_seed)
    random_rows: list[dict[str, Any]] = []
    for trade in enriched:
        side = "LONG" if rng.random() >= 0.5 else "SHORT"
        random_rows.append(
            {
                **trade,
                "side": side,
                "original_side": trade.get("side"),
                "pnl_usd": round(_pnl_for_side(trade, side), 6),
                "mode": "random",
            }
        )

    buy_hold_rows: list[dict[str, Any]] = []
    for trade in enriched:
        buy_hold_rows.append(
            {
                **trade,
                "side": "LONG",
                "original_side": trade.get("side"),
                "pnl_usd": round(_pnl_for_side(trade, "LONG"), 6),
                "mode": "buy_hold_window",
            }
        )

    blended_rows: list[dict[str, Any]] = []
    for normal, inverse in zip(normal_rows, inverse_rows):
        # Placeholder until a Twitter module emits its own trade intent. This
        # makes the future 50/50 behavior explicit without pretending it is live.
        blended_rows.append(
            {
                **normal,
                "pnl_usd": round((_safe_float(normal.get("pnl_usd")) + _safe_float(inverse.get("pnl_usd"))) / 2, 6),
                "mode": "blended_placeholder",
            }
        )

    return {
        "normal": _series_from_rows("Main predictor", normal_rows),
        "inverse": _series_from_rows(
            "Inverse diagnostic",
            inverse_rows,
            "Same trade windows with LONG/SHORT flipped. Diagnostic only.",
        ),
        "random": _series_from_rows(
            "Random baseline",
            random_rows,
            f"Seeded random side per historical trade window (seed={random_seed}).",
        ),
        "buy_hold": _series_from_rows(
            "Always-long baseline",
            buy_hold_rows,
            "Uses the same historical trade windows but always takes the long side.",
        ),
        "blended": _series_from_rows(
            "50/50 placeholder",
            blended_rows,
            "Temporary numeric/inverse placeholder until Twitter predictions exist.",
        ),
    }
