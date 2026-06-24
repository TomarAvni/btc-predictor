"""Recover and merge trading artifacts from git history and local disk.

Use when trades.json or journal.json were truncated during branch merges,
backtest overwrites, or CI data-file conflicts. Merges closed trades by
stable ``id`` and journal rows by timestamp/action/trade_number/reason.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

TRADES_REL = "data/trading/trades.json"
JOURNAL_REL = "data/trading/journal.json"
PORTFOLIO_REL = "data/trading/portfolio.json"


def _read_json(path: Path, fallback: Any) -> Any:
    if not path.exists():
        return fallback
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return fallback


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _git_json(ref: str, rel_path: str) -> Any:
    try:
        raw = subprocess.check_output(
            ["git", "show", f"{ref}:{rel_path}"],
            cwd=ROOT,
            stderr=subprocess.DEVNULL,
        )
        return json.loads(raw.decode("utf-8"))
    except (subprocess.CalledProcessError, FileNotFoundError, json.JSONDecodeError):
        return None


def _git_refs_touching(rel_path: str) -> list[str]:
    try:
        log = subprocess.check_output(
            ["git", "log", "--all", "--format=%H", "--", rel_path],
            cwd=ROOT,
            stderr=subprocess.DEVNULL,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return []
    return [line.strip() for line in log.decode().strip().splitlines() if line.strip()]


def _trade_key(trade: dict[str, Any]) -> str | None:
    trade_id = trade.get("id")
    return str(trade_id) if trade_id else None


def _journal_key(entry: dict[str, Any]) -> tuple[Any, ...]:
    reason = entry.get("reason") or ""
    if not reason and entry.get("reasons"):
        reason = "; ".join(str(r) for r in entry.get("reasons", []))
    return (
        entry.get("timestamp"),
        entry.get("action"),
        entry.get("trade_number"),
        str(entry.get("prediction_id") or ""),
        str(reason)[:120],
        entry.get("position_side") or entry.get("side"),
        entry.get("timeframe"),
    )


def _merge_trades(sources: list[list[dict[str, Any]]]) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for source in sources:
        for trade in source:
            key = _trade_key(trade)
            if key is None:
                continue
            merged[key] = trade
    return sorted(merged.values(), key=lambda t: str(t.get("exit_time") or ""))


def _merge_journal(sources: list[list[dict[str, Any]]]) -> list[dict[str, Any]]:
    merged: dict[tuple[Any, ...], dict[str, Any]] = {}
    for source in sources:
        for entry in source:
            merged[_journal_key(entry)] = entry
    return sorted(merged.values(), key=lambda e: str(e.get("timestamp") or ""))


def _merge_portfolio(sources: list[dict[str, Any]]) -> dict[str, Any] | None:
    candidates = [p for p in sources if isinstance(p, dict)]
    if not candidates:
        return None

    def sort_key(portfolio: dict[str, Any]) -> str:
        return str(portfolio.get("updated_at") or "")

    return max(candidates, key=sort_key)


def recover(
    git_refs: list[str] | None = None,
    trades_path: Path | None = None,
    journal_path: Path | None = None,
    portfolio_path: Path | None = None,
    include_disk: bool = True,
) -> dict[str, Any]:
    trades_path = trades_path or ROOT / TRADES_REL
    journal_path = journal_path or ROOT / JOURNAL_REL
    portfolio_path = portfolio_path or ROOT / PORTFOLIO_REL

    refs = list(git_refs or [])
    if not refs:
        refs = ["origin/master", "HEAD"]
        refs.extend(_git_refs_touching(TRADES_REL))
        refs.extend(_git_refs_touching(JOURNAL_REL))
    # Preserve order while deduplicating refs.
    refs = list(dict.fromkeys(refs))

    trade_sources: list[list[dict[str, Any]]] = []
    journal_sources: list[list[dict[str, Any]]] = []
    portfolio_sources: list[dict[str, Any]] = []

    if include_disk:
        disk_trades = _read_json(trades_path, [])
        disk_journal = _read_json(journal_path, [])
        disk_portfolio = _read_json(portfolio_path, None)
        if isinstance(disk_trades, list):
            trade_sources.append(disk_trades)
        if isinstance(disk_journal, list):
            journal_sources.append(disk_journal)
        if isinstance(disk_portfolio, dict):
            portfolio_sources.append(disk_portfolio)

    for ref in refs:
        git_trades = _git_json(ref, TRADES_REL)
        git_journal = _git_json(ref, JOURNAL_REL)
        git_portfolio = _git_json(ref, PORTFOLIO_REL)
        if isinstance(git_trades, list):
            trade_sources.append(git_trades)
        if isinstance(git_journal, list):
            journal_sources.append(git_journal)
        if isinstance(git_portfolio, dict):
            portfolio_sources.append(git_portfolio)

    trades_before = len(_read_json(trades_path, [])) if isinstance(_read_json(trades_path, []), list) else 0
    journal_before = len(_read_json(journal_path, [])) if isinstance(_read_json(journal_path, []), list) else 0

    merged_trades = _merge_trades(trade_sources)
    merged_journal = _merge_journal(journal_sources)
    merged_portfolio = _merge_portfolio(portfolio_sources)

    _write_json(trades_path, merged_trades)
    _write_json(journal_path, merged_journal)
    if merged_portfolio is not None:
        _write_json(portfolio_path, merged_portfolio)

    return {
        "git_refs": refs,
        "closed_trades_before": trades_before,
        "closed_trades_after": len(merged_trades),
        "journal_entries_before": journal_before,
        "journal_entries_after": len(merged_journal),
        "portfolio_updated_at": merged_portfolio.get("updated_at") if merged_portfolio else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--git-ref",
        action="append",
        default=None,
        help="Git ref(s) to pull trading rows from (repeatable). Defaults to all refs touching the files.",
    )
    parser.add_argument(
        "--no-disk",
        action="store_true",
        help="Merge git refs only; ignore current on-disk trading files.",
    )
    args = parser.parse_args()

    summary = recover(git_refs=args.git_ref, include_disk=not args.no_disk)
    print("Trading data recovery complete:")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
