"""On-chain flow intelligence from public Bitcoin APIs.

This is the free/basic layer: it measures large BTC movements and applies
transparent heuristics. Exact entity labels (Binance, Coinbase, miners, etc.)
require a maintained address-label file or a paid intelligence API.
"""

from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
from typing import Any

import httpx
import pandas as pd

from src import DATA_DIR
from src.collectors import BaseCollector
from src.utils.cache import get_cached, set_cached
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

BLOCKCHAIR_BASE = "https://api.blockchair.com/bitcoin"
BLOCKCHAIN_INFO_UNCONFIRMED = "https://blockchain.info/unconfirmed-transactions"
DEFAULT_WHALE_THRESHOLD_BTC = 100.0
HISTORY_PATH = DATA_DIR / "history" / "onchain_flows.parquet"
LATEST_PATH = DATA_DIR / "history" / "onchain_flows_latest.json"
LABELS_PATH = DATA_DIR / "onchain" / "address_labels.json"


FLOW_FEATURE_COLUMNS = [
    "whale_btc_moved_1h",
    "whale_btc_moved_6h",
    "whale_btc_moved_24h",
    "whale_btc_moved_7d",
    "whale_tx_count_24h",
    "largest_whale_tx_btc_24h",
    "exchange_inflow_btc_24h",
    "exchange_outflow_btc_24h",
    "net_exchange_flow_btc_24h",
    "miner_outflow_btc_24h",
    "miner_to_exchange_btc_24h",
    "unknown_large_flow_btc_24h",
    "cold_storage_like_btc_24h",
    "distribution_like_btc_24h",
    "flow_accumulation_score",
]


def _empty_labels() -> dict[str, dict[str, str]]:
    return {
        "addresses": {},
        "notes": {
            "schema": "address -> {entity, category}; categories: exchange, miner, custody, cold_wallet, whale",
            "free_mode": "Unknown addresses stay unknown until labels are added or a labeling API is connected.",
        },
    }


class OnChainFlowCollector(BaseCollector):
    """Measure large BTC movements and coarse destination/source categories."""

    name = "onchain_flows"
    tier = 3
    update_interval_seconds = 3600

    def __init__(
        self,
        min_btc: float = DEFAULT_WHALE_THRESHOLD_BTC,
        detail_limit: int = 10,
        labels_path: Path | None = None,
    ) -> None:
        self.min_btc = min_btc
        self.detail_limit = detail_limit
        self.labels_path = labels_path or LABELS_PATH

    def _ensure_labels_file(self) -> None:
        self.labels_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.labels_path.exists():
            self.labels_path.write_text(json.dumps(_empty_labels(), indent=2), encoding="utf-8")

    def _load_labels(self) -> dict[str, dict[str, str]]:
        self._ensure_labels_file()
        try:
            data = json.loads(self.labels_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}
        addresses = data.get("addresses", {})
        return addresses if isinstance(addresses, dict) else {}

    async def _fetch_large_transactions(self) -> list[dict[str, Any]]:
        min_satoshi = int(self.min_btc * 1e8)
        cache_key = f"onchain_flow_large_txs_{min_satoshi}"
        cached = get_cached(cache_key, max_age_minutes=15)
        if cached:
            for tx in cached:
                tx["timestamp"] = pd.to_datetime(tx["timestamp"], utc=True)
            return cached

        async with httpx.AsyncClient(timeout=30) as client:
            try:
                resp = await client.get(
                    f"{BLOCKCHAIR_BASE}/transactions",
                    params={
                        "q": f"output_total({min_satoshi}..)",
                        "s": "time(desc)",
                        "limit": 100,
                    },
                )
                resp.raise_for_status()
                txs = resp.json().get("data", [])
                records = [self._record_from_blockchair_tx(tx) for tx in txs]
            except httpx.HTTPStatusError as exc:
                logger.info(
                    "Blockchair large transaction feed unavailable (%s); using blockchain.info fallback",
                    exc.response.status_code,
                )
                records = await self._fetch_blockchain_info_unconfirmed(client)
            set_cached(cache_key, records)
            return records

    @staticmethod
    def _record_from_blockchair_tx(tx: dict[str, Any]) -> dict[str, Any]:
        return {
            "timestamp": pd.to_datetime(tx.get("time"), utc=True),
            "tx_hash": tx.get("hash", ""),
            "total_btc": float(tx.get("output_total", 0) or 0) / 1e8,
            "fee_btc": float(tx.get("fee", 0) or 0) / 1e8,
            "input_count": int(tx.get("input_count", 0) or 0),
            "output_count": int(tx.get("output_count", 0) or 0),
        }

    async def _fetch_blockchain_info_unconfirmed(
        self, client: httpx.AsyncClient
    ) -> list[dict[str, Any]]:
        resp = await client.get(BLOCKCHAIN_INFO_UNCONFIRMED, params={"format": "json"})
        resp.raise_for_status()
        txs = resp.json().get("txs", [])
        records: list[dict[str, Any]] = []
        for tx in txs:
            outputs = tx.get("out", []) or []
            inputs = tx.get("inputs", []) or []
            output_total = sum(float(out.get("value", 0) or 0) for out in outputs)
            if output_total < self.min_btc * 1e8:
                continue
            input_addresses = [
                str(item.get("prev_out", {}).get("addr"))
                for item in inputs
                if item.get("prev_out", {}).get("addr")
            ]
            output_addresses = [
                str(item.get("addr"))
                for item in outputs
                if item.get("addr")
            ]
            records.append({
                "timestamp": pd.to_datetime(tx.get("time"), unit="s", utc=True),
                "tx_hash": tx.get("hash", ""),
                "total_btc": output_total / 1e8,
                "fee_btc": 0.0,
                "input_count": len(inputs),
                "output_count": len(outputs),
                "input_addresses": input_addresses,
                "output_addresses": output_addresses,
            })
        return records

    async def _fetch_transaction_details(self, tx_hashes: list[str]) -> dict[str, dict[str, Any]]:
        if not tx_hashes:
            return {}
        cache_key = "onchain_flow_tx_details_" + "_".join(tx_hashes[: self.detail_limit])
        cached = get_cached(cache_key, max_age_minutes=60)
        if cached:
            return cached

        details: dict[str, dict[str, Any]] = {}
        async with httpx.AsyncClient(timeout=30) as client:
            for tx_hash in tx_hashes[: self.detail_limit]:
                try:
                    resp = await client.get(f"{BLOCKCHAIR_BASE}/dashboards/transaction/{tx_hash}")
                    if resp.status_code != 200:
                        continue
                    data = resp.json().get("data", {}).get(tx_hash, {})
                    details[tx_hash] = data
                except Exception as exc:  # pragma: no cover - network defensive
                    logger.debug("Transaction detail fetch failed for %s: %s", tx_hash, exc)
        set_cached(cache_key, details)
        return details

    @staticmethod
    def _addresses_from_io(items: list[dict[str, Any]]) -> list[str]:
        addresses: list[str] = []
        for item in items or []:
            recipient = item.get("recipient") or item.get("spending_signature_hex")
            if recipient:
                addresses.append(str(recipient))
        return addresses

    def _classify_transaction(
        self,
        tx: dict[str, Any],
        details: dict[str, Any] | None,
        labels: dict[str, dict[str, str]],
    ) -> dict[str, Any]:
        categories = set()
        entities = set()
        if details:
            input_addresses = self._addresses_from_io(details.get("inputs", []))
            output_addresses = self._addresses_from_io(details.get("outputs", []))
            input_addresses.extend(details.get("input_addresses", []) or [])
            output_addresses.extend(details.get("output_addresses", []) or [])
            for address in input_addresses + output_addresses:
                label = labels.get(address)
                if label:
                    categories.add(str(label.get("category", "unknown")))
                    entities.add(str(label.get("entity", address)))

        output_count = int(tx.get("output_count", 0) or 0)
        if output_count <= 2:
            heuristic = "cold_storage_like"
        elif output_count > 5:
            heuristic = "distribution_like"
        else:
            heuristic = "neutral"

        return {
            "category": ",".join(sorted(categories)) if categories else "unknown",
            "entity": ",".join(sorted(entities)) if entities else "unknown",
            "heuristic": heuristic,
        }

    @staticmethod
    def _window(records: list[dict[str, Any]], now: pd.Timestamp, hours: int) -> list[dict[str, Any]]:
        cutoff = now - pd.Timedelta(hours=hours)
        return [r for r in records if r["timestamp"] >= cutoff]

    def _summarize(self, records: list[dict[str, Any]], now: pd.Timestamp) -> dict[str, Any]:
        summary: dict[str, Any] = {}
        for hours, suffix in [(1, "1h"), (6, "6h"), (24, "24h"), (168, "7d")]:
            sub = self._window(records, now, hours)
            summary[f"whale_btc_moved_{suffix}"] = round(sum(r["total_btc"] for r in sub), 4)
            summary[f"whale_tx_count_{suffix}"] = len(sub)

        last_24h = self._window(records, now, 24)
        by_category: dict[str, float] = defaultdict(float)
        by_heuristic: dict[str, float] = defaultdict(float)
        for rec in last_24h:
            by_category[str(rec.get("category", "unknown"))] += rec["total_btc"]
            by_heuristic[str(rec.get("heuristic", "neutral"))] += rec["total_btc"]

        exchange_in = by_category.get("exchange", 0.0)
        exchange_out = by_category.get("exchange_outflow", 0.0)
        miner_out = by_category.get("miner", 0.0)
        miner_to_exchange = by_category.get("miner_to_exchange", 0.0)
        unknown = by_category.get("unknown", 0.0)
        cold_like = by_heuristic.get("cold_storage_like", 0.0)
        distribution_like = by_heuristic.get("distribution_like", 0.0)
        total_directional = cold_like + distribution_like
        accumulation_score = (
            (cold_like - distribution_like) / total_directional
            if total_directional > 0
            else 0.0
        )

        summary.update({
            "largest_whale_tx_btc_24h": round(max((r["total_btc"] for r in last_24h), default=0.0), 4),
            "exchange_inflow_btc_24h": round(exchange_in, 4),
            "exchange_outflow_btc_24h": round(exchange_out, 4),
            "net_exchange_flow_btc_24h": round(exchange_in - exchange_out, 4),
            "miner_outflow_btc_24h": round(miner_out, 4),
            "miner_to_exchange_btc_24h": round(miner_to_exchange, 4),
            "unknown_large_flow_btc_24h": round(unknown, 4),
            "cold_storage_like_btc_24h": round(cold_like, 4),
            "distribution_like_btc_24h": round(distribution_like, 4),
            "flow_accumulation_score": round(accumulation_score, 4),
            "label_source": "local_address_labels",
            "label_coverage_note": "free_mode_unknown_unless_address_label_exists",
            "top_transactions": [
                {
                    "timestamp": str(r["timestamp"]),
                    "tx_hash": r["tx_hash"],
                    "total_btc": round(r["total_btc"], 4),
                    "category": r.get("category", "unknown"),
                    "heuristic": r.get("heuristic", "neutral"),
                }
                for r in sorted(last_24h, key=lambda item: item["total_btc"], reverse=True)[:10]
            ],
        })
        return summary

    def _persist_snapshot(self, summary: dict[str, Any], timestamp: pd.Timestamp) -> None:
        HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
        LATEST_PATH.parent.mkdir(parents=True, exist_ok=True)

        serializable = dict(summary)
        serializable["timestamp"] = timestamp.isoformat()
        LATEST_PATH.write_text(json.dumps(serializable, indent=2), encoding="utf-8")

        numeric = {k: v for k, v in summary.items() if isinstance(v, (int, float))}
        row = pd.DataFrame([numeric], index=[timestamp])
        row.index.name = "timestamp"
        if HISTORY_PATH.exists():
            try:
                existing = pd.read_parquet(HISTORY_PATH)
                combined = pd.concat([existing, row]).sort_index()
                combined = combined[~combined.index.duplicated(keep="last")]
            except Exception:
                combined = row
        else:
            combined = row
        combined.to_parquet(HISTORY_PATH)

    async def _collect(self) -> pd.DataFrame:
        now = pd.Timestamp.now(tz="UTC")
        try:
            labels = self._load_labels()
            txs = await self._fetch_large_transactions()
            tx_hashes = [tx["tx_hash"] for tx in txs if tx.get("tx_hash")]
            details = await self._fetch_transaction_details(tx_hashes)
            enriched = []
            for tx in txs:
                tx_hash = tx.get("tx_hash")
                detail = details.get(tx_hash, {}) if tx_hash else {}
                if not detail and (
                    tx.get("input_addresses") or tx.get("output_addresses")
                ):
                    detail = {
                        "input_addresses": tx.get("input_addresses", []),
                        "output_addresses": tx.get("output_addresses", []),
                    }
                classified = self._classify_transaction(tx, detail, labels)
                enriched.append({**tx, **classified})
            summary = self._summarize(enriched, now)
            self._persist_snapshot(summary, now)
            numeric = {k: v for k, v in summary.items() if isinstance(v, (int, float))}
            return pd.DataFrame([numeric], index=[now])
        except Exception as exc:
            logger.warning("On-chain flow collection failed: %s", exc)
            return pd.DataFrame()

    async def collect_history(self, start: str, end: str | None = None) -> pd.DataFrame:
        if not HISTORY_PATH.exists():
            return pd.DataFrame()
        df = pd.read_parquet(HISTORY_PATH)
        mask = df.index >= pd.Timestamp(start, tz="UTC")
        if end:
            mask &= df.index <= pd.Timestamp(end, tz="UTC")
        return df[mask]
