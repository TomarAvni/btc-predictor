"""Export sentiment seed data for external scenario/memory experiments.

This script intentionally does not import MiroFish or OpenViking. It only
packages this repo's own JSON/parquet-backed context into a portable JSON file
for manual, process-isolated research.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "experiments" / "output" / "sentiment_sandbox_seed.json"


def _read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def build_seed(history_dir: Path, performance_dir: Path) -> dict[str, Any]:
    state = _read_json(history_dir / "twitter_market_state.json", {})
    events = _read_json(history_dir / "twitter_notable_events.json", [])
    rolling_accuracy = _read_json(performance_dir / "rolling_accuracy.json", {})

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": "btc-predictor",
        "license_boundary": (
            "Data export only. Do not import AGPL tools into the predictor runtime "
            "without a separate licensing review."
        ),
        "promotion_gate": (
            "Scenario outputs are research context only until compared against "
            "mature held-out prediction scores."
        ),
        "market_state": state,
        "notable_events": events[-50:] if isinstance(events, list) else [],
        "rolling_accuracy": rolling_accuracy,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export sentiment sandbox seed data")
    parser.add_argument(
        "--history-dir",
        type=Path,
        default=ROOT / "data" / "history",
        help="Directory containing twitter_market_state.json and twitter_notable_events.json",
    )
    parser.add_argument(
        "--performance-dir",
        type=Path,
        default=ROOT / "data" / "performance",
        help="Directory containing rolling_accuracy.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output JSON path",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    seed = build_seed(args.history_dir, args.performance_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(seed, indent=2), encoding="utf-8")
    print(f"Wrote sandbox seed: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
