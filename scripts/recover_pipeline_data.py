"""Recover and merge append-only pipeline JSONL stores from git history.

Use when workflow data was lost during a branch merge or when performance
artifacts on disk diverge from the canonical git-tracked history on master.
Merges by stable keys (run_number for predictions; run:timeframe:model_source
for scores), optionally pulls rows from git refs, re-scores any newly mature
predictions, and rebuilds derived performance/training artifacts.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.engine.scorer import (  # noqa: E402
    SCORES_PATH,
    _score_key,
    run_scorer,
    score_mature_predictions,
)
from src.output.jsonl_logger import PREDICTIONS_JSONL_PATH  # noqa: E402
from src.training.labeled_store import (  # noqa: E402
    LABELED_STORE_PATH,
    _store_key,
    update_labeled_store_from_scores,
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def _git_jsonl(ref: str, rel_path: str) -> list[dict[str, Any]]:
    try:
        raw = subprocess.check_output(
            ["git", "show", f"{ref}:{rel_path}"],
            cwd=ROOT,
            stderr=subprocess.DEVNULL,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return []
    rows: list[dict[str, Any]] = []
    for line in raw.decode("utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def _merge_rows(
    sources: list[list[dict[str, Any]]],
    key_fn: Callable[[dict[str, Any]], str | None],
    sort_key: Callable[[dict[str, Any]], Any] | None = None,
) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for source in sources:
        for row in source:
            key = key_fn(row)
            if key is None:
                continue
            merged[key] = row
    rows = list(merged.values())
    if sort_key is not None:
        rows.sort(key=sort_key)
    return rows


def _prediction_key(row: dict[str, Any]) -> str | None:
    run = row.get("run_number")
    return str(run) if run is not None else None


def _score_row_key(row: dict[str, Any]) -> str | None:
    run = row.get("run_number")
    tf = row.get("timeframe")
    if run is None or tf is None:
        return None
    return _score_key(run, str(tf), row.get("model_source"))


def _labeled_key(row: dict[str, Any]) -> str | None:
    key = _store_key(row)
    return key if key.count(":") == 2 else None


def recover(
    git_refs: list[str],
    predictions_path: Path,
    scores_path: Path,
    labeled_path: Path,
    rescore: bool,
) -> dict[str, Any]:
    pred_sources = [_read_jsonl(predictions_path)]
    score_sources = [_read_jsonl(scores_path)]
    labeled_sources = [_read_jsonl(labeled_path)]

    for ref in git_refs:
        pred_sources.append(_git_jsonl(ref, "data/predictions/predictions.jsonl"))
        score_sources.append(_git_jsonl(ref, "data/performance/prediction_scores.jsonl"))
        labeled_sources.append(_git_jsonl(ref, "data/training_data/labeled.jsonl"))

    predictions = _merge_rows(
        pred_sources,
        _prediction_key,
        sort_key=lambda r: (int(r.get("run_number") or 0), str(r.get("timestamp") or "")),
    )
    scores = _merge_rows(
        score_sources,
        _score_row_key,
        sort_key=lambda r: (
            str(r.get("prediction_timestamp") or ""),
            int(r.get("run_number") or 0),
            str(r.get("timeframe") or ""),
        ),
    )

    _write_jsonl(predictions_path, predictions)
    _write_jsonl(scores_path, scores)

    new_scores = 0
    if rescore:
        new_scores = len(score_mature_predictions(scores_path=scores_path))

    labeled_before = len(_read_jsonl(labeled_path))
    _write_jsonl(
        labeled_path,
        _merge_rows(
            labeled_sources,
            _labeled_key,
            sort_key=lambda r: (
                str(r.get("prediction_timestamp") or ""),
                int(r.get("run_number") or 0),
                str(r.get("timeframe") or ""),
            ),
        ),
    )
    labeled_added = update_labeled_store_from_scores(
        scores_path=scores_path,
        store_path=labeled_path,
    )
    scorer_result = run_scorer()

    return {
        "prediction_rows": len(predictions),
        "score_rows": len(_read_jsonl(scores_path)),
        "labeled_rows": len(_read_jsonl(labeled_path)),
        "new_scores": new_scores,
        "labeled_rows_added": labeled_added,
        "labeled_rows_before_merge": labeled_before,
        "git_refs": git_refs,
        "scorer": scorer_result,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--git-ref",
        action="append",
        default=["650e550", "origin/master"],
        help="Git ref(s) to pull historical JSONL rows from (repeatable).",
    )
    parser.add_argument(
        "--no-rescore",
        action="store_true",
        help="Skip scoring newly mature predictions after merge.",
    )
    args = parser.parse_args()

    summary = recover(
        git_refs=args.git_ref,
        predictions_path=PREDICTIONS_JSONL_PATH,
        scores_path=SCORES_PATH,
        labeled_path=LABELED_STORE_PATH,
        rescore=not args.no_rescore,
    )
    print("Pipeline data recovery complete:")
    for key, value in summary.items():
        if key != "scorer":
            print(f"  {key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
