"""Closed-loop auto-training entry point.

Runs the "learn from your own mistakes" loop on top of the realized prediction
outcomes the scorer has graded:

    score_predictions.py   (enriched scoring -> prediction_scores.jsonl)
            |
            v
    labeled_store          (data/training_data/labeled.jsonl, append-only)
            |
            v
    autotrain.py  ──►  refit probability calibrator on LIVE outcomes  (always-on)
                  └─►  blend live rows into model retraining           (gated)
                             │
                             └─ SAFETY GUARD: train into staging, only promote
                                if not worse than the serving artifact on a
                                held-out slice of recent live outcomes.

Designed for stateless cloud (GitHub Actions) runs: it reads the git-tracked
JSONL stores, writes promoted artifacts back into ``data/validation/models/``
(the serving location the predictor loads from), and no-ops cleanly when little
or no data has matured yet.

Usage::

    python autotrain.py                       # calibrator loop (+ model blend if eligible)
    python autotrain.py --dry-run             # evaluate + report, write nothing
    python autotrain.py --no-retrain-models   # calibrator loop only
    python autotrain.py --refresh-scores      # also run the scorer first
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src import DATA_DIR
from src.training.closed_loop import (
    DEFAULT_ACCURACY_TOLERANCE,
    DEFAULT_CALIBRATION_TOLERANCE,
    DEFAULT_HOLDOUT_FRAC,
    MIN_CALIBRATION_ROWS,
    MIN_RETRAIN_ROWS,
    refit_live_calibrator,
    retrain_models_with_live_blend,
)
from src.training.labeled_store import (
    LABELED_STORE_PATH,
    load_labeled_rows,
    per_horizon_counts,
    update_labeled_store_from_scores,
)
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

SERVING_MODEL_DIR = DATA_DIR / "validation" / "models"
STAGING_MODEL_DIR = DATA_DIR / "training_data" / "staging" / "models"
AUTOTRAIN_REPORT_PATH = DATA_DIR / "training_data" / "autotrain_report.json"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="BTC Predictor closed-loop auto-trainer")
    parser.add_argument(
        "--refresh-scores", action="store_true",
        help="Run the scorer first to grade any newly matured predictions",
    )
    parser.add_argument(
        "--no-retrain-models", action="store_true",
        help="Only refit the calibrator from live outcomes (skip model retraining)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Evaluate + report promotion decisions without writing any artifact",
    )
    parser.add_argument(
        "--min-calibration-rows", type=int, default=MIN_CALIBRATION_ROWS,
        help=f"Min live rows/horizon to refit calibrator (default {MIN_CALIBRATION_ROWS})",
    )
    parser.add_argument(
        "--min-retrain-rows", type=int, default=MIN_RETRAIN_ROWS,
        help=f"Min live rows/horizon to blend into retraining (default {MIN_RETRAIN_ROWS})",
    )
    parser.add_argument(
        "--holdout-frac", type=float, default=DEFAULT_HOLDOUT_FRAC,
        help=f"Recent live fraction held out to judge promotion (default {DEFAULT_HOLDOUT_FRAC})",
    )
    parser.add_argument(
        "--calibration-tolerance", type=float, default=DEFAULT_CALIBRATION_TOLERANCE,
        help="Brier tolerance for calibrator promotion (default 0.0 = strict)",
    )
    parser.add_argument(
        "--accuracy-tolerance", type=float, default=DEFAULT_ACCURACY_TOLERANCE,
        help=f"Directional-accuracy tolerance for model promotion (default {DEFAULT_ACCURACY_TOLERANCE})",
    )
    parser.add_argument(
        "--skip-tft", action="store_true",
        help="Skip optional TFT during model retraining (CPU-only CI convention)",
    )
    return parser.parse_args(argv)


def run_autotrain(args: argparse.Namespace) -> dict[str, Any]:
    """Execute the closed loop and return a machine-readable summary."""
    summary: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dry_run": bool(args.dry_run),
    }

    if args.refresh_scores:
        try:
            from src.engine.scorer import run_scorer

            scorer_result = run_scorer()
            summary["scorer"] = {
                "new_scores": scorer_result.get("new_scores", 0),
                "labeled_rows_added": scorer_result.get("labeled_rows_added", 0),
            }
        except Exception as exc:
            logger.warning("Score refresh failed (non-fatal): %s", exc)
            summary["scorer"] = {"error": str(exc)}

    # Make sure the labeled store reflects every graded score before we learn.
    added = update_labeled_store_from_scores()
    summary["labeled_rows_added"] = added

    labeled_rows = load_labeled_rows()
    counts = per_horizon_counts()
    summary["labeled_total_rows"] = len(labeled_rows)
    summary["labeled_per_horizon"] = counts

    if not labeled_rows:
        logger.info("No labeled live data yet -- closed loop is a no-op.")
        summary["status"] = "no_data"
        summary["calibration"] = {"status": "no_data"}
        summary["models"] = {"status": "no_data"}
        return summary

    # 1) Calibrator closed loop (always attempted; the modest-data win).
    summary["calibration"] = refit_live_calibrator(
        labeled_rows,
        serving_model_dir=SERVING_MODEL_DIR,
        min_rows=args.min_calibration_rows,
        holdout_frac=args.holdout_frac,
        tolerance=args.calibration_tolerance,
        dry_run=args.dry_run,
    )

    # 2) Model retraining with a live blend (gated on having enough matured data).
    if args.no_retrain_models:
        summary["models"] = {"status": "disabled"}
    else:
        summary["models"] = retrain_models_with_live_blend(
            labeled_rows,
            serving_model_dir=SERVING_MODEL_DIR,
            staging_model_dir=STAGING_MODEL_DIR,
            min_rows_per_horizon=args.min_retrain_rows,
            holdout_frac=args.holdout_frac,
            tolerance=args.accuracy_tolerance,
            skip_tft=args.skip_tft,
            dry_run=args.dry_run,
        )

    summary["status"] = "ran"
    return summary


def _print_summary(summary: dict[str, Any]) -> None:
    print("\n" + "=" * 70)
    print("  CLOSED-LOOP AUTO-TRAIN SUMMARY")
    print("=" * 70)
    print(f"  Labeled rows total : {summary.get('labeled_total_rows', 0)}")
    print(f"  Rows added this run: {summary.get('labeled_rows_added', 0)}")
    if summary.get("dry_run"):
        print("  MODE               : DRY-RUN (nothing written)")

    cal = summary.get("calibration", {})
    if cal.get("status") == "no_data":
        print("  Calibrator         : no data yet")
    elif cal:
        print(
            f"  Calibrator         : {cal.get('n_promoted', 0)} promoted, "
            f"{cal.get('n_kept', 0)} kept, {cal.get('n_skipped', 0)} skipped"
        )
        for tf, h in cal.get("horizons", {}).items():
            if h.get("status") in ("promote", "keep"):
                print(
                    f"      {tf:<6s} {h['status']:>7s}  "
                    f"Brier {h.get('current_brier')} -> {h.get('candidate_brier')} "
                    f"(n={h.get('n')})"
                )

    models = summary.get("models", {})
    mstatus = models.get("status")
    if mstatus == "disabled":
        print("  Model retrain      : disabled (--no-retrain-models)")
    elif mstatus == "skipped":
        print(f"  Model retrain      : skipped ({models.get('reason')})")
    elif mstatus == "ran":
        print(f"  Model retrain      : {models.get('n_promoted', 0)} horizon(s) promoted")
    else:
        print("  Model retrain      : no data yet")
    print("=" * 70 + "\n")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = run_autotrain(args)

    AUTOTRAIN_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    AUTOTRAIN_REPORT_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("Autotrain report written -> %s", AUTOTRAIN_REPORT_PATH)

    _print_summary(summary)
    print(f"Labeled store: {LABELED_STORE_PATH}")
    print(f"Report:        {AUTOTRAIN_REPORT_PATH}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception as exc:  # pragma: no cover - top-level guard
        logger.exception("Autotrain failed: %s", exc)
        print(f"\n[ERROR] Autotrain failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
