"""BTC Predictor -- one-command demo.

Downloads recent price data, runs a quick training cycle, backtests
the trading agent, and launches the dashboard.

Usage:
    python run_demo.py
"""

from __future__ import annotations

import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path


def run(cmd: list[str], description: str) -> bool:
    """Run a command, print status, return success."""
    print(f"\n{'='*70}")
    print(f"  {description}")
    print(f"{'='*70}\n")
    result = subprocess.run(cmd, cwd=Path(__file__).parent)
    if result.returncode != 0:
        print(f"\n[WARN] Step failed (exit {result.returncode}), continuing...")
        return False
    return True


def main() -> None:
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║              BTC PRICE PREDICTOR -- DEMO                           ║
║                                                                    ║
║  This script runs the full pipeline:                               ║
║    1. Download recent price history                                ║
║    2. Run 80/20 train-test validation                              ║
║    3. Show validation results                                      ║
║    4. Run trading agent backtest                                   ║
║    5. Launch dashboard                                             ║
╚══════════════════════════════════════════════════════════════════════╝
    """)

    py = sys.executable

    # Step 1: Download price data
    run(
        [py, "main.py", "--download"],
        "Step 1/5: Downloading price history...",
    )

    # Step 2: Run 80/20 validation
    run(
        [py, "validate.py", "--split", "0.8", "--output", "data/validation/",
         "--step-hours", "6"],
        "Step 2/5: Running 80/20 train-test validation...",
    )

    # Step 3: Show validation results
    print(f"\n{'='*70}")
    print("  Step 3/5: Validation Results Summary")
    print(f"{'='*70}\n")
    report_path = Path(__file__).parent / "data" / "validation" / "report.txt"
    if report_path.exists():
        print(report_path.read_text(encoding="utf-8"))
    else:
        print("  [INFO] No validation report found -- check step 2 output above.")

    # Step 4: Run trading agent backtest (last 30 days for speed)
    end_date = datetime.utcnow().strftime("%Y-%m-%d")
    start_date = (datetime.utcnow() - timedelta(days=30)).strftime("%Y-%m-%d")
    run(
        [py, "trade.py", "--backtest", "--start", start_date, "--end", end_date],
        f"Step 4/5: Running trading backtest ({start_date} to {end_date})...",
    )

    # Step 5: Launch dashboard
    print(f"\n{'='*70}")
    print("  Step 5/5: Launching dashboard...")
    print(f"{'='*70}\n")
    print("  Dashboard will open at http://localhost:8501")
    print("  Press Ctrl+C to stop.\n")

    try:
        subprocess.run(
            [py, "-m", "streamlit", "run", "dashboard/app.py"],
            cwd=Path(__file__).parent,
        )
    except KeyboardInterrupt:
        print("\n\nDemo complete.")


if __name__ == "__main__":
    main()
