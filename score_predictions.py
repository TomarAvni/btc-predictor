"""CLI entry point to score mature predictions against actual price moves."""

from src.engine.scorer import main

if __name__ == "__main__":
    raise SystemExit(main())
