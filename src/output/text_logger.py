"""Prediction output logger -- appends formatted reports to a rolling text file."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class PredictionLogger:
    """Appends formatted prediction reports to ``predictions.log``.

    Handles log rotation when the file exceeds *max_size_mb* and tracks
    the run counter across sessions.
    """

    def __init__(
        self, output_path: str = "predictions.log", max_size_mb: int = 50
    ) -> None:
        self.output_path = Path(output_path)
        self.max_size_mb = max_size_mb
        self._run_counter = self._read_run_count()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def log_prediction(
        self,
        predictions: list[dict[str, Any]],
        signals: dict[str, dict[str, Any]],
    ) -> None:
        """Write a prediction report to the log file.

        Works with partial data -- if either *predictions* or *signals*
        is empty the corresponding section just says "(none available)".
        """
        self._rotate_if_needed()
        self._run_counter += 1

        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        sep = "=" * 80

        lines = [
            "",
            sep,
            f"[{now}] -- Prediction Run #{self._run_counter}",
            sep,
            "",
            "PREDICTIONS:",
        ]

        if predictions:
            for p in predictions:
                direction = p["direction"].upper()
                sign = "+" if direction == "UP" else "-"
                mag = f"{sign}{p['magnitude']:.1f}%"
                conf = f"{p['confidence']:.0f}%"
                lines.append(
                    f"  {p['timeframe']:<6}| {direction:<6}| {mag:<9}| Confidence: {conf}"
                )
        else:
            lines.append("  (no model predictions available yet)")

        lines.append("")
        lines.append("SIGNAL SUMMARY:")

        if signals:
            for name, info in signals.items():
                value = str(info.get("value", "N/A"))
                interp = info.get("interpretation", "")
                line = f"  {name:<18}: {value}"
                if interp:
                    line += f" -- {interp}"
                lines.append(line)
        else:
            lines.append("  (no signals collected yet)")

        lines.extend([
            "",
            "NOTE: Confidence decreases with longer timeframes.",
            sep,
            "",
        ])

        report = "\n".join(lines)

        with open(self.output_path, "a", encoding="utf-8") as f:
            f.write(report)

        logger.info("Prediction #%d logged to %s", self._run_counter, self.output_path)

    def get_latest(self, n: int = 1) -> str:
        """Read the last *n* prediction reports from the file."""
        if not self.output_path.exists():
            return "No predictions yet."

        content = self.output_path.read_text(encoding="utf-8")
        sep = "=" * 80
        blocks = content.split(sep)

        predictions: list[str] = []
        i = 0
        while i < len(blocks) - 2:
            if "Prediction Run #" in blocks[i + 1]:
                pred = sep + blocks[i + 1] + sep + blocks[i + 2] + sep
                predictions.append(pred)
                i += 3
            else:
                i += 1

        if not predictions:
            return "No predictions found."

        return "\n".join(predictions[-n:])

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _read_run_count(self) -> int:
        if not self.output_path.exists():
            return 0
        try:
            content = self.output_path.read_text(encoding="utf-8")
            return content.count("Prediction Run #")
        except OSError:
            return 0

    def _rotate_if_needed(self) -> None:
        if not self.output_path.exists():
            return
        try:
            size_mb = self.output_path.stat().st_size / (1024 * 1024)
        except OSError:
            return
        if size_mb > self.max_size_mb:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            archive = self.output_path.with_suffix(f".{ts}.log")
            self.output_path.rename(archive)
            logger.info("Rotated predictions log -> %s", archive)


TextLogger = PredictionLogger
