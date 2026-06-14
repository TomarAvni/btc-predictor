"""Prediction output formatter.

Renders prediction results and signal summaries into human-readable
text blocks used by both the console and the log file.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


class PredictionFormatter:
    """Formats prediction dicts into the standard text report."""

    SEPARATOR = "=" * 80

    def format_report(
        self,
        predictions: list[dict[str, Any]],
        signals: dict[str, dict[str, Any]],
        *,
        run_number: int | None = None,
    ) -> str:
        """Build the full text report block.

        Args:
            predictions: list of dicts with timeframe/direction/magnitude/confidence.
            signals: dict of signal_name -> {value, interpretation}.
            run_number: optional run counter to include in the header.
        """
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        header = f"[{now}]"
        if run_number is not None:
            header += f" -- Prediction Run #{run_number}"

        lines = [
            "",
            self.SEPARATOR,
            header,
            self.SEPARATOR,
            "",
        ]

        lines.append(self._format_predictions(predictions))
        lines.append(self._format_signals(signals))

        lines.extend([
            "NOTE: Confidence decreases with longer timeframes.",
            self.SEPARATOR,
            "",
        ])

        return "\n".join(lines)

    @staticmethod
    def _format_predictions(predictions: list[dict[str, Any]]) -> str:
        if not predictions:
            return "PREDICTIONS:\n  (no model predictions available yet)\n"

        rows = ["PREDICTIONS:"]
        for p in predictions:
            direction = p["direction"].upper()
            sign = "+" if direction == "UP" else "-"
            mag = f"{sign}{p['magnitude']:.1f}%"
            conf = f"{p['confidence']:.0f}%"
            rows.append(
                f"  {p['timeframe']:<6}| {direction:<6}| {mag:<9}| Confidence: {conf}"
            )
        rows.append("")
        return "\n".join(rows)

    @staticmethod
    def _format_signals(signals: dict[str, dict[str, Any]]) -> str:
        if not signals:
            return "SIGNAL SUMMARY:\n  (no signals collected yet)\n"

        rows = ["SIGNAL SUMMARY:"]
        for name, info in signals.items():
            value = str(info.get("value", "N/A"))
            interp = info.get("interpretation", "")
            line = f"  {name:<18}: {value}"
            if interp:
                line += f" -- {interp}"
            rows.append(line)
        rows.append("")
        return "\n".join(rows)

    def format_status(self, status: dict[str, Any]) -> str:
        """Format the data-status dict as a brief summary."""
        if status.get("status") == "no_data":
            return "No price data available. Run with --download first."

        return (
            f"Data status: {status['candles']:,} hourly candles\n"
            f"  Range: {status['first']}  ->  {status['last']}\n"
            f"  Span:  {status['timespan_days']:,} days"
        )


Formatter = PredictionFormatter
