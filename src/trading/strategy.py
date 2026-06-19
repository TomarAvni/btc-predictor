"""Trading strategy -- entry/exit conditions and multi-timeframe alignment.

Determines WHEN and HOW to trade based on prediction signals,
timeframe alignment, and current market conditions.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional

from src.horizons import HORIZON_HOURS
from src.trading.simulator import OrderSimulator


# Horizon label -> hours, sourced from the single source of truth
# (src/horizons.py) so the trading layer always matches the engine.
TIMEFRAME_HOURS = dict(HORIZON_HOURS)


@dataclass
class StrategySignal:
    """Output of strategy evaluation."""

    should_enter: bool
    direction: str  # "LONG" or "SHORT" or "NONE"
    timeframe: str  # Selected timeframe for this trade
    confidence: float  # Confidence of the selected timeframe
    magnitude: float  # Expected magnitude (%)
    alignment_score: float  # Multi-timeframe alignment
    alignment_label: str  # "Strong", "Moderate", "Weak"
    reasons: list[str]  # Human-readable reasoning


@dataclass
class ExitSignal:
    """Signal to exit a position."""

    should_exit: bool
    reason: str
    exit_type: str  # "stop_loss", "take_profit", "time_expired", "signal_flip", "circuit_breaker"


class TradingStrategy:
    """Evaluates predictions and generates trade signals."""

    MIN_CONFIDENCE: float = 55.0
    ROUND_TRIP_COST_PCT: float = OrderSimulator.ROUND_TRIP_COST_PCT
    MIN_EDGE_BUFFER_PCT: float = 0.10
    MIN_ACTIONABLE_MOVE_PCT: float = ROUND_TRIP_COST_PCT + MIN_EDGE_BUFFER_PCT
    MIN_EVIDENCE_SAMPLES: int = 20
    ALIGNMENT_STRONG_THRESHOLD: float = 1.5
    ALIGNMENT_WEAK_THRESHOLD: float = 0.5

    def __init__(
        self,
        horizon_stats: Optional[dict[str, dict]] = None,
        side_stats: Optional[dict[str, dict]] = None,
    ) -> None:
        # Optional diagnostic gates. Callers can pass realized performance
        # summaries once enough samples exist; with no stats, the strategy
        # behaves normally except for the cost-aware edge threshold.
        self.horizon_stats = horizon_stats or {}
        self.side_stats = side_stats or {}

    def evaluate_entry(
        self,
        predictions: list[dict],
        current_price: float,
        open_positions: list,
        last_trade_time: Optional[datetime] = None,
        timestamp: Optional[datetime] = None,
    ) -> StrategySignal:
        """Evaluate predictions for a potential new entry.

        Args:
            predictions: List of prediction dicts with keys:
                timeframe, direction, magnitude, confidence
            current_price: Current BTC price.
            open_positions: List of currently open positions.
            last_trade_time: When the last trade was made.
            timestamp: Current timestamp (for backtesting).

        Returns:
            StrategySignal indicating whether to enter and with what parameters.
        """
        now = timestamp or datetime.now(timezone.utc)

        if not predictions:
            return self._no_trade("No predictions available")

        confidence_viable = [
            p for p in predictions if p.get("confidence", 0) >= self.MIN_CONFIDENCE
        ]
        if not confidence_viable:
            return self._no_trade(
                f"No predictions above minimum confidence ({self.MIN_CONFIDENCE}%)"
            )

        # Trade only when the expected move clears round-trip costs plus a small
        # safety buffer. This avoids taking high-confidence but economically tiny
        # predictions that cannot reliably pay for slippage/fees.
        viable = [
            p for p in confidence_viable
            if abs(float(p.get("magnitude", 0.0) or 0.0)) >= self.MIN_ACTIONABLE_MOVE_PCT
        ]
        if not viable:
            return self._no_trade(
                "No predictions clear cost-aware edge threshold "
                f"({self.MIN_ACTIONABLE_MOVE_PCT:.2f}% minimum move)"
            )

        primary = max(viable, key=lambda p: self._entry_score(predictions, p))
        primary_direction = primary["direction"]
        primary_timeframe = primary["timeframe"]
        primary_confidence = primary["confidence"]
        primary_magnitude = primary["magnitude"]

        # Calculate multi-timeframe alignment
        alignment_score = self._calculate_alignment(predictions, primary_direction)
        alignment_label = self._alignment_label(alignment_score)

        # Check for conflicting strong signals
        if alignment_score < self.ALIGNMENT_WEAK_THRESHOLD:
            strong_conflicts = [
                p for p in predictions
                if p["direction"] != primary_direction and p["confidence"] >= 65
            ]
            if strong_conflicts:
                return self._no_trade(
                    f"Strong conflicting signals: {[p['timeframe'] for p in strong_conflicts]} "
                    f"say {strong_conflicts[0]['direction']} (alignment: {alignment_score:.2f})"
                )

        # Check if we already have a position in the same direction
        same_direction_positions = [
            p for p in open_positions
            if (p.side == "LONG" and primary_direction == "UP")
            or (p.side == "SHORT" and primary_direction == "DOWN")
        ]
        if len(same_direction_positions) >= 2:
            return self._no_trade(
                f"Already have {len(same_direction_positions)} positions in same direction"
            )

        direction = "LONG" if primary_direction == "UP" else "SHORT"

        evidence_reason = self._evidence_gate_reason(primary_timeframe, direction)
        if evidence_reason:
            return self._no_trade(evidence_reason)

        reasons = [
            (
                f"{primary_timeframe} prediction: {primary_direction} "
                f"{primary_magnitude:+.2f}% @ {primary_confidence}% confidence"
            ),
            (
                f"Cost-aware edge: {abs(primary_magnitude) - self.ROUND_TRIP_COST_PCT:.2f}% "
                f"after estimated {self.ROUND_TRIP_COST_PCT:.2f}% round-trip cost"
            ),
            f"Multi-timeframe alignment: {alignment_label} (score {alignment_score:.2f})",
        ]

        # Note other supporting timeframes
        supporters = [
            p for p in predictions
            if p["direction"] == primary_direction
            and p["timeframe"] != primary_timeframe
            and p["confidence"] >= 50
        ]
        if supporters:
            support_str = ", ".join(
                f"{p['timeframe']} ({p['confidence']}%)" for p in supporters
            )
            reasons.append(f"Supporting timeframes: {support_str}")

        return StrategySignal(
            should_enter=True,
            direction=direction,
            timeframe=primary_timeframe,
            confidence=primary_confidence,
            magnitude=primary_magnitude,
            alignment_score=alignment_score,
            alignment_label=alignment_label,
            reasons=reasons,
        )

    def evaluate_exit(
        self,
        position,
        current_price: float,
        latest_predictions: Optional[list[dict]] = None,
        timestamp: Optional[datetime] = None,
    ) -> ExitSignal:
        """Check if a position should be exited.

        Args:
            position: The Position to evaluate.
            current_price: Current BTC price.
            latest_predictions: Most recent predictions (to detect signal flips).
            timestamp: Current timestamp.

        Returns:
            ExitSignal indicating if/why to exit.
        """
        now = timestamp or datetime.now(timezone.utc)

        # Check time-based expiration
        target_hours = TIMEFRAME_HOURS.get(position.timeframe, 24)
        holding_time = now - position.entry_time
        if holding_time > timedelta(hours=target_hours * 1.5):
            return ExitSignal(
                should_exit=True,
                reason=f"Holding period expired ({holding_time.total_seconds()/3600:.1f}h > {target_hours*1.5:.0f}h target)",
                exit_type="time_expired",
            )

        # Check if prediction has flipped against the position
        if latest_predictions:
            relevant = [
                p for p in latest_predictions
                if p["timeframe"] == position.timeframe
            ]
            if relevant:
                pred = relevant[0]
                position_direction = "UP" if position.side == "LONG" else "DOWN"
                if pred["direction"] != position_direction and pred["confidence"] >= 65:
                    return ExitSignal(
                        should_exit=True,
                        reason=(
                            f"Signal flipped: {position.timeframe} now says "
                            f"{pred['direction']} @ {pred['confidence']}% confidence"
                        ),
                        exit_type="signal_flip",
                    )

        return ExitSignal(should_exit=False, reason="", exit_type="")

    def _calculate_alignment(
        self, predictions: list[dict], primary_direction: str
    ) -> float:
        """Calculate multi-timeframe alignment score.

        For each timeframe:
          - If direction matches primary: +confidence/100
          - If direction conflicts: -confidence/100
        """
        score = 0.0
        for p in predictions:
            conf = p.get("confidence", 0) / 100.0
            if p["direction"] == primary_direction:
                score += conf
            else:
                score -= conf
        return score

    def _evidence_gate_reason(self, timeframe: str, side: str) -> Optional[str]:
        """Block statistically populated horizons/sides with negative expectancy."""
        for label, stats in (("horizon", self.horizon_stats.get(timeframe)), ("side", self.side_stats.get(side))):
            if not stats:
                continue
            n = int(stats.get("n", stats.get("total_trades", 0)) or 0)
            expectancy = float(stats.get("expectancy", stats.get("expectancy_usd", 0.0)) or 0.0)
            if n >= self.MIN_EVIDENCE_SAMPLES and expectancy <= 0:
                return (
                    f"Evidence gate: {label} {timeframe if label == 'horizon' else side} "
                    f"has non-positive expectancy (${expectancy:.2f}) over {n} samples"
                )
        return None

    def _entry_score(self, predictions: list[dict], candidate: dict) -> float:
        """Rank candidates by confidence, net move, and timeframe alignment."""
        confidence_edge = max(0.0, (float(candidate.get("confidence", 0.0)) - 50.0) / 50.0)
        net_move = max(
            0.0,
            abs(float(candidate.get("magnitude", 0.0) or 0.0)) - self.ROUND_TRIP_COST_PCT,
        )
        alignment = max(
            0.1,
            self._calculate_alignment(predictions, str(candidate.get("direction", "")).upper()),
        )
        return confidence_edge * net_move * alignment

    def _alignment_label(self, score: float) -> str:
        if score > self.ALIGNMENT_STRONG_THRESHOLD:
            return "Strong"
        elif score >= self.ALIGNMENT_WEAK_THRESHOLD:
            return "Moderate"
        return "Weak"

    def _no_trade(self, reason: str) -> StrategySignal:
        return StrategySignal(
            should_enter=False,
            direction="NONE",
            timeframe="",
            confidence=0.0,
            magnitude=0.0,
            alignment_score=0.0,
            alignment_label="",
            reasons=[reason],
        )
