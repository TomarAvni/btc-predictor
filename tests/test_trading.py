"""Unit tests for the demo trading agent.

Focus areas:
  * SHORT position P&L math, stop-loss / take-profit orientation, and the
    full open -> cover lifecycle through the agent.
  * The pre-trade safety gate (real ML required + non-zero magnitude).

Run with either::

    python -m unittest discover -s tests
    python tests/test_trading.py

No third-party deps required (stdlib unittest only).
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from datetime import datetime, timezone

# Make the repo root importable when run directly.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.trading.agent import TradingAgent  # noqa: E402
from src.trading.order import Position  # noqa: E402
from src.trading.portfolio import Portfolio  # noqa: E402
from src.trading.position_sizer import PositionSizer  # noqa: E402
from src.trading.risk_manager import RiskManager  # noqa: E402
from src.trading.simulator import OrderSimulator  # noqa: E402
from src.trading.strategy import TradingStrategy  # noqa: E402


def _short_prediction(confidence: float = 80.0, magnitude: float = 3.0) -> list[dict]:
    secondary_magnitude = magnitude + 2 if magnitude else 0.0
    return [
        {"timeframe": "24h", "direction": "DOWN", "magnitude": magnitude, "confidence": confidence},
        {"timeframe": "7d", "direction": "DOWN", "magnitude": secondary_magnitude, "confidence": confidence - 5},
    ]


def _long_prediction(confidence: float = 80.0, magnitude: float = 3.0) -> list[dict]:
    secondary_magnitude = magnitude + 2 if magnitude else 0.0
    return [
        {"timeframe": "24h", "direction": "UP", "magnitude": magnitude, "confidence": confidence},
        {"timeframe": "7d", "direction": "UP", "magnitude": secondary_magnitude, "confidence": confidence - 5},
    ]


class TestShortPnLMath(unittest.TestCase):
    """Pure P&L math on the Position model (no disk I/O)."""

    def test_short_profits_when_price_falls(self) -> None:
        pos = Position(entry_price=100.0, amount_btc=2.0, amount_usd=200.0, side="SHORT")
        # Price drops 100 -> 90: short should profit (100 - 90) * 2 = 20
        self.assertAlmostEqual(pos.unrealized_pnl_at(90.0), 20.0)
        self.assertGreater(pos.unrealized_pnl_at(90.0), 0)

    def test_short_loses_when_price_rises(self) -> None:
        pos = Position(entry_price=100.0, amount_btc=2.0, amount_usd=200.0, side="SHORT")
        # Price rises 100 -> 110: short should lose (100 - 110) * 2 = -20
        self.assertAlmostEqual(pos.unrealized_pnl_at(110.0), -20.0)
        self.assertLess(pos.unrealized_pnl_at(110.0), 0)

    def test_long_and_short_are_mirror_images(self) -> None:
        long_pos = Position(entry_price=100.0, amount_btc=1.0, amount_usd=100.0, side="LONG")
        short_pos = Position(entry_price=100.0, amount_btc=1.0, amount_usd=100.0, side="SHORT")
        self.assertAlmostEqual(
            long_pos.unrealized_pnl_at(120.0), -short_pos.unrealized_pnl_at(120.0)
        )

    def test_short_pnl_pct_sign(self) -> None:
        pos = Position(entry_price=100.0, amount_btc=1.0, amount_usd=100.0, side="SHORT")
        self.assertAlmostEqual(pos.unrealized_pnl_pct(95.0), 5.0)


class TestRiskManagerShorts(unittest.TestCase):
    def setUp(self) -> None:
        self.rm = RiskManager()

    def test_short_stop_loss_is_above_entry(self) -> None:
        sl = self.rm.calculate_stop_loss(entry_price=100.0, predicted_magnitude_pct=3.0, side="SHORT")
        self.assertGreater(sl, 100.0)

    def test_short_take_profit_is_below_entry(self) -> None:
        tp = self.rm.calculate_take_profit(entry_price=100.0, predicted_magnitude_pct=3.0, side="SHORT")
        self.assertLess(tp, 100.0)

    def test_long_orientation_is_opposite(self) -> None:
        sl = self.rm.calculate_stop_loss(entry_price=100.0, predicted_magnitude_pct=3.0, side="LONG")
        tp = self.rm.calculate_take_profit(entry_price=100.0, predicted_magnitude_pct=3.0, side="LONG")
        self.assertLess(sl, 100.0)
        self.assertGreater(tp, 100.0)

    def test_short_stop_loss_trigger(self) -> None:
        pos = Position(entry_price=100.0, amount_btc=1.0, side="SHORT", stop_loss=106.0, take_profit=94.0)
        self.assertTrue(self.rm.check_stop_loss(pos, 107.0))   # price above SL -> stop
        self.assertFalse(self.rm.check_stop_loss(pos, 105.0))
        self.assertTrue(self.rm.check_take_profit(pos, 93.0))  # price below TP -> profit
        self.assertFalse(self.rm.check_take_profit(pos, 95.0))


class TestSimulatorShorts(unittest.TestCase):
    def setUp(self) -> None:
        self.sim = OrderSimulator()

    def test_short_cover_profit_when_price_falls(self) -> None:
        _, position = self.sim.execute_short(
            amount_usd=200.0, current_price=100.0, prediction_id="p1",
            timeframe="24h", confidence=80, stop_loss=106.0, take_profit=94.0, reason="t",
        )
        self.assertEqual(position.side, "SHORT")
        # Cover at a lower price -> profit
        _, trade = self.sim.execute_close(position=position, current_price=90.0, reason="take_profit")
        self.assertEqual(trade.side, "SHORT")
        self.assertGreater(trade.pnl_usd, 0)

    def test_short_cover_loss_when_price_rises(self) -> None:
        _, position = self.sim.execute_short(
            amount_usd=200.0, current_price=100.0, prediction_id="p1",
            timeframe="24h", confidence=80, stop_loss=106.0, take_profit=94.0, reason="t",
        )
        _, trade = self.sim.execute_close(position=position, current_price=110.0, reason="stop_loss")
        self.assertLess(trade.pnl_usd, 0)

    def test_check_triggers_short(self) -> None:
        pos = Position(entry_price=100.0, amount_btc=1.0, side="SHORT", stop_loss=106.0, take_profit=94.0)
        self.assertEqual(self.sim.check_triggers(pos, high_price=107.0, low_price=101.0), "stop_loss")
        self.assertEqual(self.sim.check_triggers(pos, high_price=99.0, low_price=93.0), "take_profit")
        self.assertIsNone(self.sim.check_triggers(pos, high_price=104.0, low_price=96.0))


class TestPositionSizerDirectionAgnostic(unittest.TestCase):
    def test_sizing_identical_for_long_and_short(self) -> None:
        sizer = PositionSizer()
        # The sizer only sees confidence/alignment; direction is irrelevant.
        r1 = sizer.calculate(confidence=80.0, portfolio_value=2000.0, alignment_score=1.6)
        r2 = sizer.calculate(confidence=80.0, portfolio_value=2000.0, alignment_score=1.6)
        self.assertTrue(r1.should_trade)
        self.assertEqual(r1.amount_usd, r2.amount_usd)
        self.assertGreater(r1.amount_usd, 0)


class TestStrategyCostAwareEntry(unittest.TestCase):
    def test_blocks_high_confidence_tiny_move(self) -> None:
        strategy = TradingStrategy()
        signal = strategy.evaluate_entry(
            predictions=[
                {"timeframe": "6h", "direction": "UP", "magnitude": 0.05, "confidence": 90},
            ],
            current_price=100_000.0,
            open_positions=[],
        )
        self.assertFalse(signal.should_enter)
        self.assertIn("cost-aware edge", signal.reasons[0])

    def test_prefers_larger_net_edge_over_raw_confidence(self) -> None:
        strategy = TradingStrategy()
        signal = strategy.evaluate_entry(
            predictions=[
                {"timeframe": "6h", "direction": "UP", "magnitude": 0.5, "confidence": 80},
                {"timeframe": "24h", "direction": "UP", "magnitude": 3.0, "confidence": 70},
            ],
            current_price=100_000.0,
            open_positions=[],
        )
        self.assertTrue(signal.should_enter)
        self.assertEqual(signal.timeframe, "24h")

    def test_optional_evidence_gate_blocks_bad_horizon(self) -> None:
        strategy = TradingStrategy(horizon_stats={"24h": {"n": 25, "expectancy": -1.0}})
        signal = strategy.evaluate_entry(
            predictions=[
                {"timeframe": "24h", "direction": "UP", "magnitude": 3.0, "confidence": 80},
            ],
            current_price=100_000.0,
            open_positions=[],
        )
        self.assertFalse(signal.should_enter)
        self.assertIn("Evidence gate", signal.reasons[0])


class _AgentTestBase(unittest.TestCase):
    """Base that isolates disk writes to a temp working directory."""

    def setUp(self) -> None:
        self._prev_cwd = os.getcwd()
        self._tmp = tempfile.mkdtemp(prefix="btc_trade_test_")
        os.chdir(self._tmp)

    def tearDown(self) -> None:
        os.chdir(self._prev_cwd)

    def _make_agent(self) -> TradingAgent:
        agent = TradingAgent(portfolio=Portfolio(load_existing=False))
        agent.reset()
        return agent


class TestAgentShortLifecycle(_AgentTestBase):
    def test_short_opens_on_down_prediction(self) -> None:
        agent = self._make_agent()
        ts = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
        result = agent.on_new_prediction(_short_prediction(), current_price=100_000.0, timestamp=ts, used_ml=True)
        actions = result["actions"]
        self.assertTrue(any(a["action"] == "SHORT" for a in actions), result)
        pos = agent.portfolio.positions[0]
        self.assertEqual(pos.side, "SHORT")
        self.assertGreater(pos.stop_loss, pos.entry_price)   # SL above entry
        self.assertLess(pos.take_profit, pos.entry_price)    # TP below entry

    def test_short_covers_with_profit_when_price_drops(self) -> None:
        agent = self._make_agent()
        ts = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
        agent.on_new_prediction(_short_prediction(), current_price=100_000.0, timestamp=ts, used_ml=True)
        self.assertEqual(agent.portfolio.open_position_count, 1)

        # Price falls hard -> take-profit should trigger and realize a gain.
        exits = agent.on_price_update(price=90_000.0, high=90_500.0, low=89_500.0,
                                      timestamp=datetime(2026, 1, 2, 12, 0, tzinfo=timezone.utc))
        self.assertTrue(exits)
        self.assertEqual(agent.portfolio.open_position_count, 0)
        closed = agent.portfolio.closed_trades[-1]
        self.assertEqual(closed.side, "SHORT")
        self.assertGreater(closed.pnl_usd, 0)
        # Covering a profitable short should leave cash above the start.
        self.assertGreater(agent.portfolio.cash, agent.portfolio.STARTING_BALANCE)


class TestPreTradeSafetyGate(_AgentTestBase):
    def test_gate_blocks_when_no_ml(self) -> None:
        agent = self._make_agent()
        ts = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
        result = agent.on_new_prediction(_long_prediction(), current_price=100_000.0, timestamp=ts, used_ml=False)
        self.assertEqual(agent.portfolio.open_position_count, 0)
        self.assertEqual(result["actions"], [])

    def test_gate_blocks_on_zero_magnitude(self) -> None:
        agent = self._make_agent()
        ts = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
        preds = _long_prediction(confidence=80.0, magnitude=0.0)
        result = agent.on_new_prediction(preds, current_price=100_000.0, timestamp=ts, used_ml=True)
        self.assertEqual(agent.portfolio.open_position_count, 0)
        self.assertEqual(result["actions"], [])

    def test_gate_allows_ml_with_magnitude(self) -> None:
        agent = self._make_agent()
        ts = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
        result = agent.on_new_prediction(_long_prediction(), current_price=100_000.0, timestamp=ts, used_ml=True)
        self.assertEqual(agent.portfolio.open_position_count, 1)
        self.assertTrue(any(a["action"] == "BUY" for a in result["actions"]))

    def test_gate_helper_reasons(self) -> None:
        agent = self._make_agent()

        class _Sig:
            magnitude = 3.0

        class _ZeroSig:
            magnitude = 0.0

        self.assertIsNone(agent._pretrade_gate_reason(_Sig(), used_ml=True))
        self.assertIsNotNone(agent._pretrade_gate_reason(_Sig(), used_ml=False))
        self.assertIsNotNone(agent._pretrade_gate_reason(_ZeroSig(), used_ml=True))


class TestPortfolioShortValue(_AgentTestBase):
    def test_total_value_reflects_short_gain(self) -> None:
        pf = Portfolio(load_existing=False)
        pf.reset()
        pf.update_price(100.0)
        pos = Position(entry_price=100.0, amount_btc=1.0, amount_usd=100.0, side="SHORT",
                       stop_loss=106.0, take_profit=94.0)
        pf.open_position(pos)
        start_value = pf.total_value_usd
        # Price drops -> unrealized short gain should raise total value.
        pf.update_price(90.0)
        self.assertGreater(pf.total_value_usd, start_value)


if __name__ == "__main__":
    unittest.main(verbosity=2)
