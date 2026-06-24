"""Tests for aggressive paper profile metadata."""

from __future__ import annotations

from src.trading.paper_profile import PAPER_ONLY, PROFILE_LABEL, get_paper_profile_summary


def test_paper_profile_is_simulated_only():
    assert PAPER_ONLY is True
    assert "simulated" in PROFILE_LABEL.lower()


def test_paper_profile_summary_keys():
    summary = get_paper_profile_summary()
    assert summary["paper_only"] is True
    assert summary["min_confidence_pct"] == 50.0
    assert summary["max_position_pct"] == 95.0
    assert summary["max_exposure_pct"] == 200.0
