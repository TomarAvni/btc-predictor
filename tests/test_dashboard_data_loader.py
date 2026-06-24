"""Tests for dashboard prediction-history merging."""

from __future__ import annotations

from dashboard.data_loader import (
    _jsonl_record_to_run,
    _merge_prediction_runs,
    _parse_signal_summary_value,
)


def test_parse_signal_summary_splits_dash_value():
    parsed = _parse_signal_summary_value("66.4 -- Neutral")
    assert parsed == {"value": "66.4", "interpretation": "Neutral"}


def test_jsonl_record_to_run_maps_predictions_and_signals():
    record = {
        "run_number": 45,
        "timestamp": "2026-06-22T12:20:55Z",
        "predictions": [
            {"timeframe": "24h", "direction": "UP", "magnitude": 0.67, "confidence": 56},
        ],
        "signals_summary": {"RSI (14)": "66.4 -- Neutral"},
    }
    run = _jsonl_record_to_run(record)
    assert run["run_number"] == 45
    assert run["predictions"][0]["direction"] == "UP"
    assert run["signals"]["RSI (14)"]["value"] == "66.4"


def test_merge_prediction_runs_prefers_jsonl_on_conflict():
    log_runs = [
        {
            "run_number": 19,
            "timestamp": "2026-06-15 18:46 UTC",
            "predictions": [{"timeframe": "24h", "direction": "DOWN", "magnitude": 1.0, "confidence": 10}],
            "signals": {},
        }
    ]
    jsonl_records = [
        {
            "run_number": 19,
            "timestamp": "2026-06-16T10:00:00Z",
            "predictions": [{"timeframe": "24h", "direction": "UP", "magnitude": 2.0, "confidence": 70}],
            "signals_summary": {},
        },
        {
            "run_number": 20,
            "timestamp": "2026-06-16T11:00:00Z",
            "predictions": [{"timeframe": "24h", "direction": "UP", "magnitude": 3.0, "confidence": 80}],
            "signals_summary": {},
        },
    ]
    merged = _merge_prediction_runs(log_runs, jsonl_records)
    assert [r["run_number"] for r in merged] == [19, 20]
    assert merged[0]["predictions"][0]["direction"] == "UP"
    assert merged[0]["timestamp"] == "2026-06-16T10:00:00Z"
