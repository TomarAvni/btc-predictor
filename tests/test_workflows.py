"""Regression tests for GitHub Actions workflow contracts."""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


ROOT = Path(__file__).resolve().parents[1]


def load_workflow(name: str) -> dict:
    with (ROOT / ".github" / "workflows" / name).open(encoding="utf-8") as f:
        return yaml.load(f, Loader=yaml.BaseLoader)


class TestPipelineWatchdogWorkflow(unittest.TestCase):
    def setUp(self) -> None:
        self.workflow = load_workflow("pipeline-watchdog.yml")

    def test_watchdog_has_actions_write_permission(self) -> None:
        permissions = self.workflow["permissions"]
        self.assertEqual(permissions["actions"], "write")
        self.assertEqual(permissions["contents"], "read")

    def test_watchdog_runs_off_peak_every_fifteen_minutes(self) -> None:
        schedules = self.workflow["on"]["schedule"]
        self.assertEqual(schedules[0]["cron"], "13,28,43,58 * * * *")

    def test_watchdog_checks_prediction_log_freshness_for_one_hour(self) -> None:
        steps = self.workflow["jobs"]["predict-freshness"]["steps"]
        freshness_step = next(
            step for step in steps if step.get("name") == "Check prediction log freshness"
        )
        run_block = freshness_step["run"]

        self.assertIn("prediction_freshness", run_block)
        self.assertIn("--max-age-hours 1", run_block)

    def test_watchdog_dispatches_predict_when_stale(self) -> None:
        steps = self.workflow["jobs"]["predict-freshness"]["steps"]
        dispatch_step = next(
            step for step in steps if step.get("name") == "Dispatch Predict recovery run"
        )

        self.assertEqual(
            dispatch_step["if"],
            "steps.freshness.outputs.stale == 'true' && steps.active.outputs.count == '0'",
        )
        self.assertIn("gh workflow run predict.yml", dispatch_step["run"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
