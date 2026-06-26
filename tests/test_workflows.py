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
        self.assertEqual(schedules[0]["cron"], "11,26,41,56 * * * *")

    def test_watchdog_checks_prediction_log_freshness(self) -> None:
        steps = self.workflow["jobs"]["predict-freshness"]["steps"]
        freshness_step = next(
            step for step in steps if step.get("name") == "Check prediction log freshness"
        )
        run = freshness_step["run"]
        self.assertIn("--max-age-minutes 45", run)

    def test_watchdog_dispatches_predict_when_stale(self) -> None:
        steps = self.workflow["jobs"]["predict-freshness"]["steps"]
        dispatch_step = next(
            step for step in steps if step.get("name") == "Dispatch Predict if stale"
        )
        run = dispatch_step["run"]
        self.assertIn("gh workflow run predict.yml", run)


class TestPredictWatchdogWorkflow(unittest.TestCase):
    def setUp(self) -> None:
        self.workflow = load_workflow("predict-watchdog.yml")

    def test_watchdog_runs_every_fifteen_minutes(self) -> None:
        schedules = self.workflow["on"]["schedule"]
        self.assertEqual(schedules[0]["cron"], "4,19,34,49 * * * *")

    def test_watchdog_uses_forty_five_minute_staleness_threshold(self) -> None:
        steps = self.workflow["jobs"]["watchdog"]["steps"]
        freshness_step = next(
            step for step in steps if step.get("name") == "Check prediction log freshness"
        )
        self.assertIn("--max-age-minutes 45", freshness_step["run"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
