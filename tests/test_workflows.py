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

    def test_watchdog_dispatches_predict_after_one_hour(self) -> None:
        step = self.workflow["jobs"]["predict-freshness"]["steps"][0]
        script = step["with"]["script"]

        self.assertIn('const workflowId = "predict.yml";', script)
        self.assertIn("const staleAfterMs = 1 * 60 * 60 * 1000;", script)
        self.assertIn("listWorkflowRuns", script)
        self.assertIn("createWorkflowDispatch", script)

    def test_watchdog_chains_off_predict_completion(self) -> None:
        workflow_run = self.workflow["on"]["workflow_run"]
        self.assertIn("Predict", workflow_run["workflows"])
        self.assertIn("Predict Watchdog", workflow_run["workflows"])
        self.assertEqual(workflow_run["types"], ["completed"])


class TestPredictWatchdogWorkflow(unittest.TestCase):
    def setUp(self) -> None:
        self.workflow = load_workflow("predict-watchdog.yml")

    def test_watchdog_runs_every_fifteen_minutes(self) -> None:
        schedules = self.workflow["on"]["schedule"]
        self.assertEqual(schedules[0]["cron"], "12,27,42,57 * * * *")

    def test_watchdog_checks_fifty_minute_freshness_window(self) -> None:
        steps = self.workflow["jobs"]["watchdog"]["steps"]
        freshness_step = next(
            step for step in steps if step.get("name") == "Check prediction log freshness"
        )
        run_cmd = freshness_step["run"]
        self.assertIn("src/utils/prediction_freshness.py", run_cmd)
        self.assertIn("--max-age-hours 0.84", run_cmd)

    def test_watchdog_chains_off_predict_completion(self) -> None:
        workflow_run = self.workflow["on"]["workflow_run"]
        self.assertEqual(workflow_run["workflows"], ["Predict"])
        self.assertEqual(workflow_run["types"], ["completed"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
