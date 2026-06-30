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


def step_script(step: dict) -> str:
    return step.get("run") or step.get("with", {}).get("script", "")


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
        freshness_step = next(step for step in steps if step.get("id") == "freshness")
        script = step_script(freshness_step)

        self.assertIn("python3 src/utils/prediction_freshness.py --max-age-hours 1", script)
        self.assertIn('echo "stale=false"', script)
        self.assertIn('echo "stale=true"', script)

    def test_watchdog_dispatches_predict_when_stale(self) -> None:
        steps = self.workflow["jobs"]["predict-freshness"]["steps"]
        dispatch_step = next(
            step for step in steps if step.get("name") == "Dispatch Predict recovery run"
        )
        script = step_script(dispatch_step)

        self.assertIn("gh workflow run predict.yml", script)
        self.assertEqual(
            dispatch_step["if"],
            "steps.freshness.outputs.stale == 'true' && steps.active.outputs.count == '0'",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
