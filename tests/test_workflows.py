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


def workflow_step_scripts(workflow: dict) -> list[str]:
    scripts: list[str] = []
    for job in workflow.get("jobs", {}).values():
        for step in job.get("steps", []):
            if "run" in step:
                scripts.append(step["run"])
            script = step.get("with", {}).get("script")
            if script:
                scripts.append(script)
    return scripts


class TestPipelineWatchdogWorkflow(unittest.TestCase):
    def setUp(self) -> None:
        self.workflow = load_workflow("pipeline-watchdog.yml")

    def test_watchdog_has_actions_write_permission(self) -> None:
        permissions = self.workflow["permissions"]
        self.assertEqual(permissions["actions"], "write")
        self.assertEqual(permissions["contents"], "read")

    def test_watchdog_runs_every_fifteen_minutes(self) -> None:
        schedules = self.workflow["on"]["schedule"]
        self.assertEqual(schedules[0]["cron"], "11,26,41,56 * * * *")

    def test_watchdog_checks_prediction_log_freshness(self) -> None:
        scripts = workflow_step_scripts(self.workflow)
        freshness_scripts = [s for s in scripts if "prediction_freshness.py" in s]
        self.assertTrue(freshness_scripts, "expected a log freshness check step")
        self.assertIn("--max-age-minutes 60", freshness_scripts[0])

    def test_watchdog_dispatches_predict_when_stale(self) -> None:
        scripts = workflow_step_scripts(self.workflow)
        dispatch_scripts = [s for s in scripts if "gh workflow run predict.yml" in s]
        self.assertTrue(dispatch_scripts, "expected a Predict dispatch step")


class TestPredictWatchdogWorkflow(unittest.TestCase):
    def setUp(self) -> None:
        self.workflow = load_workflow("predict-watchdog.yml")

    def test_watchdog_runs_every_fifteen_minutes(self) -> None:
        schedules = self.workflow["on"]["schedule"]
        self.assertEqual(schedules[0]["cron"], "4,19,34,49 * * * *")

    def test_watchdog_checks_prediction_log_freshness(self) -> None:
        scripts = workflow_step_scripts(self.workflow)
        freshness_scripts = [s for s in scripts if "prediction_freshness.py" in s]
        self.assertTrue(freshness_scripts, "expected a log freshness check step")
        self.assertIn("--max-age-minutes 60", freshness_scripts[0])

    def test_watchdog_uses_direct_python3_without_setup_python(self) -> None:
        steps = self.workflow["jobs"]["watchdog"]["steps"]
        uses = [step.get("uses", "") for step in steps]
        self.assertNotIn("actions/setup-python@v5", uses)
        freshness_step = next(step for step in steps if step.get("name") == "Check prediction log freshness")
        self.assertIn("python3 src/utils/prediction_freshness.py", freshness_step["run"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
