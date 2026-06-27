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


def step_by_name(workflow: dict, job: str, name: str) -> dict:
    for step in workflow["jobs"][job]["steps"]:
        if step.get("name") == name:
            return step
    raise AssertionError(f"Step {name!r} not found in job {job!r}")


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

    def test_watchdog_checks_committed_prediction_log(self) -> None:
        step = step_by_name(self.workflow, "predict-freshness", "Check prediction log freshness")
        run = step["run"]

        self.assertIn("python3 src/utils/prediction_freshness.py --max-age-minutes 60", run)
        self.assertIn('echo "stale=false"', run)

    def test_watchdog_dispatches_predict_when_log_is_stale(self) -> None:
        dispatch = step_by_name(self.workflow, "predict-freshness", "Dispatch Predict if stale")
        run = dispatch["run"]

        self.assertEqual(
            dispatch["if"],
            "steps.freshness.outputs.stale == 'true' && steps.active.outputs.count == '0'",
        )
        self.assertIn("gh workflow run predict.yml", run)


class TestPredictWatchdogWorkflow(unittest.TestCase):
    def setUp(self) -> None:
        self.workflow = load_workflow("predict-watchdog.yml")

    def test_watchdog_runs_every_fifteen_minutes(self) -> None:
        schedules = self.workflow["on"]["schedule"]
        self.assertEqual(schedules[0]["cron"], "4,19,34,49 * * * *")

    def test_watchdog_does_not_use_setup_python(self) -> None:
        steps = self.workflow["jobs"]["watchdog"]["steps"]
        uses = [step.get("uses", "") for step in steps]
        self.assertFalse(any("setup-python" in value for value in uses))

    def test_watchdog_checks_committed_prediction_log(self) -> None:
        step = step_by_name(self.workflow, "watchdog", "Check prediction log freshness")
        run = step["run"]

        self.assertIn("python3 src/utils/prediction_freshness.py --max-age-minutes 60", run)


if __name__ == "__main__":
    unittest.main(verbosity=2)
