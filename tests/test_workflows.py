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

    def test_watchdog_checks_log_freshness_after_one_hour(self) -> None:
        steps = self.workflow["jobs"]["predict-freshness"]["steps"]
        freshness_step = next(step for step in steps if step.get("id") == "freshness")
        run = freshness_step["run"]

        self.assertIn("python3 src/utils/prediction_freshness.py --max-age-hours 1", run)
        self.assertIn('echo "stale=false"', run)
        self.assertIn('echo "stale=true"', run)

    def test_watchdog_dispatches_predict_when_stale_and_idle(self) -> None:
        steps = self.workflow["jobs"]["predict-freshness"]["steps"]
        dispatch_step = next(
            step for step in steps if step.get("name") == "Dispatch Predict recovery run"
        )

        self.assertEqual(
            dispatch_step["if"],
            "steps.freshness.outputs.stale == 'true' && steps.active.outputs.count == '0'",
        )
        self.assertIn("gh workflow run predict.yml", dispatch_step["run"])


class TestPredictWatchdogWorkflow(unittest.TestCase):
    def setUp(self) -> None:
        self.workflow = load_workflow("predict-watchdog.yml")

    def test_watchdog_uses_direct_freshness_script(self) -> None:
        steps = self.workflow["jobs"]["watchdog"]["steps"]
        freshness_step = next(step for step in steps if step.get("id") == "freshness")
        run = freshness_step["run"]

        self.assertIn("python3 src/utils/prediction_freshness.py --max-age-hours 1", run)
        self.assertFalse(any(step.get("uses") == "actions/setup-python@v5" for step in steps))


if __name__ == "__main__":
    unittest.main(verbosity=2)
