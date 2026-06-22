"""Smoke tests for the PromptFoo tweet-reader eval harness."""

from __future__ import annotations

import importlib.util
import json
import sys
import unittest
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = ROOT / "evals" / "promptfoo"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class TestPromptfooEvalHarness(unittest.TestCase):
    def test_config_points_to_local_provider_and_assertions(self) -> None:
        config = yaml.safe_load((EVAL_DIR / "promptfooconfig.yaml").read_text(encoding="utf-8"))
        self.assertEqual(config["providers"][0]["id"], "file://provider.py")
        self.assertFalse(config["providers"][0]["config"]["allow_live"])
        assertion = config["defaultTest"]["assert"][0]
        self.assertEqual(assertion["type"], "python")
        self.assertEqual(assertion["value"], "file://assertions.py:validate_grounded_reader")

    def test_cases_include_dev_and_frozen_splits(self) -> None:
        cases = yaml.safe_load((EVAL_DIR / "cases.yaml").read_text(encoding="utf-8"))
        splits = {case["vars"]["split"] for case in cases}
        self.assertEqual(splits, {"dev", "frozen"})

    def test_provider_outputs_pass_assertions_in_mock_mode(self) -> None:
        provider = _load_module("promptfoo_provider", EVAL_DIR / "provider.py")
        assertions = _load_module("promptfoo_assertions", EVAL_DIR / "assertions.py")
        cases = yaml.safe_load((EVAL_DIR / "cases.yaml").read_text(encoding="utf-8"))

        for case in cases:
            with self.subTest(case=case["description"]):
                context = {"vars": case["vars"]}
                result = provider.call_api(
                    case["vars"]["tweets_json"],
                    {"config": {"allow_live": False}},
                    context,
                )
                json.loads(result["output"])
                grade = assertions.validate_grounded_reader(result["output"], context)
                self.assertTrue(grade["pass"], grade.get("reason"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
