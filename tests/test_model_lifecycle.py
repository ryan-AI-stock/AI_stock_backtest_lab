import copy
import importlib.util
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('lifecycle', ROOT / 'scripts/validate_model_lifecycle.py')
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

class ModelLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.policy = json.loads((ROOT / 'configs/model_lifecycle.json').read_text(encoding='utf-8'))

    def test_current_policy(self):
        self.assertEqual(MODULE.validate(self.policy), [])

    def test_each_report_is_required(self):
        for report in MODULE.REPORTS:
            value = copy.deepcopy(self.policy)
            value['protected_reports'].remove(report)
            self.assertTrue(MODULE.validate(value))

    def test_risk_cannot_be_promoted(self):
        self.policy['models']['c6_risk']['status'] = 'frozen_formal'
        self.assertTrue(MODULE.validate(self.policy))

    def test_evidence_gate_cannot_be_removed(self):
        self.policy['legacy_removal_requires'].remove('historical_evidence')
        self.assertTrue(MODULE.validate(self.policy))

    def test_separate_tasks_not_required(self):
        self.policy['required_independent_tasks'] = True
        self.assertTrue(MODULE.validate(self.policy))

    def test_schedule_repository_protected(self):
        self.policy['protected_repositories'].remove('AI_stock_schedule_rules')
        self.assertTrue(MODULE.validate(self.policy))
