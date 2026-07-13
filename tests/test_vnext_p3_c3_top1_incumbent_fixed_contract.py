import json
import unittest

import pandas as pd

from backtest_lab import vnext_p3_c3_top1_incumbent_fixed_contract as subject


class C3Top1FixedContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        subject.run()
        cls.ready = json.loads((subject.OUT / "readiness_for_C3_top1_incumbent_fixed_contract.json").read_text(encoding="utf-8"))
        cls.winners = pd.read_csv(subject.OUT / "p3_C3_daily_top1_second_candidate.csv")

    def test_governance(self):
        self.assertTrue(self.ready["operational_supply_gate_pass"])
        self.assertFalse(self.ready["calibration_supply_gate_pass"])
        self.assertFalse(self.ready["performance_authorized"])

    def test_one_top1_per_date(self):
        self.assertFalse(self.winners.decision_date.duplicated().any())
        self.assertEqual(len(self.winners), 482)

    def test_execution_keys_complete(self):
        self.assertEqual(self.ready["top1_execution_ready"], 123)
        self.assertEqual(self.ready["top1_execution_blocked"], 0)
        self.assertTrue(self.ready["daily_target_materialization_feasible"])


if __name__ == "__main__":
    unittest.main()
