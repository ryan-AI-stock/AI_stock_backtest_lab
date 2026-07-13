import json
import unittest

from backtest_lab import vnext_p3_all80_adjusted_hlc_feasibility_absorption as subject


class FeasibilityAbsorptionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        subject.run()
        cls.plan = json.loads((subject.OUT / "bounded_delta_acquisition_planning.json").read_text(encoding="utf-8"))

    def test_exact_reconciliation(self):
        self.assertEqual(136491 + 1200 + 11678, 149369)
        self.assertEqual(self.plan["bounded_delta_rows"], 11678)

    def test_governance(self):
        self.assertTrue(self.plan["ready_for_bounded_delta_acquisition"])
        self.assertFalse(self.plan["ready_for_state_supply_rerun"])
        self.assertFalse(self.plan["performance_authorized"])
        self.assertEqual(self.plan["future_data_violation_count"], 0)


if __name__ == "__main__":
    unittest.main()
