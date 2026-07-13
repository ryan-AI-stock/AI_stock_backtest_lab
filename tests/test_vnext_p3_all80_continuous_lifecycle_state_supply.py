import json
import unittest

import pandas as pd

from backtest_lab import vnext_p3_all80_continuous_lifecycle_state_supply as subject


class All80LifecycleSupplyTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        subject.run()
        cls.readiness = json.loads((subject.OUT / "readiness_for_p3_all80_continuous_lifecycle_state_supply.json").read_text(encoding="utf-8"))
        cls.gate = pd.read_csv(subject.OUT / "p3_all80_supply_gate.csv")

    def test_scope_and_governance(self):
        self.assertEqual(self.readiness["P3_1_dates"], 482)
        self.assertEqual(self.readiness["candidate_rows"], 38560)
        self.assertFalse(self.readiness["performance_authorized"])
        self.assertEqual(self.readiness["future_data_violation_count"], 0)

    def test_six_frozen_platforms(self):
        self.assertEqual(len(self.gate), 6)
        self.assertSetEqual(set(self.gate.platform), {"L1", "L2", "L3"})
        self.assertSetEqual(set(self.gate.persistence), {"2of3", "3of5"})


if __name__ == "__main__":
    unittest.main()
