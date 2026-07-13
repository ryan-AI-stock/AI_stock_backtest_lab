import json
import unittest

import pandas as pd

from backtest_lab import vnext_p3_all80_entry_funnel_setup_latch_audit as subject


class EntryFunnelLatchAuditTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        subject.run()
        cls.ready = json.loads((subject.OUT / "readiness_for_entry_funnel_setup_latch_audit.json").read_text(encoding="utf-8"))
        cls.latch = pd.read_csv(subject.OUT / "p3_all80_setup_latch_supply_counterfactual.csv")

    def test_scope(self):
        self.assertEqual(self.ready["P3_1_dates"], 482)
        self.assertEqual(self.ready["candidate_rows"], 38560)
        self.assertFalse(self.ready["future_outcome_read"])
        self.assertFalse(self.ready["performance_authorized"])

    def test_frozen_latch_platforms(self):
        self.assertSetEqual(set(self.latch.latch_window_TD), {5, 10, 20})
        self.assertEqual(len(self.latch), 54)


if __name__ == "__main__":
    unittest.main()
