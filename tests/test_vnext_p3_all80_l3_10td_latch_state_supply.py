import json
import unittest

import pandas as pd

from backtest_lab import vnext_p3_all80_l3_10td_latch_state_supply as subject


class L310TDStateSupplyTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        subject.run()
        cls.ready = json.loads((subject.OUT / "readiness_for_L3_10TD_latch_state_supply.json").read_text(encoding="utf-8"))
        cls.transitions = pd.read_csv(subject.OUT / "p3_all80_L3_10TD_transition_events.csv")

    def test_scope_and_policy(self):
        self.assertEqual(self.ready["P3_1_dates"], 482)
        self.assertEqual(self.ready["candidate_rows"], 38560)
        self.assertTrue(self.ready["sequence_memory_unique"])
        self.assertFalse(self.ready["performance_authorized"])

    def test_no_state_jump(self):
        forbidden = {("S1", "S3"), ("S5", "S7")}
        self.assertFalse(any((row.from_state, row.to_state) in forbidden for row in self.transitions.itertuples()))


if __name__ == "__main__":
    unittest.main()
