import json
import unittest

import pandas as pd

from backtest_lab.vnext_p3_rank1_sequential_lifecycle_contract import OUT, run


class Rank1SequentialLifecycleContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        run()

    def test_exact_rank1_daily_scope(self):
        frame = pd.read_csv(OUT / "p3_rank1_sequential_continuous_feature_matrix.csv.gz", dtype={"ticker": str})
        self.assertEqual(len(frame), 715)
        self.assertEqual(frame.decision_date.nunique(), 715)
        self.assertTrue(frame.pool_rank.eq(1).all())

    def test_state_order_and_prohibitions(self):
        states = pd.read_csv(OUT / "p3_rank1_sequential_state_definition.csv")
        transitions = pd.read_csv(OUT / "p3_rank1_sequential_transition_contract.csv")
        self.assertEqual(states.state.tolist(), [f"S{i}" for i in range(8)])
        self.assertTrue(((transitions.from_state == "S0") & (transitions.to_state == "S3") & ~transitions.allowed).any())
        self.assertTrue(((transitions.from_state == "S5") & (transitions.to_state == "S7") & ~transitions.allowed).any())

    def test_blocked_without_continuous_kd_self_history(self):
        ready = json.loads((OUT / "readiness_for_p3_rank1_sequential_lifecycle.json").read_text(encoding="utf-8"))
        self.assertFalse(ready["KD_3_6_12M_self_percentiles_ready"])
        self.assertFalse(ready["sufficient_for_walk_forward"])
        self.assertFalse(ready["ready_for_experiments"])
        self.assertFalse(ready["P3_2_outcome_read"])


if __name__ == "__main__":
    unittest.main()
