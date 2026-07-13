import json
import unittest

import pandas as pd

from backtest_lab import vnext_p3_rank1_dynamic_kd_price_range_stage_a as subject


class Rank1DynamicRangeStageATest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ready = json.loads((subject.OUT / "readiness_for_rank1_dynamic_self_range_stage_A.json").read_text(encoding="utf-8"))
        cls.platforms = pd.read_csv(subject.OUT / "p3_rank1_dynamic_self_range_12x4_platform_supply.csv")
        cls.audit = pd.read_csv(subject.OUT / "p3_rank1_dynamic_self_range_future_PIT_audit.csv")

    def test_exact_frozen_platforms(self):
        self.assertEqual(len(self.platforms), 48)
        self.assertEqual(set(self.platforms.range_window_TD), {60, 120})
        self.assertEqual(set(self.platforms.low_zone), {0.1, 0.2, 0.3})
        self.assertEqual(set(self.platforms.latch_TD), {5, 10})
        self.assertEqual(set(self.platforms.minimum_K_range_threshold), {0, 20, 25, 30})

    def test_scope_governance(self):
        self.assertTrue(self.ready["diagnostic_subproblem"])
        self.assertFalse(self.ready["market_controller_used"])
        self.assertFalse(self.ready["all80_rerank"])
        self.assertFalse(self.ready["Top3"])
        self.assertFalse(self.ready["P3_2_outcome_read_authorized"])
        self.assertTrue(self.ready["K_range_gate_entry_only"])
        self.assertTrue(self.ready["price_range_pct_audit_only"])
        self.assertEqual(int(self.audit.violations.sum()), 0)


if __name__ == "__main__":
    unittest.main()
