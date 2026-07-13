import json
import unittest

import pandas as pd

from backtest_lab import vnext_p3_lifecycle_v1_stage_a as subject


class LifecycleV1StageATest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ready = json.loads((subject.OUT / "readiness_for_V1_stage_A.json").read_text(encoding="utf-8"))
        cls.top = pd.read_csv(subject.OUT / "p3_V1_daily_top1_second_materialization.csv")

    def test_scope_and_no_future(self):
        self.assertEqual(self.ready["P3_1_dates"], 482)
        self.assertFalse(self.ready["P3_2_outcome_read_authorized"])
        self.assertFalse(self.ready["performance_authorized"])
        self.assertEqual(self.ready["future_data_violation_count"], 0)

    def test_one_top1_row_per_date(self):
        self.assertEqual(len(self.top), 482)
        self.assertFalse(self.top.decision_date.duplicated().any())

    def test_stopped_checkpoint_governance(self):
        self.assertTrue(self.ready["non_representative_of_current_rank1_stock_only_timing_stage"])
        self.assertFalse(self.ready["may_be_used_to_reject_stock_only_low_buy_high_sell_hypothesis"])
        self.assertTrue(self.ready["follow_up_stopped"])
        self.assertFalse(self.ready["ready_for_experiments"])
        self.assertEqual(self.ready["allowed_role"], "supply_reference_only")


if __name__ == "__main__":
    unittest.main()
