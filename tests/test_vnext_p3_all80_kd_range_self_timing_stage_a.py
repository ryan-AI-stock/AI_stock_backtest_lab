import json,unittest
import pandas as pd
from backtest_lab import vnext_p3_all80_kd_range_self_timing_stage_a as subject

class All80KDRangeStageATest(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.ready=json.loads((subject.OUT/"readiness_for_all80_KD_range_stage_A.json").read_text(encoding="utf-8")); cls.summary=pd.read_csv(subject.OUT/"p3_all80_KD_range_48_platform_supply.csv"); cls.audit=pd.read_csv(subject.OUT/"p3_all80_KD_range_future_PIT_audit.csv")
 def test_scope(self):
  self.assertEqual(len(self.summary),48); self.assertTrue(self.ready["Layer5_all80_K_range_comparison"]); self.assertFalse(self.ready["full_market_controller"]); self.assertFalse(self.ready["normal_switch"]); self.assertFalse(self.ready["Top3"]); self.assertFalse(self.ready["P3_2_outcome_read_authorized"])
 def test_exact80_and_pit(self):
  self.assertEqual(self.ready["primary80_membership_rows"],self.ready["P3_1_decision_dates"]*80); self.assertEqual(int(self.audit.violations.sum()),0)

if __name__=="__main__": unittest.main()
