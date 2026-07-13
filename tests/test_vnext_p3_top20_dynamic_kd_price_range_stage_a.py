import json
import unittest
from backtest_lab import vnext_p3_top20_dynamic_kd_price_range_stage_a as subject

class Top20DynamicRangeStageATest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ready=json.loads((subject.OUT/"readiness_for_top20_dynamic_self_range_stage_A.json").read_text(encoding="utf-8"))
    def test_stopped_checkpoint(self):
        self.assertTrue(self.ready["superseded_by_all80_K_range_eligibility_comparison"])
        self.assertTrue(self.ready["non_representative_of_current_scope"])
        self.assertTrue(self.ready["follow_up_stopped"])
        self.assertFalse(self.ready["ready_for_experiments"])
        self.assertFalse(self.ready["materialization_executed"])

if __name__=="__main__": unittest.main()
