import unittest

from backtest_lab.vnext_p3_weekly_rank1_entry_hold_exit_state_machine import OUT, run


class Rank1EntryHoldExitTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        run()

    def test_normal_switch_is_disabled(self):
        import pandas as pd
        actions = pd.read_csv(OUT / "p3_rank1_entry_hold_exit_daily_action_trace.csv")
        self.assertFalse(actions.normal_switch_enabled.astype(bool).any())

    def test_nav_does_not_use_cross_asset_nominal_return(self):
        import pandas as pd
        recon = pd.read_csv(OUT / "p3_rank1_entry_hold_exit_NAV_reconciliation.csv")
        self.assertFalse(recon.cross_asset_nominal_price_return_used.astype(bool).any())

    def test_contract_is_superseded_for_target_architecture(self):
        import json
        readiness = json.loads((OUT / "readiness_for_p3_rank1_entry_hold_exit_state_machine.json").read_text(encoding="utf-8"))
        self.assertTrue(readiness["superseded_for_target_architecture"])
        self.assertFalse(readiness["ready_for_experiments"])


if __name__ == "__main__":
    unittest.main()
