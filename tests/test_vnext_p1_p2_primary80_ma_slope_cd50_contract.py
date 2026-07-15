import unittest

from backtest_lab.vnext_p1_p2_primary80_ma_slope_cd50_contract import COOLDOWNS, SIGNALS, parameter_matrix


class Primary80MASlopeCD50ContractTest(unittest.TestCase):
    def test_exact_fifty_variants(self):
        frame = parameter_matrix()
        self.assertEqual(len(frame), 50)
        self.assertEqual(frame.variant_id.nunique(), 50)
        self.assertEqual(set(frame.signal_family), set(SIGNALS))
        self.assertEqual(set(frame.post_buy_exit_lock_trading_days), set(COOLDOWNS))

    def test_fixed_governance(self):
        frame = parameter_matrix()
        self.assertFalse(frame.market_controller_used.any())
        self.assertTrue(frame.buy_rule.str.contains("close>MA", regex=False).all())
        self.assertTrue(frame.sell_rule.str.contains("close<MA", regex=False).all())


if __name__ == "__main__":
    unittest.main()
