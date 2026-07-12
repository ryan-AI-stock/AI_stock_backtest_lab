import unittest

from backtest_lab.vnext_p3_weekly_rank1_single_candidate_contract import etf_net, rank1_authority, stock_net


class WeeklyRank1ContractTest(unittest.TestCase):
    def test_canonical_rank1_is_unique(self):
        rank1 = rank1_authority()
        self.assertEqual(len(rank1), 154)
        self.assertEqual(rank1.snapshot_date.nunique(), 154)
        self.assertTrue(rank1.pool_rank.eq(1).all())

    def test_stock_cost_exceeds_etf_cost(self):
        self.assertLess(stock_net(0.0, 10), etf_net(0.0, 10))


if __name__ == "__main__":
    unittest.main()
