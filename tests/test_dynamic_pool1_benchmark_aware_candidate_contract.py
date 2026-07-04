import tempfile
import unittest
from pathlib import Path

import pandas as pd

from backtest_lab.dynamic_pool1_benchmark_aware_candidate_contract import run_benchmark_aware_candidate_contract


class DynamicPool1BenchmarkAwareCandidateContractTest(unittest.TestCase):
    def test_builds_primary_and_sensitivity_filters_without_formal_flags(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            panel = root / "repair.csv"
            pd.DataFrame(
                [
                    {
                        "candidate_month": "2024-01",
                        "candidate_as_of_date": "2024-01-31",
                        "ticker": "2330",
                        "candidate_rank": 1,
                        "candidate_score": 0.9,
                        "candidate_layer": "core",
                        "price_ready_flag": True,
                        "benchmark_0050_ready_flag": True,
                        "benchmark_00631l_ready_flag": True,
                        "ret_60d_vs_0050_trailing": 1.0,
                        "ret_60d_vs_00631L_trailing": 2.0,
                        "ret_20d_vs_0050_trailing": -1.0,
                        "ret_20d_vs_00631L_trailing": 1.0,
                        "benchmark_blocked_reason": "",
                        "uses_cross_section_median_as_primary_benchmark": False,
                    }
                ]
            ).to_csv(panel, index=False)
            manifest = run_benchmark_aware_candidate_contract(
                repo_root=root,
                repair_panel=panel,
                output_dir=root / "out",
            )
            self.assertEqual(manifest["primary_filter"], "rs60_positive_vs_both")
            self.assertFalse(manifest["formal_model_changed"])
            contract = pd.read_csv(root / "out" / "dynamic_pool1_benchmark_aware_candidate_contract.csv")
            self.assertTrue(bool(contract.loc[0, "rs60_positive_vs_both"]))
            self.assertFalse(bool(contract.loc[0, "rs20_and_rs60_positive_vs_both"]))
            self.assertTrue(bool(contract.loc[0, "top10_and_rs60_positive_vs_both"]))
            self.assertFalse(bool(contract.loc[0, "portfolio_replay_executed"]))


if __name__ == "__main__":
    unittest.main()
