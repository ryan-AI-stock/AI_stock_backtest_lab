import tempfile
import unittest
from pathlib import Path

import pandas as pd

from backtest_lab.dynamic_pool1_explicit_benchmark_context_contract import run_explicit_benchmark_context


class DynamicPool1ExplicitBenchmarkContextContractTest(unittest.TestCase):
    def test_adds_explicit_benchmark_flags_without_formal_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            liquidity = root / "liquidity" / "shards"
            liquidity.mkdir(parents=True)
            panel = root / "candidate_panel.csv"
            cache = root / "backtest_cache"
            cache.mkdir()
            rows = []
            for idx in range(70):
                date = pd.Timestamp("2024-01-02") + pd.offsets.BDay(idx)
                rows.append({"date": date.strftime("%Y-%m-%d"), "ticker": "2330", "close": 100 + idx})
            pd.DataFrame(rows).to_csv(liquidity / "accepted_liquidity_rows_2024_01.csv", index=False)
            pd.DataFrame(
                [
                    {
                        "year_month": "2024-04",
                        "ticker": "2330",
                        "name": "台積電",
                        "dynamic_pool1_score_v0": 0.8,
                        "candidate_rank_v0": 1,
                        "candidate_layer": "core",
                        "selected_for_pool_v0": True,
                    }
                ]
            ).to_csv(panel, index=False)
            bench_rows = []
            for idx in range(70):
                date = pd.Timestamp("2024-01-02") + pd.offsets.BDay(idx)
                bench_rows.append({"date": date.strftime("%Y-%m-%d"), "close": 50 + idx})
            pd.DataFrame(bench_rows).to_csv(cache / "0050_TW.csv", index=False)
            pd.DataFrame(bench_rows).to_csv(cache / "00631L_TW.csv", index=False)

            manifest = run_explicit_benchmark_context(
                repo_root=root,
                candidate_panel=panel,
                liquidity_dir=root / "liquidity",
                output_dir=root / "out",
            )

            self.assertEqual(manifest["status"], "completed_explicit_benchmark_context_contract")
            self.assertFalse(manifest["formal_model_changed"])
            out = pd.read_csv(root / "out" / "dynamic_pool1_candidate_panel_with_explicit_benchmark.csv")
            self.assertIn("benchmark_0050_ready_flag", out.columns)
            self.assertFalse(bool(out.loc[0, "uses_cross_section_median_as_primary_benchmark"]))
            self.assertFalse(bool(out.loc[0, "portfolio_replay_executed"]))


if __name__ == "__main__":
    unittest.main()
