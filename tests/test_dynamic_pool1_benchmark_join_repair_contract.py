import tempfile
import unittest
from pathlib import Path

import pandas as pd

from backtest_lab.dynamic_pool1_benchmark_join_repair_contract import run_benchmark_join_repair_contract


class DynamicPool1BenchmarkJoinRepairContractTest(unittest.TestCase):
    def test_repair_source_improves_benchmark_readiness_without_formal_flags(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            liquidity = root / "liquidity" / "shards"
            liquidity.mkdir(parents=True)
            context = root / "context"
            context.mkdir()
            audit = root / "audit"
            audit.mkdir()
            panel = root / "candidate_panel.csv"

            candidate_rows = []
            benchmark_rows = []
            for idx in range(70):
                date = pd.Timestamp("2024-01-02") + pd.offsets.BDay(idx)
                date_text = date.strftime("%Y-%m-%d")
                candidate_rows.append({"date": date_text, "ticker": "2330", "close": 100 + idx})
                benchmark_rows.append({"date": date_text, "close": 50 + idx, "adj_close": 50 + idx})
            pd.DataFrame(candidate_rows).to_csv(liquidity / "accepted_liquidity_rows_2024_01.csv", index=False)
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
            pd.DataFrame(
                [
                    {
                        "candidate_month": "2024-04",
                        "rows": 1,
                        "benchmark_0050_ready_rows": 0,
                        "benchmark_00631l_ready_rows": 0,
                        "explicit_0050_ready_rate": 0,
                        "explicit_00631l_ready_rate": 0,
                    }
                ]
            ).to_csv(context / "benchmark_readiness_summary.csv", index=False)
            for ticker in ["0050", "00631L"]:
                primary = root / "backtest_cache" / f"{ticker}_TW.csv"
                repair = root / "backtest_cache" / "stock_pool_observations" / f"{ticker}_TW.csv"
                primary.parent.mkdir(parents=True, exist_ok=True)
                repair.parent.mkdir(parents=True, exist_ok=True)
                pd.DataFrame(benchmark_rows).to_csv(primary, index=False)
                pd.DataFrame(benchmark_rows).to_csv(repair, index=False)

            manifest = run_benchmark_join_repair_contract(
                repo_root=root,
                candidate_panel=panel,
                liquidity_dir=root / "liquidity",
                context_dir=context,
                audit_dir=audit,
                output_dir=root / "out",
            )

            self.assertEqual(manifest["status"], "completed_benchmark_join_repair_contract")
            self.assertFalse(manifest["formal_model_changed"])
            repaired = pd.read_csv(root / "out" / "benchmark_join_repair_panel.csv")
            self.assertTrue(bool(repaired.loc[0, "benchmark_0050_ready_flag"]))
            self.assertTrue(bool(repaired.loc[0, "benchmark_00631l_ready_flag"]))
            parity = pd.read_csv(root / "out" / "benchmark_overlap_parity_audit.csv")
            self.assertEqual(set(parity["parity_status"]), {"pass"})


if __name__ == "__main__":
    unittest.main()
