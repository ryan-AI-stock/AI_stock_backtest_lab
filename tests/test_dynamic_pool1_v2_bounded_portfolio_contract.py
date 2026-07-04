import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd

from backtest_lab.dynamic_pool1_v2_bounded_portfolio_contract import (
    run_dynamic_pool1_v2_bounded_portfolio_contract,
)


class DynamicPool1V2BoundedPortfolioContractTest(unittest.TestCase):
    def test_builds_bounded_contract_without_formal_override(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            panel = root / "v2.csv"
            rows = []
            for source in [
                "v2_primary_rs60_top15_monthly",
                "v2_primary_rs60_top10_monthly",
                "v2_broad_watchlist_rs60_all",
            ]:
                for rank, ticker in enumerate(["2330", "2454", "2308"], start=1):
                    rows.append(
                        {
                            "candidate_month": "2024-01",
                            "candidate_as_of_date": "2024-01-31",
                            "ticker": ticker,
                            "candidate_rank": rank,
                            "candidate_score": 1.0 - rank / 10,
                            "candidate_layer": "core",
                            "price_ready_flag": True,
                            "benchmark_0050_ready_flag": True,
                            "benchmark_00631l_ready_flag": True,
                            "ret_60d_vs_0050_trailing": 1.0,
                            "ret_60d_vs_00631L_trailing": 1.0,
                            "ret_20d_vs_0050_trailing": 1.0,
                            "ret_20d_vs_00631L_trailing": 1.0,
                            "rs60_positive_vs_both": True,
                            "rs20_and_rs60_positive_vs_both": True,
                            "top10_and_rs60_positive_vs_both": True,
                            "benchmark_blocked_reason": "",
                            "uses_cross_section_median_as_primary_benchmark": False,
                            "forward_return_used_as_contract_rule": False,
                            "variant_id": source,
                            "variant_role": "primary",
                        }
                    )
            pd.DataFrame(rows).to_csv(panel, index=False)
            v0_pool = root / "candidate_pool_by_month.csv"
            pd.DataFrame(
                [
                    {"year_month": "2024-01", "ticker": "2330", "market": "TWSE"},
                    {"year_month": "2024-01", "ticker": "2454", "market": "TWSE"},
                    {"year_month": "2024-01", "ticker": "2308", "market": "TWSE"},
                ]
            ).to_csv(v0_pool, index=False)

            formal = root / "formal.csv"
            pd.DataFrame(
                [
                    {
                        "signal_date": "2024-01-15",
                        "execution_date": "2024-01-16",
                        "formal_target": "CASH",
                        "target_type": "risk_control_cash",
                        "risk_off_state": "no_target_cash_all",
                    },
                    {
                        "signal_date": "2024-02-01",
                        "execution_date": "2024-02-02",
                        "formal_target": "2454.TW",
                        "target_type": "stock",
                        "risk_off_state": "formal_target_active",
                    },
                    {
                        "signal_date": "2024-02-05",
                        "execution_date": "2024-02-06",
                        "formal_target": "CASH",
                        "target_type": "risk_control_cash",
                        "risk_off_state": "no_target_cash_all",
                    },
                ]
            ).to_csv(formal, index=False)
            (root / "backtest_cache" / "stock_pool_observations").mkdir(parents=True)
            pd.DataFrame(
                [
                    {"date": "2024-01-16", "adj_close": 100},
                    {"date": "2024-02-02", "adj_close": 101},
                    {"date": "2024-02-06", "adj_close": 102},
                ]
            ).to_csv(
                root / "backtest_cache" / "stock_pool_observations" / "0050_TW.csv",
                index=False,
            )
            pd.DataFrame(
                [
                    {"date": "2024-01-16", "adj_close": 10},
                    {"date": "2024-02-02", "adj_close": 11},
                    {"date": "2024-02-06", "adj_close": 12},
                ]
            ).to_csv(
                root / "backtest_cache" / "stock_pool_observations" / "00631L_TW.csv",
                index=False,
            )

            with mock.patch("backtest_lab.dynamic_pool1_v2_bounded_portfolio_contract.FORMAL_STREAMS", [Path("formal.csv")]):
                manifest = run_dynamic_pool1_v2_bounded_portfolio_contract(
                    repo_root=root,
                    v2_member_panel=panel,
                    candidate_v0_pool=v0_pool,
                    output_dir=root / "out",
                    liquidity_calendar_dir=root / "missing_liquidity_calendar",
                )

            self.assertFalse(manifest["formal_model_changed"])
            self.assertFalse(manifest["trade_decision_changed"])
            self.assertFalse(manifest["portfolio_replay_executed"])
            self.assertEqual(manifest["future_data_violation_count"], 0)

            signal = pd.read_csv(root / "out" / "daily_signal_panel.csv")
            jan_rows = signal[signal["date"].eq("2024-01-15")]
            self.assertTrue((jan_rows["dynamic_blocked_reason"] == "blocked_no_asof_dynamic_candidate_pool").any())
            feb_rows = signal[signal["date"].eq("2024-02-01")]
            self.assertIn("blocked_formal_state_direct_stock_target_no_override", set(feb_rows["dynamic_blocked_reason"]))

            variants = pd.read_csv(root / "out" / "portfolio_variant_matrix.csv")
            self.assertNotIn("all_formal_states", ";".join(variants["dynamic_pool_variant"]))
            self.assertFalse(variants["exit_contract"].astype(str).str.contains("hold_60").any())
            weights = pd.read_csv(root / "out" / "portfolio_weight_ledger.csv")
            self.assertIn("canonical_ticker", weights.columns)
            self.assertIn("2330.TW", set(weights["canonical_ticker"]))
            trades = pd.read_csv(root / "out" / "trade_ledger.csv")
            self.assertIn("price_source_cache_key", trades.columns)
            self.assertIn("2330.TW", set(trades["price_source_cache_key"]))
            readiness = json.loads((root / "out" / "readiness_for_experiments.json").read_text(encoding="utf-8"))
            self.assertTrue(readiness["ready_for_experiments_validation"])


if __name__ == "__main__":
    unittest.main()
