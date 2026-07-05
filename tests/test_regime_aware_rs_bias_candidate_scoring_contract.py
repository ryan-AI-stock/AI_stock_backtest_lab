import tempfile
import unittest
from pathlib import Path

import pandas as pd

from backtest_lab.regime_aware_rs_bias_candidate_scoring_contract import (
    run_regime_aware_rs_bias_candidate_scoring_contract,
)


class RegimeAwareRsBiasCandidateScoringContractTest(unittest.TestCase):
    def test_builds_branch_rows_without_forward_rule_or_formal_flags(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            candidate_dir = root / "outputs" / "dynamic_pool1_benchmark_aware_candidate_contract_20260704"
            context_dir = root / "outputs" / "dynamic_pool1_candidate_panel_v0_20260704"
            candidate_dir.mkdir(parents=True)
            context_dir.mkdir(parents=True)
            dates = pd.bdate_range("2025-01-02", periods=300)
            as_of = dates[-1].strftime("%Y-%m-%d")
            tickers = ["6669", "2308", "2317", "2454"]
            pd.DataFrame(
                [
                    {
                        "candidate_month": "2026-02",
                        "candidate_as_of_date": as_of,
                        "ticker": ticker,
                        "candidate_rank": idx + 1,
                        "candidate_score": 0.9 - idx * 0.1,
                        "candidate_layer": "core",
                    }
                    for idx, ticker in enumerate(tickers)
                ]
            ).to_csv(candidate_dir / "dynamic_pool1_benchmark_aware_candidate_contract.csv", index=False)
            pd.DataFrame(
                [
                    {
                        "year_month": "2026-02",
                        "ticker": ticker,
                        "name": f"name_{ticker}",
                        "market": "TWSE",
                        "candidate_layer": "core",
                        "selected_for_pool_v0": True,
                        "ai_supply_chain_layers": "AI",
                        "mainline_theme_labels": "theme",
                    }
                    for ticker in tickers
                ]
            ).to_csv(context_dir / "candidate_pool_by_month.csv", index=False)

            shards = root / "liquidity" / "shards"
            shards.mkdir(parents=True)
            rows = []
            for ticker_idx, ticker in enumerate(tickers):
                for idx, date in enumerate(dates):
                    close = 100 + ticker_idx * 10 + idx * (1.0 + ticker_idx * 0.1)
                    rows.append(
                        {
                            "date": date.strftime("%Y-%m-%d"),
                            "ticker": ticker,
                            "name": f"name_{ticker}",
                            "market": "TWSE",
                            "close": close,
                            "turnover_value": 100_000_000 + idx,
                            "liquidity_pass": True,
                        }
                    )
            pd.DataFrame(rows).to_csv(shards / "accepted_liquidity_rows_2026_02.csv", index=False)

            bench_dir = root / "backtest_cache" / "stock_pool_observations"
            bench_dir.mkdir(parents=True)
            for filename, step in {"0050_TW.csv": 0.4, "00631L_TW.csv": 0.8}.items():
                pd.DataFrame(
                    {
                        "date": [date.strftime("%Y-%m-%d") for date in dates],
                        "close": [100 + idx * step for idx in range(len(dates))],
                        "adj_close": [100 + idx * step for idx in range(len(dates))],
                    }
                ).to_csv(bench_dir / filename, index=False)

            manifest = run_regime_aware_rs_bias_candidate_scoring_contract(
                repo_root=root,
                candidate_contract=candidate_dir / "dynamic_pool1_benchmark_aware_candidate_contract.csv",
                candidate_context=context_dir / "candidate_pool_by_month.csv",
                liquidity_dir=root / "liquidity",
                output_dir=root / "out",
            )

            self.assertEqual(manifest["future_data_violation_count"], 0)
            self.assertFalse(manifest["formal_model_changed"])
            self.assertFalse(manifest["portfolio_replay_executed"])
            contract = pd.read_csv(root / "out" / "regime_aware_rs_bias_candidate_contract.csv")
            self.assertGreaterEqual(len(contract), 16)
            self.assertEqual(
                set(contract["branch_variant"]),
                {
                    "long_strong_rs40_bias_guard",
                    "short_cycle_rs20_bias_repair",
                    "pullback_prior_strength_bias_repair",
                    "fallback_market_bias_context",
                },
            )
            self.assertFalse(bool(contract["uses_forward_return_as_rule"].any()))
            self.assertIn("rs40_vs_0050_pct", contract.columns)
            self.assertIn("stock_bias60_zscore", contract.columns)
            case = pd.read_csv(root / "out" / "case_trace_6669_2308_2317_2454.csv")
            self.assertEqual(set(case["base_ticker"].astype(str)), set(tickers))
            future = pd.read_csv(root / "out" / "future_data_audit.csv")
            self.assertFalse(bool(future["future_data_violation"].any()))


if __name__ == "__main__":
    unittest.main()
