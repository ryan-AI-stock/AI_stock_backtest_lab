import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd

from backtest_lab.dynamic_pool1_portfolio_challenger_contract import (
    run_dynamic_pool1_portfolio_contract,
)


class DynamicPool1PortfolioChallengerContractTest(unittest.TestCase):
    def test_builds_contract_without_formal_flags(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cand = root / "cand"
            cand.mkdir()
            pd.DataFrame(
                [
                    {
                        "year_month": "2024-01",
                        "ticker": "2330",
                        "candidate_rank_v0": 1,
                        "dynamic_pool1_score_v0": 0.9,
                        "feature_readiness_state": "ready",
                    },
                    {
                        "year_month": "2024-01",
                        "ticker": "2454",
                        "candidate_rank_v0": 2,
                        "dynamic_pool1_score_v0": 0.7,
                        "feature_readiness_state": "ready",
                    },
                ]
            ).to_csv(cand / "candidate_pool_by_month.csv", index=False)
            pd.DataFrame(
                [
                    {
                        "year_month": "2024-01",
                        "ticker": "2330",
                    }
                ]
            ).to_csv(cand / "candidate_panel_monthly.csv", index=False)

            formal = root / "formal.csv"
            pd.DataFrame(
                [
                    {
                        "signal_date": "2024-01-02",
                        "execution_date": "2024-01-03",
                        "formal_target": "CASH",
                    }
                ]
            ).to_csv(formal, index=False)
            (root / "backtest_cache").mkdir()
            pd.DataFrame([{"date": "2024-01-02", "close": 100}]).to_csv(root / "backtest_cache" / "0050_TW.csv", index=False)
            pd.DataFrame([{"date": "2024-01-02", "close": 10}]).to_csv(root / "backtest_cache" / "00631L_TW.csv", index=False)

            with mock.patch("backtest_lab.dynamic_pool1_portfolio_challenger_contract.FORMAL_STREAMS", [Path("formal.csv")]):
                manifest = run_dynamic_pool1_portfolio_contract(
                    repo_root=root,
                    candidate_dir=cand,
                    output_dir=root / "out",
                )

            self.assertEqual(manifest["status"], "completed_portfolio_challenger_contract_only")
            self.assertFalse(manifest["formal_model_changed"])
            daily = pd.read_csv(root / "out" / "daily_portfolio_contract_panel.csv")
            self.assertEqual(daily.loc[0, "formal_conflict_state"], "no_conflict_formal_cash")
            variants = pd.read_csv(root / "out" / "portfolio_variant_matrix.csv")
            self.assertIn("dynamic_top1_when_formal_cash_or_market_exposure", set(variants["variant_id"]))
            readiness = json.loads((root / "out" / "readiness_for_experiments.json").read_text(encoding="utf-8"))
            self.assertTrue(readiness["ready_for_experiments"])
            self.assertFalse(readiness["active_in_trade_decision"])


if __name__ == "__main__":
    unittest.main()
