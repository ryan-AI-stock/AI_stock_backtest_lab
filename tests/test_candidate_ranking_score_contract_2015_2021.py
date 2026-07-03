import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

import test_paths  # noqa: F401

from backtest_lab.candidate_ranking_score_contract_2015_2021 import (
    run_candidate_ranking_score_contract_2015_2021,
)


class CandidateRankingScoreContract2015Test(unittest.TestCase):
    def _price_frame(self, start: str, periods: int, daily_step: float) -> pd.DataFrame:
        dates = pd.bdate_range(start=start, periods=periods)
        close = [100 + (index * daily_step) for index in range(periods)]
        return pd.DataFrame(
            {
                "date": [date.strftime("%Y-%m-%d") for date in dates],
                "open": close,
                "high": close,
                "low": close,
                "close": close,
                "adj_close": close,
                "volume": [1000] * periods,
                "dividend": [0.0] * periods,
                "stock_split": [0.0] * periods,
            }
        )

    def test_builds_diagnostic_rankings_without_formal_activation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            lifecycle = root / "lifecycle"
            cache = root / "cache"
            output = root / "out"
            lifecycle.mkdir()
            cache.mkdir()

            signal_date = "2015-04-02"
            lifecycle_contract = pd.DataFrame(
                [
                    {"ticker": "2330.TW", "name": "台積電"},
                    {"ticker": "2454.TW", "name": "聯發科"},
                    {"ticker": "2308.TW", "name": "台達電"},
                ]
            )
            lifecycle_contract.to_csv(lifecycle / "pool1_ticker_lifecycle_contract.csv", index=False)
            lifecycle_daily = pd.DataFrame(
                [
                    {
                        "date": signal_date,
                        "ticker": "2330.TW",
                        "name": "台積電",
                        "has_valid_price_on_date": True,
                        "candidate_available_for_pool1_ranking": True,
                        "excluded_reason": "",
                        "synthetic_used": False,
                    },
                    {
                        "date": signal_date,
                        "ticker": "2454.TW",
                        "name": "聯發科",
                        "has_valid_price_on_date": True,
                        "candidate_available_for_pool1_ranking": True,
                        "excluded_reason": "",
                        "synthetic_used": False,
                    },
                    {
                        "date": signal_date,
                        "ticker": "2308.TW",
                        "name": "台達電",
                        "has_valid_price_on_date": True,
                        "candidate_available_for_pool1_ranking": False,
                        "excluded_reason": "",
                        "synthetic_used": False,
                    },
                ]
            )
            lifecycle_daily.to_csv(lifecycle / "pool1_date_aware_candidate_availability_daily.csv", index=False)
            self._price_frame("2015-01-01", 70, 2.0).to_csv(cache / "2330_TW.csv", index=False)
            self._price_frame("2015-01-01", 70, 1.0).to_csv(cache / "2454_TW.csv", index=False)
            self._price_frame("2015-03-20", 10, 1.0).to_csv(cache / "2308_TW.csv", index=False)

            result = run_candidate_ranking_score_contract_2015_2021(
                lifecycle_dir=lifecycle,
                price_cache_dir=cache,
                price_source_registry=root / "missing_registry.csv",
                output_dir=output,
                start_date=signal_date,
                end_date=signal_date,
            )
            self.assertEqual(result, output)

            manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], "completed_diagnostic_only")
            self.assertFalse(manifest["formal_model_changed"])
            self.assertFalse(manifest["trade_decision_changed"])
            self.assertFalse(manifest["active_in_trade_decision"])
            self.assertFalse(manifest["uses_forward_return"])

            panel = pd.read_csv(output / "candidate_ranking_panel_2015_2021.csv")
            tsmc = panel[panel["candidate_ticker"].eq("2330.TW")].iloc[0]
            self.assertEqual(int(tsmc["rank_10_30"]), 1)
            self.assertEqual(tsmc["top1_10_30"], "2330.TW")
            self.assertIn("2454.TW", tsmc["top3_10_30"])
            self.assertNotEqual(str(tsmc["rank_score_10_30"]), "")
            self.assertNotEqual(str(tsmc["score_margin_top1_top2_10_30"]), "")

            blocked = pd.read_csv(output / "blocked_rows.csv")
            delta = blocked[blocked["candidate_ticker"].eq("2308.TW")].iloc[0]
            self.assertIn("insufficient_60d_history", str(delta["blocked_reason"]))
            self.assertIn("20_60_score_not_ready", str(delta["blocked_reason"]))

            readiness = pd.read_csv(output / "data_readiness_by_date.csv").iloc[0]
            self.assertTrue(bool(readiness["ready_10_30"]))
            self.assertTrue(bool(readiness["ready_10_40"]))
            self.assertTrue(bool(readiness["ready_20_60"]))

            turnover = pd.read_csv(output / "top_name_turnover_summary.csv")
            self.assertEqual(set(turnover["variant"]), {"10_30", "10_40", "20_60"})


if __name__ == "__main__":
    unittest.main()
