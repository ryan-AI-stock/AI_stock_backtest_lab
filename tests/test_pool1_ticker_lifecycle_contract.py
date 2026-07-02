import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

import test_paths  # noqa: F401

from backtest_lab.pool1_ticker_lifecycle_contract import run_pool1_ticker_lifecycle_contract


class Pool1TickerLifecycleContractTest(unittest.TestCase):
    def test_builds_date_aware_availability_and_excludes_before_warmup(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            panel = root / "panel"
            output = root / "out"
            panel.mkdir()
            price = root / "price.csv"
            supplemental = root / "supplemental.csv"

            (panel / "manifest.json").write_text(
                json.dumps({"date_start": "2014-11-03", "date_end": "2018-02-06"}, ensure_ascii=False),
                encoding="utf-8",
            )
            pd.DataFrame(
                [
                    {"date": "2014-11-03"},
                    {"date": "2015-01-28"},
                    {"date": "2018-02-05"},
                    {"date": "2018-02-06"},
                ]
            ).to_csv(panel / "formal_policy_input_readiness.csv", index=False)
            pd.DataFrame(
                [
                    {"date": "2015-01-28", "candidate_ticker": "00631L.TW", "candidate_name": "0050正二", "rank": 1},
                    {"date": "2015-01-28", "candidate_ticker": "2330.TW", "candidate_name": "台積電", "rank": 1},
                    {"date": "2015-01-28", "candidate_ticker": "2454.TW", "candidate_name": "聯發科", "rank": 2},
                    {"date": "2015-01-28", "candidate_ticker": "2308.TW", "candidate_name": "台達電", "rank": 3},
                    {"date": "2015-01-28", "candidate_ticker": "2317.TW", "candidate_name": "鴻海", "rank": 4},
                    {"date": "2015-01-28", "candidate_ticker": "2382.TW", "candidate_name": "廣達", "rank": 5},
                    {"date": "2015-01-28", "candidate_ticker": "3231.TW", "candidate_name": "緯創", "rank": 6},
                    {"date": "2018-02-06", "candidate_ticker": "6669.TW", "candidate_name": "緯穎", "rank": 1},
                ]
            ).to_csv(panel / "pool1_daily_candidate_ranking_panel.csv", index=False)
            rows = []
            for ticker in ("2330.TW", "2454.TW", "2308.TW", "2317.TW", "2382.TW", "3231.TW"):
                rows.append(
                    {
                        "ticker": ticker,
                        "name": ticker,
                        "first_date": "2014-11-03",
                        "last_date": "2021-12-31",
                        "coverage_status": "price_only_ready",
                        "adjusted_close_available": True,
                        "ready_for_backtest_price_only": True,
                        "cache_path": f"cache/{ticker}.csv",
                        "source_type": "cache",
                        "synthetic_used": False,
                    }
                )
            rows.append(
                {
                    "ticker": "6669.TW",
                    "name": "緯穎",
                    "first_date": "2017-11-13",
                    "last_date": "2021-12-31",
                    "coverage_status": "price_only_ready",
                    "adjusted_close_available": True,
                    "ready_for_backtest_price_only": True,
                    "cache_path": "cache/6669.csv",
                    "source_type": "cache",
                    "synthetic_used": False,
                }
            )
            pd.DataFrame(rows).to_csv(price, index=False)
            pd.DataFrame(
                [
                    {
                        "ticker": "00631L.TW",
                        "combined_first_date": "2014-11-03",
                        "combined_last_date": "2021-12-31",
                        "price_source_ready": True,
                        "supplemental_synthetic_used": False,
                        "base_source": "cache/00631L.csv",
                        "supplemental_source_path": "data/00631L.csv",
                        "supplemental_source_type": "twse_stock_day_backfill",
                    }
                ]
            ).to_csv(supplemental, index=False)

            result = run_pool1_ticker_lifecycle_contract(
                panel_dir=panel,
                price_coverage_path=price,
                supplemental_coverage_path=supplemental,
                output_dir=output,
            )
            self.assertEqual(result, output)

            manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            self.assertTrue(manifest["candidate_availability_formal_ready"])
            self.assertFalse(manifest["formal_target_stream_ready"])
            self.assertFalse(manifest["no_target_cash_all_applied"])

            lifecycle = pd.read_csv(output / "pool1_ticker_lifecycle_contract.csv")
            wiwynn = lifecycle[lifecycle["ticker"].eq("6669.TW")].iloc[0]
            self.assertEqual(wiwynn["data_start"], "2017-11-13")
            self.assertEqual(wiwynn["first_pool1_scoring_date"], "2018-02-06")

            daily = pd.read_csv(output / "pool1_date_aware_candidate_availability_daily.csv")
            before = daily[daily["date"].eq("2018-02-05") & daily["ticker"].eq("6669.TW")].iloc[0]
            self.assertFalse(bool(before["candidate_available_for_pool1_ranking"]))
            self.assertEqual(before["excluded_reason"], "insufficient_60d_pool1_scoring_warmup")
            after = daily[daily["date"].eq("2018-02-06") & daily["ticker"].eq("6669.TW")].iloc[0]
            self.assertTrue(bool(after["candidate_available_for_pool1_ranking"]))

            blockers = pd.read_csv(output / "blocker_by_ticker.csv")
            self.assertTrue(blockers["candidate_availability_ready"].all())
            self.assertIn("pool1_attack_gate_state_replay", set(blockers["remaining_formal_target_blocker"]))


if __name__ == "__main__":
    unittest.main()
