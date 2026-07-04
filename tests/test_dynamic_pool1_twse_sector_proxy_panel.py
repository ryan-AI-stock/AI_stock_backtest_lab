import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

import test_paths  # noqa: F401

from backtest_lab.dynamic_pool1_twse_sector_proxy_panel import run_dynamic_pool1_twse_sector_proxy_panel


class DynamicPool1TwseSectorProxyPanelTest(unittest.TestCase):
    def test_builds_twse_only_anchor_panel_without_formalizing_it(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sector = root / "sector"
            shards = sector / "shards"
            output = root / "out"
            sector.mkdir()
            shards.mkdir()

            (sector / "readiness_for_core.json").write_text(
                json.dumps(
                    {
                        "task_id": "TASK-RADAR-DATA-DYNAMIC-POOL1-SECTOR-MAINLINE-PIT-FULL-SWEEP-AND-TPEX-REVERSE-20260703",
                        "status": "completed_partial_twse_monthly_anchor_ready_tpex_blocked",
                        "twse_only": True,
                        "tpex_included": False,
                        "mainline_theme_ready": False,
                        "twse_sector_monthly_anchor_ready": True,
                        "twse_sector_full_sweep_ready": False,
                        "sector_membership_pit_partial_ready": True,
                        "sector_breadth_pit_daily_ready": False,
                        "twse_sector_membership_rows": 2,
                        "future_data_violation_count": 0,
                    }
                ),
                encoding="utf-8",
            )
            (sector / "source_manifest.json").write_text("{}", encoding="utf-8")
            pd.DataFrame(
                [
                    {
                        "file": "shards/twse_sector_membership_2015.csv",
                        "market": "TWSE",
                        "year": 2015,
                        "rows": 2,
                        "months": 1,
                        "source_type": "official_twse_mi_index_monthly_anchor_industry_membership_candidate",
                        "formal_exact": False,
                    }
                ]
            ).to_csv(sector / "twse_sector_membership_pit_daily_manifest.csv", index=False)
            pd.DataFrame(
                [
                    _sector_row("2330", "台積電", "24", "半導體業"),
                    _sector_row("2454", "聯發科", "24", "半導體業"),
                ]
            ).to_csv(shards / "twse_sector_membership_2015.csv", index=False)
            pd.DataFrame(
                [
                    {
                        "date": "2015-01-30",
                        "candidate_ticker": "2330.TW",
                        "candidate_name": "台積電",
                        "candidate_source": "pool1_date_aware_lifecycle_universe",
                    }
                ]
            ).to_csv(root / "candidate.csv", index=False)

            run_dynamic_pool1_twse_sector_proxy_panel(
                sector_output=sector,
                candidate_panel_path=root / "candidate.csv",
                output_dir=output,
            )

            readiness = json.loads((output / "dynamic_pool1_sector_proxy_readiness.json").read_text(encoding="utf-8"))
            self.assertTrue(readiness["ready_for_experiments_twse_only_diagnostic"])
            self.assertFalse(readiness["ready_for_strategy_replay"])
            self.assertFalse(readiness["dynamic_pool1_shadow_challenger_ready"])
            self.assertTrue(readiness["twse_only"])
            self.assertFalse(readiness["tpex_included"])
            self.assertFalse(readiness["mainline_theme_ready"])
            self.assertFalse(readiness["daily_exact"])
            self.assertTrue(readiness["diagnostic_only"])

            panel = pd.read_csv(output / "dynamic_pool1_twse_sector_proxy_panel.csv")
            self.assertEqual(len(panel), 2)
            self.assertTrue(panel["diagnostic_only"].astype(bool).all())
            self.assertFalse(panel["daily_exact"].astype(bool).any())
            self.assertFalse(panel["active_in_trade_decision"].astype(bool).any())

            breadth = pd.read_csv(output / "dynamic_pool1_twse_sector_breadth_monthly_anchor.csv")
            self.assertEqual(int(breadth.iloc[0]["constituent_count"]), 2)

            context = pd.read_csv(output / "dynamic_pool1_candidate_sector_context.csv")
            self.assertEqual(len(context), 1)
            self.assertEqual(context.iloc[0]["sector_name"], "半導體業")
            self.assertEqual(context.iloc[0]["membership_policy"], "monthly_anchor")


def _sector_row(ticker: str, name: str, sector_code: str, sector_name: str) -> dict[str, object]:
    return {
        "ticker": ticker,
        "name": name,
        "market": "TWSE",
        "sector_code": sector_code,
        "sector_name": sector_name,
        "mainline": sector_name,
        "theme": "",
        "source_date": "2015-01-30",
        "effective_date": "2015-01-30",
        "as_of_date": "2015-01-30",
        "source_url": "https://www.twse.com.tw/exchangeReport/MI_INDEX",
        "source_type": "official_twse_mi_index_monthly_anchor_industry_membership_candidate",
        "formal_exact": False,
        "accepted_for_diagnostic": True,
        "accepted_for_formal": False,
        "evidence": "monthly anchor",
        "notes": "not daily exact",
    }


if __name__ == "__main__":
    unittest.main()
