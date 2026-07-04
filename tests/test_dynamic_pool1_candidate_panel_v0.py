import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from backtest_lab.dynamic_pool1_candidate_panel_v0 import SourcePaths, build_candidate_panel


class DynamicPool1CandidatePanelV0Test(unittest.TestCase):
    def test_builds_shadow_panel_without_formal_flags(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            liquidity = root / "liquidity"
            revenue = root / "revenue"
            fundamentals = root / "fundamentals"
            taxonomy = root / "taxonomy"
            pool1b = root / "pool1b"
            sector = root / "sector"
            (liquidity / "shards").mkdir(parents=True)
            (revenue / "accepted_monthly_revenue_rows_shards").mkdir(parents=True)
            (fundamentals / "shards").mkdir(parents=True)
            taxonomy.mkdir()
            (pool1b / "cache_compatible").mkdir(parents=True)
            sector.mkdir()

            liq_rows = []
            for month in range(1, 16):
                year = 2024 if month <= 12 else 2025
                mm = month if month <= 12 else month - 12
                for ticker, name, close in [("0050", "0050", 100 + month), ("2330", "台積電", 500 + month * 5), ("2454", "聯發科", 700 + month * 8)]:
                    for day in [1, 8, 15, 22, 28]:
                        liq_rows.append(
                            {
                                "date": f"{year}-{mm:02d}-{min(day, 28):02d}",
                                "ticker": ticker,
                                "name": name,
                                "market": "TWSE",
                                "turnover_value": 50_000_000,
                                "close": close + day / 100,
                                "liquidity_pass": True,
                            }
                        )
            pd.DataFrame(liq_rows).to_csv(liquidity / "shards" / "accepted_liquidity_rows_2024_01.csv", index=False)

            rev_rows = []
            for month in range(1, 16):
                year = 2024 if month <= 12 else 2025
                mm = month if month <= 12 else month - 12
                for ticker, name in [("2330", "台積電"), ("2454", "聯發科")]:
                    rev_rows.append(
                        {
                            "ticker": ticker,
                            "name": name,
                            "market": "TWSE",
                            "revenue_year_month": f"{year}-{mm:02d}",
                            "revenue_value": 1000 + month * (20 if ticker == "2454" else 10),
                            "available_date": f"{year}-{mm:02d}-10",
                            "pit_usable": True,
                        }
                    )
            pd.DataFrame(rev_rows).to_csv(
                revenue / "accepted_monthly_revenue_rows_shards" / "accepted_monthly_revenue_rows_2024.csv",
                index=False,
            )

            pd.DataFrame(
                [
                    {
                        "ticker": "2330",
                        "name": "台積電",
                        "market": "sii",
                        "fiscal_year": 2024,
                        "quarter": 1,
                        "available_date": "2024-05-15",
                        "eps": 1.0,
                        "roe": 10.0,
                        "gross_margin": 50.0,
                        "operating_margin": 40.0,
                        "net_income": 100,
                        "operating_income": 120,
                    }
                ]
            ).to_csv(fundamentals / "shards" / "accepted_quarterly_fundamentals_rows_2024.csv", index=False)
            pd.DataFrame(
                [
                    {
                        "ticker": "2330.TW",
                        "candidate_scope": "old_ai",
                        "has_accepted_evidence": True,
                        "ai_supply_chain_layers": "foundry",
                        "mainline_theme_labels": "AI",
                        "accepted_for_diagnostic": True,
                        "accepted_for_formal": False,
                        "human_review_required": True,
                    }
                ]
            ).to_csv(taxonomy / "taxonomy_evidence_by_ticker.csv", index=False)
            (pool1b / "cache_compatible" / "2454.TW.csv").write_text("date,close\n2024-01-01,1\n", encoding="utf-8")

            out = root / "out"
            manifest = build_candidate_panel(
                SourcePaths(
                    liquidity_dir=liquidity,
                    revenue_dir=revenue,
                    fundamentals_dir=fundamentals,
                    taxonomy_dir=taxonomy,
                    pool1b_price_repair_dir=pool1b,
                    sector_dir=sector,
                ),
                out,
            )

            self.assertTrue((out / "candidate_panel_monthly.csv").exists())
            self.assertTrue((out / "candidate_pool_by_month.csv").exists())
            saved = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], "completed_shadow_candidate_panel_v0")
            self.assertFalse(saved["formal_model_changed"])
            self.assertFalse(saved["trade_decision_changed"])
            self.assertFalse(saved["active_in_trade_decision"])
            self.assertTrue(saved["ready_for_experiments_shadow_replay"])


if __name__ == "__main__":
    unittest.main()
