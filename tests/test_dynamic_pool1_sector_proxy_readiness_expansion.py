import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

import test_paths  # noqa: F401

from backtest_lab.dynamic_pool1_sector_proxy_readiness_expansion import (
    run_dynamic_pool1_sector_proxy_readiness_expansion,
)


class DynamicPool1SectorProxyReadinessExpansionTest(unittest.TestCase):
    def test_audits_price_coverage_and_keeps_strategy_replay_off(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sector = root / "sector"
            cache = root / "cache"
            output = root / "out"
            sector.mkdir()
            cache.mkdir()

            _write_sector_proxy_output(sector)
            pd.DataFrame(
                [
                    {"date": "2022-01-03", "open": 1, "high": 1, "low": 1, "close": 1, "adj_close": 1},
                    {"date": "2026-07-02", "open": 2, "high": 2, "low": 2, "close": 2, "adj_close": 2},
                ]
            ).to_csv(cache / "2330_TW.csv", index=False)

            run_dynamic_pool1_sector_proxy_readiness_expansion(
                sector_proxy_output=sector,
                output_dir=output,
                price_cache_dir=cache,
                price_source_registry=root / "missing_registry.csv",
                candidate_panel_2022_latest=root / "missing_candidate.csv",
            )

            readiness = json.loads((output / "readiness_for_experiments.json").read_text(encoding="utf-8"))
            self.assertEqual(readiness["status"], "data_needed")
            self.assertFalse(readiness["strategy_replay"])
            self.assertFalse(readiness["ready_for_strategy_replay"])
            self.assertFalse(readiness["active_in_trade_decision"])
            self.assertTrue(readiness["diagnostic_only"])
            self.assertTrue(readiness["twse_only"])
            self.assertFalse(readiness["tpex_included"])
            self.assertEqual(readiness["price_coverage_total_tickers"], 2)
            self.assertEqual(readiness["price_coverage_ready_tickers"], 1)
            self.assertFalse(readiness["candidate_sector_context_2022_latest_ready"])

            coverage = pd.read_csv(output / "twse_price_coverage_audit.csv")
            missing = coverage[coverage["ticker"].eq("2454.TW")].iloc[0]
            self.assertFalse(bool(missing["price_data_ready"]))
            self.assertEqual(missing["blocked_reason"], "missing_local_price_cache_for_twse_sector_proxy_ticker")

            blockers = pd.read_csv(output / "sector_proxy_join_blockers.csv")
            self.assertIn("missing_2022_latest_candidate_panel", set(blockers["blocker_type"]))

    def test_builds_2022_candidate_context_when_panel_exists(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sector = root / "sector"
            cache = root / "cache"
            output = root / "out"
            sector.mkdir()
            cache.mkdir()

            _write_sector_proxy_output(sector, date="2022-01-31")
            pd.DataFrame(
                [
                    {"date": "2022-01-31", "candidate_ticker": "2330.TW", "candidate_name": "台積電"},
                ]
            ).to_csv(root / "candidate.csv", index=False)
            for ticker in ["2330", "2454"]:
                pd.DataFrame(
                    [
                        {"date": "2022-01-31", "open": 1, "high": 1, "low": 1, "close": 1, "adj_close": 1},
                        {"date": "2026-07-02", "open": 2, "high": 2, "low": 2, "close": 2, "adj_close": 2},
                    ]
                ).to_csv(cache / f"{ticker}_TW.csv", index=False)

            run_dynamic_pool1_sector_proxy_readiness_expansion(
                sector_proxy_output=sector,
                output_dir=output,
                price_cache_dir=cache,
                price_source_registry=root / "missing_registry.csv",
                candidate_panel_2022_latest=root / "candidate.csv",
            )

            context = pd.read_csv(output / "candidate_sector_context_2022_latest.csv")
            self.assertEqual(len(context), 1)
            self.assertEqual(context.iloc[0]["context_status"], "twse_monthly_anchor_exact_date_match")
            self.assertFalse(bool(context.iloc[0]["active_in_trade_decision"]))


def _write_sector_proxy_output(root: Path, date: str = "2022-01-03") -> None:
    pd.DataFrame(
        [
            _sector_row(date, "2330.TW", "2330", "台積電"),
            _sector_row(date, "2454.TW", "2454", "聯發科"),
        ]
    ).to_csv(root / "dynamic_pool1_twse_sector_proxy_panel.csv", index=False)
    (root / "dynamic_pool1_sector_proxy_readiness.json").write_text(
        json.dumps(
            {
                "status": "completed_twse_only_sector_proxy_diagnostic_panel",
                "future_data_violation_count": 0,
            }
        ),
        encoding="utf-8",
    )


def _sector_row(date: str, ticker_with_suffix: str, ticker: str, name: str) -> dict[str, object]:
    return {
        "as_of_date": date,
        "effective_date": date,
        "source_date": date,
        "ticker": ticker,
        "ticker_with_suffix": ticker_with_suffix,
        "name": name,
        "market": "TWSE",
        "sector_code": "24",
        "sector_name": "半導體業",
        "mainline": "半導體業",
        "theme": "",
        "membership_policy": "monthly_anchor",
        "daily_exact": False,
        "twse_only": True,
        "tpex_included": False,
        "mainline_theme_ready": False,
        "diagnostic_only": True,
        "active_in_trade_decision": False,
        "accepted_for_formal": False,
        "future_data_violation": False,
        "source_type": "official_twse_mi_index_monthly_anchor_industry_membership_candidate",
        "formal_exact": False,
        "source_url": "https://www.twse.com.tw/exchangeReport/MI_INDEX",
    }


if __name__ == "__main__":
    unittest.main()
