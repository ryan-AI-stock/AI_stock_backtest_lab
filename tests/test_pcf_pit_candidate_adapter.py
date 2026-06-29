import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from backtest_lab.pcf_pit_candidate_adapter import (
    load_0050_pcf_monthly_anchor,
    resolve_0050_constituents_for_date,
    run_0050_pit_candidate_backtest_data_readiness,
)


class PcfPitCandidateAdapterTest(unittest.TestCase):
    def test_resolves_calendar_month_and_pit_safe_anchor(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            anchor = root / "anchor.csv"
            _anchor_csv(anchor)

            loaded = load_0050_pcf_monthly_anchor(anchor)
            self.assertEqual(len(loaded), 6)
            self.assertTrue((loaded["formal_exact"].astype(str) == "false").all())

            calendar_rows, calendar_meta = resolve_0050_constituents_for_date(
                "2014-12-10",
                monthly_anchor_path=anchor,
                mode="calendar_month",
            )
            self.assertEqual(calendar_meta["effective_month"], "2014-12")
            self.assertEqual(calendar_meta["constituent_count"], 3)
            self.assertTrue(calendar_meta["anchor_after_query_date"])
            self.assertFalse(calendar_meta["pit_safe_for_query_date"])
            self.assertEqual(set(calendar_rows["ticker"]), {"2330", "2454", "2317"})

            pit_rows, pit_meta = resolve_0050_constituents_for_date(
                "2014-12-10",
                monthly_anchor_path=anchor,
                mode="pit_safe",
            )
            self.assertEqual(pit_meta["effective_month"], "2014-11")
            self.assertFalse(pit_meta["anchor_after_query_date"])
            self.assertTrue(pit_meta["pit_safe_for_query_date"])
            self.assertEqual(set(pit_rows["ticker"]), {"1101", "1102", "1216"})

    def test_builds_readiness_outputs_without_changing_formal_model(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            anchor = root / "anchor.csv"
            coverage = root / "price_coverage.csv"
            output = root / "out"
            _anchor_csv(anchor)
            pd.DataFrame(
                [
                    {
                        "ticker": "1101.TW",
                        "first_date": "2014-11-03",
                        "last_date": "2026-06-29",
                        "adjusted_close_available": True,
                        "ready_for_backtest_price_only": True,
                    },
                    {
                        "ticker": "1102.TW",
                        "first_date": "2014-11-03",
                        "last_date": "2026-06-29",
                        "adjusted_close_available": True,
                        "ready_for_backtest_price_only": True,
                    },
                    {
                        "ticker": "1216.TW",
                        "first_date": "2014-11-03",
                        "last_date": "2026-06-29",
                        "adjusted_close_available": True,
                        "ready_for_backtest_price_only": True,
                    },
                ]
            ).to_csv(coverage, index=False)

            run_0050_pit_candidate_backtest_data_readiness(
                monthly_anchor_path=anchor,
                price_coverage_path=coverage,
                output_dir=output,
                sample_dates=("2014-11-03", "2014-12-10"),
            )

            manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            self.assertTrue(manifest["monthly_anchor_readable"])
            self.assertFalse(manifest["formal_exact"])
            self.assertFalse(manifest["formal_model_changed"])
            self.assertFalse(manifest["trade_decision_changed"])
            self.assertFalse(manifest["active_in_trade_decision"])
            self.assertFalse(manifest["strategy_ready"])

            blockers = pd.read_csv(output / "backtest_data_blockers.csv")
            self.assertIn("formal_target_signal_stream_2014_2021", set(blockers["blocker"]))
            self.assertIn("execution_ledger_2014_2021", set(blockers["blocker"]))


def _anchor_csv(path: Path) -> None:
    rows = []
    for month, date, tickers in [
        ("2014-11", "2014-11-28", [("1101", "台泥"), ("1102", "亞泥"), ("1216", "統一")]),
        ("2014-12", "2014-12-31", [("2330", "台積電"), ("2454", "聯發科"), ("2317", "鴻海")]),
    ]:
        for ticker, name in tickers:
            rows.append(
                {
                    "effective_month": month,
                    "effective_date": date,
                    "holdings_date": date,
                    "source_date": date,
                    "ticker": ticker,
                    "name": name,
                    "weight": "1.0",
                    "weight_available": "true",
                    "source_url": "https://example.test",
                    "raw_source_id": f"raw#{ticker}",
                    "source_type": "source_backed_manual_candidate",
                    "formal_exact": "false",
                    "source_backed_manual_candidate": "true",
                    "membership_ready": "true",
                    "weighted_portfolio_ready": "true",
                    "row_count": "3",
                    "row_count_anomaly": "false",
                    "validation_decision": "accepted",
                    "proxy_row_used": "false",
                    "active_in_trade_decision": "false",
                }
            )
    pd.DataFrame(rows).to_csv(path, index=False)


if __name__ == "__main__":
    unittest.main()
