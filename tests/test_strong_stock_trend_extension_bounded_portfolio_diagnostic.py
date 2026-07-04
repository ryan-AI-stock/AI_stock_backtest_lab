import tempfile
import unittest
from pathlib import Path

import pandas as pd

from backtest_lab.strong_stock_trend_extension_bounded_portfolio_diagnostic import (
    run_strong_stock_trend_extension_bounded_portfolio_diagnostic,
)


class StrongStockTrendExtensionBoundedPortfolioDiagnosticTest(unittest.TestCase):
    def test_runs_bounded_diagnostic_without_formal_override_or_cap_violation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            contract = root / "contract"
            contract.mkdir()
            pd.DataFrame(
                [
                    {
                        "signal_date": "2024-01-02",
                        "next_tradable_date": "2024-01-03",
                        "action_variant": "trend_ext_slope_acceleration_primary_sleeve10_hold20_when_formal_market_exposure_or_cash",
                        "event_variant": "trend_ext_slope_acceleration",
                        "ticker": "2330.TW",
                        "action_allowed": True,
                        "case_trace_only": False,
                        "proxy_row": False,
                        "entry_date": "2024-01-03",
                        "exit_date": "2024-01-17",
                        "sleeve_weight_candidate": 0.10,
                    },
                    {
                        "signal_date": "2024-01-04",
                        "next_tradable_date": "2024-01-05",
                        "action_variant": "trend_ext_slope_acceleration_primary_sleeve10_hold20_when_formal_market_exposure_or_cash",
                        "event_variant": "trend_ext_slope_acceleration",
                        "ticker": "2308.TW",
                        "action_allowed": True,
                        "case_trace_only": False,
                        "proxy_row": False,
                        "entry_date": "2024-01-05",
                        "exit_date": "2024-01-19",
                        "sleeve_weight_candidate": 0.10,
                    },
                ]
            ).to_csv(contract / "trend_extension_event_to_action_contract.csv", index=False)

            formal_dir = root / "outputs" / "combined_formal_target_stream_20150128_20211230_20260702"
            formal_dir.mkdir(parents=True)
            formal_rows = []
            dates = pd.date_range("2024-01-02", periods=20, freq="B")
            for date in dates:
                target = "CASH"
                target_type = "risk_control_cash"
                risk = "no_target_cash_all"
                if date.strftime("%Y-%m-%d") >= "2024-01-10":
                    target = "2454.TW"
                    target_type = "stock"
                    risk = "formal_target_active"
                formal_rows.append(
                    {
                        "signal_date": date.strftime("%Y-%m-%d"),
                        "execution_date": date.strftime("%Y-%m-%d"),
                        "formal_target": target,
                        "target_type": target_type,
                        "risk_off_state": risk,
                    }
                )
            pd.DataFrame(formal_rows).to_csv(formal_dir / "combined_formal_target_stream.csv", index=False)

            shards = root / "liquidity" / "shards"
            shards.mkdir(parents=True)
            price_rows = []
            for idx, date in enumerate(dates):
                for ticker, market, close in [
                    ("2330", "TWSE", 100 + idx),
                    ("2308", "TWSE", 50 + idx),
                    ("2454", "TWSE", 80 + idx),
                ]:
                    price_rows.append(
                        {
                            "date": date.strftime("%Y-%m-%d"),
                            "ticker": ticker,
                            "market": market,
                            "close": close,
                        }
                    )
            pd.DataFrame(price_rows).to_csv(shards / "accepted_liquidity_rows_2024_01.csv", index=False)

            bench = root / "backtest_cache" / "stock_pool_observations"
            bench.mkdir(parents=True)
            for filename in ["0050_TW.csv", "00631L_TW.csv"]:
                pd.DataFrame(
                    [{"date": date.strftime("%Y-%m-%d"), "close": 100 + idx} for idx, date in enumerate(dates)]
                ).to_csv(bench / filename, index=False)

            manifest = run_strong_stock_trend_extension_bounded_portfolio_diagnostic(
                repo_root=root,
                contract_dir=contract,
                liquidity_dir=root / "liquidity",
                output_dir=root / "out",
            )

            self.assertEqual(manifest["future_data_violation_count"], 0)
            self.assertEqual(manifest["formal_direct_stock_target_override_count"], 0)
            self.assertEqual(manifest["sleeve_cap_violation_count"], 0)
            daily = pd.read_csv(root / "out" / "daily_equity_by_variant.csv")
            self.assertIn("baseline_formal_next_day", set(daily["variant"]))
            blocked = pd.read_csv(root / "out" / "blocked_rows.csv")
            self.assertIn("blocked_by_active_sleeve_no_pyramid", set(blocked["blocked_reason"].astype(str)))


if __name__ == "__main__":
    unittest.main()
