import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from backtest_lab.pool1_pool2_market_exposure_override import run_pool1_pool2_market_exposure_override


class Pool1Pool2MarketExposureOverrideTest(unittest.TestCase):
    def test_label_only_boundary_and_override_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            blocker = root / "blocker"
            prices = root / "prices"
            output = root / "out"
            source.mkdir()
            blocker.mkdir()
            prices.mkdir()
            dates = pd.date_range("2024-01-02", periods=90, freq="B")
            daily_rows = []
            event_rows = []
            for index, date in enumerate(dates):
                ticker = "00631L.TW" if index % 5 == 0 else "2330.TW"
                weights = '{"00631L.TW": 0.4}' if ticker == "00631L.TW" else '{"2330.TW": 1.0}'
                daily_rows.append(
                    {
                        "variant": "combined_cap40_confirmation1",
                        "date": date.strftime("%Y-%m-%d"),
                        "period": "2024",
                        "target_weights": weights,
                        "position_ticker": ticker,
                        "cash": 1000,
                        "equity": 1_000_000 + index * 1000,
                        "drawdown": 0.0,
                        "turnover": 10000 if index in {0, 5, 10} else 0,
                        "transaction_cost": 100 if index in {0, 5, 10} else 0,
                        "action": "rebalance" if index in {0, 5, 10} else "hold",
                    }
                )
                event_rows.append(
                    {
                        "variant": "combined_cap40_confirmation1",
                        "date": date.strftime("%Y-%m-%d"),
                        "period": "2024",
                        "pool1_vote": ticker,
                        "pool2_vote": ticker,
                        "pool2_disagreement": False,
                        "event_reason": "pool1_primary",
                        "target_weights": weights,
                    }
                )
            pd.DataFrame(daily_rows).to_csv(source / "daily_equity_by_variant.csv", index=False)
            pd.DataFrame({"variant": ["combined_cap40_confirmation1"], "date": [dates[0].strftime("%Y-%m-%d")], "ticker": ["00631L.TW"], "action": ["buy"], "gross_amount": [10000]}).to_csv(source / "trade_ledger_by_variant.csv", index=False)
            pd.DataFrame(event_rows).to_csv(source / "pool2_disagreement_variant_events.csv", index=False)
            pd.DataFrame([row for row in event_rows if "00631L" in row["target_weights"]]).to_csv(blocker / "cap40_trigger_event_panel.csv", index=False)

            for ticker, (start, step) in {"0050.TW": (20, 0.1), "00631L.TW": (10, 0.2), "2330.TW": (100, 0.05)}.items():
                closes = [start + i * step for i in range(len(dates))]
                pd.DataFrame({"date": dates.strftime("%Y-%m-%d"), "open": closes, "high": closes, "low": closes, "close": closes, "adj_close": closes, "volume": [1000] * len(dates)}).to_csv(prices / f"{ticker.replace('.', '_')}.csv", index=False)

            run_pool1_pool2_market_exposure_override(source_dir=source, blocker_dir=blocker, price_cache_dir=prices, output_dir=output)

            manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            self.assertFalse(manifest["formal_model_changed"])
            self.assertFalse(manifest["trade_decision_changed"])
            self.assertFalse(manifest["formal_absorption_ready"])
            self.assertFalse(manifest["uses_forward_return_as_rule"])
            self.assertFalse(manifest["opportunity_cost_label_active_in_trade_decision"])

            daily = pd.read_csv(output / "daily_equity_by_variant.csv")
            base = daily[daily["variant"].eq("combined_cap40_confirmation1_base")]["equity"].reset_index(drop=True)
            label = daily[daily["variant"].eq("combined_cap40_confirmation1_0050x2_opportunity_cost_label_only")]["equity"].reset_index(drop=True)
            pd.testing.assert_series_equal(base, label, check_names=False)

            audit = pd.read_csv(output / "override_trigger_audit.csv")
            self.assertFalse(audit["uses_forward_return_as_rule"].any())
            self.assertFalse(audit["opportunity_cost_label_active_in_trade_decision"].any())

            self.assertTrue((output / "market_exposure_benchmark_comparison.csv").exists())
            self.assertTrue((output / "overfit_guard_report.csv").exists())


if __name__ == "__main__":
    unittest.main()
