import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from backtest_lab.pool1_pool2_final_challenger_blockers import run_pool1_pool2_final_challenger_blockers


class Pool1Pool2FinalChallengerBlockersTest(unittest.TestCase):
    def test_blocker_outputs_keep_report_only_boundaries(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            robust = root / "robust"
            prices = root / "prices"
            output = root / "out"
            source.mkdir()
            robust.mkdir()
            prices.mkdir()
            dates = pd.date_range("2024-01-02", periods=8, freq="B")
            variants = ["combined_cap40_confirmation1", "pool1_pool2_disagree_confirmation_2", "pool1_pool2_disagree_confirmation_1"]
            daily_rows = []
            event_rows = []
            for variant in variants:
                equity = 1_000_000.0
                for index, date in enumerate(dates):
                    ticker = "00631L.TW" if index % 2 == 0 else "2330.TW"
                    equity *= 1.01
                    daily_rows.append(
                        {
                            "variant": variant,
                            "date": date.strftime("%Y-%m-%d"),
                            "period": "2024",
                            "target_weights": '{"00631L.TW": 0.4}' if ticker == "00631L.TW" else '{"2330.TW": 1.0}',
                            "position_ticker": ticker,
                            "cash": 1000,
                            "equity": equity,
                            "drawdown": 0.0,
                            "turnover": 10000 if index in {0, 2, 4} else 0,
                            "transaction_cost": 100 if index in {0, 2, 4} else 0,
                            "action": "rebalance" if index in {0, 2, 4} else "hold",
                        }
                    )
                    event_rows.append(
                        {
                            "variant": variant,
                            "date": date.strftime("%Y-%m-%d"),
                            "period": "2024",
                            "pool1_vote": ticker,
                            "pool2_vote": "0050.TW",
                            "pool2_disagreement": True,
                            "event_reason": "pool2_disagrees_confirmation_1_not_met",
                            "target_weights": '{"00631L.TW": 0.4}' if ticker == "00631L.TW" else '{"2330.TW": 1.0}',
                        }
                    )
            pd.DataFrame(daily_rows).to_csv(source / "daily_equity_by_variant.csv", index=False)
            pd.DataFrame({"variant": variants, "date": [dates[0].strftime("%Y-%m-%d")] * 3, "ticker": ["00631L.TW"] * 3, "action": ["buy"] * 3, "gross_amount": [10000] * 3}).to_csv(source / "trade_ledger_by_variant.csv", index=False)
            pd.DataFrame(event_rows).to_csv(source / "pool2_disagreement_variant_events.csv", index=False)
            pd.DataFrame([row for row in event_rows if row["variant"] == "combined_cap40_confirmation1" and "00631L" in row["target_weights"]]).to_csv(robust / "cap40_trigger_event_panel.csv", index=False)
            pd.DataFrame({"date": dates.strftime("%Y-%m-%d"), "target_drop_from_top3_next_1d": [False] * 8, "target_drop_from_top3_next_2d": [False] * 8, "target_drop_from_top3_next_3d": [False] * 8, "target_reappears_in_top3_within_5d": [True] * 8}).to_csv(root / "target_drop.csv", index=False)

            for ticker, start in {"0050.TW": 20, "00631L.TW": 10, "2330.TW": 100}.items():
                closes = [start + i for i in range(len(dates))]
                pd.DataFrame({"date": dates.strftime("%Y-%m-%d"), "open": closes, "high": closes, "low": closes, "close": closes, "adj_close": closes, "volume": [1000] * len(dates)}).to_csv(prices / f"{ticker}.csv", index=False)

            run_pool1_pool2_final_challenger_blockers(source_dir=source, robustness_dir=robust, target_drop_source_path=root / "target_drop.csv", price_cache_dir=prices, output_dir=output)

            manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            self.assertFalse(manifest["formal_model_changed"])
            self.assertFalse(manifest["trade_decision_changed"])
            self.assertFalse(manifest["formal_absorption_ready"])
            self.assertFalse(manifest["next_day_ledger_mixed_with_same_day"])
            self.assertTrue(manifest["benchmarks_include_0050x2"])

            next_day = pd.read_csv(output / "next_day_fill_ledger.csv")
            self.assertFalse(next_day["next_day_ledger_mixed_with_same_day"].any())
            self.assertIn("completed", set(next_day["status"]))

            entry = pd.read_csv(output / "entry_without_exit_recomputed.csv")
            self.assertIn("entry_without_exit_confirmation", entry.columns)

            target_drop = pd.read_csv(output / "target_drop_from_top3_recomputed.csv")
            self.assertIn("target_drop_from_top3_next_3d", target_drop.columns)

            counter = pd.read_csv(output / "cap40_actual_vs_no_cap_counterfactual.csv")
            self.assertFalse(counter.empty)
            self.assertFalse(counter["formal_model_changed"].any())


if __name__ == "__main__":
    unittest.main()
