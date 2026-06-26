import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from backtest_lab.pool1_pool2_final_challenger_robustness import run_pool1_pool2_final_challenger_robustness


class Pool1Pool2FinalChallengerRobustnessTest(unittest.TestCase):
    def test_runner_outputs_required_boundary_and_robustness_panels(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            prices = root / "prices"
            output = root / "out"
            source.mkdir()
            prices.mkdir()
            dates = pd.date_range("2024-01-02", periods=8, freq="B")
            variants = [
                "combined_cap40_confirmation1",
                "pool1_pool2_disagree_confirmation_2",
                "pool1_primary_no_overlay",
                "pool1_pool2_veto_no_cap",
            ]
            daily_rows = []
            trade_rows = []
            event_rows = []
            for variant in variants:
                equity = 1_000_000.0
                for index, date in enumerate(dates):
                    ticker = "00631L.TW" if index % 3 == 0 else "2330.TW"
                    equity *= 1.01
                    daily_rows.append(
                        {
                            "variant": variant,
                            "date": date.strftime("%Y-%m-%d"),
                            "period": "2024_now",
                            "target_weights": '{"00631L.TW": 0.4}' if variant == "combined_cap40_confirmation1" and ticker == "00631L.TW" else '{"2330.TW": 1.0}',
                            "position_ticker": ticker,
                            "cash": 1000,
                            "equity": equity,
                            "drawdown": 0.0,
                            "turnover": 10000 if index in {0, 3, 6} else 0,
                            "transaction_cost": 100 if index in {0, 3, 6} else 0,
                            "action": "rebalance" if index in {0, 3, 6} else "hold",
                        }
                    )
                    event_rows.append(
                        {
                            "variant": variant,
                            "date": date.strftime("%Y-%m-%d"),
                            "period": "2024_now",
                            "pool1_vote": ticker,
                            "pool2_vote": "0050.TW",
                            "pool2_disagreement": True,
                            "event_reason": "pool2_disagrees_confirmation_1_not_met",
                            "target_weights": '{"00631L.TW": 0.4}' if variant == "combined_cap40_confirmation1" and ticker == "00631L.TW" else '{"2330.TW": 1.0}',
                        }
                    )
                trade_rows.append({"variant": variant, "date": dates[0].strftime("%Y-%m-%d"), "ticker": "00631L.TW", "action": "buy", "gross_amount": 10000})
            pd.DataFrame(daily_rows).to_csv(source / "daily_equity_by_variant.csv", index=False)
            pd.DataFrame(trade_rows).to_csv(source / "trade_ledger_by_variant.csv", index=False)
            pd.DataFrame(event_rows).to_csv(source / "pool2_disagreement_variant_events.csv", index=False)
            pd.DataFrame({"variant": variants, "period_label": ["full"] * len(variants)}).to_csv(source / "period_performance_by_variant.csv", index=False)

            for ticker, start in {"0050.TW": 20, "00631L.TW": 10}.items():
                closes = [start + i for i in range(len(dates))]
                pd.DataFrame(
                    {
                        "date": dates.strftime("%Y-%m-%d"),
                        "open": closes,
                        "high": closes,
                        "low": closes,
                        "close": closes,
                        "adj_close": closes,
                        "volume": [1000] * len(dates),
                    }
                ).to_csv(prices / f"{ticker}.csv", index=False)

            run_pool1_pool2_final_challenger_robustness(source_dir=source, price_cache_dir=prices, output_dir=output)

            manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            self.assertFalse(manifest["formal_model_changed"])
            self.assertFalse(manifest["trade_decision_changed"])
            self.assertFalse(manifest["formal_absorption_ready"])
            self.assertTrue(manifest["cap_performance_recomputed"])
            self.assertTrue(manifest["benchmarks_include_0050x2"])
            self.assertTrue(manifest["same_date_range_for_candidates"])
            self.assertEqual(manifest["next_day_approximation_status"], "blocked_not_mixed_with_same_day")

            benchmark = pd.read_csv(output / "benchmark_comparison_0050_00631L_0050x2.csv")
            self.assertIn("0050x2", set(benchmark["benchmark"]))

            trigger = pd.read_csv(output / "cap40_trigger_attribution.csv")
            self.assertFalse(trigger.empty)
            self.assertFalse(trigger["uses_forward_return_as_rule"].any())

            execution = pd.read_csv(output / "execution_ledger_by_candidate.csv")
            self.assertIn("next_day_approximation", set(execution["execution_mode"]))
            self.assertIn("blocked", set(execution["status"]))


if __name__ == "__main__":
    unittest.main()
