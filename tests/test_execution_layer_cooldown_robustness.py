import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

import test_paths  # noqa: F401

from backtest_lab.execution_layer_cooldown_robustness import (
    MAIN_CANDIDATE,
    NEXT_DAY_BASELINE,
    SAME_DAY_REFERENCE,
    SECONDARY_CANDIDATE,
    run_execution_layer_cooldown_robustness,
)


class ExecutionLayerCooldownRobustnessTest(unittest.TestCase):
    def test_runner_builds_cooldown3_robustness_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            prices = root / "prices"
            output = root / "out"
            source.mkdir()
            prices.mkdir()
            _write_source(source)
            _write_price(prices, "0050.TW", [10, 11, 12, 13, 14, 15, 16, 17])
            _write_price(prices, "00631L.TW", [5, 6, 5.5, 7, 8, 9, 10, 11])

            result = run_execution_layer_cooldown_robustness(
                source_dir=source,
                price_cache_dir=prices,
                output_dir=output,
            )

            manifest = json.loads((result / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["main_candidate"], MAIN_CANDIDATE)
            self.assertFalse(manifest["formal_model_changed"])
            self.assertFalse(manifest["trade_decision_changed"])
            self.assertFalse(manifest["active_in_trade_decision"])
            self.assertFalse(manifest["formal_execution_layer_activated"])
            self.assertFalse(manifest["uses_forward_return_as_rule"])

            matrix = pd.read_csv(result / "candidate_parameter_matrix.csv")
            self.assertIn(MAIN_CANDIDATE, matrix["candidate"].tolist())
            self.assertFalse(matrix["active_in_trade_decision"].astype(bool).any())

            perf = pd.read_csv(result / "period_performance_by_candidate.csv")
            self.assertIn("full", perf["period_label"].tolist())
            self.assertIn(MAIN_CANDIDATE, perf["variant_id"].tolist())
            self.assertFalse(perf["active_in_trade_decision"].map(lambda value: str(value).lower() == "true").any())

            hard = pd.read_csv(result / "hard_gate_2024_attribution.csv")
            self.assertIn("excess_vs_0050x2_pct", hard.columns)

            concentration = pd.read_csv(result / "position_concentration_by_candidate.csv")
            self.assertIn("top_holding_day_share", concentration.columns)

            cost = pd.read_csv(result / "cost_sensitivity_by_candidate.csv")
            self.assertIn("cost_20bp", cost.columns)
            self.assertFalse(cost["active_in_trade_decision"].astype(bool).any())

            readiness = pd.read_csv(result / "cooldown_robustness_readiness_report.csv").iloc[0]
            self.assertFalse(bool(readiness["formal_absorption_ready"]))
            self.assertFalse(bool(readiness["active_in_trade_decision"]))


def _write_source(source: Path) -> None:
    dates = pd.date_range("2024-01-02", periods=6, freq="B")
    daily_rows = []
    trade_rows = []
    fill_rows = []
    for variant in (MAIN_CANDIDATE, SECONDARY_CANDIDATE, NEXT_DAY_BASELINE, SAME_DAY_REFERENCE):
        equity = 1_000_000.0
        for index, date in enumerate(dates):
            equity *= 1.02 if index else 1.0
            daily_rows.append(
                {
                    "variant_id": variant,
                    "date": date.strftime("%Y-%m-%d"),
                    "period": "fixture",
                    "portfolio_equity": equity,
                    "drawdown": 0,
                    "top_holding": "00631L.TW" if index % 2 else "2454.TW",
                    "cash_weight": 0.2,
                    "pending_order_count": 0,
                    "execution_diagnostic_active_in_trade_decision": False,
                }
            )
        trade_rows.append(
            {
                "variant_id": variant,
                "date": dates[1].strftime("%Y-%m-%d"),
                "ticker": "00631L.TW",
                "action": "buy",
                "gross_amount": 100_000,
                "transaction_cost": 143,
            }
        )
        fill_rows.append(
            {
                "variant_id": variant,
                "signal_date": dates[0].strftime("%Y-%m-%d"),
                "fill_date": dates[1].strftime("%Y-%m-%d"),
                "target_weights": '{"00631L.TW": 0.4}',
            }
        )
    pd.DataFrame(daily_rows).to_csv(source / "next_day_fill_full_equity_ledger.csv", index=False)
    pd.DataFrame(trade_rows).to_csv(source / "next_day_fill_trade_ledger.csv", index=False)
    pd.DataFrame(fill_rows).to_csv(source / "fill_event_panel.csv", index=False)
    pd.DataFrame(columns=["variant_id", "blocked_reason"]).to_csv(source / "blocked_execution_events.csv", index=False)
    pd.DataFrame(
        [
            {
                "alignment_state": "passed",
                "max_abs_diff": 0,
                "final_equity_diff": 0,
                "active_in_trade_decision": False,
            }
        ]
    ).to_csv(source / "baseline_alignment.csv", index=False)


def _write_price(cache: Path, ticker: str, closes: list[float]) -> None:
    rows = []
    for index, close in enumerate(closes):
        rows.append(
            {
                "date": (pd.Timestamp("2024-01-02") + pd.Timedelta(days=index)).strftime("%Y-%m-%d"),
                "open": close,
                "close": close,
                "adj_close": close,
            }
        )
    pd.DataFrame(rows).to_csv(cache / f"{ticker.replace('.', '_')}.csv", index=False)


if __name__ == "__main__":
    unittest.main()
