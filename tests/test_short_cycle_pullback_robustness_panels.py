import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

import test_paths  # noqa: F401

from backtest_lab.short_cycle_pullback_robustness_panels import run_short_cycle_pullback_robustness_panels


BEST = "ma20_reclaim_overlay_20_when_formal_cash_or_market_exposure_hold60"
BASE = "baseline_formal_next_day"


class ShortCyclePullbackRobustnessPanelsTest(unittest.TestCase):
    def test_builds_robustness_panels_without_formalizing_best_variant(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "experiments"
            contract = root / "contract"
            output = root / "out"
            source.mkdir()
            contract.mkdir()
            _write_experiments_fixture(source)
            (contract / "readiness_for_experiments.json").write_text(
                json.dumps({"eligible_event_rows": 2}),
                encoding="utf-8",
            )

            run_short_cycle_pullback_robustness_panels(
                experiments_output=source,
                contract_output=contract,
                output_dir=output,
            )

            readiness = json.loads((output / "robustness_readiness_for_experiments.json").read_text(encoding="utf-8"))
            self.assertEqual(readiness["best_variant"], BEST)
            self.assertTrue(readiness["ready_for_experiments_robustness_validation"])
            self.assertFalse(readiness["ready_for_formal_absorption"])
            self.assertFalse(readiness["formal_model_changed"])
            self.assertFalse(readiness["trade_decision_changed"])
            self.assertFalse(readiness["active_in_trade_decision"])
            self.assertTrue(readiness["diagnostic_only"])

            oos = pd.read_csv(output / "oos_period_panel.csv")
            self.assertIn("full", set(oos["period"]))
            self.assertIn("2024", set(oos["period"]))

            ablation = pd.read_csv(output / "ablation_panel.csv")
            self.assertIn("exclude_top_ticker_proxy", set(ablation["ablation_id"]))

            conflict = pd.read_csv(output / "formal_target_conflict_audit.csv")
            self.assertTrue((conflict["conflict_status"] == "pass").all())

            cost = pd.read_csv(output / "cost_slippage_sensitivity.csv")
            self.assertIn("1.5x_cost", set(cost["scenario"]))


def _write_experiments_fixture(root: Path) -> None:
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "task_id": "TASK-BACKTEST-EXPERIMENTS-SHORT-CYCLE-PULLBACK-PORTFOLIO-CHALLENGER-VALIDATION-001",
                "future_data_violation_count": 0,
            }
        ),
        encoding="utf-8",
    )
    dates = pd.date_range("2024-01-02", periods=8, freq="B")
    rows = []
    for variant, start in [(BASE, 100), (BEST, 100)]:
        equity = start
        for idx, date in enumerate(dates):
            equity *= 1.01 if variant == BEST else 1.005
            rows.append(
                {
                    "variant_id": variant,
                    "date": date.strftime("%Y-%m-%d"),
                    "equity": equity,
                    "formal_target": "CASH" if idx < 3 else "00631L.TW",
                    "sleeve_ticker": "2330.TW" if variant == BEST and idx in {2, 3} else "",
                    "sleeve_age": idx,
                    "weight_sum": 1.0,
                    "cash_weight": 0.0,
                    "cost": 10 if idx in {1, 2} else 0,
                    "turnover": 1000 if idx in {1, 2} else 0,
                    "diagnostic_only": True,
                }
            )
    pd.DataFrame(rows).to_csv(root / "daily_weight_ledger.csv", index=False)
    pd.DataFrame(
        [
            _trade(BASE, "2024-01-03", "00631L.TW", "buy", "rebalance_formal_or_sleeve", 1000, 10),
            _trade(BEST, "2024-01-03", "00631L.TW", "buy", "rebalance_formal_or_sleeve", 800, 8),
            _trade(BEST, "2024-01-04", "2330.TW", "buy", "enter_strong_stock_ma20_pullback_reclaim", 200, 2),
            _trade(BEST, "2024-01-08", "2330.TW", "sell", "exit_strong_stock_ma20_pullback_reclaim", 220, 2),
        ]
    ).to_csv(root / "trade_ledger.csv", index=False)
    pd.DataFrame(
        [
            _perf(BASE, 10.0, -5.0, 2, 100),
            _perf(BEST, 15.0, -4.0, 4, 120),
        ]
    ).to_csv(root / "portfolio_challenger_diagnostic.csv", index=False)
    pd.DataFrame(
        [
            {"variant_id": BASE, "month": "2024-01", "monthly_return_pct": 10.0},
            {"variant_id": BEST, "month": "2024-01", "monthly_return_pct": 15.0},
        ]
    ).to_csv(root / "monthly_performance.csv", index=False)
    pd.DataFrame(
        [
            {
                "variant_id": BEST,
                "ticker": "2330.TW",
                "candidate_source": "old_ai",
                "supply_chain_layer": "old_ai_seven",
                "entry_count": 1,
            }
        ]
    ).to_csv(root / "event_usage_summary.csv", index=False)
    pd.DataFrame(
        [
            {
                "variant_id": BEST,
                "entry_date": "2024-01-04",
                "ticker": "2330.TW",
                "candidate_name": "台積電",
                "candidate_source": "old_ai",
                "supply_chain_layer": "old_ai_seven",
                "formal_target": "CASH",
                "scope": "cash_or_market_exposure",
                "same_as_formal": False,
            }
        ]
    ).to_csv(root / "formal_target_overlap_audit.csv", index=False)
    pd.DataFrame(columns=["variant_id", "entry_date", "ticker"]).to_csv(root / "material_layer_case_slice.csv", index=False)
    pd.DataFrame(columns=["variant_id", "entry_date", "ticker"]).to_csv(root / "6488_case_only.csv", index=False)


def _trade(variant: str, date: str, ticker: str, side: str, reason: str, notional: float, cost: float) -> dict[str, object]:
    return {
        "variant_id": variant,
        "date": date,
        "ticker": ticker,
        "side": side,
        "from_weight": 0.0,
        "to_weight": 0.2,
        "notional": notional,
        "cost": cost,
        "reason": reason,
        "diagnostic_only": True,
    }


def _perf(variant: str, total_return: float, mdd: float, trade_legs: int, cost: float) -> dict[str, object]:
    return {
        "start_date": "2024-01-02",
        "end_date": "2024-01-11",
        "start_equity": 100.0,
        "final_equity": 100.0 + total_return,
        "total_return_pct": total_return,
        "cagr_pct": total_return,
        "mdd_pct": mdd,
        "worst_day_pct": -1.0,
        "variant_id": variant,
        "period": "full",
        "delta_return_vs_baseline_pct": 0.0 if variant == BASE else 5.0,
        "delta_mdd_vs_baseline_pct": 0.0 if variant == BASE else 1.0,
        "trade_legs": trade_legs,
        "total_cost": cost,
        "turnover": 1000.0,
    }


if __name__ == "__main__":
    unittest.main()
