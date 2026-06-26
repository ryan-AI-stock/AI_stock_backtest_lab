import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from backtest_lab.formal_model_diagnosis_and_replay import run_formal_model_diagnosis_and_replay


class FormalModelDiagnosisAndReplayTest(unittest.TestCase):
    def test_builds_report_only_diagnosis_package(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            absorption = root / "absorption"
            candidate = root / "candidate"
            comparison = root / "comparison"
            label = root / "label"
            cooldown = root / "cooldown"
            cache = root / "cache"
            output = root / "output"
            for path in (absorption, candidate, comparison, label, cooldown, cache):
                path.mkdir()

            (absorption / "manifest.json").write_text(
                json.dumps(
                    {
                        "task_id": "absorb",
                        "formal_model_target": "combined_cap40_confirmation1_base",
                        "formal_model_route": "pool1_primary_pool2_confirmation_cap40",
                        "formal_absorption_ready": True,
                        "latest_complete_common_date": "2022-01-05",
                    }
                ),
                encoding="utf-8",
            )
            pd.DataFrame(
                [
                    {
                        "variant": "combined_cap40_confirmation1",
                        "period_label": "full",
                        "status": "completed",
                        "start_date": "2022-01-03",
                        "end_date": "2022-01-05",
                        "start_equity": 1_000_000,
                        "final_equity": 1_100_000,
                        "return_pct": 10.0,
                        "max_drawdown_pct": -1.0,
                        "trade_days": 1,
                        "total_transaction_cost": 100,
                    },
                    {
                        "variant": "combined_cap40_confirmation1",
                        "period_label": "2024_hard_gate",
                        "status": "empty",
                        "start_date": "",
                        "end_date": "",
                        "start_equity": "",
                        "final_equity": "",
                        "return_pct": "",
                        "max_drawdown_pct": "",
                        "trade_days": 0,
                        "total_transaction_cost": 0,
                    },
                ]
            ).to_csv(candidate / "period_performance_by_variant.csv", index=False)
            pd.DataFrame(
                [
                    {
                        "variant": "combined_cap40_confirmation1",
                        "date": "2022-01-03",
                        "period": "2022",
                        "target_weights": "{}",
                        "position_ticker": "cash",
                        "cash": 1_000_000,
                        "equity": 1_000_000,
                        "drawdown": 0.0,
                        "turnover": 0,
                        "transaction_cost": 0,
                        "action": "hold",
                    },
                    {
                        "variant": "combined_cap40_confirmation1",
                        "date": "2022-01-04",
                        "period": "2022",
                        "target_weights": '{"2330.TW": 1.0}',
                        "position_ticker": "2330.TW",
                        "cash": 0,
                        "equity": 1_050_000,
                        "drawdown": 0.0,
                        "turnover": 1_000_000,
                        "transaction_cost": 100,
                        "action": "rebalance",
                    },
                    {
                        "variant": "combined_cap40_confirmation1",
                        "date": "2022-01-05",
                        "period": "2022",
                        "target_weights": '{"2330.TW": 1.0}',
                        "position_ticker": "2330.TW",
                        "cash": 0,
                        "equity": 1_100_000,
                        "drawdown": 0.0,
                        "turnover": 0,
                        "transaction_cost": 0,
                        "action": "hold",
                    },
                ]
            ).to_csv(candidate / "daily_equity_by_variant.csv", index=False)
            pd.DataFrame(
                {
                    "variant": ["combined_cap40_confirmation1"],
                    "date": ["2022-01-04"],
                    "ticker": ["2330.TW"],
                    "action": ["buy"],
                    "costs": [100],
                }
            ).to_csv(candidate / "trade_ledger_by_variant.csv", index=False)
            pd.DataFrame(
                [
                    {
                        "variant": "current_formal_three_pool_baseline",
                        "period_label": "full",
                        "status": "completed",
                        "start_date": "2022-01-03",
                        "end_date": "2022-01-05",
                        "start_equity": 1_000_000,
                        "final_equity": 900_000,
                        "return_pct": -10.0,
                        "max_drawdown_pct": -10.0,
                        "trade_days": 0,
                        "total_transaction_cost": 0,
                    },
                    {
                        "variant": "pool1_only_formal_replay",
                        "period_label": "full",
                        "status": "completed",
                        "start_date": "2022-01-03",
                        "end_date": "2022-01-05",
                        "start_equity": 1_000_000,
                        "final_equity": 1_200_000,
                        "return_pct": 20.0,
                        "max_drawdown_pct": -2.0,
                        "trade_days": 1,
                        "total_transaction_cost": 200,
                    },
                ]
            ).to_csv(comparison / "period_performance_by_variant.csv", index=False)
            (label / "manifest.json").write_text(json.dumps({"opportunity_cost_label_active_in_trade_decision": False}), encoding="utf-8")
            (cooldown / "manifest.json").write_text(json.dumps({"execution_label_active_in_trade_decision": False}), encoding="utf-8")
            _price_csv(cache / "0050_TW.csv", [100, 105, 110])
            _price_csv(cache / "00631L_TW.csv", [100, 120, 130])

            run_formal_model_diagnosis_and_replay(
                absorption_dir=absorption,
                formal_candidate_dir=candidate,
                comparison_dir=comparison,
                opportunity_label_dir=label,
                cooldown_label_dir=cooldown,
                price_cache_dir=cache,
                output_dir=output,
            )

            manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["formal_model_target"], "combined_cap40_confirmation1_base")
            self.assertFalse(manifest["formal_model_changed_in_this_task"])
            self.assertFalse(manifest["trade_decision_changed_in_this_task"])
            self.assertFalse(manifest["pool3_shadow_used_as_formal"])
            self.assertFalse(manifest["0050x2_opportunity_label_active_in_trade_decision"])
            self.assertFalse(manifest["execution_label_active_in_trade_decision"])
            self.assertFalse(manifest["uses_forward_return_as_rule"])
            self.assertEqual(manifest["formal_same_day_full_return_pct"], 10.0)

            diagnostics = pd.read_csv(output / "report_only_diagnostics_inventory.csv")
            self.assertFalse(diagnostics["active_in_trade_decision"].any())
            self.assertFalse(diagnostics["used_in_performance"].any())

            benchmarks = pd.read_csv(output / "formal_vs_benchmarks_performance.csv")
            self.assertIn("0050正二_00631L", set(benchmarks["benchmark"]))


def _price_csv(path: Path, adj_close: list[float]) -> None:
    pd.DataFrame(
        {
            "date": ["2022-01-03", "2022-01-04", "2022-01-05"],
            "open": adj_close,
            "close": adj_close,
            "adj_close": adj_close,
        }
    ).to_csv(path, index=False)


if __name__ == "__main__":
    unittest.main()
