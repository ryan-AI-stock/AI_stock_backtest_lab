from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from backtest_lab.live_path_tracker import run_live_path_tracker


class LivePathTrackerTests(unittest.TestCase):
    def test_runner_writes_report_outputs_without_changing_model(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scenario_dir = root / "scenario"
            output_dir = root / "out"
            _write_scenario_fixture(scenario_dir)

            manifest = run_live_path_tracker(
                scenario_dir=scenario_dir,
                output_dir=output_dir,
                report_date="2026-06-16",
                actual_portfolio_value=1_000_000,
                actual_holding_ticker="00631L.TW",
                actual_holding_name="0050正二",
                model_target_ticker="00631L.TW",
                model_target_name="0050正二",
            )

            self.assertFalse(manifest["model_changed"])
            self.assertFalse(manifest["active_in_trade_decision"])
            self.assertTrue(manifest["not_forecast"])
            self.assertTrue(manifest["not_investment_advice"])
            self.assertTrue((output_dir / "AI模型實戰路徑追蹤報告_最新版.pdf").exists())
            self.assertTrue((output_dir / "report.md").exists())
            self.assertTrue((output_dir / "live_path_tracking.csv").exists())
            self.assertTrue((output_dir / "manifest.json").exists())

            tracking = pd.read_csv(output_dir / "live_path_tracking.csv")
            self.assertEqual(tracking.iloc[-1]["deviation_status"], "normal_range")
            self.assertEqual(tracking.iloc[-1]["nearest_scenario"], "p50")

    def test_persistent_weak_path_becomes_model_assumption_warning(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scenario_dir = root / "scenario"
            output_dir = root / "out"
            prior_tracking = root / "prior.csv"
            _write_scenario_fixture(scenario_dir)
            pd.DataFrame(
                [
                    {
                        "report_date": f"2026-06-{day:02d}",
                        "scenario_step": idx,
                        "actual_portfolio_value": 970_000,
                        "p25": 980_000,
                    }
                    for idx, day in enumerate(range(11, 15), start=0)
                ]
            ).to_csv(prior_tracking, index=False)

            run_live_path_tracker(
                scenario_dir=scenario_dir,
                output_dir=output_dir,
                report_date="2026-06-16",
                actual_portfolio_value=970_000,
                actual_tracking_csv=prior_tracking,
            )

            tracking = pd.read_csv(output_dir / "live_path_tracking.csv")
            self.assertEqual(tracking.iloc[-1]["deviation_status"], "model_assumption_warning")
            events = pd.read_csv(output_dir / "deviation_events.csv")
            self.assertIn("model_assumption_warning", events["deviation_status"].tolist())


def _write_scenario_fixture(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    rows = []
    values_by_path = {
        "analog_a": [1_000_000, 1_000_000, 1_000_000],
        "analog_b": [1_000_000, 1_020_000, 1_040_000],
        "analog_c": [1_000_000, 950_000, 930_000],
        "analog_d": [1_000_000, 1_080_000, 1_120_000],
        "analog_e": [1_000_000, 980_000, 960_000],
    }
    for analog, values in values_by_path.items():
        for step, value in enumerate(values):
            rows.append(
                {
                    "analog_start": analog,
                    "step": step,
                    "historical_date": f"2025-01-{step + 2:02d}",
                    "projected_value": value,
                    "return_pct": value / 1_000_000 - 1,
                    "current_ticker": "00631L.TW",
                    "regime": "strong_bull",
                    "match_tier": "strict",
                    "similarity_score": 1.0,
                }
            )
    pd.DataFrame(rows).to_csv(root / "scenario_paths.csv", index=False)
    pd.DataFrame(
        [
            {
                "scenario": "median_p50",
                "final_value": 1_000_000,
                "total_return_pct": 0,
                "max_drawdown_pct": 0,
                "analog_count": 5,
                "note": "fixture",
            }
        ]
    ).to_csv(root / "scenario_summary.csv", index=False)
    pd.DataFrame([{"analog_start": "analog_a"}]).to_csv(root / "analog_cases.csv", index=False)
    (root / "summary.json").write_text(
        json.dumps(
            {
                "status": "ready",
                "analog_count": 5,
                "scenario_inputs": {
                    "signal_date": "2026-06-12",
                    "scenario_start": "2026-06-15",
                    "scenario_end": "2026-12-31",
                    "initial_capital": 1_000_000,
                    "target_ticker": "00631L.TW",
                    "target_label": "0050正二",
                    "current_regime": "strong_bull",
                    "current_mode": "0050_defense",
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    unittest.main()
