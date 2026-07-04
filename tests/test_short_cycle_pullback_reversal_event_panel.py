import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

import test_paths  # noqa: F401

from backtest_lab.short_cycle_pullback_reversal_event_panel import run_short_cycle_pullback_reversal_event_panel


class ShortCyclePullbackReversalEventPanelTest(unittest.TestCase):
    def test_pool1b_repaired_cache_removes_missing_price_blocker_without_formalizing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            experiments = root / "experiments"
            preliminary = root / "preliminary"
            repair = root / "repair"
            output = root / "out"
            experiments.mkdir()
            preliminary.mkdir()
            repair.mkdir()

            _write_original_experiments(experiments)
            _write_preliminary_pool1b(preliminary)
            _write_repair_package(repair)

            run_short_cycle_pullback_reversal_event_panel(
                experiments_output=experiments,
                output_dir=output,
                pool1b_repair_output=repair,
                pool1b_preliminary_rerun_output=preliminary,
            )

            manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            self.assertTrue(manifest["pool1b_repaired_cache_used"])
            self.assertFalse(manifest["pool1b_repaired_cache_adjusted_close_available"])
            self.assertFalse(manifest["preliminary_rerun_used_as_formal_validation"])
            self.assertEqual(manifest["blocked_ticker_count"], 0)
            self.assertEqual(manifest["case_6488_two_event_count"], 1)
            self.assertFalse(manifest["formal_model_changed"])
            self.assertFalse(manifest["trade_decision_changed"])
            self.assertFalse(manifest["active_in_trade_decision"])

            panel = pd.read_csv(output / "short_cycle_pullback_reversal_event_panel.csv")
            case_6488 = panel[panel["ticker"].eq("6488.TWO")].iloc[0]
            self.assertTrue(bool(case_6488["source_price_cache_repaired"]))
            self.assertTrue(bool(case_6488["repaired_cache_available"]))
            self.assertFalse(bool(case_6488["adjusted_close_available"]))
            self.assertTrue(bool(case_6488["diagnostic_only"]))
            self.assertFalse(bool(case_6488["active_in_trade_decision"]))

            price = pd.read_csv(output / "price_readiness_by_ticker.csv")
            price_6488 = price[price["ticker"].eq("6488.TWO")].iloc[0]
            self.assertTrue(bool(price_6488["price_data_ready"]))
            self.assertTrue(pd.isna(price_6488["blocked_reason"]) or price_6488["blocked_reason"] == "")
            self.assertTrue(bool(price_6488["repaired_cache_available"]))


def _write_original_experiments(root: Path) -> None:
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "task_id": "TASK-BACKTEST-EXPERIMENTS-DYNAMIC-POOL1-SHORT-CYCLE-PULLBACK-REVERSAL-DIAGNOSTIC-001",
                "price_latest": "2026-06-29",
                "formal_overlay_latest": "2026-06-12",
            }
        ),
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            _event_row("2024-07-01", "strong_stock_ma20_pullback_reclaim", "2330.TW", "old_ai"),
            _event_row("2024-07-01", "strong_stock_ma20_pullback_reclaim", "6488.TWO", "pool1b"),
        ]
    ).to_csv(root / "pullback_reversal_event_panel.csv", index=False)
    pd.DataFrame([{"blocker": "missing_price_cache", "latest_price_date": "", "source": "pool1b", "ticker": "6488.TWO"}]).to_csv(
        root / "data_blockers.csv", index=False
    )
    pd.DataFrame().to_csv(root / "concentration_by_ticker_month_sector.csv", index=False)
    pd.DataFrame().to_csv(root / "formal_target_vs_pullback_candidate_opportunity.csv", index=False)
    pd.DataFrame().to_csv(root / "old_ai_pullback_case_study.csv", index=False)


def _write_preliminary_pool1b(root: Path) -> None:
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "status": "preliminary_non_production_waiting_core_event_panel",
                "production_grade_event_panel_used": False,
                "requires_core_event_panel_rerun": True,
            }
        ),
        encoding="utf-8",
    )
    row = _event_row("2024-07-02", "strong_stock_ma20_pullback_reclaim", "6488.TWO", "pool1b")
    row["adjusted_close_available"] = False
    row["is_material_layer"] = True
    pd.DataFrame([row]).to_csv(root / "pullback_reversal_event_panel.csv", index=False)


def _write_repair_package(root: Path) -> None:
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "price_cache_candidate_ready": True,
                "completed_ticker_count": 1,
                "latest_complete_date": "2026-07-02",
                "adjusted_close_available": False,
                "adjusted_close_boundary": "Official daily OHLCV is unadjusted; adjusted_close is intentionally empty and not synthesized.",
            }
        ),
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            {
                "ticker": "6488.TWO",
                "cache_compatible_path": str(root / "cache_compatible" / "6488_TWO.csv"),
                "legacy_repaired_cache_path": "",
                "row_count": 603,
                "first_date": "2024-01-02",
                "last_date": "2026-07-02",
                "adjusted_close_available": False,
            }
        ]
    ).to_csv(root / "cache_compatible_files_manifest.csv", index=False)
    pd.DataFrame(
        [
            {
                "ticker": "6488.TWO",
                "code": "6488",
                "name": "環球晶",
                "market": "TPEx",
                "requested_start": "2024-01-01",
                "first_date": "2024-01-02",
                "last_date": "2026-07-02",
                "row_count": 603,
                "adjusted_close_available": False,
                "coverage_ready": True,
                "blocked_reason": "",
            }
        ]
    ).to_csv(root / "coverage_by_ticker.csv", index=False)


def _event_row(date: str, variant: str, ticker: str, source: str) -> dict[str, object]:
    return {
        "date": date,
        "variant_id": variant,
        "event_status": "event_candidate",
        "ticker": ticker,
        "candidate_name": ticker,
        "candidate_source": source,
        "supply_chain_layer": "Semiconductor materials" if ticker == "6488.TWO" else "",
        "sector_code": "24",
        "sector_name": "半導體業",
        "formal_target": "00631L.TW",
        "formal_target_is_market_exposure": True,
        "close": 100,
        "dist_ma20_pct": -1,
        "dist_ma60_pct": 5,
        "dist_ma120_pct": 10,
        "ma60_slope_10d_pct": 1,
        "ma120_slope_20d_pct": 1,
        "rs_vs_0050_20d_pct": 2,
        "rs_vs_0050_60d_pct": 3,
        "drawdown_from_60d_high_pct": -5,
        "peer_recovery_count": 1,
        "market_support_ok": True,
        "diagnostic_only": True,
        "forward_return_20d_pct": 1,
        "forward_path_mdd_20d_pct": -2,
        "forward_return_40d_pct": 2,
        "forward_path_mdd_40d_pct": -3,
        "forward_return_60d_pct": 3,
        "forward_path_mdd_60d_pct": -4,
    }


if __name__ == "__main__":
    unittest.main()
