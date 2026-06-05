from __future__ import annotations

import unittest

import pandas as pd

import test_paths  # noqa: F401

from backtest_lab.frozen_strategy_monitor import (
    _action,
    _append_projection_row,
    _cash_account_reference,
    _incomplete_tickers,
    _model_target_status,
    _ranking_rows,
    _score_band,
    _score_guide_lines,
    _target_is_actionable,
)
from backtest_lab.regime_mode_switch import frozen_cycle_proven_top1_v1_variant


class FrozenStrategyMonitorTest(unittest.TestCase):
    def test_frozen_variant_separates_etfs_from_attack_candidates(self) -> None:
        variant = frozen_cycle_proven_top1_v1_variant()

        self.assertEqual(variant.name, "frozen_cycle_proven_top1_v1")
        self.assertEqual(variant.attack_selection_exclude_tickers, ("0050.TW", "00631L.TW"))
        self.assertEqual(variant.market_risk_off_exposure, 0.25)
        self.assertEqual(variant.defense_anchor_ticker, "00631L.TW")

    def test_projection_row_uses_signal_close_without_corporate_actions(self) -> None:
        signal_date = pd.Timestamp("2026-05-29")
        projection_date = pd.Timestamp("2026-06-01")
        frame = pd.DataFrame(
            {
                "open": [100.0],
                "close": [105.0],
                "adj_close": [105.0],
                "dividend": [2.0],
                "stock_split": [4.0],
            },
            index=[signal_date],
        )

        projected = _append_projection_row(frame, signal_date, projection_date)

        self.assertEqual(projected.loc[projection_date, "close"], 105.0)
        self.assertEqual(projected.loc[projection_date, "dividend"], 0.0)
        self.assertEqual(projected.loc[projection_date, "stock_split"], 0.0)

    def test_incomplete_tickers_requires_all_nine_assets_on_signal_date(self) -> None:
        signal_date = "2026-05-29"
        complete = pd.DataFrame({"open": [1], "close": [1], "adj_close": [1]}, index=[pd.Timestamp(signal_date)])
        stale = pd.DataFrame({"open": [1], "close": [1], "adj_close": [1]}, index=[pd.Timestamp("2026-05-28")])

        missing = _incomplete_tickers({"0050.TW": complete, "00631L.TW": stale}, signal_date)

        self.assertEqual(missing, ["00631L.TW"])

    def test_action_distinguishes_hold_rotate_and_cash(self) -> None:
        self.assertEqual(_action("2454.TW", "2454.TW", 0.99, 1.0), "維持目前模型部位")
        self.assertEqual(_action("2454.TW", "6669.TW", 1.0, 1.0), "模型輪動至新標的")
        self.assertEqual(_action("2454.TW", "cash", 1.0, 0.0), "模型轉為現金觀察")

    def test_cash_account_reference_distinguishes_ranking_from_actionable_target(self) -> None:
        self.assertEqual(_model_target_status("cash", 0.0), "沒有合格持股目標，模型目標為現金觀察")
        self.assertEqual(_model_target_status("2454.TW", 1.0), "有合格模型目標")
        self.assertFalse(_target_is_actionable("cash", 0.0))
        self.assertTrue(_target_is_actionable("2454.TW", 1.0))
        self.assertIn("模型不建立新部位", _cash_account_reference("cash", "現金", 0.0))
        self.assertIn("聯發科", _cash_account_reference("2454.TW", "聯發科", 1.0))

    def test_score_band_and_guide_make_ranking_non_actionable_by_itself(self) -> None:
        self.assertEqual(_score_band(0.81), "極強勢")
        self.assertEqual(_score_band(0.5), "強勢觀察")
        self.assertEqual(_score_band(0.25), "中性偏強")
        self.assertEqual(_score_band(0), "弱勢或未明顯領先")
        self.assertEqual(_score_band(-0.01), "弱勢/風險偏高")
        self.assertIn("不是單看分數", _score_guide_lines()[0])

    def test_ranking_rows_include_score_band(self) -> None:
        rows = _ranking_rows({"2454.TW": 0.9, "0050.TW": 0.1}, {"2454.TW": "聯發科", "0050.TW": "0050"})

        self.assertEqual(rows[0]["ticker"], "2454.TW")
        self.assertEqual(rows[0]["score_band"], "極強勢")
        self.assertEqual(rows[1]["role"], "市場訊號/等待工具")


if __name__ == "__main__":
    unittest.main()
