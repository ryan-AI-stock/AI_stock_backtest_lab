from __future__ import annotations

import unittest
import tempfile
from dataclasses import replace
from pathlib import Path

import pandas as pd

import test_paths  # noqa: F401

from backtest_lab.costs import TaiwanCostModel
from backtest_lab.frozen_strategy_monitor import (
    DETAIL_BOTTOM_Y,
    DETAIL_START_Y,
    FrozenStrategySignal,
    _action,
    _append_projection_row,
    _cash_account_reference,
    _detail_section_height,
    _detail_sections,
    _incomplete_tickers,
    _model_target_status,
    _paginate_detail_sections,
    _personal_exposure_summary,
    _report_mode,
    _report_mode_label,
    _ranking_rows,
    _score_band,
    _score_guide_lines,
    _target_is_actionable,
    _write_signal_pdf,
    attach_personal_portfolio,
)
from backtest_lab.portfolio_app import PortfolioStore
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

    def test_attach_personal_portfolio_adds_actual_exposure_summary(self) -> None:
        signal = _sample_signal()
        with tempfile.TemporaryDirectory() as tmp:
            store_path = Path(tmp) / "portfolio_store.json"
            store = PortfolioStore(store_path)
            store.replace_portfolio(
                user_id="default",
                cash_twd=50_000,
                positions=[{"ticker": "2454.TW", "shares": 10, "avg_cost": 4000}],
            )

            updated = attach_personal_portfolio(
                signal,
                portfolio_store=store_path,
                asset_types={"2454.TW": "stock"},
                cost_model=TaiwanCostModel(),
            )

        summary = _personal_exposure_summary(updated)
        self.assertIsNotNone(updated.personal_portfolio)
        self.assertEqual(_report_mode(updated), "personalized")
        self.assertIn("個人化版", _report_mode_label(updated))
        self.assertAlmostEqual(summary["target_actual_exposure"], 43_000 / 93_000)
        self.assertGreater(summary["target_gap_exposure"], 0.5)

    def test_report_mode_is_general_without_portfolio_file(self) -> None:
        signal = _sample_signal()

        self.assertEqual(_report_mode(signal), "general")
        self.assertIn("一般版", _report_mode_label(signal))

    def test_signal_pdf_is_written_as_image_backed_pdf(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "report.pdf"

            _write_signal_pdf(path, _sample_signal())

            content = path.read_bytes()
            self.assertTrue(content.startswith(b"%PDF"))
            self.assertIn(b"/Subtype /Image", content)

    def test_detail_pages_stay_above_footer_safe_area(self) -> None:
        signal = replace(
            _sample_signal(),
            personal_portfolio={
                "total_value_twd": 4_000_000,
                "cash_twd": 800_000,
                "market_value_twd": 3_200_000,
                "positions": [{"ticker": "2454.TW", "market_value_twd": 1_000_000}],
            },
            personal_recommendations=[
                {"action": "buy", "ticker": "2454.TW", "shares": 100, "reference_price": 4300.0},
                {"action": "sell", "ticker": "2330.TW", "shares": 200, "reference_price": 1200.0},
                {"action": "buy", "ticker": "6669.TW", "shares": 10, "reference_price": 5600.0},
                {"action": "hold", "ticker": "00631L.TW", "shares": 0, "reference_price": 380.0},
            ],
        )

        pages = _paginate_detail_sections(_detail_sections(signal))
        available_height = DETAIL_START_Y - DETAIL_BOTTOM_Y

        self.assertGreaterEqual(len(pages), 2)
        for page in pages:
            used_height = sum(_detail_section_height(lines) for _, lines in page)
            self.assertLessEqual(used_height, available_height)
        self.assertEqual(pages[-1][-1][0], "風險聲明")


def _sample_signal() -> FrozenStrategySignal:
    return FrozenStrategySignal(
        strategy_id="frozen_cycle_proven_top1_v1",
        signal_date="2026-06-05",
        execution_timing="下一個台股交易日，由投資人自行決定是否執行",
        market_regime="strong_bull",
        market_regime_label="強多頭",
        current_ticker="2454.TW",
        current_label="聯發科",
        current_exposure=1.0,
        target_ticker="2454.TW",
        target_label="聯發科",
        target_exposure=1.0,
        action="維持目前模型部位",
        target_is_actionable=True,
        model_target_status="有合格模型目標",
        cash_account_reference="若目前全現金且選擇跟隨模型，模型目標是聯發科，目標曝險約 100%。",
        attack_gate_active=True,
        attack_gate_ever_activated=True,
        risk_off_active=False,
        model_total_value_twd=1_000_000,
        close_prices={"2454.TW": 4300.0, "0050.TW": 60.0},
        ranking=[
            {
                "rank": 1,
                "ticker": "2454.TW",
                "label": "聯發科",
                "score": 0.9,
                "score_band": "極強勢",
                "role": "進攻候選",
            },
            {
                "rank": 2,
                "ticker": "0050.TW",
                "label": "0050",
                "score": 0.1,
                "score_band": "弱勢或未明顯領先",
                "role": "市場訊號/等待工具",
            },
        ],
        projected_trades=[],
    )


if __name__ == "__main__":
    unittest.main()
