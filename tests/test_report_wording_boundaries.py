from __future__ import annotations

import unittest
from types import SimpleNamespace

import test_paths  # noqa: F401

from backtest_lab.frozen_report_content import (
    markdown_report,
    personal_pdf_section,
    report_mode_label,
    shadow_mode_pdf_section,
)
from backtest_lab.frozen_report_pdf import detail_sections


class ReportWordingBoundariesTest(unittest.TestCase):
    def test_markdown_report_uses_observation_wording_and_flags_00631l_as_leveraged(self) -> None:
        report = markdown_report(
            _signal(),
            report_name="AI股票最佳策略每日觀察報告",
            report_variant_label="最佳版 v20260605",
            score_guide_lines=["強弱排名不是買入資格。"],
        )

        self.assertNotIn("每日 AI 輔助操作建議", report)
        self.assertIn("每日 AI 輔助市場觀察與紀律提醒", report)
        self.assertIn("0050正二是槓桿大盤訊號、積極等待與曝險工具", report)
        self.assertNotIn("低風險防守資產", report)

    def test_pdf_detail_sections_use_observation_wording(self) -> None:
        sections = detail_sections(
            _signal(),
            personal_pdf_section=personal_pdf_section,
            shadow_pdf_section=shadow_mode_pdf_section,
        )
        flattened = "\n".join(line for _, lines in sections for line in lines)

        self.assertNotIn("每日 AI 輔助操作建議", flattened)
        self.assertIn("每日 AI 輔助市場觀察與紀律提醒", flattened)


def _signal() -> SimpleNamespace:
    return SimpleNamespace(
        signal_date="2026-06-18",
        execution_timing="下一交易日開盤",
        market_regime_label="強多頭",
        action="維持目前模型部位",
        model_target_status="有合格模型目標",
        current_label="0050正二",
        current_exposure=1.0,
        target_label="0050正二",
        target_ticker="00631L.TW",
        target_exposure=1.0,
        cash_account_reference="全現金帳戶以模型目標狀態作為觀察參考。",
        attack_gate_active=False,
        attack_gate_ever_activated=True,
        risk_off_active=False,
        model_total_value_twd=1_000_000,
        projected_trades=[],
        shadow_modes=[],
        personal_portfolio=None,
        personal_recommendations=[],
        ranking=[
            {
                "rank": 1,
                "label": "0050正二",
                "ticker": "00631L.TW",
                "score": 0.37,
                "score_band": "中性偏強",
                "role": "市場訊號/等待工具",
            }
        ],
    )


if __name__ == "__main__":
    unittest.main()
