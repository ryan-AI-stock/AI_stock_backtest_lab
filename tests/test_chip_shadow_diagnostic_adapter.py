import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from backtest_lab.chip_shadow_diagnostic_adapter import (
    EXCLUDED_SIGNALS,
    FORBIDDEN_WORDING,
    build_chip_diagnostic_panel,
    build_manifest,
    build_signal_summary,
    diagnostic_wording,
    run_chip_shadow_diagnostic_adapter,
    validate_wording,
)


class ChipShadowDiagnosticAdapterTests(unittest.TestCase):
    def test_builds_shadow_diagnostic_flags_without_formal_decision(self):
        panel = build_chip_diagnostic_panel(_sample_institutional_frame())

        self.assertTrue((panel["active_in_trade_decision"] == False).all())  # noqa: E712
        self.assertNotIn("day_ratio_top10", panel.columns)
        self.assertNotIn("margin_and_day_overheat_flag", panel.columns)

        positive = panel.loc[panel["ticker"] == "2454.TW"].iloc[0]
        self.assertTrue(bool(positive["inst_total_net_positive"]))
        self.assertTrue(bool(positive["foreign_trust_sync_buy"]))
        self.assertFalse(bool(positive["h1_negative_or_h2_sell_pressure"]))

        risk = panel.loc[panel["ticker"] == "2308.TW"].iloc[0]
        self.assertTrue(bool(risk["foreign_sell_ge3"]))
        self.assertTrue(bool(risk["foreign_sell_ge5"]))
        self.assertTrue(bool(risk["foreign_trust_sync_sell"]))
        self.assertTrue(bool(risk["inst_total_net_negative"]))
        self.assertTrue(bool(risk["h1_negative_or_h2_sell_pressure"]))

    def test_summary_keeps_h3_and_valuation_excluded(self):
        panel = build_chip_diagnostic_panel(_sample_institutional_frame())
        summary = build_signal_summary(panel)
        excluded = summary.loc[summary["signal_id"].isin(EXCLUDED_SIGNALS)]

        self.assertEqual(set(excluded["status"]), {"excluded_from_core_adapter"})
        self.assertTrue((excluded["active_in_trade_decision"] == False).all())  # noqa: E712
        self.assertEqual(int(excluded["triggered_rows"].sum()), 0)

    def test_manifest_marks_shadow_or_diagnostic_only(self):
        panel = build_chip_diagnostic_panel(_sample_institutional_frame())
        summary = build_signal_summary(panel)
        manifest = build_manifest(
            sources=_sources(),
            panel=panel,
            summary=summary,
            start_date="2024-01-02",
            end_date="2024-01-03",
        )

        self.assertEqual(manifest["decision_layer"], "shadow_or_diagnostic")
        self.assertFalse(manifest["active_in_trade_decision"])
        self.assertFalse(manifest["formal_trade_decision_changed"])
        self.assertFalse(manifest["frozen_baseline_changed"])
        self.assertEqual(manifest["valuation_status"], "excluded_pit_blocked")

    def test_run_rejects_valuation_source(self):
        with tempfile.TemporaryDirectory() as temp:
            institutional = Path(temp) / "institutional.csv"
            margin = Path(temp) / "margin.csv"
            day_trading = Path(temp) / "day.csv"
            _sample_institutional_frame().to_csv(institutional, index=False)
            margin.write_text("date,ticker\n2024-01-02,2454.TW\n", encoding="utf-8")
            day_trading.write_text("date,ticker\n2024-01-02,2454.TW\n", encoding="utf-8")

            with self.assertRaises(ValueError):
                run_chip_shadow_diagnostic_adapter(
                    institutional_source=str(institutional),
                    margin_source=str(margin),
                    day_trading_source=str(day_trading),
                    valuation_source=str(Path(temp) / "valuation.csv"),
                    output_dir=str(Path(temp) / "out"),
                    start_date="2024-01-02",
                    end_date="2024-01-03",
                )

    def test_run_writes_replayable_outputs(self):
        with tempfile.TemporaryDirectory() as temp:
            institutional = Path(temp) / "institutional.csv"
            margin = Path(temp) / "margin.csv"
            day_trading = Path(temp) / "day.csv"
            output = Path(temp) / "out"
            _sample_institutional_frame().to_csv(institutional, index=False)
            margin.write_text("date,ticker\n2024-01-02,2454.TW\n", encoding="utf-8")
            day_trading.write_text("date,ticker\n2024-01-02,2454.TW\n", encoding="utf-8")

            result = run_chip_shadow_diagnostic_adapter(
                institutional_source=str(institutional),
                margin_source=str(margin),
                day_trading_source=str(day_trading),
                valuation_source=None,
                output_dir=str(output),
                start_date="2024-01-02",
                end_date="2024-01-03",
            )

            self.assertEqual(result, output)
            self.assertTrue((output / "chip_diagnostic_panel.csv").exists())
            self.assertTrue((output / "chip_diagnostic_summary.csv").exists())
            self.assertTrue((output / "chip_shadow_diagnostic_manifest.json").exists())
            self.assertTrue((output / "run_log.csv").exists())
            self.assertEqual((output / "completed.txt").read_text(encoding="utf-8"), "completed")
            manifest = json.loads((output / "chip_shadow_diagnostic_manifest.json").read_text(encoding="utf-8"))
            self.assertFalse(manifest["active_in_trade_decision"])

    def test_wording_stays_observational(self):
        text = diagnostic_wording(
            attack_confirmation_score=1,
            sell_pressure_warning_score=2,
            shadow_risk=True,
        )
        validate_wording(text)
        for word in FORBIDDEN_WORDING:
            self.assertNotIn(word, text)


def _sources():
    from backtest_lab.chip_shadow_diagnostic_adapter import ChipDiagnosticSources

    return ChipDiagnosticSources(
        institutional_source="institutional.csv",
        margin_source="margin.csv",
        day_trading_source="day.csv",
        valuation_source=None,
    )


def _sample_institutional_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "date": pd.Timestamp("2024-01-02"),
                "ticker": "2454.TW",
                "name": "聯發科",
                "foreign_net_buy_shares": 100.0,
                "investment_trust_net_buy_shares": 50.0,
                "dealer_net_buy_shares": 10.0,
                "foreign_consecutive_buy_days": 2,
                "foreign_consecutive_sell_days": 0,
                "trust_consecutive_buy_days": 2,
                "trust_consecutive_sell_days": 0,
            },
            {
                "date": pd.Timestamp("2024-01-02"),
                "ticker": "2308.TW",
                "name": "台達電",
                "foreign_net_buy_shares": -200.0,
                "investment_trust_net_buy_shares": -30.0,
                "dealer_net_buy_shares": -5.0,
                "foreign_consecutive_buy_days": 0,
                "foreign_consecutive_sell_days": 5,
                "trust_consecutive_buy_days": 0,
                "trust_consecutive_sell_days": 3,
            },
            {
                "date": pd.Timestamp("2024-01-02"),
                "ticker": "0050.TW",
                "name": "元大台灣50",
                "foreign_net_buy_shares": 1000.0,
                "investment_trust_net_buy_shares": 0.0,
                "dealer_net_buy_shares": 0.0,
                "foreign_consecutive_buy_days": 1,
                "foreign_consecutive_sell_days": 0,
                "trust_consecutive_buy_days": 0,
                "trust_consecutive_sell_days": 0,
            },
        ]
    )


if __name__ == "__main__":
    unittest.main()
