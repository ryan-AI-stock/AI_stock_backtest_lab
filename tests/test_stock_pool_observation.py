from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

import test_paths  # noqa: F401

from backtest_lab.decision_layers import CANDIDATE_SOURCE, DATA_READINESS
from backtest_lab.stock_pool_observation import (
    _attach_cashflow_report_boundary,
    _attach_chip_context_report_boundary,
    _attach_live_risk_regime_warning_boundary,
    _attach_target_stability_warning_boundary,
    _cashflow_report_boundary,
    _chip_context_report_boundary,
    _live_risk_regime_warning_boundary,
    _load_observation_price_frames,
    _resolve_dynamic_observation_pool,
    _sanitize_visible_report_reason,
    _set_formal_report_readiness,
    _top_candidate_rows,
    _target_stability_warning_boundary,
    _user_facing_candidate_reason,
    _wrap_text_lines,
    build_dispatched_stock_pool_observation,
    build_stock_pool_observation,
    markdown_observation_batch_report,
    run_stock_pool_observation_batch,
    write_stock_pool_observation,
)
from backtest_lab.stock_pool_store import symbol_entry
from backtest_lab.valuation_source import ValuationSignal


class StockPoolObservationTest(unittest.TestCase):
    def test_visible_reason_wording_is_user_facing_chinese(self) -> None:
        self.assertEqual(
            _sanitize_visible_report_reason("No market data for signal date 2026-06-26; latest available is 2026-06-25"),
            "資料不足：2026-06-26 還沒有完整市場資料，目前可用到 2026-06-25。",
        )
        readable = _user_facing_candidate_reason("大型廣度池 v1：base=Y；60日相對0050超額=2.3%(Y)；20/60動能品質=N")
        self.assertIn("基本條件通過", readable)
        self.assertIn("60日表現相對0050=2.3%，通過", readable)
        self.assertIn("20日與60日動能品質：未通過", readable)
        self.assertNotIn("base=Y", readable)
        self.assertNotIn("(Y)", readable)

    def test_pdf_text_wrap_preserves_full_visible_text(self) -> None:
        text = "沒有合格持股目標，模型目標為現金"
        lines = _wrap_text_lines(text, max_units=15)

        self.assertGreater(len(lines), 1)
        self.assertEqual("".join(lines), text)
        self.assertTrue(all("…" not in line for line in lines))

    def test_cashflow_boundary_uses_150k_report_only_reference(self) -> None:
        manifest: dict[str, object] = {}
        _attach_cashflow_report_boundary(manifest)

        self.assertEqual(manifest["cashflow_objective_capital_twd"], 4_000_000)
        self.assertEqual(manifest["cashflow_monthly_target_twd"], 150_000)
        self.assertEqual(manifest["cashflow_target_source"], "user_updated_2026_06_27")
        self.assertEqual(manifest["cashflow_policy_reference"], "capped_profit_withdrawal_150k")
        self.assertEqual(manifest["cashflow_cash_buffer_required_twd"], 2_400_000)
        self.assertEqual(manifest["cashflow_boundary"], "report_only")
        self.assertFalse(manifest["cashflow_active_in_trade_decision"])
        self.assertFalse(manifest["formal_model_changed"])
        self.assertFalse(manifest["trade_decision_changed"])

    def test_cashflow_visible_wording_avoids_income_promise(self) -> None:
        cashflow = _cashflow_report_boundary()
        manifest = {
            "signal_date": "2026-06-26",
            "report_ready": True,
            "report_wording_boundary": {"formal_baseline": {"description": "主攻池優先，確認池做風險確認"}},
            "cashflow_report_boundary": cashflow,
            "generated": [],
        }
        markdown = markdown_observation_batch_report(manifest, [])

        self.assertIn("月生活費目標上限：150,000 元", markdown)
        self.assertIn("外部現金緩衝需求：2,400,000 元", markdown)
        self.assertIn("完整領到15萬的月份約45.28%", markdown)
        self.assertIn("舊20萬高壓測試", markdown)
        for forbidden in ("穩定月領", "保證收入", "固定可提領", "report-only"):
            self.assertNotIn(forbidden, markdown)

    def test_target_stability_warning_boundary_is_report_only(self) -> None:
        manifest = _target_stability_manifest(score_a=1.00, score_b=0.94)
        _attach_target_stability_warning_boundary(manifest)

        self.assertEqual(manifest["target_stability_warning_state"], "low_score_margin_watch")
        self.assertTrue(manifest["low_score_gap_proxy_flag"])
        self.assertFalse(manifest["target_stability_warning_active_in_trade_decision"])
        self.assertEqual(manifest["target_stability_warning_boundary"], "report_only")
        self.assertFalse(manifest["formal_model_changed"])
        self.assertFalse(manifest["trade_decision_changed"])
        self.assertFalse(manifest["active_in_trade_decision"])
        self.assertIn("不是正式候選排序契約", manifest["target_stability_proxy_contract"])

    def test_target_stability_ignores_pool2_disagreement_and_any_low_confidence(self) -> None:
        manifest = _target_stability_manifest(score_a=1.00, score_b=0.70)
        manifest["pool2_disagreement"] = True
        manifest["any_low_confidence_warning"] = True

        warning = _target_stability_warning_boundary(manifest)

        self.assertEqual(warning["target_stability_warning_state"], "mixed_or_insufficient")
        self.assertFalse(warning["pool2_disagreement_negative_warning_used"])
        self.assertFalse(warning["any_low_confidence_warning_used"])

    def test_target_stability_visible_wording_keeps_proxy_boundary(self) -> None:
        manifest = _target_stability_manifest(score_a=1.00, score_b=0.70)
        _attach_target_stability_warning_boundary(manifest)
        manifest.update(
            {
                "signal_date": "2026-06-26",
                "report_ready": True,
                "report_wording_boundary": {"formal_baseline": {"description": "主攻池優先，確認池做風險確認"}},
                "cashflow_report_boundary": _cashflow_report_boundary(),
                "generated": manifest["generated"],
            }
        )
        markdown = markdown_observation_batch_report(manifest, [])

        self.assertIn("標的穩定度提醒（僅供診斷）", markdown)
        self.assertIn("不是正式候選排序契約", markdown)
        self.assertIn("不代表換倉指令", markdown)
        for forbidden in ("應賣出", "禁止換倉", "明天一定轉弱", "target_drop_from_top3", "pool_signal_panel", "score margin contract", "target drop proxy"):
            self.assertNotIn(forbidden, markdown)

    def test_live_risk_regime_warning_boundary_is_report_only(self) -> None:
        manifest = _live_risk_manifest(risk_off_active=True, attack_gate_active=False)
        _attach_live_risk_regime_warning_boundary(manifest)

        self.assertEqual(manifest["live_risk_regime_state"], "risk_off_watch")
        self.assertEqual(manifest["live_risk_regime_boundary"], "report_only")
        self.assertEqual(manifest["live_risk_regime_breadth_readiness"], "breadth_not_ready")
        self.assertFalse(manifest["live_risk_regime_active_in_trade_decision"])
        self.assertFalse(manifest["formal_model_changed"])
        self.assertFalse(manifest["trade_decision_changed"])
        self.assertFalse(manifest["active_in_trade_decision"])

    def test_live_risk_regime_marks_data_insufficient(self) -> None:
        warning = _live_risk_regime_warning_boundary({"generated": [], "signal_date": "2026-06-26"})

        self.assertEqual(warning["live_risk_regime_state"], "data_insufficient")
        self.assertEqual(warning["live_risk_regime_boundary"], "report_only")
        self.assertFalse(warning["live_risk_regime_active_in_trade_decision"])

    def test_live_risk_regime_visible_wording_is_not_trade_instruction(self) -> None:
        manifest = _live_risk_manifest(risk_off_active=False, attack_gate_active=False)
        _attach_live_risk_regime_warning_boundary(manifest)
        manifest.update(
            {
                "signal_date": "2026-06-26",
                "report_ready": True,
                "report_wording_boundary": {"formal_baseline": {"description": "主攻池優先，確認池做風險確認"}},
                "cashflow_report_boundary": _cashflow_report_boundary(),
                "target_stability_warning": _target_stability_warning_boundary(manifest),
                "generated": manifest["generated"],
            }
        )
        markdown = markdown_observation_batch_report(manifest, [])

        self.assertIn("市場風險環境提醒（僅供診斷）", markdown)
        self.assertIn("市場環境偏弱", markdown)
        self.assertIn("市場廣度資料尚未納入正式契約", markdown)
        self.assertIn("目前沒有啟用正式降曝險規則", markdown)
        for forbidden in ("應降曝險", "禁止買進", "系統已切換防守", "此訊號會提升收益"):
            self.assertNotIn(forbidden, markdown)

    def test_chip_context_boundary_is_report_only_and_discloses_coverage(self) -> None:
        manifest = _chip_context_manifest(
            signal_date="2026-06-26",
            bullish=1.0,
            institutional_risk=0.0,
            flow_risk=0.0,
            coverage_end="2026-06-26",
        )
        _attach_chip_context_report_boundary(manifest)

        self.assertEqual(manifest["chip_context_state"], "h1_positive_context")
        self.assertEqual(manifest["chip_data_coverage_end"], "2026-06-26")
        self.assertFalse(manifest["chip_neutral_reference_available"])
        self.assertFalse(manifest["chip_context_active_in_trade_decision"])
        self.assertEqual(manifest["chip_context_boundary"], "report_only")
        self.assertFalse(manifest["formal_model_changed"])
        self.assertFalse(manifest["trade_decision_changed"])
        self.assertFalse(manifest["active_in_trade_decision"])

    def test_chip_context_h1_h2_states_do_not_become_rules(self) -> None:
        h1 = _chip_context_report_boundary(_chip_context_manifest(signal_date="2026-05-24", bullish=1.0, institutional_risk=0.0, flow_risk=0.0, coverage_end="2026-05-24"))
        h2 = _chip_context_report_boundary(_chip_context_manifest(signal_date="2026-05-24", bullish=0.0, institutional_risk=1.0, flow_risk=1.0, coverage_end="2026-05-24"))

        self.assertEqual(h1["chip_context_state"], "h1_positive_context")
        self.assertEqual(h2["chip_context_state"], "h2_sell_pressure_observation")
        self.assertIn("不提高正式權重", h1["chip_context_reason"])
        self.assertIn("不支持作為正式否決或降權", h2["chip_context_reason"])
        self.assertFalse(h1["chip_context_active_in_trade_decision"])
        self.assertFalse(h2["chip_context_active_in_trade_decision"])

    def test_chip_context_stale_data_blocks_formal_report_readiness(self) -> None:
        manifest = _chip_context_manifest(
            signal_date="2026-06-26",
            bullish=1.0,
            institutional_risk=0.0,
            flow_risk=0.0,
            coverage_end="2026-05-26",
        )
        _attach_chip_context_report_boundary(manifest)
        manifest.update(
            {
                "report_ready": True,
                "require_fresh_institutional_flow": True,
                "report_wording_boundary": {"formal_baseline": {"description": "主攻池優先，確認池做風險確認"}},
                "cashflow_report_boundary": _cashflow_report_boundary(),
                "target_stability_warning": _target_stability_warning_boundary(manifest),
                "live_risk_regime_warning": _live_risk_regime_warning_boundary(manifest),
                "generated": manifest["generated"],
            }
        )
        _set_formal_report_readiness(manifest)
        markdown = markdown_observation_batch_report(manifest, [])

        self.assertIn("籌碼背景觀察（僅供診斷）", markdown)
        self.assertIn("籌碼資料截止日：2026-05-26", markdown)
        self.assertIn("停止發布，等待完整資料", markdown)
        self.assertFalse(manifest["formal_report_ready"])
        self.assertEqual(manifest["chip_context_state"], "chip_data_insufficient")

    def test_chip_context_visible_wording_is_not_trade_instruction_when_fresh(self) -> None:
        manifest = _chip_context_manifest(signal_date="2026-05-24", bullish=1.0, institutional_risk=0.0, flow_risk=0.0, coverage_end="2026-05-24")
        _attach_chip_context_report_boundary(manifest)
        manifest.update(
            {
                "report_ready": True,
                "report_wording_boundary": {"formal_baseline": {"description": "主攻池優先，確認池做風險確認"}},
                "cashflow_report_boundary": _cashflow_report_boundary(),
                "target_stability_warning": _target_stability_warning_boundary(manifest),
                "live_risk_regime_warning": _live_risk_regime_warning_boundary(manifest),
                "generated": manifest["generated"],
            }
        )
        markdown = markdown_observation_batch_report(manifest, [])

        self.assertIn("籌碼背景觀察（僅供診斷）", markdown)
        self.assertIn("籌碼資料截止日：2026-05-24", markdown)
        self.assertIn("中性對照組：目前不可用", markdown)
        self.assertIn("籌碼賣壓不作正式否決", markdown)
        for forbidden in ("籌碼確認買進", "籌碼賣壓否決", "法人買超提高權重", "賣壓觸發降權", "H1 positive", "H2 sell pressure"):
            self.assertNotIn(forbidden, markdown)

    def test_price_loader_uses_current_cache_when_refresh_fails(self) -> None:
        dates = pd.bdate_range("2025-04-01", periods=320)
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp) / "cache"
            cache_dir.mkdir()
            _trend_frame(dates, start=80, step=0.2, volume=5_000_000).reset_index(names="date").to_csv(
                cache_dir / "6919_TW.csv",
                index=False,
            )

            with patch("backtest_lab.stock_pool_observation.download_yfinance_prices", side_effect=ValueError("refresh failed")):
                prices, missing = _load_observation_price_frames(
                    tickers=["6919.TW"],
                    start_date="2020-01-02",
                    end_date=dates[-1].strftime("%Y-%m-%d"),
                    cache_dir=cache_dir,
                )

            self.assertIn("6919.TW", prices)
            self.assertEqual(missing, [])
            self.assertEqual(prices["6919.TW"].index.max(), dates[-1])

    def test_price_loader_fills_signal_date_from_twse_when_yfinance_lags(self) -> None:
        dates = pd.bdate_range("2025-04-01", periods=320)
        lagged = _trend_frame(dates[:-1], start=80, step=0.2, volume=5_000_000)

        def fake_fill(prices, signal_date, tickers):
            filled = dict(prices)
            for ticker in tickers:
                frame = filled[ticker].copy()
                frame.loc[pd.Timestamp(signal_date)] = {
                    "open": 150.0,
                    "high": 152.0,
                    "low": 149.0,
                    "close": 151.0,
                    "adj_close": 151.0,
                    "volume": 10_000_000,
                    "dividend": 0.0,
                    "stock_split": 0.0,
                }
                filled[ticker] = frame.sort_index()
            return filled

        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp) / "cache"
            with patch("backtest_lab.stock_pool_observation.download_yfinance_prices", return_value={"0050.TW": lagged}), patch(
                "backtest_lab.stock_pool_observation.fill_signal_date_from_twse",
                side_effect=fake_fill,
            ):
                prices, missing = _load_observation_price_frames(
                    tickers=["0050.TW"],
                    start_date="2020-01-02",
                    end_date=dates[-1].strftime("%Y-%m-%d"),
                    cache_dir=cache_dir,
                )

            self.assertEqual(missing, [])
            self.assertIn(dates[-1], prices["0050.TW"].index)
            self.assertEqual(float(prices["0050.TW"].loc[dates[-1], "close"]), 151.0)
            self.assertTrue((cache_dir / "0050_TW.csv").exists())

    def test_price_loader_recovers_lagged_history_before_twse_fill(self) -> None:
        dates = pd.bdate_range("2025-04-01", periods=320)
        lagged = _trend_frame(dates[:-1], start=80, step=0.2, volume=5_000_000)

        def fake_download(*, tickers, start_date, end_date, cache_dir, allow_edge_gap):
            if not allow_edge_gap:
                raise ValueError("strict signal date missing")
            return {tickers[0]: lagged}

        def fake_fill(prices, signal_date, tickers):
            filled = dict(prices)
            for ticker in tickers:
                frame = filled[ticker].copy()
                frame.loc[pd.Timestamp(signal_date)] = {
                    "open": 150.0,
                    "high": 152.0,
                    "low": 149.0,
                    "close": 151.0,
                    "adj_close": 151.0,
                    "volume": 10_000_000,
                    "dividend": 0.0,
                    "stock_split": 0.0,
                }
                filled[ticker] = frame.sort_index()
            return filled

        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp) / "cache"
            with patch("backtest_lab.stock_pool_observation.download_yfinance_prices", side_effect=fake_download), patch(
                "backtest_lab.stock_pool_observation.fill_signal_date_from_twse",
                side_effect=fake_fill,
            ):
                prices, missing = _load_observation_price_frames(
                    tickers=["0050.TW"],
                    start_date="2020-01-02",
                    end_date=dates[-1].strftime("%Y-%m-%d"),
                    cache_dir=cache_dir,
                )

            self.assertEqual(missing, [])
            self.assertIn(dates[-1], prices["0050.TW"].index)
            self.assertEqual(float(prices["0050.TW"].loc[dates[-1], "close"]), 151.0)

    def test_build_observation_outputs_unified_schema_and_top_candidate(self) -> None:
        dates = pd.bdate_range("2025-01-02", periods=160)
        tsmc = symbol_entry("2330.TW", source="manual")
        tsmc["market_cap_twd"] = 2_000_000_000_000
        mediatek = symbol_entry("2454.TW", source="manual")
        mediatek["market_cap_twd"] = 200_000_000_000
        pool = {
            "pool_id": "custom_ai_pool",
            "name": "自訂AI觀察池",
            "strategy_preset": "universal_pool_custom",
            "resolved_symbols": [tsmc, mediatek],
        }
        prices = {
            "2330.TW": _trend_frame(dates, start=100, step=0.2, volume=20_000_000),
            "2454.TW": _trend_frame(dates, start=100, step=0.8, volume=20_000_000),
        }

        observation = build_stock_pool_observation(
            pool=pool,
            prices_by_ticker=prices,
            signal_date=dates[-1],
        )

        self.assertEqual(observation.schema_version, 1)
        self.assertEqual(observation.pool_id, "custom_ai_pool")
        self.assertEqual(observation.signal_date, dates[-1].strftime("%Y-%m-%d"))
        self.assertEqual(observation.candidate_count, 2)
        self.assertEqual(observation.action_state, "watch_candidate")
        self.assertEqual(observation.decision_layer, CANDIDATE_SOURCE)
        self.assertFalse(observation.active_in_trade_decision)
        self.assertEqual(observation.top_ticker, "2454.TW")
        self.assertEqual(observation.top_display, "聯發科(2454)")
        self.assertEqual(observation.top_asset_type, "stock")
        self.assertTrue(observation.attack_gate_open)
        self.assertTrue(observation.eligible_for_pool_selection)
        self.assertEqual(observation.selection_layer, "formal_candidate")
        self.assertEqual(observation.rank_score, observation.top_score)
        self.assertTrue(observation.base_pool_passed)
        self.assertEqual(observation.gate_rule_id, "universal_pool_base_gate_v1")
        self.assertIn("通用池基礎 gate", observation.gate_reason)
        self.assertGreaterEqual(observation.passed_count, 1)
        scores = {candidate.ticker: candidate for candidate in observation.candidates}
        self.assertEqual(scores["2330.TW"].size_profile, "large_cap")
        self.assertEqual(scores["2454.TW"].size_profile, "mid_cap")
        self.assertEqual(scores["2454.TW"].market_cap_twd, 200_000_000_000)

    def test_etf_candidate_is_market_exposure_tool_not_stock_attack_candidate(self) -> None:
        dates = pd.bdate_range("2025-01-02", periods=160)
        pool = {
            "pool_id": "etf_pool",
            "name": "ETF曝險池",
            "strategy_preset": "universal_pool_custom",
            "resolved_symbols": [symbol_entry("00631L.TW", source="fixed")],
        }
        prices = {"00631L.TW": _trend_frame(dates, start=100, step=0.6, volume=20_000_000)}

        observation = build_stock_pool_observation(
            pool=pool,
            prices_by_ticker=prices,
            signal_date=dates[-1],
        )
        rows = observation.to_dict()["candidates"]
        top_rows = [row for row in _top_candidate_rows(observation) if row["ticker"] == "00631L.TW"]

        self.assertEqual(observation.top_ticker, "00631L.TW")
        self.assertEqual(observation.top_asset_type, "etf")
        self.assertIsNone(observation.attack_gate_open)
        self.assertTrue(observation.eligible_for_pool_selection)
        self.assertEqual(observation.selection_layer, "market_exposure_tool")
        self.assertTrue(rows[0]["passed"])
        self.assertEqual(top_rows[0]["asset_type"], "etf")
        self.assertEqual(top_rows[0]["selection_layer"], "market_exposure_tool")
        self.assertTrue(top_rows[0]["eligible_for_pool_selection"])
        self.assertEqual(top_rows[0]["gate_rule_id"], "universal_pool_base_gate_v1")
        self.assertIn("通用池基礎 gate", top_rows[0]["gate_reason"])

    def test_tw50_pool_uses_own_gate_rule_not_formal_attack_gate(self) -> None:
        dates = pd.bdate_range("2025-01-02", periods=160)
        yageo = symbol_entry("2327.TW", source="tw50_history_csv")
        yageo["market_cap_twd"] = 900_000_000_000
        tsmc = symbol_entry("2330.TW", source="tw50_history_csv")
        tsmc["market_cap_twd"] = 20_000_000_000_000
        pool = {
            "pool_id": "tw50_dynamic_constituents_v0",
            "name": "大型市場廣度池 v0",
            "strategy_preset": "universal_pool_custom",
            "resolved_symbols": [yageo, tsmc],
            "dynamic_constituents": {"source": "tw50_history_csv", "path": "data/tw50_constituents.csv"},
        }
        prices = {
            "2327.TW": _trend_frame(dates, start=100, step=1.0, volume=20_000_000),
            "2330.TW": _trend_frame(dates, start=100, step=0.2, volume=20_000_000),
            "0050.TW": _trend_frame(dates, start=100, step=0.0, volume=20_000_000),
        }

        observation = build_stock_pool_observation(
            pool=pool,
            prices_by_ticker=prices,
            signal_date=dates[-1],
        )
        top_rows = _top_candidate_rows(observation)

        self.assertEqual(observation.top_ticker, "2327.TW")
        self.assertEqual(observation.gate_rule_id, "tw50_large_breadth_attack_gate_v1")
        self.assertTrue(observation.base_pool_passed)
        self.assertTrue(observation.attack_gate_open)
        self.assertTrue(observation.eligible_for_pool_selection)
        self.assertEqual(observation.selection_layer, "formal_candidate")
        self.assertIn("大型廣度池 v1", observation.gate_reason)
        self.assertIn("60日相對0050超額", observation.gate_reason)
        self.assertEqual(top_rows[0]["gate_rule_id"], "tw50_large_breadth_attack_gate_v1")
        self.assertTrue(top_rows[0]["base_pool_passed"])
        self.assertTrue(top_rows[0]["attack_gate_open"])
        self.assertTrue(top_rows[0]["eligible_for_pool_selection"])

    def test_formal_pool_excludes_0050_candidate_and_uses_company_display_names(self) -> None:
        dates = pd.bdate_range("2025-01-02", periods=180)
        pool = {
            "pool_id": "tw50_dynamic_constituents_v0",
            "name": "大型市場廣度池 v0",
            "strategy_preset": "universal_pool_custom",
            "resolved_symbols": [
                {"ticker": "0050.TW", "display": "0050", "asset_type": "etf"},
                {"ticker": "2303.TW", "display": "2303", "asset_type": "stock"},
                {"ticker": "2344.TW", "display": "2344", "asset_type": "stock"},
                {"ticker": "2327.TW", "display": "2327", "asset_type": "stock"},
                {"ticker": "5876.TW", "display": "5876", "asset_type": "stock"},
                {"ticker": "6919.TW", "display": "6919", "asset_type": "stock"},
                {"ticker": "00631L.TW", "display": "0050正二", "asset_type": "etf"},
            ],
            "dynamic_constituents": {"source": "tw50_history_csv", "path": "data/tw50_constituents.csv"},
        }
        prices = {
            "0050.TW": _trend_frame(dates, start=100, step=1.5, volume=20_000_000),
            "2303.TW": _trend_frame(dates, start=100, step=0.9, volume=20_000_000),
            "2344.TW": _trend_frame(dates, start=100, step=1.0, volume=20_000_000),
            "2327.TW": _trend_frame(dates, start=100, step=0.8, volume=20_000_000),
            "5876.TW": _trend_frame(dates, start=100, step=0.7, volume=20_000_000),
            "6919.TW": _trend_frame(dates, start=100, step=0.6, volume=20_000_000),
            "00631L.TW": _trend_frame(dates, start=100, step=1.1, volume=20_000_000),
        }

        observation = build_stock_pool_observation(
            pool=pool,
            prices_by_ticker=prices,
            signal_date=dates[-1],
        )
        top_rows = _top_candidate_rows(observation, limit=7)
        candidate_tickers = [candidate.ticker for candidate in observation.candidates]
        displays = [row["display"] for row in top_rows]

        self.assertNotIn("0050.TW", candidate_tickers)
        self.assertNotIn("0050.TW", [row["ticker"] for row in top_rows])
        self.assertIn("00631L.TW", candidate_tickers)
        self.assertIn("聯電(2303)", displays)
        self.assertIn("華邦電(2344)", displays)
        self.assertIn("國巨(2327)", displays)
        self.assertIn("上海商銀(5876)", displays)
        self.assertIn("康霈*(6919)", displays)
        self.assertNotIn("2303(2303)", displays)
        self.assertNotIn("2344(2344)", displays)
        self.assertNotIn("2327(2327)", displays)
        self.assertNotIn("5876(5876)", displays)
        self.assertNotIn("6919(6919)", displays)

    def test_tw50_pool_blocks_high_rank_when_benchmark_margin_is_insufficient(self) -> None:
        dates = pd.bdate_range("2025-01-02", periods=160)
        pool = _tw50_test_pool(["2327.TW", "2330.TW"])
        prices = {
            "2327.TW": _trend_frame(dates, start=100, step=0.30, volume=20_000_000),
            "2330.TW": _trend_frame(dates, start=100, step=0.10, volume=20_000_000),
            "0050.TW": _trend_frame(dates, start=100, step=0.28, volume=20_000_000),
        }

        observation = build_stock_pool_observation(
            pool=pool,
            prices_by_ticker=prices,
            signal_date=dates[-1],
        )
        top_rows = _top_candidate_rows(observation)

        self.assertIsNone(observation.top_ticker)
        self.assertFalse(observation.eligible_for_pool_selection)
        self.assertEqual(observation.selection_layer, "no_selection")
        self.assertEqual(top_rows[0]["ticker"], "2327.TW")
        self.assertFalse(top_rows[0]["eligible_for_pool_selection"])
        self.assertFalse(top_rows[0]["attack_gate_open"])
        self.assertIn("60日相對0050超額", top_rows[0]["gate_reason"])

    def test_tw50_pool_blocks_when_momentum_quality_is_insufficient(self) -> None:
        dates = pd.bdate_range("2025-01-02", periods=160)
        pool = _tw50_test_pool(["2327.TW", "2330.TW"])
        prices = {
            "2327.TW": _trend_frame(dates, start=100, step=0.08, volume=20_000_000),
            "2330.TW": _trend_frame(dates, start=100, step=0.02, volume=20_000_000),
            "0050.TW": _trend_frame(dates, start=100, step=0.00, volume=20_000_000),
        }

        observation = build_stock_pool_observation(
            pool=pool,
            prices_by_ticker=prices,
            signal_date=dates[-1],
        )
        top_rows = _top_candidate_rows(observation)

        self.assertIsNone(observation.top_ticker)
        self.assertFalse(top_rows[0]["eligible_for_pool_selection"])
        self.assertFalse(top_rows[0]["attack_gate_open"])
        self.assertIn("20/60動能品質=N", top_rows[0]["gate_reason"])

    def test_tw50_pool_blocks_when_persistence_is_insufficient(self) -> None:
        dates = pd.bdate_range("2025-01-02", periods=160)
        pool = _tw50_test_pool(["2327.TW", "2330.TW"])
        prices = {
            "2327.TW": _final_surge_frame(dates, flat_days=157, final_gain=0.55, volume=20_000_000),
            "2330.TW": _trend_frame(dates, start=100, step=0.10, volume=20_000_000),
            "0050.TW": _trend_frame(dates, start=100, step=0.00, volume=20_000_000),
        }

        observation = build_stock_pool_observation(
            pool=pool,
            prices_by_ticker=prices,
            signal_date=dates[-1],
        )
        top_rows = _top_candidate_rows(observation)

        self.assertIsNone(observation.top_ticker)
        self.assertFalse(top_rows[0]["eligible_for_pool_selection"])
        self.assertFalse(top_rows[0]["attack_gate_open"])
        self.assertIn("持續性=", top_rows[0]["gate_reason"])
        self.assertIn("(N)", top_rows[0]["gate_reason"])

    def test_core_defensive_pool_uses_resilience_gate_for_stock_candidate(self) -> None:
        dates = pd.bdate_range("2025-01-02", periods=160)
        pool = _core_defensive_test_pool(["2882.TW", "2330.TW"])
        prices = {
            "2882.TW": _trend_frame(dates, start=100, step=0.35, volume=20_000_000),
            "2330.TW": _trend_frame(dates, start=100, step=0.10, volume=20_000_000),
            "0050.TW": _trend_frame(dates, start=100, step=0.20, volume=20_000_000),
        }

        observation = build_stock_pool_observation(
            pool=pool,
            prices_by_ticker=prices,
            signal_date=dates[-1],
        )
        top_rows = _top_candidate_rows(observation)

        self.assertEqual(observation.top_ticker, "2882.TW")
        self.assertEqual(observation.gate_rule_id, "core_style_complement_opportunity_gate_v1")
        self.assertTrue(observation.attack_gate_open)
        self.assertTrue(observation.eligible_for_pool_selection)
        self.assertEqual(observation.selection_layer, "formal_candidate")
        self.assertIn("風格補強池 v1", observation.gate_reason)
        self.assertIn("60日相對0050強度", observation.gate_reason)
        self.assertIn("120日機會成本", observation.gate_reason)
        self.assertTrue(top_rows[0]["eligible_for_pool_selection"])

    def test_core_defensive_pool_blocks_when_lagging_benchmark_too_much(self) -> None:
        dates = pd.bdate_range("2025-01-02", periods=160)
        pool = _core_defensive_test_pool(["2882.TW", "2330.TW"])
        prices = {
            "2882.TW": _trend_frame(dates, start=100, step=0.12, volume=20_000_000),
            "2330.TW": _trend_frame(dates, start=100, step=0.02, volume=20_000_000),
            "0050.TW": _trend_frame(dates, start=100, step=0.30, volume=20_000_000),
        }

        observation = build_stock_pool_observation(
            pool=pool,
            prices_by_ticker=prices,
            signal_date=dates[-1],
        )
        top_rows = _top_candidate_rows(observation)

        self.assertIsNone(observation.top_ticker)
        self.assertFalse(top_rows[0]["eligible_for_pool_selection"])
        self.assertFalse(top_rows[0]["attack_gate_open"])
        self.assertIn("60日相對0050強度", top_rows[0]["gate_reason"])

    def test_core_defensive_pool_blocks_when_trend_resilience_is_insufficient(self) -> None:
        dates = pd.bdate_range("2025-01-02", periods=160)
        pool = _core_defensive_test_pool(["2882.TW", "2330.TW"])
        prices = {
            "2882.TW": _trend_frame(dates, start=100, step=0.04, volume=20_000_000),
            "2330.TW": _trend_frame(dates, start=100, step=0.01, volume=20_000_000),
            "0050.TW": _trend_frame(dates, start=100, step=0.00, volume=20_000_000),
        }

        observation = build_stock_pool_observation(
            pool=pool,
            prices_by_ticker=prices,
            signal_date=dates[-1],
        )
        top_rows = _top_candidate_rows(observation)

        self.assertIsNone(observation.top_ticker)
        self.assertFalse(top_rows[0]["eligible_for_pool_selection"])
        self.assertIn("60/120中期上攻力=N", top_rows[0]["gate_reason"])

    def test_core_defensive_pool_blocks_when_opportunity_cost_is_too_high(self) -> None:
        dates = pd.bdate_range("2025-01-02", periods=160)
        pool = _core_defensive_test_pool(["2882.TW", "2330.TW"])
        prices = {
            "2882.TW": _final_surge_frame(dates, flat_days=95, final_gain=0.35, volume=20_000_000),
            "2330.TW": _trend_frame(dates, start=100, step=0.01, volume=20_000_000),
            "0050.TW": _trend_frame(dates, start=100, step=0.45, volume=20_000_000),
        }

        observation = build_stock_pool_observation(
            pool=pool,
            prices_by_ticker=prices,
            signal_date=dates[-1],
        )
        top_rows = _top_candidate_rows(observation)

        self.assertIsNone(observation.top_ticker)
        self.assertFalse(top_rows[0]["eligible_for_pool_selection"])
        self.assertFalse(top_rows[0]["attack_gate_open"])
        self.assertIn("120日機會成本", top_rows[0]["gate_reason"])
        self.assertIn("(N)", top_rows[0]["gate_reason"])

    def test_core_defensive_pool_falls_back_to_market_exposure_when_no_stock_passes(self) -> None:
        dates = pd.bdate_range("2025-01-02", periods=160)
        pool = _core_defensive_test_pool(["00631L.TW", "2882.TW", "0050.TW"])
        prices = {
            "00631L.TW": _trend_frame(dates, start=100, step=0.80, volume=20_000_000),
            "2882.TW": _final_surge_frame(dates, flat_days=95, final_gain=0.35, volume=20_000_000),
            "0050.TW": _trend_frame(dates, start=100, step=0.45, volume=20_000_000),
        }

        observation = build_stock_pool_observation(
            pool=pool,
            prices_by_ticker=prices,
            signal_date=dates[-1],
        )
        top_rows = _top_candidate_rows(observation)
        rows_by_ticker = {row["ticker"]: row for row in top_rows}

        self.assertEqual(observation.top_ticker, "00631L.TW")
        self.assertEqual(observation.top_asset_type, "etf")
        self.assertEqual(observation.selection_layer, "market_exposure_tool")
        self.assertTrue(observation.eligible_for_pool_selection)
        self.assertIsNone(observation.attack_gate_open)
        self.assertEqual(rows_by_ticker["00631L.TW"]["selection_layer"], "market_exposure_tool")
        self.assertFalse(rows_by_ticker["2882.TW"]["eligible_for_pool_selection"])
        self.assertIn("非AI風格個股無合格", observation.gate_reason)

    def test_core_defensive_pool_prefers_qualified_stock_over_market_exposure_tool(self) -> None:
        dates = pd.bdate_range("2025-01-02", periods=160)
        pool = _core_defensive_test_pool(["00631L.TW", "2882.TW", "0050.TW"])
        prices = {
            "00631L.TW": _trend_frame(dates, start=100, step=0.80, volume=20_000_000),
            "2882.TW": _trend_frame(dates, start=100, step=0.35, volume=20_000_000),
            "0050.TW": _trend_frame(dates, start=100, step=0.20, volume=20_000_000),
        }

        observation = build_stock_pool_observation(
            pool=pool,
            prices_by_ticker=prices,
            signal_date=dates[-1],
        )

        self.assertEqual(observation.top_ticker, "2882.TW")
        self.assertEqual(observation.top_asset_type, "stock")
        self.assertEqual(observation.selection_layer, "formal_candidate")
        self.assertTrue(observation.attack_gate_open)
        self.assertTrue(observation.eligible_for_pool_selection)

    def test_core_defensive_pool_resolves_one_representative_per_style_bucket(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "core_defensive_candidates.csv"
            path.write_text(
                "\n".join(
                    [
                        "effective_date,ticker,symbol,name,style_role,review_status,is_current_member,defensive_score,stability_score,cross_sector_score,fundamental_score,review_reason",
                        "2026-06-01,0050.TW,0050,0050,大型市場核心ETF,active,true,88,86,95,80,etf",
                        "2026-06-01,00631L.TW,00631L,0050正二,大型市場槓桿ETF,active,true,65,45,80,70,leveraged",
                        "2026-06-01,2881.TW,2881,富邦金,金融核心,active,true,78,76,84,78,financial",
                        "2026-06-01,2882.TW,2882,國泰金,金融核心,active,true,78,75,84,77,financial",
                        "2026-06-01,2412.TW,2412,中華電,電信防守核心,active,true,92,94,82,84,telecom",
                        "2026-06-01,3045.TW,3045,台灣大,電信防守核心,active,true,93,90,80,82,telecom",
                        "2026-06-01,2330.TW,2330,台積電,市場核心半導體,active,true,82,80,88,92,semi",
                        "2026-06-01,1301.TW,1301,台塑,傳產核心,watch,false,99,99,99,99,watch",
                    ]
                ),
                encoding="utf-8",
            )
            pool = {
                "pool_id": "large_core_bluechip_v0",
                "name": "核心風格補強池 v1",
                "strategy_preset": "core_defensive_style_v1",
                "resolved_symbols": [symbol_entry("2882.TW", source="fixed"), symbol_entry("2412.TW", source="fixed")],
                "candidate_review_config": {
                    "source_mode": "core_defensive_candidate_csv",
                    "path": str(path),
                },
            }

            resolved = _resolve_dynamic_observation_pool(
                pool,
                signal_date="2026-06-12",
                radar_snapshot_dir=None,
                radar_data_dir=None,
                radar_top_n=20,
                tw50_constituents_path=None,
            )

        tickers = [item["ticker"] for item in resolved["resolved_symbols"]]
        self.assertEqual(tickers, ["0050.TW", "00631L.TW", "2881.TW", "3045.TW"])
        self.assertNotIn("2882.TW", tickers)
        self.assertNotIn("2412.TW", tickers)
        self.assertNotIn("2330.TW", tickers)
        self.assertEqual(resolved["core_defensive_style_selection_mode"], "one_representative_per_style_bucket_v1")
        buckets = {item["ticker"]: item["style_bucket"] for item in resolved["resolved_symbols"]}
        self.assertEqual(buckets["2881.TW"], "financial_core")
        self.assertEqual(buckets["3045.TW"], "telecom_defensive")

    def test_core_defensive_pool_blocks_when_drawdown_control_fails(self) -> None:
        dates = pd.bdate_range("2025-01-02", periods=160)
        pool = _core_defensive_test_pool(["2882.TW"])
        prices = {
            "2882.TW": _pullback_frame(dates, peak_gain=0.50, final_drawdown=0.14, volume=20_000_000),
            "0050.TW": _trend_frame(dates, start=100, step=0.00, volume=20_000_000),
        }

        observation = build_stock_pool_observation(
            pool=pool,
            prices_by_ticker=prices,
            signal_date=dates[-1],
        )
        top_rows = _top_candidate_rows(observation)

        self.assertIsNone(observation.top_ticker)
        self.assertFalse(top_rows[0]["eligible_for_pool_selection"])
        self.assertIn("20日回撤控管", top_rows[0]["gate_reason"])
        self.assertIn("(N)", top_rows[0]["gate_reason"])

    def test_build_observation_preserves_valuation_signal_fields(self) -> None:
        dates = pd.bdate_range("2025-01-02", periods=160)
        tsmc = symbol_entry("2330.TW", source="manual")
        tsmc["market_cap_twd"] = 2_000_000_000_000
        pool = {
            "pool_id": "custom_ai_pool",
            "name": "自訂AI觀察池",
            "strategy_preset": "universal_pool_custom",
            "resolved_symbols": [tsmc],
        }
        prices = {"2330.TW": _trend_frame(dates, start=100, step=0.2, volume=20_000_000)}

        observation = build_stock_pool_observation(
            pool=pool,
            prices_by_ticker=prices,
            signal_date=dates[-1],
            valuation_signal_by_ticker={
                "2330.TW": ValuationSignal(
                    ticker="2330.TW",
                    fair_price=1200,
                    buy_price=1080,
                    safety_margin_pct=0.12,
                    gate_passed=True,
                    score_adjustment=0.02,
                    reason="估值仍有安全邊際",
                    signal_date="2025-08-01",
                )
            },
        )

        candidate = observation.candidates[0]
        self.assertEqual(candidate.valuation_reason, "估值仍有安全邊際")
        self.assertEqual(candidate.valuation_fair_price, 1200)
        self.assertEqual(candidate.valuation_buy_price, 1080)
        self.assertEqual(candidate.valuation_source_date, "2025-08-01")

    def test_core_defensive_preset_applies_stricter_overheat_filter(self) -> None:
        dates = pd.bdate_range("2025-01-02", periods=160)
        tsmc = symbol_entry("2330.TW", source="manual")
        tsmc["market_cap_twd"] = 2_000_000_000_000
        pool = {
            "pool_id": "large_core_bluechip_v0",
            "name": "核心風格補強池 v1",
            "strategy_preset": "core_defensive_style_v1",
            "resolved_symbols": [tsmc],
        }
        prices = {"2330.TW": _late_surge_frame(dates, volume=20_000_000)}

        observation = build_stock_pool_observation(
            pool=pool,
            prices_by_ticker=prices,
            signal_date=dates[-1],
        )

        candidate = observation.candidates[0]
        self.assertFalse(candidate.passed)
        self.assertEqual(candidate.reason, "20日漲幅過熱")
        self.assertEqual(candidate.applied_score_mode, "risk_adjusted")

    def test_observation_resolves_to_previous_common_trading_date(self) -> None:
        dates = pd.bdate_range("2025-01-02", periods=160)
        pool = {
            "pool_id": "custom_ai_pool",
            "name": "自訂AI觀察池",
            "strategy_preset": "universal_pool_custom",
            "resolved_symbols": [symbol_entry("2330.TW", source="manual")],
        }
        prices = {"2330.TW": _trend_frame(dates, start=100, step=0.2, volume=20_000_000)}

        observation = build_stock_pool_observation(
            pool=pool,
            prices_by_ticker=prices,
            signal_date=(dates[-1] + pd.Timedelta(days=3)).strftime("%Y-%m-%d"),
        )

        self.assertEqual(observation.signal_date, dates[-1].strftime("%Y-%m-%d"))
        self.assertEqual(observation.data_end_date, dates[-1].strftime("%Y-%m-%d"))

    def test_observation_can_require_exact_signal_date(self) -> None:
        dates = pd.bdate_range("2025-01-02", periods=160)
        pool = {
            "pool_id": "custom_ai_pool",
            "name": "自訂AI觀察池",
            "strategy_preset": "universal_pool_custom",
            "resolved_symbols": [symbol_entry("2330.TW", source="manual")],
        }
        prices = {"2330.TW": _trend_frame(dates, start=100, step=0.2, volume=20_000_000)}

        with self.assertRaisesRegex(ValueError, "No exact common price data"):
            build_stock_pool_observation(
                pool=pool,
                prices_by_ticker=prices,
                signal_date=(dates[-1] + pd.Timedelta(days=3)).strftime("%Y-%m-%d"),
                require_exact_signal_date=True,
            )

    def test_write_observation_outputs_json_and_candidates_csv(self) -> None:
        dates = pd.bdate_range("2025-01-02", periods=160)
        pool = {
            "pool_id": "custom_ai_pool",
            "name": "自訂AI觀察池",
            "strategy_preset": "universal_pool_custom",
            "resolved_symbols": [symbol_entry("2330.TW", source="manual")],
        }
        observation = build_stock_pool_observation(
            pool=pool,
            prices_by_ticker={"2330.TW": _trend_frame(dates, start=100, step=0.2, volume=20_000_000)},
            signal_date=dates[-1],
        )
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            write_stock_pool_observation(output_dir, observation)

            self.assertTrue((output_dir / "stock_pool_observation.json").exists())
            self.assertTrue((output_dir / "stock_pool_observation_candidates.csv").exists())

    def test_best_preset_dispatches_to_frozen_strategy_observation(self) -> None:
        dates = pd.bdate_range("2025-01-02", periods=160)
        pool = {
            "pool_id": "large_cap_best_v20260605",
            "name": "AI中大型權值股池最佳版 v20260605",
            "strategy_preset": "best_v20260605",
            "resolved_symbols": [symbol_entry("2454.TW", source="fixed")],
        }
        prices = {"2454.TW": _trend_frame(dates, start=100, step=0.2, volume=20_000_000)}
        expected = build_stock_pool_observation(
            pool={**pool, "strategy_preset": "universal_pool_custom"},
            prices_by_ticker=prices,
            signal_date=dates[-1],
        )

        with patch("backtest_lab.stock_pool_observation._build_regime_signal_observation", return_value=expected) as mocked:
            observation = build_dispatched_stock_pool_observation(
                pool=pool,
                prices_by_ticker=prices,
                signal_date=dates[-1],
                warmup_start=dates[0].strftime("%Y-%m-%d"),
            )

        self.assertIs(observation, expected)
        self.assertTrue(mocked.called)

    def test_batch_writes_manifest_and_skips_empty_pool(self) -> None:
        dates = pd.bdate_range("2025-01-02", periods=160)
        source_cache = {
            "2330.TW": _trend_frame(dates, start=100, step=0.2, volume=20_000_000),
        }
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp) / "cache"
            cache_dir.mkdir()
            source_cache["2330.TW"].reset_index(names="date").to_csv(cache_dir / "2330_TW.csv", index=False)
            manifest = run_stock_pool_observation_batch(
                pools=[
                    {
                        "pool_id": "custom_ai_pool",
                        "name": "自訂AI觀察池",
                        "strategy_preset": "universal_pool_custom",
                        "role_name": "測試專家",
                        "role_description": "確認角色會進入輸出",
                        "candidate_review_frequency": "monthly",
                        "candidate_update_policy": "測試更新政策",
                        "resolved_symbols": [symbol_entry("2330.TW", source="manual")],
                    },
                    {
                        "pool_id": "empty_radar_pool",
                        "name": "空雷達池",
                        "strategy_preset": "radar_core_mid_small_calibrated_v1",
                        "resolved_symbols": [],
                    },
                ],
                signal_date=dates[-1].strftime("%Y-%m-%d"),
                warmup_start=dates[0].strftime("%Y-%m-%d"),
                cache_dir=cache_dir,
                output_root=Path(tmp) / "out",
            )

            self.assertEqual(len(manifest["generated"]), 1)
            self.assertEqual(manifest["generated"][0]["role_name"], "測試專家")
            self.assertEqual(manifest["generated"][0]["candidate_review_frequency"], "monthly")
            self.assertEqual(manifest["generated"][0]["candidate_update_policy"], "測試更新政策")
            self.assertEqual(manifest["generated"][0]["decision_layer"], CANDIDATE_SOURCE)
            self.assertFalse(manifest["generated"][0]["active_in_trade_decision"])
            self.assertEqual(manifest["generated"][0]["top_asset_type"], "stock")
            self.assertIn("rank_score", manifest["generated"][0])
            self.assertTrue(manifest["generated"][0]["base_pool_passed"])
            self.assertTrue(manifest["generated"][0]["attack_gate_open"])
            self.assertTrue(manifest["generated"][0]["eligible_for_pool_selection"])
            self.assertEqual(manifest["generated"][0]["selection_layer"], "formal_candidate")
            self.assertEqual(manifest["generated"][0]["gate_rule_id"], "universal_pool_base_gate_v1")
            self.assertIn("通用池基礎 gate", manifest["generated"][0]["gate_reason"])
            self.assertEqual(manifest["generated"][0]["candidate_review"]["frequency"], "monthly")
            self.assertEqual(manifest["generated"][0]["candidate_review"]["source_status"], "manual_review_required")
            self.assertEqual(len(manifest["generated"][0]["top_candidates"]), 1)
            self.assertEqual(manifest["generated"][0]["top_candidates"][0]["display"], "台積電(2330)")
            self.assertTrue(manifest["generated"][0]["top_candidates"][0]["base_pool_passed"])
            self.assertEqual(manifest["generated"][0]["top_candidates"][0]["selection_layer"], "formal_candidate")
            self.assertEqual(manifest["generated"][0]["top_candidates"][0]["gate_rule_id"], "universal_pool_base_gate_v1")
            self.assertEqual(len(manifest["skipped"]), 1)
            self.assertEqual(manifest["skipped"][0]["reason"], "missing_formal_radar_candidates")
            self.assertEqual(manifest["skipped"][0]["decision_layer"], DATA_READINESS)
            self.assertIn("candidate_review", manifest["skipped"][0])
            self.assertIn("model_layer_audit", manifest)
            self.assertIn("report_wording_boundary", manifest)
            self.assertIn("decision_first_report_contract", manifest)
            decision = manifest["decision_first_report_contract"]
            self.assertEqual(decision["decision_first_state"], "formal_target_available")
            self.assertEqual(decision["data_completeness_state"], "complete")
            self.assertIn("台積電(2330)", decision["formal_target_display"])
            self.assertEqual(decision["switch_signal_state"], "previous_target_missing")
            self.assertEqual(decision["score_margin_state"], "formal_candidate_ranking_contract_missing")
            wording = manifest["report_wording_boundary"]
            self.assertFalse(wording["formal_model_changed"])
            self.assertFalse(wording["trade_decision_changed"])
            self.assertTrue(wording["formal_baseline"]["active_in_trade_decision"])
            self.assertFalse(wording["diagnostic_boundary"]["active_in_trade_decision"])
            self.assertFalse(wording["execution_boundary"]["active_in_trade_decision"])
            self.assertIn("非正式診斷", wording["diagnostic_boundary"]["description"])
            self.assertIn("換倉與出場規則", wording["execution_boundary"]["description"])
            self.assertTrue(manifest["formal_report_ready"])
            self.assertEqual(manifest["formal_report_blocker_count"], 0)
            self.assertNotIn("pool3_radar_attack_satellite", manifest)
            manifest_path = Path(manifest["output_root"]) / "stock_pool_observation_manifest.json"
            self.assertTrue(manifest_path.exists())
            self.assertTrue((Path(manifest["output_root"]) / "model_layer_audit.json").exists())
            self.assertTrue((Path(manifest["output_root"]) / "stock_pool_observation_summary.csv").exists())
            self.assertTrue((Path(manifest["output_root"]) / "stock_pool_candidate_reviews.csv").exists())
            self.assertTrue((Path(manifest["output_root"]) / "stock_pool_candidate_reviews.json").exists())
            self.assertFalse((Path(manifest["output_root"]) / "pool3_radar_attack_satellite.json").exists())
            self.assertFalse((Path(manifest["output_root"]) / "pool3_radar_attack_satellite.csv").exists())
            self.assertFalse((Path(manifest["output_root"]) / "pool3_radar_attack_satellite.md").exists())
            self.assertTrue((Path(manifest["output_root"]) / "stock_pool_observation_report.md").exists())
            report_text = (Path(manifest["output_root"]) / "stock_pool_observation_report.md").read_text(encoding="utf-8")
            self.assertIn("測試專家", report_text)
            self.assertIn("月頻", report_text)
            self.assertNotIn("三池", report_text)
            self.assertNotIn("風格補強池", report_text)
            self.assertNotIn("Pool3 Radar Top10 攻擊衛星觀察", report_text)
            self.assertIn("正式模型基準", report_text)
            self.assertIn("正式報告狀態：可發布", report_text)
            self.assertNotIn("使用邊界", report_text)
            self.assertNotIn("診斷邊界", report_text)
            self.assertNotIn("執行邊界", report_text)
            self.assertTrue((Path(manifest["output_root"]) / "AI股票池觀察總覽_最新版_v20260612.pdf").exists())
            self.assertTrue(
                (Path(manifest["generated"][0]["output_dir"]) / "stock_pool_observation.json").exists()
            )

    def test_decision_first_contract_detects_maintained_previous_target(self) -> None:
        dates = pd.bdate_range("2025-01-02", periods=160)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache_dir = root / "cache"
            cache_dir.mkdir()
            _trend_frame(dates, start=100, step=0.2, volume=20_000_000).reset_index(names="date").to_csv(
                cache_dir / "2330_TW.csv",
                index=False,
            )
            previous_dir = root / "out" / "20250731"
            previous_dir.mkdir(parents=True)
            (previous_dir / "stock_pool_observation_manifest.json").write_text(
                json.dumps(
                    {
                        "formal_report_ready": True,
                        "actual_signal_date": "2025-07-31",
                        "decision_first_report_contract": {
                            "formal_target_display": "台積電(2330)",
                            "formal_target_ticker": "2330.TW",
                        },
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

            manifest = run_stock_pool_observation_batch(
                pools=[
                    {
                        "pool_id": "custom_ai_pool",
                        "name": "自訂AI觀察池",
                        "strategy_preset": "universal_pool_custom",
                        "resolved_symbols": [symbol_entry("2330.TW", source="manual")],
                    }
                ],
                signal_date=dates[-1].strftime("%Y-%m-%d"),
                warmup_start=dates[0].strftime("%Y-%m-%d"),
                cache_dir=cache_dir,
                output_root=root / "out",
            )

            decision = manifest["decision_first_report_contract"]
            self.assertEqual(decision["previous_formal_target_ticker"], "2330.TW")
            self.assertEqual(decision["switch_signal_state"], "maintain_formal_target")
            self.assertIn("維持不變", decision["switch_signal_wording_zh"])

    def test_decision_first_contract_detects_formal_target_change(self) -> None:
        dates = pd.bdate_range("2025-01-02", periods=160)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache_dir = root / "cache"
            cache_dir.mkdir()
            _trend_frame(dates, start=100, step=0.2, volume=20_000_000).reset_index(names="date").to_csv(
                cache_dir / "2330_TW.csv",
                index=False,
            )
            previous_dir = root / "out" / "20250731"
            previous_dir.mkdir(parents=True)
            (previous_dir / "stock_pool_observation_manifest.json").write_text(
                json.dumps(
                    {
                        "formal_report_ready": True,
                        "actual_signal_date": "2025-07-31",
                        "decision_first_report_contract": {
                            "formal_target_display": "鴻海(2317)",
                            "formal_target_ticker": "2317.TW",
                        },
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

            manifest = run_stock_pool_observation_batch(
                pools=[
                    {
                        "pool_id": "custom_ai_pool",
                        "name": "自訂AI觀察池",
                        "strategy_preset": "universal_pool_custom",
                        "resolved_symbols": [symbol_entry("2330.TW", source="manual")],
                    }
                ],
                signal_date=dates[-1].strftime("%Y-%m-%d"),
                warmup_start=dates[0].strftime("%Y-%m-%d"),
                cache_dir=cache_dir,
                output_root=root / "out",
            )

            decision = manifest["decision_first_report_contract"]
            self.assertEqual(decision["previous_formal_target_ticker"], "2317.TW")
            self.assertEqual(decision["switch_signal_state"], "formal_target_changed")
            self.assertIn("正式目標已從 鴻海(2317)", decision["switch_signal_wording_zh"])
            report_text = (Path(manifest["output_root"]) / "stock_pool_observation_report.md").read_text(encoding="utf-8")
            self.assertIn("前一份正式報告標的：鴻海(2317)（2025-07-31）", report_text)

    def test_batch_excludes_non_operational_scorecard_pool_by_default(self) -> None:
        dates = pd.bdate_range("2025-01-02", periods=160)
        prices = {"2330.TW": _trend_frame(dates, start=100, step=0.5, volume=20_000_000)}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache_dir = root / "cache"
            cache_dir.mkdir()

            with patch("backtest_lab.stock_pool_observation.download_yfinance_prices", return_value=prices):
                manifest = run_stock_pool_observation_batch(
                    pools=[
                        {
                            "pool_id": "large_cap_best_v20260605",
                            "name": "AI中大型權值股池最佳版 v20260605",
                            "strategy_preset": "universal_pool_custom",
                            "operational_observation": True,
                            "resolved_symbols": [symbol_entry("2330.TW", source="fixed")],
                        },
                        {
                            "pool_id": "model_scorecard_ep10",
                            "name": "模型延遲公開成績單池",
                            "strategy_preset": "delayed_public_scorecard_v1",
                            "operational_observation": False,
                            "resolved_symbols": [symbol_entry("2330.TW", source="dynamic")],
                        },
                    ],
                    signal_date=dates[-1].strftime("%Y-%m-%d"),
                    warmup_start=dates[0].strftime("%Y-%m-%d"),
                    cache_dir=cache_dir,
                    output_root=root / "out",
                )

            self.assertEqual([item["pool_id"] for item in manifest["generated"]], ["large_cap_best_v20260605"])
            self.assertIn("top_candidates", manifest["generated"][0])
            self.assertIn("strength_rank", manifest["generated"][0]["top_candidates"][0])
            report = (Path(manifest["output_root"]) / "stock_pool_observation_report.md").read_text(encoding="utf-8")
            self.assertNotIn("模型延遲公開成績單池", report)

    def test_batch_report_hides_0050_candidate_and_repeated_code_display(self) -> None:
        dates = pd.bdate_range("2025-01-02", periods=180)
        prices = {
            "0050.TW": _trend_frame(dates, start=100, step=1.5, volume=20_000_000),
            "2303.TW": _trend_frame(dates, start=100, step=0.9, volume=20_000_000),
            "2344.TW": _trend_frame(dates, start=100, step=1.0, volume=20_000_000),
            "00631L.TW": _trend_frame(dates, start=100, step=1.1, volume=20_000_000),
        }

        def fake_download(*, tickers, **kwargs):
            return {ticker: prices[ticker] for ticker in tickers if ticker in prices}

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch("backtest_lab.stock_pool_observation.download_yfinance_prices", side_effect=fake_download):
                manifest = run_stock_pool_observation_batch(
                    pools=[
                        {
                            "pool_id": "tw50_dynamic_constituents_v0",
                            "name": "大型市場廣度池 v0",
                            "strategy_preset": "universal_pool_custom",
                            "operational_observation": True,
                            "resolved_symbols": [
                                {"ticker": "0050.TW", "display": "0050", "asset_type": "etf"},
                                {"ticker": "2303.TW", "display": "2303", "asset_type": "stock"},
                                {"ticker": "2344.TW", "display": "2344", "asset_type": "stock"},
                                {"ticker": "00631L.TW", "display": "0050正二", "asset_type": "etf"},
                            ],
                            "dynamic_constituents": {"source": "tw50_history_csv", "path": "data/tw50_constituents.csv"},
                        }
                    ],
                    signal_date=dates[-1].strftime("%Y-%m-%d"),
                    warmup_start=dates[0].strftime("%Y-%m-%d"),
                    cache_dir=root / "cache",
                    output_root=root / "out",
                )

            top_rows = manifest["generated"][0]["top_candidates"]
            report = (Path(manifest["output_root"]) / "stock_pool_observation_report.md").read_text(encoding="utf-8")

            self.assertNotIn("0050.TW", [row["ticker"] for row in top_rows])
            self.assertNotIn("0050(0050)", report)
            self.assertNotIn("動態0050成分股", report)
            self.assertIn("大型權值成分股", report)
            self.assertIn("聯電(2303)", report)
            self.assertIn("華邦電(2344)", report)
            self.assertNotIn("2303(2303)", report)
            self.assertNotIn("2344(2344)", report)
            self.assertNotIn("combined_cap40_confirmation1_base", report)
            self.assertNotIn("pool1_primary_pool2_confirmation_cap40", report)
            self.assertNotIn("Pool1+Pool2 formal baseline", report)
            self.assertNotIn("PIT-ready", report)
            self.assertNotIn("正式 target", report)
            self.assertNotIn("selector", report)
            self.assertNotIn("三池", report)
            self.assertNotIn("Pool3", report)
            self.assertNotIn("風格補強池", report)
            self.assertNotIn("large_core_bluechip_v0", report)

    def test_visible_report_sanitizes_0050_from_skipped_pool_reason(self) -> None:
        dates = pd.bdate_range("2025-01-02", periods=180)

        def fake_download(*, tickers, **kwargs):
            return {}

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch("backtest_lab.stock_pool_observation.download_yfinance_prices", side_effect=fake_download):
                manifest = run_stock_pool_observation_batch(
                    pools=[
                        {
                            "pool_id": "tw50_dynamic_constituents_v0",
                            "name": "大型市場廣度池 v0",
                            "strategy_preset": "universal_pool_custom",
                            "operational_observation": True,
                            "resolved_symbols": [
                                {"ticker": "0050.TW", "display": "0050", "asset_type": "etf"},
                                {"ticker": "2303.TW", "display": "2303", "asset_type": "stock"},
                            ],
                            "dynamic_constituents": {"source": "tw50_history_csv", "path": "data/tw50_constituents.csv"},
                        }
                    ],
                    signal_date=dates[-1].strftime("%Y-%m-%d"),
                    warmup_start=dates[0].strftime("%Y-%m-%d"),
                    cache_dir=root / "cache",
                    output_root=root / "out",
                )

            report = (Path(manifest["output_root"]) / "stock_pool_observation_report.md").read_text(encoding="utf-8")

            self.assertIn("2303.TW", report)
            self.assertIn("暫無正式觀察", report)
            self.assertIn("資料不足：缺少價格資料", report)
            self.assertIn("正式報告狀態：停止發布，等待完整資料", report)
            self.assertFalse(manifest["formal_report_ready"])
            self.assertEqual(manifest["formal_report_blocker_count"], 1)
            self.assertFalse((Path(manifest["output_root"]) / "AI股票池觀察總覽_最新版_v20260612.pdf").exists())
            self.assertNotIn("0050.TW", report)
            self.assertNotIn("0050(0050)", report)
            self.assertNotIn("跳過股票池", report)
            self.assertNotIn("未納入摘要", report)
            self.assertNotIn("No price data available", report)
            self.assertNotIn("使用邊界", report)

    def test_batch_resolves_radar_pool_from_formal_radar_metrics(self) -> None:
        dates = pd.bdate_range("2025-01-02", periods=160)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache_dir = root / "cache"
            radar_data_dir = root / "radar_data"
            cache_dir.mkdir()
            _trend_frame(dates, start=100, step=0.5, volume=20_000_000).reset_index(names="date").to_csv(
                cache_dir / "1111_TW.csv",
                index=False,
            )
            _write_formal_candidate_interface(
                radar_data_dir / "formal_radar_candidates.latest.csv",
                [
                    {
                        "report_date": dates[-1].strftime("%Y-%m-%d"),
                        "symbol": "1111",
                        "name": "測試記憶體",
                        "sector": "記憶體",
                        "score": "70.0",
                        "bucket_key": "watch",
                        "rank_in_bucket": "1",
                        "selected_for_backtest_pool": "true",
                        "market_cap_twd": "80,000,000,000",
                    }
                ],
            )
            pd.DataFrame(
                [
                    {
                        "date": dates[-1].strftime("%Y-%m-%d"),
                        "ticker": "1111.TW",
                        "margin_balance_5d_change_pct": 18.0,
                        "margin_overheat_flag": "true",
                    }
                ]
            ).to_csv(radar_data_dir / "margin_short.latest.csv", index=False, encoding="utf-8-sig")
            pd.DataFrame(
                [
                    {
                        "date": dates[-1].strftime("%Y-%m-%d"),
                        "ticker": "1111.TW",
                        "day_trading_volume_ratio": 42.0,
                    }
                ]
            ).to_csv(radar_data_dir / "day_trading.latest.csv", index=False, encoding="utf-8-sig")

            manifest = run_stock_pool_observation_batch(
                pools=[
                    {
                        "pool_id": "radar_mid_small_calibrated_v1",
                        "name": "雷達中小型校準版",
                        "strategy_preset": "radar_core_mid_small_calibrated_v1",
                        "resolved_symbols": [],
                    }
                ],
                signal_date=dates[-1].strftime("%Y-%m-%d"),
                warmup_start=dates[0].strftime("%Y-%m-%d"),
                cache_dir=cache_dir,
                output_root=root / "out",
                radar_data_dir=radar_data_dir,
                radar_top_n=5,
            )

            self.assertEqual(len(manifest["generated"]), 1)
            self.assertEqual(manifest["generated"][0]["pool_id"], "radar_mid_small_calibrated_v1")
            self.assertTrue(manifest["market_cap_source"].endswith("formal_radar_candidates.latest.csv"))
            self.assertEqual(manifest["market_cap_count"], 1)
            self.assertIn("margin_short", manifest["risk_factor_sources"])
            self.assertIn("day_trading", manifest["risk_factor_sources"])
            self.assertEqual(manifest["risk_factor_count"], 1)
            self.assertEqual(manifest["generated"][0]["top_ticker"], "1111.TW")
            self.assertTrue(manifest["generated"][0]["eligible_for_pool_selection"])
            self.assertEqual(manifest["generated"][0]["selection_layer"], "formal_candidate")
            self.assertEqual(manifest["generated"][0]["top_candidates"][0]["display"], "測試記憶體(1111)")
            self.assertEqual(manifest["generated"][0]["top_candidates"][0]["selection_layer"], "formal_candidate")
            self.assertEqual(
                manifest["generated"][0]["source_metadata"]["candidate_displays"],
                ["測試記憶體(1111)"],
            )
            self.assertEqual(manifest["skipped"], [])
            candidates = pd.read_csv(
                Path(manifest["generated"][0]["output_dir"]) / "stock_pool_observation_candidates.csv"
            )
            self.assertEqual(candidates.loc[0, "size_profile"], "mid_cap")
            self.assertEqual(candidates.loc[0, "market_cap_twd"], 80_000_000_000)
            self.assertGreater(candidates.loc[0, "flow_risk_score"], 0)
            self.assertIn("融資短線升溫", candidates.loc[0, "flow_risk_reasons"])
            self.assertIn("當沖比42.0%", candidates.loc[0, "flow_risk_reasons"])
            report = (Path(manifest["output_root"]) / "stock_pool_observation_report.md").read_text(encoding="utf-8")
            self.assertFalse(manifest["formal_report_ready"])
            self.assertNotIn("RADAR正式候選", report)
            self.assertNotIn("測試記憶體(1111)", report)

    def test_batch_does_not_emit_pool3_radar_artifacts_or_change_consensus(self) -> None:
        dates = pd.bdate_range("2025-01-02", periods=160)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache_dir = root / "cache"
            radar_data_dir = root / "radar_data"
            cache_dir.mkdir()
            radar_data_dir.mkdir()
            for ticker, step in {"2330.TW": 0.4, "2327.TW": 0.35, "1111.TW": 0.5, "2222.TW": 0.45}.items():
                _trend_frame(dates, start=100, step=step, volume=20_000_000).reset_index(names="date").to_csv(
                    cache_dir / f"{ticker.replace('.', '_')}.csv",
                    index=False,
                )
            pd.DataFrame(
                [
                    {"name": "記憶體", "capital_inflow_rank": 100, "capital_share": 20, "turnover_share_change": 30},
                    {"name": "PCB/載板", "capital_inflow_rank": 80, "capital_share": 15, "turnover_share_change": 20},
                    {"name": "ASIC/IP", "capital_inflow_rank": 70, "capital_share": 10, "turnover_share_change": 10},
                ]
            ).to_csv(radar_data_dir / "sector_metrics.refreshed.csv", index=False, encoding="utf-8-sig")
            _write_formal_candidate_interface(
                radar_data_dir / "formal_radar_candidates.latest.csv",
                [
                    {
                        "report_date": dates[-1].strftime("%Y-%m-%d"),
                        "symbol": "1111",
                        "name": "衛星一",
                        "sector": "記憶體",
                        "score": "80.0",
                        "bucket_key": "watch",
                        "rank_in_bucket": "1",
                        "selected_for_backtest_pool": "false",
                    },
                    {
                        "report_date": dates[-1].strftime("%Y-%m-%d"),
                        "symbol": "2222",
                        "name": "衛星二",
                        "sector": "PCB/載板",
                        "score": "70.0",
                        "bucket_key": "watch",
                        "rank_in_bucket": "2",
                        "selected_for_backtest_pool": "false",
                    },
                ],
            )
            readiness_dir = radar_data_dir / "formal_sources"
            readiness_dir.mkdir(parents=True)
            (readiness_dir / "date_aware_theme_membership_v2_readiness.json").write_text(
                json.dumps(
                    {
                        "ready": False,
                        "formal_top3_status": "formal_blocked",
                        "blocking_issues": ["no_accepted_v2_evidence_yet"],
                    }
                ),
                encoding="utf-8",
            )

            manifest = run_stock_pool_observation_batch(
                pools=[
                    {
                        "pool_id": "pool1",
                        "name": "池1",
                        "strategy_preset": "universal_pool_custom",
                        "vote_group": "three_perspective_v1",
                        "resolved_symbols": [symbol_entry("2330.TW", source="manual")],
                    },
                    {
                        "pool_id": "pool2",
                        "name": "池2",
                        "strategy_preset": "universal_pool_custom",
                        "vote_group": "three_perspective_v1",
                        "resolved_symbols": [symbol_entry("2327.TW", source="manual")],
                    },
                ],
                signal_date=dates[-1].strftime("%Y-%m-%d"),
                warmup_start=dates[0].strftime("%Y-%m-%d"),
                cache_dir=cache_dir,
                output_root=root / "out",
                radar_data_dir=radar_data_dir,
            )

        output_root = Path(manifest["output_root"])
        self.assertNotIn("pool3_radar_attack_satellite", manifest)
        self.assertFalse((output_root / "pool3_radar_attack_satellite.json").exists())
        self.assertFalse((output_root / "pool3_radar_attack_satellite.csv").exists())
        self.assertFalse((output_root / "pool3_radar_attack_satellite.md").exists())
        self.assertEqual(len(manifest["consensus"]["voters"]), 2)
        self.assertNotIn("1111.TW", [vote["ticker"] for vote in manifest["consensus"]["votes"]])

    def test_batch_resolves_dynamic_tw50_constituents_from_point_in_time_csv(self) -> None:
        dates = pd.bdate_range("2025-01-02", periods=160)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache_dir = root / "cache"
            cache_dir.mkdir()
            _trend_frame(dates, start=100, step=0.4, volume=20_000_000).reset_index(names="date").to_csv(
                cache_dir / "2330_TW.csv",
                index=False,
            )
            _trend_frame(dates, start=100, step=0.0, volume=20_000_000).reset_index(names="date").to_csv(
                cache_dir / "0050_TW.csv",
                index=False,
            )
            tw50_path = root / "tw50_constituents.csv"
            pd.DataFrame(
                [
                    {"effective_date": "2025-01-01", "ticker": "2330.TW", "name": "台積電"},
                    {"effective_date": "2026-01-01", "ticker": "2454.TW", "name": "聯發科"},
                ]
            ).to_csv(tw50_path, index=False, encoding="utf-8-sig")

            manifest = run_stock_pool_observation_batch(
                pools=[
                    {
                        "pool_id": "tw50_dynamic_constituents_v0",
                        "name": "動態0050成分股池 v0",
                        "strategy_preset": "universal_pool_custom",
                        "operational_observation": True,
                        "vote_group": "three_perspective_v1",
                        "resolved_symbols": [],
                        "dynamic_constituents": {"source": "tw50_history_csv", "path": str(tw50_path)},
                    }
                ],
                signal_date=dates[-1].strftime("%Y-%m-%d"),
                warmup_start=dates[0].strftime("%Y-%m-%d"),
                cache_dir=cache_dir,
                output_root=root / "out",
            )

            self.assertEqual(len(manifest["generated"]), 1)
            self.assertEqual(manifest["generated"][0]["top_ticker"], "2330.TW")
            self.assertEqual(manifest["generated"][0]["vote_group"], "three_perspective_v1")
            self.assertEqual(manifest["consensus"]["result_state"], "insufficient_votes")
            self.assertIsNone(manifest["consensus"]["winner_ticker"])
            manifest_payload = (Path(manifest["output_root"]) / "stock_pool_observation_manifest.json").read_text(encoding="utf-8")
            self.assertIn('"consensus"', manifest_payload)
            report = (Path(manifest["output_root"]) / "stock_pool_consensus_report.md").read_text(encoding="utf-8")
            self.assertIn("候選分歧診斷摘要", report)
            self.assertNotIn("三立場股票池表決摘要", report)
            self.assertNotIn("三池", report)
            summary_report = (Path(manifest["output_root"]) / "stock_pool_observation_report.md").read_text(encoding="utf-8")
            self.assertIn("正式模型基準", summary_report)
            self.assertNotIn("三池", summary_report)
            self.assertNotIn("風格補強池", summary_report)
            self.assertNotIn("舊三池診斷", summary_report)
            self.assertIn("大型廣度池", summary_report)

    def test_batch_generates_pool_with_partial_price_coverage(self) -> None:
        dates = pd.bdate_range("2025-01-02", periods=160)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache_dir = root / "cache"
            cache_dir.mkdir()
            _trend_frame(dates, start=100, step=0.5, volume=20_000_000).reset_index(names="date").to_csv(
                cache_dir / "1111_TW.csv",
                index=False,
            )

            def fake_download(*, tickers, **kwargs):
                if tickers == ["9999.TW"]:
                    raise ValueError("missing test price")
                return {"1111.TW": _trend_frame(dates, start=100, step=0.5, volume=20_000_000)}

            with patch("backtest_lab.stock_pool_observation.download_yfinance_prices", side_effect=fake_download):
                manifest = run_stock_pool_observation_batch(
                    pools=[
                        {
                            "pool_id": "partial_pool",
                            "name": "部分價格覆蓋池",
                            "strategy_preset": "universal_pool_custom",
                            "resolved_symbols": [
                                {
                                    "ticker": "1111.TW",
                                    "display": "可用(1111)",
                                    "asset_type": "stock",
                                },
                                {
                                    "ticker": "9999.TW",
                                    "display": "缺價(9999)",
                                    "asset_type": "stock",
                                },
                            ],
                        }
                    ],
                    signal_date=dates[-1].strftime("%Y-%m-%d"),
                    warmup_start=dates[0].strftime("%Y-%m-%d"),
                    cache_dir=cache_dir,
                    output_root=root / "out",
                )

            self.assertEqual(len(manifest["generated"]), 1)
            self.assertEqual(manifest["generated"][0]["top_ticker"], "1111.TW")
            self.assertEqual(manifest["generated"][0]["missing_price_tickers"], ["9999.TW"])
            self.assertEqual(manifest["skipped"], [])
            report = (Path(manifest["output_root"]) / "stock_pool_observation_report.md").read_text(encoding="utf-8")
            self.assertIn("缺價股票", report)
            self.assertIn("9999.TW", report)

    def test_batch_skips_without_latest_pdf_when_exact_signal_date_missing(self) -> None:
        dates = pd.bdate_range("2025-01-02", periods=160)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache_dir = root / "cache"
            cache_dir.mkdir()
            prices = {"1111.TW": _trend_frame(dates, start=100, step=0.5, volume=20_000_000)}

            with patch("backtest_lab.stock_pool_observation.download_yfinance_prices", return_value=prices):
                manifest = run_stock_pool_observation_batch(
                    pools=[
                        {
                            "pool_id": "strict_pool",
                            "name": "嚴格日期池",
                            "strategy_preset": "universal_pool_custom",
                            "resolved_symbols": [{"ticker": "1111.TW", "display": "測試(1111)", "asset_type": "stock"}],
                        }
                    ],
                    signal_date=(dates[-1] + pd.Timedelta(days=3)).strftime("%Y-%m-%d"),
                    warmup_start=dates[0].strftime("%Y-%m-%d"),
                    cache_dir=cache_dir,
                    output_root=root / "out",
                    require_exact_signal_date=True,
                )

            self.assertEqual(manifest["generated"], [])
            self.assertEqual(len(manifest["skipped"]), 1)
            self.assertIn("No exact common price data", manifest["skipped"][0]["reason"])
            self.assertFalse((Path(manifest["output_root"]) / "AI股票池觀察總覽_最新版_v20260612.pdf").exists())

    def test_batch_fallback_keeps_manifest_but_does_not_publish_latest_pdf(self) -> None:
        dates = pd.bdate_range("2025-01-02", periods=160)
        requested_signal_date = (dates[-1] + pd.Timedelta(days=3)).strftime("%Y-%m-%d")
        actual_signal_date = dates[-1].strftime("%Y-%m-%d")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache_dir = root / "cache"
            cache_dir.mkdir()
            prices = {"1111.TW": _trend_frame(dates, start=100, step=0.5, volume=20_000_000)}

            with patch("backtest_lab.stock_pool_observation.download_yfinance_prices", return_value=prices):
                manifest = run_stock_pool_observation_batch(
                    pools=[
                        {
                            "pool_id": "manual_fallback_pool",
                            "name": "手動補跑池",
                            "strategy_preset": "universal_pool_custom",
                            "resolved_symbols": [{"ticker": "1111.TW", "display": "測試(1111)", "asset_type": "stock"}],
                        }
                    ],
                    signal_date=requested_signal_date,
                    warmup_start=dates[0].strftime("%Y-%m-%d"),
                    cache_dir=cache_dir,
                    output_root=root / "out",
                    require_exact_signal_date=False,
                )

            output_root = Path(manifest["output_root"])
            manifest_payload = json.loads((output_root / "stock_pool_observation_manifest.json").read_text(encoding="utf-8"))

            self.assertEqual(len(manifest["generated"]), 1)
            self.assertEqual(manifest["requested_signal_date"], requested_signal_date)
            self.assertEqual(manifest["actual_signal_date"], actual_signal_date)
            self.assertEqual(manifest["signal_date"], actual_signal_date)
            self.assertTrue(manifest["signal_date_fallback_used"])
            self.assertIn(requested_signal_date, manifest["fallback_reason"])
            self.assertIn(actual_signal_date, manifest["fallback_reason"])
            self.assertEqual(manifest_payload["requested_signal_date"], requested_signal_date)
            self.assertEqual(manifest_payload["actual_signal_date"], actual_signal_date)
            self.assertFalse(manifest["formal_report_ready"])
            self.assertFalse((output_root / "AI股票池觀察總覽_最新版_v20260612.pdf").exists())
            report_text = (output_root / "stock_pool_observation_report.md").read_text(encoding="utf-8")
            self.assertIn(f"要求訊號日：{requested_signal_date}", report_text)
            self.assertIn(f"訊號日：{actual_signal_date}", report_text)
            self.assertIn("正式報告狀態：停止發布，等待完整資料", report_text)


def _trend_frame(dates: pd.DatetimeIndex, *, start: float, step: float, volume: int) -> pd.DataFrame:
    closes = [start + index * step for index in range(len(dates))]
    return pd.DataFrame(
        {
            "open": closes,
            "high": [value * 1.01 for value in closes],
            "low": [value * 0.99 for value in closes],
            "close": closes,
            "adj_close": closes,
            "volume": [volume] * len(dates),
            "dividend": [0.0] * len(dates),
            "stock_split": [0.0] * len(dates),
        },
        index=dates,
    )


def _late_surge_frame(dates: pd.DatetimeIndex, *, volume: int) -> pd.DataFrame:
    closes = [100.0] * (len(dates) - 20) + [100.0 + index * 3.0 for index in range(1, 21)]
    return pd.DataFrame(
        {
            "open": closes,
            "high": [value * 1.01 for value in closes],
            "low": [value * 0.99 for value in closes],
            "close": closes,
            "adj_close": closes,
            "volume": [volume] * len(dates),
            "dividend": [0.0] * len(dates),
            "stock_split": [0.0] * len(dates),
        },
        index=dates,
    )


def _final_surge_frame(
    dates: pd.DatetimeIndex,
    *,
    flat_days: int,
    final_gain: float,
    volume: int,
) -> pd.DataFrame:
    surge_days = len(dates) - flat_days
    closes = [100.0] * flat_days
    closes += [100.0 * (1 + final_gain * (index + 1) / surge_days) for index in range(surge_days)]
    return pd.DataFrame(
        {
            "open": closes,
            "high": [value * 1.01 for value in closes],
            "low": [value * 0.99 for value in closes],
            "close": closes,
            "adj_close": closes,
            "volume": [volume] * len(dates),
            "dividend": [0.0] * len(dates),
            "stock_split": [0.0] * len(dates),
        },
        index=dates,
    )


def _pullback_frame(
    dates: pd.DatetimeIndex,
    *,
    peak_gain: float,
    final_drawdown: float,
    volume: int,
) -> pd.DataFrame:
    stable_days = len(dates) - 20
    closes = [100.0] * stable_days
    peak = 100.0 * (1 + peak_gain)
    pullback_low = peak * (1 - final_drawdown - 0.03)
    final_close = peak * (1 - final_drawdown)
    closes += [peak] + [pullback_low] * 18 + [final_close]
    return pd.DataFrame(
        {
            "open": closes,
            "high": [value * 1.01 for value in closes],
            "low": [value * 0.99 for value in closes],
            "close": closes,
            "adj_close": closes,
            "volume": [volume] * len(dates),
            "dividend": [0.0] * len(dates),
            "stock_split": [0.0] * len(dates),
        },
        index=dates,
    )


def _tw50_test_pool(tickers: list[str]) -> dict[str, object]:
    symbols = []
    for ticker in tickers:
        symbol = symbol_entry(ticker, source="tw50_history_csv")
        symbol["market_cap_twd"] = 900_000_000_000
        symbols.append(symbol)
    return {
        "pool_id": "tw50_dynamic_constituents_v0",
        "name": "大型市場廣度池 v0",
        "strategy_preset": "universal_pool_custom",
        "resolved_symbols": symbols,
        "dynamic_constituents": {"source": "tw50_history_csv", "path": "data/tw50_constituents.csv"},
    }


def _core_defensive_test_pool(tickers: list[str]) -> dict[str, object]:
    symbols = []
    for ticker in tickers:
        symbol = symbol_entry(ticker, source="fixed")
        symbol["market_cap_twd"] = 900_000_000_000
        symbols.append(symbol)
    return {
        "pool_id": "large_core_bluechip_v0",
        "name": "核心風格補強池 v1",
        "strategy_preset": "core_defensive_style_v1",
        "resolved_symbols": symbols,
    }


def _write_stock_metrics(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    defaults = {
        "symbol": "",
        "name": "",
        "sector": "",
        "close": 100,
        "pullback_quality": 50,
        "chip_cleanliness": 50,
        "foreign_5d": 0,
        "trust_5d": 0,
        "margin_change_5d": 0,
        "pe": 0,
        "sector_pe_low": 0,
        "sector_pe_avg": 0,
        "sector_pe_high": 0,
        "fair_value_low": 0,
        "fair_value_avg": 0,
        "fair_value_high": 0,
        "revenue_yoy": 0,
        "revenue_mom": 0,
        "technical_setup": 50,
        "liquidity": 100,
        "risk_heat": 50,
        "thesis": "",
        "risk_reason": "",
    }
    frame = pd.DataFrame([{**defaults, **row} for row in rows])
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def _write_formal_candidate_interface(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")


def _target_stability_manifest(*, score_a: float, score_b: float) -> dict[str, object]:
    return {
        "generated": [
            {
                "pool_id": "ai_theme_large_cap_v20260613",
                "pool_name": "AI主線池",
                "active_in_trade_decision": True,
                "top_candidates": [
                    {"rank": 1, "ticker": "6669.TW", "display": "緯穎(6669)", "score": score_a},
                    {"rank": 2, "ticker": "00631L.TW", "display": "0050正二(00631L)", "score": score_b},
                ],
            },
            {
                "pool_id": "tw50_dynamic_constituents_v0",
                "pool_name": "大型廣度池",
                "active_in_trade_decision": False,
                "top_candidates": [],
            },
        ],
        "skipped": [],
    }


def _live_risk_manifest(*, risk_off_active: bool, attack_gate_active: bool) -> dict[str, object]:
    return {
        "signal_date": "2026-06-26",
        "actual_signal_date": "2026-06-26",
        "generated": [
            {
                "pool_id": "ai_theme_large_cap_v20260613",
                "pool_name": "AI主線池",
                "data_end_date": "2026-06-26",
                "active_in_trade_decision": True,
                "source_metadata": {
                    "risk_off_active": risk_off_active,
                    "attack_gate_active": attack_gate_active,
                    "market_regime_label": "正式風險診斷",
                },
                "top_candidates": [
                    {"rank": 1, "ticker": "6669.TW", "display": "緯穎(6669)", "score": 1.0},
                    {"rank": 2, "ticker": "00631L.TW", "display": "0050正二(00631L)", "score": 0.8},
                ],
            },
            {
                "pool_id": "tw50_dynamic_constituents_v0",
                "pool_name": "大型廣度池",
                "active_in_trade_decision": False,
                "top_candidates": [],
            },
        ],
        "skipped": [],
    }


def _chip_context_manifest(
    *,
    signal_date: str,
    bullish: float,
    institutional_risk: float,
    flow_risk: float,
    coverage_end: str = "",
) -> dict[str, object]:
    manifest: dict[str, object] = {
        "signal_date": signal_date,
        "actual_signal_date": signal_date,
        "generated": [
            {
                "pool_id": "ai_theme_large_cap_v20260613",
                "pool_name": "AI主線池",
                "data_end_date": signal_date,
                "active_in_trade_decision": True,
                "top_candidates": [
                    {
                        "rank": 1,
                        "ticker": "6669.TW",
                        "display": "緯穎(6669)",
                        "score": 1.0,
                        "is_model_target": True,
                        "bullish_flow_score": bullish,
                        "institutional_risk": institutional_risk,
                        "flow_risk_score": flow_risk,
                    }
                ],
            },
            {
                "pool_id": "tw50_dynamic_constituents_v0",
                "pool_name": "大型廣度池",
                "active_in_trade_decision": False,
                "top_candidates": [],
            },
        ],
        "skipped": [],
    }
    if coverage_end:
        manifest["risk_factor_coverage_end_by_kind"] = {"institutional": coverage_end}
    return manifest


if __name__ == "__main__":
    unittest.main()
