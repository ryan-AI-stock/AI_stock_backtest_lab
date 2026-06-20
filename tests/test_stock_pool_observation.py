from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

import test_paths  # noqa: F401

from backtest_lab.decision_layers import CANDIDATE_SOURCE, DATA_READINESS
from backtest_lab.stock_pool_observation import (
    _top_candidate_rows,
    build_dispatched_stock_pool_observation,
    build_stock_pool_observation,
    run_stock_pool_observation_batch,
    write_stock_pool_observation,
)
from backtest_lab.stock_pool_store import symbol_entry
from backtest_lab.valuation_source import ValuationSignal


class StockPoolObservationTest(unittest.TestCase):
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
        self.assertEqual(observation.gate_rule_id, "core_defensive_resilience_gate_v1")
        self.assertTrue(observation.attack_gate_open)
        self.assertTrue(observation.eligible_for_pool_selection)
        self.assertEqual(observation.selection_layer, "formal_candidate")
        self.assertIn("核心防守池 v1", observation.gate_reason)
        self.assertIn("60日相對0050韌性", observation.gate_reason)
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
        self.assertIn("60日相對0050韌性", top_rows[0]["gate_reason"])

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
        self.assertIn("60/120趨勢韌性=N", top_rows[0]["gate_reason"])

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
            "name": "核心防守風格池 v1",
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
            manifest_path = Path(manifest["output_root"]) / "stock_pool_observation_manifest.json"
            self.assertTrue(manifest_path.exists())
            self.assertTrue((Path(manifest["output_root"]) / "model_layer_audit.json").exists())
            self.assertTrue((Path(manifest["output_root"]) / "stock_pool_observation_summary.csv").exists())
            self.assertTrue((Path(manifest["output_root"]) / "stock_pool_candidate_reviews.csv").exists())
            self.assertTrue((Path(manifest["output_root"]) / "stock_pool_candidate_reviews.json").exists())
            self.assertTrue((Path(manifest["output_root"]) / "stock_pool_observation_report.md").exists())
            report_text = (Path(manifest["output_root"]) / "stock_pool_observation_report.md").read_text(encoding="utf-8")
            self.assertIn("測試專家", report_text)
            self.assertIn("月頻", report_text)
            self.assertTrue((Path(manifest["output_root"]) / "AI股票池觀察總覽_最新版_v20260612.pdf").exists())
            self.assertTrue(
                (Path(manifest["generated"][0]["output_dir"]) / "stock_pool_observation.json").exists()
            )

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
            self.assertIn("RADAR正式候選", report)
            self.assertIn("測試記憶體(1111)", report)

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
            self.assertIn("三立場股票池表決摘要", report)
            summary_report = (Path(manifest["output_root"]) / "stock_pool_observation_report.md").read_text(encoding="utf-8")
            self.assertIn("三池共識", summary_report)
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
        "name": "核心防守風格池 v1",
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


if __name__ == "__main__":
    unittest.main()
