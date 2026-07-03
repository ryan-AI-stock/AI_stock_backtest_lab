import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

import test_paths  # noqa: F401

from backtest_lab.dynamic_pool1_pit_readiness_contract import (
    run_dynamic_pool1_pit_readiness_contract,
)


class DynamicPool1PitReadinessContractTest(unittest.TestCase):
    def test_builds_readiness_contract_without_accepting_current_snapshots_as_formal(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache = root / "cache"
            radar = root / "radar"
            output = root / "out"
            cache.mkdir()
            radar.mkdir()

            pd.DataFrame(
                {
                    "date": ["2015-01-05", "2015-01-06"],
                    "open": [100, 101],
                    "high": [101, 102],
                    "low": [99, 100],
                    "close": [100, 101],
                    "adj_close": [100, 101],
                    "volume": [1000, 1200],
                }
            ).to_csv(cache / "2330_TW.csv", index=False)
            pd.DataFrame(
                [
                    {
                        "ticker": "00631L.TW",
                        "source_id": "00631l_twse_stock_day",
                        "source_path": "data/normalized_prices/00631L.csv",
                        "source_type": "twse_stock_day_backfill",
                        "first_date": "2014-11-03",
                        "last_date": "2015-12-31",
                        "price_source_ready": True,
                        "strategy_ready": False,
                        "synthetic_used": False,
                        "provenance": "test",
                        "notes": "price-only",
                    }
                ]
            ).to_csv(root / "price_source_registry.csv", index=False)
            pd.DataFrame(
                [
                    {
                        "effective_date": "2025-06-23",
                        "ticker": "2330.TW",
                        "name": "台積電",
                        "source": "seed_snapshot",
                        "source_updated_at": "2026-06-13",
                    }
                ]
            ).to_csv(root / "tw50_constituents.csv", index=False)
            pd.DataFrame(
                [
                    {
                        "effective_date": "2026-06-01",
                        "ticker": "2330.TW",
                        "symbol": "2330",
                        "name": "台積電",
                        "theme_role": "AI半導體",
                        "review_status": "active",
                    }
                ]
            ).to_csv(root / "ai_theme_candidates.csv", index=False)
            pd.DataFrame(
                [
                    {
                        "symbol": "2330",
                        "sector": "semiconductor",
                        "source_date": "2026-07-01",
                    }
                ]
            ).to_csv(radar / "sector_map.csv", index=False)
            pd.DataFrame(
                [
                    {
                        "symbol": "2330",
                        "market_cap_twd": 1000000000000,
                        "source_date": "2026-07-01",
                    }
                ]
            ).to_csv(radar / "stock_metrics.refreshed.csv", index=False)

            result = run_dynamic_pool1_pit_readiness_contract(
                output_dir=output,
                price_cache_dir=cache,
                price_source_registry=root / "price_source_registry.csv",
                tw50_constituents_path=root / "tw50_constituents.csv",
                ai_theme_candidates_path=root / "ai_theme_candidates.csv",
                radar_data_dir=radar,
            )
            self.assertEqual(result, output)

            required = [
                "all_listed_liquid_universe_pit_daily.csv",
                "monthly_revenue_pit.csv",
                "quarterly_fundamentals_pit.csv",
                "market_cap_pit.csv",
                "sector_membership_pit.csv",
                "sector_breadth_pit_daily.csv",
                "candidate_data_readiness_by_date.csv",
                "future_data_violation_audit.csv",
                "source_manifest.json",
                "readiness.json",
            ]
            for name in required:
                self.assertTrue((output / name).exists(), name)

            readiness = json.loads((output / "readiness.json").read_text(encoding="utf-8"))
            self.assertFalse(readiness["formal_model_changed"])
            self.assertFalse(readiness["trade_decision_changed"])
            self.assertFalse(readiness["active_in_trade_decision"])
            self.assertFalse(readiness["dynamic_pool1_shadow_challenger_ready"])
            self.assertEqual(readiness["future_data_violation_count"], 0)
            self.assertEqual(readiness["table_status"]["monthly_revenue_pit"]["status"], "blocked")
            self.assertEqual(readiness["table_status"]["sector_membership_pit"]["status"], "blocked")

            sector = pd.read_csv(output / "sector_membership_pit.csv")
            self.assertTrue(sector["diagnostic_only"].astype(bool).all())
            self.assertFalse(sector["accepted_for_formal"].astype(bool).any())

            audit = pd.read_csv(output / "future_data_violation_audit.csv")
            self.assertFalse(audit["future_data_violation"].astype(bool).any())
            self.assertFalse(audit["current_snapshot_used_as_historical"].astype(bool).any())

            by_date = pd.read_csv(output / "candidate_data_readiness_by_date.csv")
            self.assertEqual(set(by_date["year_bucket"]), {"2015-2021", "2022-2023", "2024-latest"})
            self.assertIn("source_date", pd.read_csv(output / "all_listed_liquid_universe_pit_daily.csv").columns)
            self.assertIn("release_date", pd.read_csv(output / "monthly_revenue_pit.csv").columns)
            self.assertIn("effective_date", pd.read_csv(output / "quarterly_fundamentals_pit.csv").columns)

    def test_liquidity_full_sweep_downgrades_universe_blocker_to_partial_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache = root / "cache"
            radar = root / "radar"
            sweep = root / "sweep"
            listing = root / "listing"
            output = root / "out"
            cache.mkdir()
            radar.mkdir()
            sweep.mkdir()
            listing.mkdir()

            pd.DataFrame(
                {
                    "date": ["2015-01-05"],
                    "open": [100],
                    "high": [101],
                    "low": [99],
                    "close": [100],
                    "adj_close": [100],
                    "volume": [1000],
                }
            ).to_csv(cache / "2330_TW.csv", index=False)
            pd.DataFrame(
                [
                    {
                        "shard_file": str(sweep / "shards" / "accepted_liquidity_rows_2015_01.csv"),
                        "row_count": 100,
                        "first_date": "2015-01-05",
                        "last_date": "2015-01-30",
                        "markets": "TWSE;TPEx",
                        "ticker_count": 10,
                        "git_tracked": False,
                    }
                ]
            ).to_csv(sweep / "accepted_liquidity_shard_manifest.csv", index=False)
            pd.DataFrame(
                [
                    {
                        "year": 2015,
                        "market": "TWSE",
                        "expected_weekday_attempts": 1,
                        "attempted": 1,
                        "rows_found_attempts": 1,
                        "no_rows_attempts": 0,
                        "failed_attempts": 0,
                        "missing_attempts": 0,
                        "accepted_liquidity_rows": 100,
                        "coverage_status": "complete",
                    }
                ]
            ).to_csv(sweep / "coverage_by_year_market.csv", index=False)
            pd.DataFrame(
                [
                    {
                        "source_id": "twse_mi_index_daily_presence",
                        "official_proxy_manual": "official",
                        "status": "accepted_for_daily_presence_and_liquidity_only",
                        "notes": "not a listing master",
                    }
                ]
            ).to_csv(sweep / "listing_status_source_inventory.csv", index=False)
            (sweep / "readiness_for_core.json").write_text(
                json.dumps(
                    {
                        "covered_date_range": {"start": "2015-01-05", "end": "2026-07-02"},
                        "accepted_liquidity_rows": 100,
                        "accepted_shard_count": 1,
                        "all_listed_liquid_universe_pit_daily_full_range_ready": True,
                        "listing_delisting_suspension_metadata_ready": False,
                        "ready_for_core_rerun": True,
                        "ready_for_strategy_replay": False,
                        "dynamic_pool1_shadow_challenger_ready": False,
                        "future_data_violation_count": 0,
                    }
                ),
                encoding="utf-8",
            )
            (sweep / "manifest.json").write_text("{}", encoding="utf-8")
            pd.DataFrame(
                [
                    {
                        "ticker": "7827",
                        "name": "漢康-KY創",
                        "market": "TWSE",
                        "event_type": "listing",
                        "event_date": "2026-05-29",
                        "source_date": "2026-07-03",
                        "formal_ready": False,
                    }
                ]
            ).to_csv(listing / "accepted_listing_metadata_rows.csv", index=False)
            pd.DataFrame(
                [
                    {
                        "ticker": "1788",
                        "name": "杏昌",
                        "market": "TPEx",
                        "event_type": "suspension",
                        "event_date": "2026-06-18",
                        "source_date": "2026-07-03",
                        "formal_ready": False,
                    }
                ]
            ).to_csv(listing / "accepted_suspension_event_rows.csv", index=False)
            pd.DataFrame(
                [
                    {
                        "dataset": "listing_delisting_suspension_master",
                        "market": "TWSE",
                        "blocked_requirement": "complete historical suspension/resumption master",
                        "formal_ready": False,
                    }
                ]
            ).to_csv(listing / "blocked_source_rows.csv", index=False)
            (listing / "readiness_for_core.json").write_text(
                json.dumps(
                    {
                        "status": "completed_partial_event_sources_but_master_ready_false",
                        "accepted_listing_metadata_rows": 1,
                        "accepted_suspension_event_rows": 1,
                        "proxy_source_rows": 3,
                        "blocked_source_rows": 1,
                        "listing_delisting_suspension_metadata_ready": False,
                        "ready_for_core_rerun": True,
                        "ready_for_strategy_replay": False,
                        "dynamic_pool1_shadow_challenger_ready": False,
                        "future_data_violation_count": 0,
                    }
                ),
                encoding="utf-8",
            )
            (listing / "manifest.json").write_text("{}", encoding="utf-8")

            run_dynamic_pool1_pit_readiness_contract(
                output_dir=output,
                price_cache_dir=cache,
                price_source_registry=root / "missing_registry.csv",
                tw50_constituents_path=root / "missing_tw50.csv",
                ai_theme_candidates_path=root / "missing_ai.csv",
                radar_data_dir=radar,
                liquidity_sweep_output=sweep,
                listing_metadata_output=listing,
            )

            readiness = json.loads((output / "readiness.json").read_text(encoding="utf-8"))
            self.assertEqual(readiness["table_status"]["all_listed_liquid_universe_pit_daily"]["status"], "partial")
            self.assertFalse(readiness["ready_for_strategy_replay"])
            self.assertFalse(readiness["dynamic_pool1_shadow_challenger_ready"])
            self.assertFalse(readiness["liquidity_full_sweep"]["listing_delisting_suspension_metadata_ready"])
            self.assertEqual(readiness["table_status"]["listing_delisting_suspension_metadata"]["status"], "partial")
            self.assertFalse(readiness["listing_metadata"]["listing_delisting_suspension_metadata_ready"])
            self.assertEqual(readiness["listing_metadata"]["accepted_event_rows"], 2)

            delta = pd.read_csv(output / "blocker_delta_after_liquidity_full_sweep.csv")
            universe_delta = delta[delta["blocker"].eq("all_listed_liquid_universe_pit_daily")].iloc[0]
            self.assertEqual(universe_delta["before_status"], "blocked")
            self.assertEqual(universe_delta["after_status"], "partial")

            summary = pd.read_csv(output / "dataset_readiness_summary.csv")
            universe_summary = summary[summary["dataset"].eq("all_listed_liquid_universe_pit_daily")].iloc[0]
            self.assertEqual(universe_summary["readiness_status"], "partial")

            listing_delta = pd.read_csv(output / "blocker_delta_after_listing_metadata.csv")
            listing_row = listing_delta[listing_delta["blocker"].eq("listing_delisting_suspension_metadata")].iloc[0]
            self.assertEqual(listing_row["after_status"], "partial")
            self.assertFalse(bool(listing_row["ready_for_strategy_replay"]))

    def test_listing_master_completion_is_stronger_partial_not_strategy_ready(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache = root / "cache"
            radar = root / "radar"
            listing = root / "listing_completion"
            output = root / "out"
            cache.mkdir()
            radar.mkdir()
            listing.mkdir()

            pd.DataFrame(
                {
                    "date": ["2015-01-05"],
                    "open": [100],
                    "high": [101],
                    "low": [99],
                    "close": [100],
                    "adj_close": [100],
                    "volume": [1000],
                }
            ).to_csv(cache / "2330_TW.csv", index=False)
            pd.DataFrame([{"ticker": "2330", "event_type": "listing"}]).to_csv(
                listing / "accepted_listing_metadata_rows.csv",
                index=False,
            )
            pd.DataFrame([{"ticker": "2330", "event_type": "suspension"}]).to_csv(
                listing / "accepted_suspension_event_rows.csv",
                index=False,
            )
            pd.DataFrame([{"ticker": "2330", "event_type": "code_name_change_candidate"}]).to_csv(
                listing / "accepted_code_name_change_rows.csv",
                index=False,
            )
            pd.DataFrame(columns=["ticker", "event_type"]).to_csv(
                listing / "accepted_transfer_listing_rows.csv",
                index=False,
            )
            pd.DataFrame(
                [
                    {
                        "dataset": "listing_delisting_suspension_master",
                        "market": "TPEx",
                        "blocked_requirement": "2015-2025 historical status",
                        "formal_ready": False,
                    }
                ]
            ).to_csv(listing / "blocked_source_rows.csv", index=False)
            (listing / "readiness_for_core.json").write_text(
                json.dumps(
                    {
                        "status": "completed_partial_improved_twse_status_coverage_but_master_ready_false",
                        "previous_accepted_event_rows": 557,
                        "accepted_listing_metadata_rows": 379,
                        "accepted_suspension_event_rows": 5582,
                        "accepted_code_name_change_rows": 33,
                        "accepted_transfer_listing_rows": 0,
                        "accepted_event_rows_total": 5994,
                        "new_or_carried_forward_event_rows_delta_vs_previous": 5437,
                        "proxy_source_rows": 2153,
                        "blocked_source_rows": 4,
                        "twse_suspension_resumption_range_sweep_candidate": True,
                        "twse_altered_trading_monthly_anchor_candidate": True,
                        "tpex_historical_listing_delisting_master_ready": False,
                        "tpex_historical_suspension_resumption_master_ready": False,
                        "listing_delisting_suspension_metadata_ready": False,
                        "ready_for_core_rerun": True,
                        "ready_for_strategy_replay": False,
                        "dynamic_pool1_shadow_challenger_ready": False,
                        "future_data_violation_count": 0,
                    }
                ),
                encoding="utf-8",
            )
            (listing / "manifest.json").write_text("{}", encoding="utf-8")

            run_dynamic_pool1_pit_readiness_contract(
                output_dir=output,
                price_cache_dir=cache,
                price_source_registry=root / "missing_registry.csv",
                tw50_constituents_path=root / "missing_tw50.csv",
                ai_theme_candidates_path=root / "missing_ai.csv",
                radar_data_dir=radar,
                liquidity_sweep_output=root / "missing_sweep",
                listing_metadata_output=listing,
            )

            readiness = json.loads((output / "readiness.json").read_text(encoding="utf-8"))
            self.assertEqual(readiness["table_status"]["listing_delisting_suspension_metadata"]["status"], "stronger_partial")
            self.assertEqual(readiness["listing_metadata"]["accepted_event_rows"], 5994)
            self.assertEqual(readiness["listing_metadata"]["delta_vs_previous"], 5437)
            self.assertTrue(readiness["listing_metadata"]["twse_only_diagnostic_possible"])
            self.assertFalse(readiness["ready_for_strategy_replay"])
            self.assertFalse(readiness["dynamic_pool1_shadow_challenger_ready"])

            completion_delta = pd.read_csv(output / "blocker_delta_after_listing_master_completion.csv")
            row = completion_delta[completion_delta["blocker"].eq("listing_delisting_suspension_metadata")].iloc[0]
            self.assertEqual(row["after_status"], "stronger_partial")
            self.assertEqual(int(row["accepted_event_rows_total"]), 5994)
            self.assertTrue(bool(row["twse_only_diagnostic_possible"]))
            self.assertFalse(bool(row["cross_market_strategy_replay_ready"]))

    def test_tpex_blocker_evidence_does_not_accept_current_rows_as_historical(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache = root / "cache"
            radar = root / "radar"
            tpex = root / "tpex"
            output = root / "out"
            cache.mkdir()
            radar.mkdir()
            tpex.mkdir()

            pd.DataFrame(
                {
                    "date": ["2015-01-05"],
                    "open": [100],
                    "high": [101],
                    "low": [99],
                    "close": [100],
                    "adj_close": [100],
                    "volume": [1000],
                }
            ).to_csv(cache / "2330_TW.csv", index=False)
            pd.DataFrame(
                [
                    {
                        "ticker": "1788",
                        "event_type": "suspension",
                        "event_date": "2026-06-18",
                        "source_date": "2026-07-03",
                    }
                ]
            ).to_csv(tpex / "accepted_current_or_carried_tpex_rows.csv", index=False)
            pd.DataFrame(
                [
                    {
                        "dataset": "tpex_historical_listing_status_master",
                        "market": "TPEx",
                        "blocked_requirement": "2015-2025 historical listing/status route",
                        "formal_ready": False,
                    }
                ]
            ).to_csv(tpex / "blocked_source_rows.csv", index=False)
            pd.DataFrame([{"route": "tpex_spendi_history", "status": "no_historical_rows"}]).to_csv(
                tpex / "source_probe_attempts.csv",
                index=False,
            )
            (tpex / "readiness_for_core.json").write_text(
                json.dumps(
                    {
                        "status": "blocked_with_attempt_evidence",
                        "accepted_historical_rows": 0,
                        "accepted_listing_metadata_rows": 0,
                        "accepted_suspension_event_rows": 0,
                        "accepted_status_snapshot_rows": 0,
                        "accepted_current_or_carried_tpex_rows": 236,
                        "source_probe_attempts": 7,
                        "blocked_source_rows": 4,
                        "tpex_2015_2025_historical_listing_status_ready": False,
                        "full_cross_market_listing_master_ready": False,
                        "ready_for_core_rerun": True,
                        "ready_for_strategy_replay": False,
                        "dynamic_pool1_shadow_challenger_ready": False,
                        "future_data_violation_count": 0,
                    }
                ),
                encoding="utf-8",
            )
            (tpex / "manifest.json").write_text("{}", encoding="utf-8")

            run_dynamic_pool1_pit_readiness_contract(
                output_dir=output,
                price_cache_dir=cache,
                price_source_registry=root / "missing_registry.csv",
                tw50_constituents_path=root / "missing_tw50.csv",
                ai_theme_candidates_path=root / "missing_ai.csv",
                radar_data_dir=radar,
                liquidity_sweep_output=root / "missing_sweep",
                listing_metadata_output=root / "missing_listing",
                tpex_status_output=tpex,
            )

            readiness = json.loads((output / "readiness.json").read_text(encoding="utf-8"))
            self.assertEqual(readiness["table_status"]["tpex_historical_listing_status"]["status"], "blocked_with_attempt_evidence")
            self.assertEqual(readiness["tpex_historical_listing_status"]["accepted_historical_rows"], 0)
            self.assertEqual(readiness["tpex_historical_listing_status"]["accepted_current_or_carried_tpex_rows"], 236)
            self.assertFalse(readiness["tpex_historical_listing_status"]["current_or_carried_rows_used_as_historical"])
            self.assertFalse(readiness["dynamic_pool1_shadow_challenger_ready"])

            delta = pd.read_csv(output / "blocker_delta_after_tpex_blocker_evidence.csv")
            row = delta[delta["blocker"].eq("tpex_historical_listing_status")].iloc[0]
            self.assertEqual(row["after_status"], "blocked_with_attempt_evidence")
            self.assertEqual(int(row["accepted_historical_rows"]), 0)
            self.assertEqual(int(row["accepted_current_or_carried_tpex_rows"]), 236)
            self.assertFalse(bool(row["current_or_carried_rows_used_as_historical"]))

    def test_tpex_static_reverse_partial_accepts_sample_rows_but_not_full_ready(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache = root / "cache"
            radar = root / "radar"
            tpex = root / "tpex_static_reverse"
            output = root / "out"
            cache.mkdir()
            radar.mkdir()
            tpex.mkdir()

            pd.DataFrame(
                {
                    "date": ["2015-01-05"],
                    "open": [100],
                    "high": [101],
                    "low": [99],
                    "close": [100],
                    "adj_close": [100],
                    "volume": [1000],
                }
            ).to_csv(cache / "2330_TW.csv", index=False)
            pd.DataFrame([{"ticker": "7402", "event_type": "listing", "event_date": "2015-12-29"}]).to_csv(
                tpex / "accepted_listing_metadata_rows.csv",
                index=False,
            )
            pd.DataFrame(
                [
                    {
                        "ticker": "3066",
                        "status_date": "2015-01-05",
                        "is_altered_trading": True,
                        "is_suspended": False,
                    }
                ]
            ).to_csv(tpex / "accepted_status_snapshot_rows.csv", index=False)
            pd.DataFrame(columns=["ticker", "event_type"]).to_csv(
                tpex / "accepted_suspension_event_rows.csv",
                index=False,
            )
            pd.DataFrame(
                [
                    {
                        "source": "TPEx bulletin/sprc",
                        "blocked_component": "suspension/resumption event history",
                        "blocked_reason": "current_only_no_historical_date_parameter",
                    }
                ]
            ).to_csv(tpex / "blocked_source_rows.csv", index=False)
            pd.DataFrame([{"route": "company/latest", "status": "accepted_sample"}]).to_csv(
                tpex / "route_probe_attempts.csv",
                index=False,
            )
            (tpex / "readiness_for_core.json").write_text(
                json.dumps(
                    {
                        "status": "completed_partial_with_accepted_historical_rows",
                        "accepted_listing_metadata_rows": 150,
                        "accepted_status_snapshot_rows": 107,
                        "accepted_suspension_event_rows": 0,
                        "route_probe_attempts": 32,
                        "future_data_violation_count": 0,
                        "tpex_static_reverse_contract_extracted": True,
                        "tpex_sample_historical_listing_delisting_ready": True,
                        "tpex_sample_historical_status_snapshot_ready": True,
                        "tpex_full_2015_2025_master_ready": False,
                        "listing_delisting_suspension_master_full_ready": False,
                        "ready_for_core_rerun": True,
                        "ready_for_strategy_replay": False,
                        "dynamic_pool1_shadow_challenger_ready": False,
                    }
                ),
                encoding="utf-8",
            )
            (tpex / "manifest.json").write_text("{}", encoding="utf-8")

            run_dynamic_pool1_pit_readiness_contract(
                output_dir=output,
                price_cache_dir=cache,
                price_source_registry=root / "missing_registry.csv",
                tw50_constituents_path=root / "missing_tw50.csv",
                ai_theme_candidates_path=root / "missing_ai.csv",
                radar_data_dir=radar,
                liquidity_sweep_output=root / "missing_sweep",
                listing_metadata_output=root / "missing_listing",
                tpex_status_output=tpex,
            )

            readiness = json.loads((output / "readiness.json").read_text(encoding="utf-8"))
            self.assertEqual(
                readiness["table_status"]["tpex_historical_listing_status"]["status"],
                "partial_with_accepted_historical_rows",
            )
            self.assertEqual(readiness["tpex_historical_listing_status"]["accepted_historical_rows"], 257)
            self.assertFalse(readiness["tpex_historical_listing_status"]["tpex_full_2015_2025_master_ready"])
            self.assertFalse(readiness["ready_for_strategy_replay"])
            self.assertFalse(readiness["dynamic_pool1_shadow_challenger_ready"])

            delta = pd.read_csv(output / "blocker_delta_after_tpex_blocker_evidence.csv")
            row = delta[delta["blocker"].eq("tpex_historical_listing_status")].iloc[0]
            self.assertEqual(row["delta"], "partial_historical_rows_accepted")
            self.assertEqual(int(row["accepted_historical_rows"]), 257)
            self.assertFalse(bool(row["full_2015_2025_master_ready"]))

    def test_tpex_full_route_coverage_remains_partial_without_transition_events(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache = root / "cache"
            radar = root / "radar"
            tpex = root / "tpex_full_route"
            output = root / "out"
            cache.mkdir()
            radar.mkdir()
            tpex.mkdir()

            pd.DataFrame(
                {
                    "date": ["2015-01-05"],
                    "open": [100],
                    "high": [101],
                    "low": [99],
                    "close": [100],
                    "adj_close": [100],
                    "volume": [1000],
                }
            ).to_csv(cache / "2330_TW.csv", index=False)
            (tpex / "readiness_for_core.json").write_text(
                json.dumps(
                    {
                        "status": "completed_full_route_coverage_suspension_events_still_blocked",
                        "covered_start": "2015-01-01",
                        "covered_end": "2025-12-31",
                        "route_request_attempts": 4040,
                        "failed_attempts": 0,
                        "accepted_listing_metadata_rows": 294,
                        "accepted_delisting_metadata_rows": 104,
                        "accepted_status_snapshot_rows": 90865,
                        "accepted_suspension_event_rows": 0,
                        "accepted_historical_rows": 91263,
                        "future_data_violation_count": 0,
                        "full_tpex_2015_2025_route_coverage_ready": True,
                        "full_tpex_2015_2025_master_ready": False,
                        "listing_delisting_suspension_master_full_ready": False,
                        "ready_for_core_rerun": True,
                        "ready_for_strategy_replay": False,
                        "dynamic_pool1_shadow_challenger_ready": False,
                    }
                ),
                encoding="utf-8",
            )
            pd.DataFrame([{"source": "TPEx bulletin/sprc", "blocked_component": "transition event ledger"}]).to_csv(
                tpex / "blocked_source_rows.csv",
                index=False,
            )
            pd.DataFrame([{"year": 2015, "coverage_status": "complete_route_coverage"}]).to_csv(
                tpex / "coverage_by_year.csv",
                index=False,
            )
            pd.DataFrame([{"dataset": "status_snapshot", "accepted_rows": 90865}]).to_csv(
                tpex / "accepted_rows_summary.csv",
                index=False,
            )
            (tpex / "manifest.json").write_text("{}", encoding="utf-8")

            run_dynamic_pool1_pit_readiness_contract(
                output_dir=output,
                price_cache_dir=cache,
                price_source_registry=root / "missing_registry.csv",
                tw50_constituents_path=root / "missing_tw50.csv",
                ai_theme_candidates_path=root / "missing_ai.csv",
                radar_data_dir=radar,
                liquidity_sweep_output=root / "missing_sweep",
                listing_metadata_output=root / "missing_listing",
                tpex_status_output=tpex,
            )

            readiness = json.loads((output / "readiness.json").read_text(encoding="utf-8"))
            tpex_status = readiness["tpex_historical_listing_status"]
            self.assertEqual(
                readiness["table_status"]["tpex_historical_listing_status"]["status"],
                "route_coverage_ready_status_snapshot_partial",
            )
            self.assertEqual(tpex_status["accepted_historical_rows"], 91263)
            self.assertTrue(tpex_status["daily_status_snapshot_asof_ready"])
            self.assertFalse(tpex_status["explicit_transition_event_ledger_ready"])
            self.assertFalse(tpex_status["ready_for_strategy_replay"])
            self.assertFalse(readiness["dynamic_pool1_shadow_challenger_ready"])

            delta = pd.read_csv(output / "blocker_delta_after_tpex_full_route_coverage.csv")
            row = delta[delta["blocker"].eq("tpex_historical_listing_status")].iloc[0]
            self.assertEqual(row["after_status"], "route_coverage_ready_status_snapshot_partial")
            self.assertEqual(int(row["accepted_historical_rows"]), 91263)
            self.assertTrue(bool(row["daily_status_snapshot_asof_ready"]))
            self.assertFalse(bool(row["explicit_transition_event_ledger_ready"]))

    def test_tpex_transition_candidates_remain_unverified_partial(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache = root / "cache"
            radar = root / "radar"
            tpex = root / "tpex_full_route"
            transition = root / "tpex_transition"
            output = root / "out"
            cache.mkdir()
            radar.mkdir()
            tpex.mkdir()
            transition.mkdir()

            pd.DataFrame(
                {
                    "date": ["2015-01-05"],
                    "open": [100],
                    "high": [101],
                    "low": [99],
                    "close": [100],
                    "adj_close": [100],
                    "volume": [1000],
                }
            ).to_csv(cache / "2330_TW.csv", index=False)
            (tpex / "readiness_for_core.json").write_text(
                json.dumps(
                    {
                        "status": "completed_full_route_coverage_suspension_events_still_blocked",
                        "covered_start": "2015-01-01",
                        "covered_end": "2025-12-31",
                        "route_request_attempts": 4040,
                        "accepted_status_snapshot_rows": 90865,
                        "accepted_historical_rows": 91263,
                        "accepted_suspension_event_rows": 0,
                        "full_tpex_2015_2025_route_coverage_ready": True,
                        "ready_for_strategy_replay": False,
                    }
                ),
                encoding="utf-8",
            )
            (tpex / "manifest.json").write_text("{}", encoding="utf-8")
            pd.DataFrame([{"source": "TPEx bulletin/sprc", "blocked_component": "transition event ledger"}]).to_csv(
                tpex / "blocked_source_rows.csv",
                index=False,
            )

            (transition / "readiness_for_core.json").write_text(
                json.dumps(
                    {
                        "status": "completed_transition_candidate_ledger_unverified",
                        "transition_candidate_count": 2295,
                        "announcement_verification_attempts": 160,
                        "announcement_verified_event_count": 0,
                        "unverified_transition_candidate_count": 2295,
                        "future_data_violation_count": 0,
                        "ready_for_core_rerun": True,
                        "ready_for_strategy_replay": False,
                        "dynamic_pool1_shadow_challenger_ready": False,
                    }
                ),
                encoding="utf-8",
            )
            (transition / "manifest.json").write_text("{}", encoding="utf-8")
            pd.DataFrame(
                [
                    {
                        "event_id": "tpex_transition_000001",
                        "ticker": "020003",
                        "event_date": "2022-04-28",
                        "source_type": "inferred_from_daily_status_snapshot",
                        "verification_status": "unverified_candidate",
                    }
                ]
            ).to_csv(transition / "transition_event_candidates.csv", index=False)
            pd.DataFrame(
                columns=[
                    "event_id",
                    "ticker",
                    "event_date",
                    "source_type",
                    "verification_status",
                ]
            ).to_csv(transition / "announcement_verified_events.csv", index=False)
            pd.DataFrame([{"event_id": "tpex_transition_000001"}]).to_csv(
                transition / "unverified_transition_candidates.csv",
                index=False,
            )
            pd.DataFrame([{"attempt": 1}]).to_csv(transition / "announcement_verification_attempts.csv", index=False)

            run_dynamic_pool1_pit_readiness_contract(
                output_dir=output,
                price_cache_dir=cache,
                price_source_registry=root / "missing_registry.csv",
                tw50_constituents_path=root / "missing_tw50.csv",
                ai_theme_candidates_path=root / "missing_ai.csv",
                radar_data_dir=radar,
                liquidity_sweep_output=root / "missing_sweep",
                listing_metadata_output=root / "missing_listing",
                tpex_status_output=tpex,
                tpex_transition_output=transition,
            )

            readiness = json.loads((output / "readiness.json").read_text(encoding="utf-8"))
            transition_status = readiness["tpex_transition_candidates"]
            self.assertEqual(transition_status["status"], "partial_unverified_inferred_transition_candidates")
            self.assertEqual(transition_status["transition_candidate_count"], 2295)
            self.assertEqual(transition_status["announcement_verified_event_count"], 0)
            self.assertFalse(transition_status["official_explicit_transition_event_ledger_ready"])
            self.assertFalse(transition_status["inferred_candidates_used_as_official_events"])
            self.assertFalse(readiness["ready_for_strategy_replay"])
            self.assertFalse(readiness["dynamic_pool1_shadow_challenger_ready"])

            delta = pd.read_csv(output / "blocker_delta_after_tpex_transition_candidates.csv")
            row = delta[delta["blocker"].eq("tpex_explicit_transition_event_ledger")].iloc[0]
            self.assertEqual(row["after_status"], "partial_unverified_inferred_transition_candidates")
            self.assertEqual(int(row["transition_candidate_count"]), 2295)
            self.assertEqual(int(row["announcement_verified_event_count"]), 0)
            self.assertFalse(bool(row["inferred_candidates_used_as_official_events"]))
            self.assertFalse(bool(row["official_explicit_transition_event_ledger_ready"]))

    def test_monthly_revenue_full_universe_is_source_candidate_not_strategy_ready(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache = root / "cache"
            radar = root / "radar"
            monthly = root / "monthly_revenue"
            output = root / "out"
            cache.mkdir()
            radar.mkdir()
            monthly.mkdir()

            pd.DataFrame(
                {
                    "date": ["2015-01-05"],
                    "open": [100],
                    "high": [101],
                    "low": [99],
                    "close": [100],
                    "adj_close": [100],
                    "volume": [1000],
                }
            ).to_csv(cache / "2330_TW.csv", index=False)
            (monthly / "readiness_for_core.json").write_text(
                json.dumps(
                    {
                        "status": "completed_full_universe_candidate",
                        "monthly_revenue_pit_full_universe_ready": True,
                        "monthly_revenue_pit_partial_ready": True,
                        "period_start": "2015-01",
                        "period_end": "2026-05",
                        "route_request_attempts": 548,
                        "failed_attempts": 0,
                        "accepted_rows": 244620,
                        "symbol_count": 1971,
                        "accepted_month_market": 274,
                        "expected_month_market": 274,
                        "coverage_ratio_month_market": 1,
                        "future_data_violation_count": 0,
                        "ready_for_core_rerun": True,
                        "ready_for_strategy_replay": False,
                    }
                ),
                encoding="utf-8",
            )
            (monthly / "manifest.json").write_text("{}", encoding="utf-8")
            pd.DataFrame(
                [
                    {
                        "shard_file": "accepted_monthly_revenue_rows_shards/accepted_monthly_revenue_rows_2015.csv",
                        "year": 2015,
                        "rows": 19045,
                        "symbol_count": 1612,
                        "period_start": "2015-01",
                        "period_end": "2015-12",
                    }
                ]
            ).to_csv(monthly / "accepted_monthly_revenue_rows_manifest.csv", index=False)
            pd.DataFrame(
                [
                    {
                        "ticker": "1259",
                        "name": "安心",
                        "market": "TPEx",
                        "revenue_year_month": "2015-01",
                        "revenue_value": 341286000,
                        "source_date": "2015-02-10",
                        "release_date": "2015-02-10",
                        "available_date": "2015-02-10",
                        "formal_exact": False,
                        "pit_usable": True,
                    }
                ]
            ).to_csv(monthly / "accepted_monthly_revenue_rows_sample.csv", index=False)
            pd.DataFrame([{"year": 2015, "rows": 19045}]).to_csv(
                monthly / "accepted_monthly_revenue_rows.csv",
                index=False,
            )

            run_dynamic_pool1_pit_readiness_contract(
                output_dir=output,
                price_cache_dir=cache,
                price_source_registry=root / "missing_registry.csv",
                tw50_constituents_path=root / "missing_tw50.csv",
                ai_theme_candidates_path=root / "missing_ai.csv",
                radar_data_dir=radar,
                liquidity_sweep_output=root / "missing_sweep",
                listing_metadata_output=root / "missing_listing",
                tpex_status_output=root / "missing_tpex",
                tpex_transition_output=root / "missing_transition",
                monthly_revenue_output=monthly,
            )

            readiness = json.loads((output / "readiness.json").read_text(encoding="utf-8"))
            monthly_status = readiness["monthly_revenue"]
            self.assertEqual(readiness["table_status"]["monthly_revenue_pit"]["status"], "source_candidate_ready")
            self.assertEqual(monthly_status["accepted_rows"], 244620)
            self.assertFalse(monthly_status["formal_exact"])
            self.assertFalse(readiness["ready_for_strategy_replay"])
            self.assertFalse(readiness["dynamic_pool1_shadow_challenger_ready"])

            delta = pd.read_csv(output / "blocker_delta_after_mops_monthly_revenue.csv")
            row = delta[delta["blocker"].eq("monthly_revenue_pit")].iloc[0]
            self.assertEqual(row["after_status"], "source_candidate_ready")
            self.assertEqual(int(row["accepted_rows"]), 244620)
            self.assertFalse(bool(row["formal_exact"]))
            self.assertFalse(bool(row["ready_for_strategy_replay"]))

    def test_quarterly_fundamentals_route_unlock_is_partial_not_full_ready(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache = root / "cache"
            radar = root / "radar"
            quarterly = root / "quarterly"
            output = root / "out"
            cache.mkdir()
            radar.mkdir()
            quarterly.mkdir()

            pd.DataFrame(
                {
                    "date": ["2015-01-05"],
                    "open": [100],
                    "high": [101],
                    "low": [99],
                    "close": [100],
                    "adj_close": [100],
                    "volume": [1000],
                }
            ).to_csv(cache / "2330_TW.csv", index=False)
            (quarterly / "readiness_for_core.json").write_text(
                json.dumps(
                    {
                        "status": "completed_route_unlocked_sample_rows",
                        "quarterly_fundamentals_route_unlocked": True,
                        "quarterly_fundamentals_pit_full_universe_ready": False,
                        "sample_rows": 320,
                        "full_source_rows_observed": 320,
                        "tested_periods": ["2015Q4", "2024Q4"],
                        "tested_markets": ["otc", "sii"],
                        "source_type": "source_candidate_no_exact_filing_date",
                        "formal_exact": False,
                        "filing_date_available": False,
                        "available_date_policy": "conservative statutory deadline, not exact filing timestamp",
                        "future_data_violation_count": 0,
                        "ready_for_core_rerun": True,
                        "ready_for_strategy_replay": False,
                    }
                ),
                encoding="utf-8",
            )
            (quarterly / "manifest.json").write_text("{}", encoding="utf-8")
            pd.DataFrame(
                [
                    {
                        "ticker": "2330",
                        "period": "2015Q4",
                        "market": "sii",
                        "source_type": "source_candidate_no_exact_filing_date",
                        "formal_exact": False,
                    }
                ]
            ).to_csv(quarterly / "accepted_quarterly_fundamentals_sample_rows.csv", index=False)
            pd.DataFrame([{"api": "ajax_t163sb04", "status": "ok"}]).to_csv(
                quarterly / "route_probe_attempts.csv",
                index=False,
            )

            run_dynamic_pool1_pit_readiness_contract(
                output_dir=output,
                price_cache_dir=cache,
                price_source_registry=root / "missing_registry.csv",
                tw50_constituents_path=root / "missing_tw50.csv",
                ai_theme_candidates_path=root / "missing_ai.csv",
                radar_data_dir=radar,
                liquidity_sweep_output=root / "missing_sweep",
                listing_metadata_output=root / "missing_listing",
                tpex_status_output=root / "missing_tpex",
                tpex_transition_output=root / "missing_transition",
                monthly_revenue_output=root / "missing_monthly",
                quarterly_fundamentals_output=quarterly,
            )

            readiness = json.loads((output / "readiness.json").read_text(encoding="utf-8"))
            quarterly_status = readiness["quarterly_fundamentals"]
            self.assertEqual(
                readiness["table_status"]["quarterly_fundamentals_pit"]["status"],
                "route_unlocked_source_candidate_partial",
            )
            self.assertEqual(quarterly_status["sample_rows"], 320)
            self.assertFalse(quarterly_status["quarterly_fundamentals_pit_full_universe_ready"])
            self.assertFalse(quarterly_status["formal_exact"])
            self.assertFalse(quarterly_status["filing_date_available"])
            self.assertFalse(readiness["ready_for_strategy_replay"])
            self.assertFalse(readiness["dynamic_pool1_shadow_challenger_ready"])

            delta = pd.read_csv(output / "blocker_delta_after_quarterly_fundamentals_route_unlock.csv")
            row = delta[delta["blocker"].eq("quarterly_fundamentals_pit")].iloc[0]
            self.assertEqual(row["after_status"], "route_unlocked_source_candidate_partial")
            self.assertEqual(int(row["sample_rows"]), 320)
            self.assertFalse(bool(row["formal_exact"]))
            self.assertFalse(bool(row["filing_date_available"]))
            self.assertFalse(bool(row["ready_for_strategy_replay"]))

    def test_quarterly_fundamentals_full_sweep_is_source_candidate_not_exact(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache = root / "cache"
            radar = root / "radar"
            quarterly = root / "quarterly_full"
            output = root / "out"
            cache.mkdir()
            radar.mkdir()
            quarterly.mkdir()

            pd.DataFrame(
                {
                    "date": ["2015-01-05"],
                    "open": [100],
                    "high": [101],
                    "low": [99],
                    "close": [100],
                    "adj_close": [100],
                    "volume": [1000],
                }
            ).to_csv(cache / "2330_TW.csv", index=False)
            (quarterly / "readiness_for_core.json").write_text(
                json.dumps(
                    {
                        "status": "completed_full_sweep_source_candidate",
                        "quarterly_fundamentals_full_sweep_ready": True,
                        "covered_start": "2015Q1",
                        "covered_end": "2026Q1",
                        "accepted_rows": 79046,
                        "symbol_count": 1970,
                        "failed_or_missing_period_markets": 0,
                        "route_request_attempts": 180,
                        "source_type": "source_candidate_no_exact_filing_date",
                        "formal_exact": False,
                        "filing_date_available": False,
                        "future_data_violation_count": 0,
                        "ready_for_core_rerun": True,
                        "ready_for_strategy_replay": False,
                    }
                ),
                encoding="utf-8",
            )
            (quarterly / "manifest.json").write_text("{}", encoding="utf-8")
            pd.DataFrame(
                [
                    {
                        "shard_path": "shards/accepted_quarterly_fundamentals_rows_2015.csv",
                        "fiscal_year": 2015,
                        "rows": 6204,
                    }
                ]
            ).to_csv(quarterly / "accepted_quarterly_fundamentals_rows_manifest.csv", index=False)
            pd.DataFrame([{"market": "sii", "rows": 43705, "symbol_count": 1079}]).to_csv(
                quarterly / "coverage_by_market.csv",
                index=False,
            )
            pd.DataFrame([{"period": "2015Q1", "rows": 1000}]).to_csv(
                quarterly / "coverage_by_year_quarter.csv",
                index=False,
            )
            pd.DataFrame(columns=["period", "market", "reason"]).to_csv(
                quarterly / "missing_or_failed_periods.csv",
                index=False,
            )

            run_dynamic_pool1_pit_readiness_contract(
                output_dir=output,
                price_cache_dir=cache,
                price_source_registry=root / "missing_registry.csv",
                tw50_constituents_path=root / "missing_tw50.csv",
                ai_theme_candidates_path=root / "missing_ai.csv",
                radar_data_dir=radar,
                liquidity_sweep_output=root / "missing_sweep",
                listing_metadata_output=root / "missing_listing",
                tpex_status_output=root / "missing_tpex",
                tpex_transition_output=root / "missing_transition",
                monthly_revenue_output=root / "missing_monthly",
                quarterly_fundamentals_output=quarterly,
            )

            readiness = json.loads((output / "readiness.json").read_text(encoding="utf-8"))
            quarterly_status = readiness["quarterly_fundamentals"]
            self.assertEqual(
                readiness["table_status"]["quarterly_fundamentals_pit"]["status"],
                "full_sweep_source_candidate_ready",
            )
            self.assertEqual(quarterly_status["accepted_rows"], 79046)
            self.assertEqual(quarterly_status["symbol_count"], 1970)
            self.assertFalse(quarterly_status["formal_exact"])
            self.assertFalse(quarterly_status["filing_date_available"])
            self.assertFalse(readiness["ready_for_strategy_replay"])
            self.assertFalse(readiness["dynamic_pool1_shadow_challenger_ready"])

            delta = pd.read_csv(output / "blocker_delta_after_quarterly_fundamentals_route_unlock.csv")
            row = delta[delta["blocker"].eq("quarterly_fundamentals_pit")].iloc[0]
            self.assertEqual(row["after_status"], "full_sweep_source_candidate_ready")
            self.assertEqual(int(row["accepted_rows"]), 79046)
            self.assertFalse(bool(row["formal_exact"]))
            self.assertFalse(bool(row["ready_for_strategy_replay"]))

    def test_market_cap_tpex_partial_is_not_full_or_free_float_ready(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache = root / "cache"
            radar = root / "radar"
            market_cap = root / "market_cap"
            output = root / "out"
            cache.mkdir()
            radar.mkdir()
            market_cap.mkdir()

            pd.DataFrame(
                {
                    "date": ["2015-01-05"],
                    "open": [100],
                    "high": [101],
                    "low": [99],
                    "close": [100],
                    "adj_close": [100],
                    "volume": [1000],
                }
            ).to_csv(cache / "2330_TW.csv", index=False)
            (market_cap / "readiness_for_core.json").write_text(
                json.dumps(
                    {
                        "status": "completed_partial_tpex_total_market_cap_proxy_sample_twse_blocked",
                        "market_cap_pit_ready": False,
                        "market_cap_pit_partial_ready": True,
                        "accepted_rows": 240,
                        "accepted_markets": ["TPEx"],
                        "blocked_markets": ["TWSE"],
                        "source_type": "shares_derived_official_daily_candidate",
                        "formal_exact": False,
                        "free_float_market_cap_ready": False,
                        "future_data_violation_count": 0,
                        "ready_for_core_rerun": True,
                        "ready_for_strategy_replay": False,
                    }
                ),
                encoding="utf-8",
            )
            (market_cap / "manifest.json").write_text("{}", encoding="utf-8")
            pd.DataFrame(
                [
                    {
                        "ticker": "1259",
                        "market": "TPEx",
                        "date": "2015-01-05",
                        "market_cap": 1000000000,
                        "source_type": "shares_derived_official_daily_candidate",
                        "formal_exact": False,
                    }
                ]
            ).to_csv(market_cap / "accepted_market_cap_rows.csv", index=False)
            pd.DataFrame([{"market": "TPEx", "rows": 240}]).to_csv(
                market_cap / "accepted_market_cap_rows_manifest.csv",
                index=False,
            )
            pd.DataFrame([{"market": "TPEx", "coverage": "sample"}]).to_csv(
                market_cap / "coverage_by_market.csv",
                index=False,
            )

            run_dynamic_pool1_pit_readiness_contract(
                output_dir=output,
                price_cache_dir=cache,
                price_source_registry=root / "missing_registry.csv",
                tw50_constituents_path=root / "missing_tw50.csv",
                ai_theme_candidates_path=root / "missing_ai.csv",
                radar_data_dir=radar,
                liquidity_sweep_output=root / "missing_sweep",
                listing_metadata_output=root / "missing_listing",
                tpex_status_output=root / "missing_tpex",
                tpex_transition_output=root / "missing_transition",
                monthly_revenue_output=root / "missing_monthly",
                quarterly_fundamentals_output=root / "missing_quarterly",
                market_cap_output=market_cap,
            )

            readiness = json.loads((output / "readiness.json").read_text(encoding="utf-8"))
            market_status = readiness["market_cap"]
            self.assertEqual(
                readiness["table_status"]["market_cap_pit"]["status"],
                "tpex_partial_total_market_cap_source_candidate",
            )
            self.assertEqual(market_status["accepted_rows"], 240)
            self.assertEqual(market_status["accepted_markets"], ["TPEx"])
            self.assertEqual(market_status["blocked_markets"], ["TWSE"])
            self.assertFalse(market_status["formal_exact"])
            self.assertFalse(market_status["free_float_market_cap_ready"])
            self.assertFalse(readiness["ready_for_strategy_replay"])
            self.assertFalse(readiness["dynamic_pool1_shadow_challenger_ready"])

            delta = pd.read_csv(output / "blocker_delta_after_market_cap_partial.csv")
            row = delta[delta["blocker"].eq("market_cap_pit")].iloc[0]
            self.assertEqual(row["after_status"], "tpex_partial_total_market_cap_source_candidate")
            self.assertEqual(int(row["accepted_rows"]), 240)
            self.assertFalse(bool(row["formal_exact"]))
            self.assertFalse(bool(row["free_float_market_cap_ready"]))

    def test_market_cap_tpex_full_sweep_still_blocks_cross_market_replay(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache = root / "cache"
            radar = root / "radar"
            market_cap = root / "market_cap"
            output = root / "out"
            cache.mkdir()
            radar.mkdir()
            market_cap.mkdir()

            pd.DataFrame(
                {
                    "date": ["2015-01-05"],
                    "open": [100],
                    "high": [101],
                    "low": [99],
                    "close": [100],
                    "adj_close": [100],
                    "volume": [1000],
                }
            ).to_csv(cache / "2330_TW.csv", index=False)
            (market_cap / "readiness_for_core.json").write_text(
                json.dumps(
                    {
                        "status": "completed_partial_tpex_full_candidate_twse_blocked",
                        "market_cap_pit_ready": False,
                        "market_cap_pit_partial_ready": True,
                        "tpex_full_sweep_completed": True,
                        "tpex_completed_dates": 3002,
                        "tpex_expected_weekday_dates": 3002,
                        "tpex_missing_dates": 0,
                        "tpex_accepted_rows": 2123871,
                        "accepted_markets": ["TPEx"],
                        "blocked_markets": ["TWSE"],
                        "source_type": "shares_derived_official_daily_candidate",
                        "formal_exact": False,
                        "twse_route_unlocked": False,
                        "free_float_market_cap_ready": False,
                        "future_data_violation_count": 0,
                        "ready_for_core_rerun": True,
                        "ready_for_strategy_replay": False,
                    }
                ),
                encoding="utf-8",
            )
            (market_cap / "manifest.json").write_text("{}", encoding="utf-8")
            pd.DataFrame([{"market": "TPEx", "rows": 2123871, "shard_count": 139}]).to_csv(
                market_cap / "accepted_market_cap_rows_manifest.csv",
                index=False,
            )
            pd.DataFrame(
                [
                    {
                        "market": "TPEx",
                        "accepted_rows": 2123871,
                        "covered_dates": 3002,
                        "status": "full_sweep_completed",
                    },
                    {
                        "market": "TWSE",
                        "accepted_rows": 0,
                        "covered_dates": 0,
                        "status": "blocked_route_not_unlocked_for_historical_market_cap",
                    },
                ]
            ).to_csv(market_cap / "coverage_by_market.csv", index=False)

            run_dynamic_pool1_pit_readiness_contract(
                output_dir=output,
                price_cache_dir=cache,
                price_source_registry=root / "missing_registry.csv",
                tw50_constituents_path=root / "missing_tw50.csv",
                ai_theme_candidates_path=root / "missing_ai.csv",
                radar_data_dir=radar,
                liquidity_sweep_output=root / "missing_sweep",
                listing_metadata_output=root / "missing_listing",
                tpex_status_output=root / "missing_tpex",
                tpex_transition_output=root / "missing_transition",
                monthly_revenue_output=root / "missing_monthly",
                quarterly_fundamentals_output=root / "missing_quarterly",
                market_cap_output=market_cap,
            )

            readiness = json.loads((output / "readiness.json").read_text(encoding="utf-8"))
            market_status = readiness["market_cap"]
            self.assertEqual(
                readiness["table_status"]["market_cap_pit"]["status"],
                "tpex_full_total_market_cap_source_candidate",
            )
            self.assertEqual(market_status["accepted_rows"], 2123871)
            self.assertEqual(market_status["accepted_markets"], ["TPEx"])
            self.assertEqual(market_status["blocked_markets"], ["TWSE"])
            self.assertTrue(market_status["tpex_full_sweep_completed"])
            self.assertEqual(market_status["tpex_completed_dates"], 3002)
            self.assertEqual(market_status["tpex_missing_dates"], 0)
            self.assertFalse(market_status["formal_exact"])
            self.assertFalse(market_status["free_float_market_cap_ready"])
            self.assertFalse(readiness["ready_for_strategy_replay"])
            self.assertFalse(readiness["dynamic_pool1_shadow_challenger_ready"])

            delta = pd.read_csv(output / "blocker_delta_after_tpex_market_cap_full_sweep.csv")
            row = delta[delta["blocker"].eq("market_cap_pit")].iloc[0]
            self.assertEqual(row["after_status"], "tpex_full_total_market_cap_source_candidate")
            self.assertEqual(
                row["delta"],
                "tpex_full_total_market_cap_source_candidate_ready_twse_free_float_blocked",
            )
            self.assertEqual(int(row["accepted_rows"]), 2123871)
            self.assertEqual(int(row["tpex_completed_dates"]), 3002)
            self.assertEqual(int(row["tpex_missing_dates"]), 0)
            self.assertFalse(bool(row["formal_exact"]))
            self.assertFalse(bool(row["free_float_market_cap_ready"]))


if __name__ == "__main__":
    unittest.main()
