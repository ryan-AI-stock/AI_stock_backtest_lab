from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


TASK_ID = "TASK-BACKTEST-DATA-DYNAMIC-POOL1-PIT-READINESS-CONTRACT-001"
DEFAULT_OUTPUT_DIR = "outputs/dynamic_pool1_pit_readiness_contract_20260703"
DEFAULT_PRICE_CACHE_DIR = "backtest_cache/stock_pool_observations"
DEFAULT_PRICE_SOURCE_REGISTRY = "data/price_source_registry.csv"
DEFAULT_TW50_CONSTITUENTS = "data/tw50_constituents.csv"
DEFAULT_AI_THEME_CANDIDATES = "data/ai_theme_candidates.csv"
DEFAULT_RADAR_DATA_DIR = (
    "C:/Users/zergv/Documents/Codex/2026-05-23/ai-stock-rotation-radar-https-docs/data"
)
DEFAULT_LIQUIDITY_SWEEP_OUTPUT = (
    "C:/Users/zergv/Documents/Codex/2026-05-23/ai-stock-rotation-radar-https-docs/outputs/"
    "radar_dynamic_pool1_all_listed_liquid_universe_full_sweep_20260703"
)
DEFAULT_LISTING_METADATA_OUTPUT = (
    "C:/Users/zergv/Documents/Codex/2026-05-23/ai-stock-rotation-radar-https-docs/outputs/"
    "radar_dynamic_pool1_listing_delisting_suspension_master_20260703"
)
DEFAULT_TPEX_STATUS_OUTPUT = (
    "C:/Users/zergv/Documents/Codex/2026-05-23/ai-stock-rotation-radar-https-docs/outputs/"
    "radar_dynamic_pool1_tpex_historical_listing_status_master_20260703"
)
DEFAULT_TPEX_TRANSITION_OUTPUT = ""
DEFAULT_MONTHLY_REVENUE_OUTPUT = ""
DEFAULT_QUARTERLY_FUNDAMENTALS_OUTPUT = ""
DEFAULT_MARKET_CAP_OUTPUT = ""

YEAR_BUCKETS = (
    ("2015-2021", "2015-01-01", "2021-12-31"),
    ("2022-2023", "2022-01-01", "2023-12-31"),
    ("2024-latest", "2024-01-01", "2026-07-03"),
)

TABLE_SPECS = (
    "all_listed_liquid_universe_pit_daily",
    "monthly_revenue_pit",
    "quarterly_fundamentals_pit",
    "market_cap_pit",
    "sector_membership_pit",
    "sector_breadth_pit_daily",
)


def run_dynamic_pool1_pit_readiness_contract(
    *,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    price_cache_dir: str | Path = DEFAULT_PRICE_CACHE_DIR,
    price_source_registry: str | Path = DEFAULT_PRICE_SOURCE_REGISTRY,
    tw50_constituents_path: str | Path = DEFAULT_TW50_CONSTITUENTS,
    ai_theme_candidates_path: str | Path = DEFAULT_AI_THEME_CANDIDATES,
    radar_data_dir: str | Path = DEFAULT_RADAR_DATA_DIR,
    liquidity_sweep_output: str | Path | None = None,
    listing_metadata_output: str | Path | None = None,
    tpex_status_output: str | Path | None = None,
    tpex_transition_output: str | Path | None = None,
    monthly_revenue_output: str | Path | None = None,
    quarterly_fundamentals_output: str | Path | None = None,
    market_cap_output: str | Path | None = None,
) -> Path:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    run_log: list[dict[str, str]] = []

    def log(step: str, status: str, detail: str = "") -> None:
        run_log.append(
            {
                "timestamp": pd.Timestamp.now(tz="Asia/Taipei").strftime("%Y-%m-%d %H:%M:%S%z"),
                "step": step,
                "status": status,
                "detail": detail,
            }
        )
        pd.DataFrame(run_log).to_csv(output / "run_log.csv", index=False, encoding="utf-8-sig")
        (output / "current_step.txt").write_text(f"{step}:{status}\n{detail}", encoding="utf-8")

    try:
        log("inventory_sources", "started", "")
        liquidity_sweep = _load_liquidity_sweep(liquidity_sweep_output)
        listing_metadata = _load_listing_metadata(listing_metadata_output)
        tpex_status = _load_tpex_status_blocker(tpex_status_output)
        tpex_transition = _load_tpex_transition_candidates(tpex_transition_output)
        monthly_revenue = _load_monthly_revenue_pit(monthly_revenue_output)
        quarterly_fundamentals = _load_quarterly_fundamentals_route_unlock(quarterly_fundamentals_output)
        market_cap = _load_market_cap_pit(market_cap_output)
        source_inventory = _source_inventory(
            price_cache_dir=Path(price_cache_dir),
            price_source_registry=Path(price_source_registry),
            tw50_constituents_path=Path(tw50_constituents_path),
            ai_theme_candidates_path=Path(ai_theme_candidates_path),
            radar_data_dir=Path(radar_data_dir),
            liquidity_sweep=liquidity_sweep,
            listing_metadata=listing_metadata,
            tpex_status=tpex_status,
            tpex_transition=tpex_transition,
            monthly_revenue=monthly_revenue,
            quarterly_fundamentals=quarterly_fundamentals,
            market_cap=market_cap,
        )
        price_coverage = _price_cache_coverage(Path(price_cache_dir), Path(price_source_registry))

        log("build_contract_tables", "started", "")
        tables = _build_contract_tables(
            source_inventory, price_coverage, liquidity_sweep, monthly_revenue, quarterly_fundamentals, market_cap
        )
        readiness_by_date = _candidate_data_readiness_by_date(
            source_inventory, price_coverage, liquidity_sweep, monthly_revenue, quarterly_fundamentals, market_cap
        )
        violation_audit = _future_data_violation_audit(source_inventory)
        source_manifest = _source_manifest(source_inventory, price_coverage)
        readiness = _readiness_json(
            source_inventory,
            price_coverage,
            readiness_by_date,
            violation_audit,
            liquidity_sweep,
            listing_metadata,
            tpex_status,
            tpex_transition,
            monthly_revenue,
            quarterly_fundamentals,
            market_cap,
        )
        dataset_summary = _dataset_readiness_summary(readiness)
        blocker_delta = _blocker_delta_after_liquidity_full_sweep(liquidity_sweep)
        listing_delta = _blocker_delta_after_listing_metadata(listing_metadata)
        listing_completion_delta = _blocker_delta_after_listing_master_completion(listing_metadata)
        tpex_delta = _blocker_delta_after_tpex_blocker_evidence(tpex_status)
        tpex_full_route_delta = _blocker_delta_after_tpex_full_route_coverage(tpex_status)
        tpex_transition_delta = _blocker_delta_after_tpex_transition_candidates(tpex_status, tpex_transition)
        monthly_revenue_delta = _blocker_delta_after_mops_monthly_revenue(monthly_revenue)
        quarterly_delta = _blocker_delta_after_quarterly_fundamentals_route_unlock(quarterly_fundamentals)
        market_cap_delta = _blocker_delta_after_market_cap_partial(market_cap)

        log("write_outputs", "started", str(output))
        for table_name, frame in tables.items():
            frame.to_csv(output / f"{table_name}.csv", index=False, encoding="utf-8-sig")
        readiness_by_date.to_csv(output / "candidate_data_readiness_by_date.csv", index=False, encoding="utf-8-sig")
        dataset_summary.to_csv(output / "dataset_readiness_summary.csv", index=False, encoding="utf-8-sig")
        blocker_delta.to_csv(output / "blocker_delta_after_liquidity_full_sweep.csv", index=False, encoding="utf-8-sig")
        listing_delta.to_csv(output / "blocker_delta_after_listing_metadata.csv", index=False, encoding="utf-8-sig")
        listing_completion_delta.to_csv(
            output / "blocker_delta_after_listing_master_completion.csv",
            index=False,
            encoding="utf-8-sig",
        )
        tpex_delta.to_csv(output / "blocker_delta_after_tpex_blocker_evidence.csv", index=False, encoding="utf-8-sig")
        tpex_full_route_delta.to_csv(
            output / "blocker_delta_after_tpex_full_route_coverage.csv",
            index=False,
            encoding="utf-8-sig",
        )
        tpex_transition_delta.to_csv(
            output / "blocker_delta_after_tpex_transition_candidates.csv",
            index=False,
            encoding="utf-8-sig",
        )
        monthly_revenue_delta.to_csv(
            output / "blocker_delta_after_mops_monthly_revenue.csv",
            index=False,
            encoding="utf-8-sig",
        )
        quarterly_delta.to_csv(
            output / "blocker_delta_after_quarterly_fundamentals_route_unlock.csv",
            index=False,
            encoding="utf-8-sig",
        )
        quarterly_delta.to_csv(
            output / "blocker_delta_after_quarterly_fundamentals_full_sweep.csv",
            index=False,
            encoding="utf-8-sig",
        )
        market_cap_delta.to_csv(
            output / "blocker_delta_after_market_cap_partial.csv",
            index=False,
            encoding="utf-8-sig",
        )
        market_cap_delta.to_csv(
            output / "blocker_delta_after_tpex_market_cap_full_sweep.csv",
            index=False,
            encoding="utf-8-sig",
        )
        market_cap_delta.to_csv(
            output / "blocker_delta_after_twse_capital_stock_route_partial.csv",
            index=False,
            encoding="utf-8-sig",
        )
        market_cap_delta.to_csv(
            output / "blocker_delta_after_twse_capital_stock_full_sweep_proxy_contract.csv",
            index=False,
            encoding="utf-8-sig",
        )
        violation_audit.to_csv(output / "future_data_violation_audit.csv", index=False, encoding="utf-8-sig")
        (output / "source_manifest.json").write_text(
            json.dumps(source_manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (output / "readiness.json").write_text(json.dumps(readiness, ensure_ascii=False, indent=2), encoding="utf-8")
        (output / "next_step_handoff.md").write_text(_next_step_handoff(readiness), encoding="utf-8")
        (output / "final_summary_zh.md").write_text(_final_summary(readiness), encoding="utf-8")
        (output / "manifest.json").write_text(json.dumps(_manifest(readiness), ensure_ascii=False, indent=2), encoding="utf-8")
        pd.DataFrame([{"step": TASK_ID, "status": "completed_readiness_contract", "output_dir": str(output.resolve())}]).to_csv(
            output / "completed.csv",
            index=False,
            encoding="utf-8-sig",
        )
        pd.DataFrame(columns=["step", "status", "reason"]).to_csv(output / "failed.csv", index=False, encoding="utf-8-sig")
        log("completed", "completed", str(output.resolve()))
        (output / "current_step.txt").write_text("completed\n", encoding="utf-8")
        return output
    except Exception as exc:
        pd.DataFrame([{"step": TASK_ID, "status": "failed", "reason": str(exc)}]).to_csv(
            output / "failed.csv",
            index=False,
            encoding="utf-8-sig",
        )
        log("failed", "failed", str(exc))
        raise


def _source_inventory(
    *,
    price_cache_dir: Path,
    price_source_registry: Path,
    tw50_constituents_path: Path,
    ai_theme_candidates_path: Path,
    radar_data_dir: Path,
    liquidity_sweep: dict[str, Any],
    listing_metadata: dict[str, Any],
    tpex_status: dict[str, Any],
    tpex_transition: dict[str, Any],
    monthly_revenue: dict[str, Any],
    quarterly_fundamentals: dict[str, Any],
    market_cap: dict[str, Any],
) -> list[dict[str, Any]]:
    sources = [
        {
            "data_area": "price_adjusted_ohlcv",
            "source_name": "local_stock_pool_observation_price_cache",
            "path": price_cache_dir,
            "source_family": "local_cache",
            "pit_acceptance": "partial",
            "diagnostic_only": False,
            "blocked_reason": "ticker universe is not all-listed PIT; adjusted-close availability varies by file",
        },
        {
            "data_area": "price_supplemental_registry",
            "source_name": "price_source_registry",
            "path": price_source_registry,
            "source_family": "local_registry",
            "pit_acceptance": "partial",
            "diagnostic_only": False,
            "blocked_reason": "some sources are price-only or unadjusted-only; not enough for dynamic Pool1 formal replay",
        },
        {
            "data_area": "tw50_pit_candidate",
            "source_name": "tw50_constituents_current_snapshot",
            "path": tw50_constituents_path,
            "source_family": "current_snapshot",
            "pit_acceptance": "diagnostic_only",
            "diagnostic_only": True,
            "blocked_reason": "current/proxy snapshot cannot be used as historical all-listed or sector PIT universe",
        },
        {
            "data_area": "ai_theme_candidates",
            "source_name": "ai_theme_candidates_current_snapshot",
            "path": ai_theme_candidates_path,
            "source_family": "current_snapshot",
            "pit_acceptance": "diagnostic_only",
            "diagnostic_only": True,
            "blocked_reason": "2026 AI theme list cannot be backfilled to 2015 as dynamic Pool1 PIT universe",
        },
    ]
    radar_candidates = {
        "market_cap_pit": [
            "market_cap.latest.csv",
            "market_caps.latest.csv",
            "stock_metrics.refreshed.csv",
            "formal_radar_candidates.latest.csv",
        ],
        "sector_membership_pit": [
            "sector_map.csv",
            "sector_map.generated.csv",
            "sector_universe.csv",
            "theme_map.csv",
            "theme_universe.csv",
        ],
        "sector_breadth_pit_daily": [
            "sector_metrics.csv",
            "sector_metrics.refreshed.csv",
            "theme_history.generated.csv",
            "hot_sector_symbols.generated.csv",
        ],
        "all_listed_liquid_universe_pit_daily": [
            "market_universe.generated.csv",
            "market_quotes.generated.csv",
            "stock_metrics.csv",
            "stock_metrics.tracked.refreshed.csv",
        ],
    }
    for data_area, names in radar_candidates.items():
        for name in names:
            sources.append(
                {
                    "data_area": data_area,
                    "source_name": name,
                    "path": radar_data_dir / name,
                    "source_family": "radar_current_or_generated",
                    "pit_acceptance": "diagnostic_only",
                    "diagnostic_only": True,
                    "blocked_reason": "source lacks accepted historical release/effective-date PIT contract for 2015 backfill",
                }
            )

    if liquidity_sweep.get("exists"):
        readiness = liquidity_sweep.get("readiness", {})
        sources.append(
            {
                "data_area": "all_listed_liquid_universe_pit_daily",
                "source_name": "radar_dynamic_pool1_all_listed_liquid_universe_full_sweep",
                "path": liquidity_sweep.get("path", ""),
                "source_family": "radar_official_daily_liquidity_sweep",
                "pit_acceptance": "partial",
                "diagnostic_only": False,
                "blocked_reason": (
                    "accepted for daily liquidity/trading-table presence; listing/delisting/suspension metadata "
                    "is still not ready"
                ),
                "sweep_start": readiness.get("covered_date_range", {}).get("start", ""),
                "sweep_end": readiness.get("covered_date_range", {}).get("end", ""),
                "accepted_liquidity_rows": readiness.get("accepted_liquidity_rows", 0),
                "accepted_shard_count": readiness.get("accepted_shard_count", 0),
            }
        )

    if listing_metadata.get("exists"):
        readiness = listing_metadata.get("readiness", {})
        sources.append(
            {
                "data_area": "listing_delisting_suspension_metadata",
                "source_name": "radar_dynamic_pool1_listing_delisting_suspension_master",
                "path": listing_metadata.get("path", ""),
                "source_family": "radar_official_partial_event_sources",
                "pit_acceptance": "partial",
                "diagnostic_only": False,
                "blocked_reason": (
                    "partial listing/delisting/suspension event rows are available, but complete historical master "
                    "metadata is still not ready"
                ),
                "accepted_listing_metadata_rows": readiness.get("accepted_listing_metadata_rows", 0),
                "accepted_suspension_event_rows": readiness.get("accepted_suspension_event_rows", 0),
                "proxy_source_rows": readiness.get("proxy_source_rows", 0),
                "blocked_source_rows": readiness.get("blocked_source_rows", 0),
            }
        )

    if tpex_status.get("exists"):
        readiness = tpex_status.get("readiness", {})
        sources.append(
            {
                "data_area": "tpex_historical_listing_status",
                "source_name": "radar_dynamic_pool1_tpex_historical_listing_status_master",
                "path": tpex_status.get("path", ""),
                "source_family": "radar_blocker_evidence_package",
                "pit_acceptance": "blocked_with_attempt_evidence",
                "diagnostic_only": True,
                "blocked_reason": (
                    "TPEx 2015-2025 historical listing/status probe produced zero accepted historical rows; "
                    "current/carried 2026 rows are not accepted as historical PIT"
                ),
                "accepted_historical_rows": readiness.get("accepted_historical_rows", 0),
                "accepted_current_or_carried_tpex_rows": readiness.get("accepted_current_or_carried_tpex_rows", 0),
                "source_probe_attempts": readiness.get("source_probe_attempts", 0),
                "blocked_source_rows": readiness.get("blocked_source_rows", 0),
            }
        )

    if tpex_transition.get("exists"):
        readiness = tpex_transition.get("readiness", {})
        sources.append(
            {
                "data_area": "tpex_transition_event_candidates",
                "source_name": "radar_dynamic_pool1_tpex_suspension_transition_event_ledger",
                "path": tpex_transition.get("path", ""),
                "source_family": "radar_inferred_transition_candidates",
                "pit_acceptance": "partial_unverified_candidate",
                "diagnostic_only": True,
                "blocked_reason": (
                    "transition candidates are inferred from daily status snapshot diffs; announcement verified "
                    "events remain 0, so this is not an official explicit transition event ledger"
                ),
                "transition_candidate_count": readiness.get("transition_candidate_count", 0),
                "announcement_verified_event_count": readiness.get("announcement_verified_event_count", 0),
                "unverified_transition_candidate_count": readiness.get("unverified_transition_candidate_count", 0),
            }
        )

    if monthly_revenue.get("exists"):
        readiness = monthly_revenue.get("readiness", {})
        sources.append(
            {
                "data_area": "monthly_revenue_pit",
                "source_name": "radar_dynamic_pool1_mops_monthly_revenue_full_universe_pit",
                "path": monthly_revenue.get("path", ""),
                "source_family": "mops_static_monthly_revenue_conservative_available_date",
                "pit_acceptance": "source_candidate_ready",
                "diagnostic_only": False,
                "blocked_reason": (
                    "full-universe monthly revenue PIT source candidate is available with conservative "
                    "available_date, but formal_exact=false because per-company filing timestamps are not exact"
                ),
                "accepted_rows": readiness.get("accepted_rows", 0),
                "symbol_count": readiness.get("symbol_count", 0),
                "period_start": readiness.get("period_start", ""),
                "period_end": readiness.get("period_end", ""),
                "formal_exact": False,
            }
        )

    if quarterly_fundamentals.get("exists"):
        readiness = quarterly_fundamentals.get("readiness", {})
        sources.append(
            {
                "data_area": "quarterly_fundamentals_pit",
                "source_name": "radar_dynamic_pool1_quarterly_fundamentals_route_unlock",
                "path": quarterly_fundamentals.get("path", ""),
                "source_family": "mops_t163sb04_quarterly_fundamentals_source_candidate",
                "pit_acceptance": "source_candidate_partial",
                "diagnostic_only": True,
                "blocked_reason": (
                    "MOPS quarterly fundamentals source candidate is available, but formal_exact=false and "
                    "exact per-company filing_date is not ready"
                ),
                "sample_rows": readiness.get("sample_rows", 0),
                "accepted_rows": readiness.get("accepted_rows", 0),
                "symbol_count": readiness.get("symbol_count", 0),
                "formal_exact": False,
                "filing_date_available": False,
            }
        )

    if market_cap.get("exists"):
        summary = _market_cap_summary(market_cap)
        sources.append(
            {
                "data_area": "market_cap_pit",
                "source_name": "radar_dynamic_pool1_market_cap_pit",
                "path": market_cap.get("path", ""),
                "source_family": summary["source_type"],
                "pit_acceptance": summary["status"],
                "diagnostic_only": True,
                "blocked_reason": summary["remaining_blocker"],
                "accepted_rows": summary["accepted_rows"],
                "accepted_markets": ",".join(summary["accepted_markets"]),
                "blocked_markets": ",".join(summary["blocked_markets"]),
                "twse_capital_stock_sample_rows": summary["twse_capital_stock_sample_rows"],
                "formal_exact": False,
                "free_float_market_cap_ready": False,
            }
        )

    inventory: list[dict[str, Any]] = []
    for source in sources:
        inventory.append(_source_row(source))
    return inventory


def _source_row(source: dict[str, Any]) -> dict[str, Any]:
    path = Path(source["path"])
    exists = path.exists()
    columns: list[str] = []
    row_count = 0
    min_date = ""
    max_date = ""
    if exists and path.is_file() and path.suffix.lower() == ".csv":
        columns = _csv_columns(path)
        row_count = _count_csv_rows(path)
        min_date, max_date = _date_range_from_csv(path, columns)
    elif exists and path.is_dir():
        row_count = len(list(path.glob("*.csv")))
    has_source_date = any(column in columns for column in ("source_date", "source_updated_at", "date", "report_date"))
    has_release_date = "release_date" in columns
    has_effective_date = "effective_date" in columns
    return {
        **{key: value for key, value in source.items() if key != "path"},
        "path": str(path),
        "exists": exists,
        "row_count": row_count,
        "columns": columns,
        "min_date": min_date,
        "max_date": max_date,
        "has_source_date": has_source_date,
        "has_release_date": has_release_date,
        "has_effective_date": has_effective_date,
        "accepted_for_formal": False,
        "future_data_violation_count": 0,
    }


def _load_liquidity_sweep(liquidity_sweep_output: str | Path | None) -> dict[str, Any]:
    if liquidity_sweep_output is None:
        return {"exists": False, "path": ""}
    root = Path(liquidity_sweep_output)
    if not root.exists():
        return {"exists": False, "path": str(root), "missing_reason": "liquidity sweep output path does not exist"}
    readiness = _load_json(root / "readiness_for_core.json")
    manifest = _load_json(root / "manifest.json")
    shard_manifest_path = root / "accepted_liquidity_shard_manifest.csv"
    coverage_path = root / "coverage_by_year_market.csv"
    listing_inventory_path = root / "listing_status_source_inventory.csv"
    future_audit_path = root / "future_data_violation_audit.csv"
    shard_manifest = _read_csv_if_exists(shard_manifest_path)
    coverage = _read_csv_if_exists(coverage_path)
    listing_inventory = _read_csv_if_exists(listing_inventory_path)
    future_audit = _read_csv_if_exists(future_audit_path)
    return {
        "exists": True,
        "path": str(root),
        "readiness": readiness,
        "manifest": manifest,
        "shard_manifest_path": str(shard_manifest_path),
        "coverage_path": str(coverage_path),
        "listing_inventory_path": str(listing_inventory_path),
        "future_audit_path": str(future_audit_path),
        "shard_manifest_rows": int(len(shard_manifest)),
        "coverage_rows": int(len(coverage)),
        "listing_inventory_rows": int(len(listing_inventory)),
        "future_audit_rows": int(len(future_audit)),
        "shard_manifest": shard_manifest,
        "coverage": coverage,
        "listing_inventory": listing_inventory,
        "future_audit": future_audit,
    }


def _load_listing_metadata(listing_metadata_output: str | Path | None) -> dict[str, Any]:
    if listing_metadata_output is None:
        return {"exists": False, "path": ""}
    root = Path(listing_metadata_output)
    if not root.exists():
        return {"exists": False, "path": str(root), "missing_reason": "listing metadata output path does not exist"}
    readiness = _load_json(root / "readiness_for_core.json")
    manifest = _load_json(root / "manifest.json")
    accepted_listing_path = root / "accepted_listing_metadata_rows.csv"
    accepted_suspension_path = root / "accepted_suspension_event_rows.csv"
    blocked_path = root / "blocked_source_rows.csv"
    future_audit_path = root / "future_data_violation_audit.csv"
    return {
        "exists": True,
        "path": str(root),
        "readiness": readiness,
        "manifest": manifest,
        "accepted_listing_path": str(accepted_listing_path),
        "accepted_suspension_path": str(accepted_suspension_path),
        "blocked_path": str(blocked_path),
        "future_audit_path": str(future_audit_path),
        "accepted_listing_rows": int(len(_read_csv_if_exists(accepted_listing_path))),
        "accepted_suspension_rows": int(len(_read_csv_if_exists(accepted_suspension_path))),
        "blocked_rows": int(len(_read_csv_if_exists(blocked_path))),
        "future_audit_rows": int(len(_read_csv_if_exists(future_audit_path))),
    }


def _load_tpex_status_blocker(tpex_status_output: str | Path | None) -> dict[str, Any]:
    if tpex_status_output is None:
        return {"exists": False, "path": ""}
    root = Path(tpex_status_output)
    if not root.exists():
        return {"exists": False, "path": str(root), "missing_reason": "TPEx status output path does not exist"}
    readiness = _load_json(root / "readiness_for_core.json")
    manifest = _load_json(root / "manifest.json")
    current_path = root / "accepted_current_or_carried_tpex_rows.csv"
    blocked_path = root / "blocked_source_rows.csv"
    attempts_path = root / "source_probe_attempts.csv"
    future_audit_path = root / "future_data_violation_audit.csv"
    return {
        "exists": True,
        "path": str(root),
        "readiness": readiness,
        "manifest": manifest,
        "current_or_carried_path": str(current_path),
        "blocked_path": str(blocked_path),
        "attempts_path": str(attempts_path),
        "future_audit_path": str(future_audit_path),
        "current_or_carried_rows": int(len(_read_csv_if_exists(current_path))),
        "blocked_rows": int(len(_read_csv_if_exists(blocked_path))),
        "attempt_rows": int(len(_read_csv_if_exists(attempts_path))),
        "future_audit_rows": int(len(_read_csv_if_exists(future_audit_path))),
    }


def _load_tpex_transition_candidates(tpex_transition_output: str | Path | None) -> dict[str, Any]:
    if not tpex_transition_output:
        return {"exists": False, "path": ""}
    root = Path(tpex_transition_output)
    if not root.exists():
        return {"exists": False, "path": str(root), "missing_reason": "TPEx transition output path does not exist"}
    readiness = _load_json(root / "readiness_for_core.json")
    manifest = _load_json(root / "manifest.json")
    candidates_path = root / "transition_event_candidates.csv"
    verified_path = root / "announcement_verified_events.csv"
    unverified_path = root / "unverified_transition_candidates.csv"
    attempts_path = root / "announcement_verification_attempts.csv"
    blocked_path = root / "blocked_source_rows.csv"
    future_audit_path = root / "future_data_violation_audit.csv"
    return {
        "exists": True,
        "path": str(root),
        "readiness": readiness,
        "manifest": manifest,
        "candidates_path": str(candidates_path),
        "verified_path": str(verified_path),
        "unverified_path": str(unverified_path),
        "attempts_path": str(attempts_path),
        "blocked_path": str(blocked_path),
        "future_audit_path": str(future_audit_path),
        "candidate_rows": int(len(_read_csv_if_exists(candidates_path))),
        "verified_rows": int(len(_read_csv_if_exists(verified_path))),
        "unverified_rows": int(len(_read_csv_if_exists(unverified_path))),
        "attempt_rows": int(len(_read_csv_if_exists(attempts_path))),
        "blocked_rows": int(len(_read_csv_if_exists(blocked_path))),
        "future_audit_rows": int(len(_read_csv_if_exists(future_audit_path))),
    }


def _load_monthly_revenue_pit(monthly_revenue_output: str | Path | None) -> dict[str, Any]:
    if not monthly_revenue_output:
        return {"exists": False, "path": ""}
    root = Path(monthly_revenue_output)
    if not root.exists():
        return {"exists": False, "path": str(root), "missing_reason": "monthly revenue output path does not exist"}
    readiness = _load_json(root / "readiness_for_core.json")
    manifest = _load_json(root / "manifest.json")
    shard_manifest_path = root / "accepted_monthly_revenue_rows_manifest.csv"
    sample_path = root / "accepted_monthly_revenue_rows_sample.csv"
    index_path = root / "accepted_monthly_revenue_rows.csv"
    coverage_market_path = root / "coverage_by_market.csv"
    coverage_month_path = root / "coverage_by_year_month.csv"
    future_audit_path = root / "future_data_violation_audit.csv"
    return {
        "exists": True,
        "path": str(root),
        "readiness": readiness,
        "manifest": manifest,
        "shard_manifest_path": str(shard_manifest_path),
        "sample_path": str(sample_path),
        "index_path": str(index_path),
        "coverage_market_path": str(coverage_market_path),
        "coverage_month_path": str(coverage_month_path),
        "future_audit_path": str(future_audit_path),
        "shard_manifest_rows": int(len(_read_csv_if_exists(shard_manifest_path))),
        "sample_rows": int(len(_read_csv_if_exists(sample_path))),
        "index_rows": int(len(_read_csv_if_exists(index_path))),
        "coverage_market_rows": int(len(_read_csv_if_exists(coverage_market_path))),
        "coverage_month_rows": int(len(_read_csv_if_exists(coverage_month_path))),
        "future_audit_rows": int(len(_read_csv_if_exists(future_audit_path))),
    }


def _load_quarterly_fundamentals_route_unlock(quarterly_output: str | Path | None) -> dict[str, Any]:
    if not quarterly_output:
        return {"exists": False, "path": ""}
    root = Path(quarterly_output)
    if not root.exists():
        return {"exists": False, "path": str(root), "missing_reason": "quarterly fundamentals output path does not exist"}
    readiness = _load_json(root / "readiness_for_core.json")
    manifest = _load_json(root / "manifest.json")
    sample_path = root / "accepted_quarterly_fundamentals_sample_rows.csv"
    shard_manifest_path = root / "accepted_quarterly_fundamentals_rows_manifest.csv"
    attempts_path = root / "route_probe_attempts.csv"
    coverage_market_path = root / "coverage_by_market.csv"
    coverage_quarter_path = root / "coverage_by_year_quarter.csv"
    missing_path = root / "missing_or_failed_periods.csv"
    blocked_path = root / "blocked_source_rows.csv"
    future_audit_path = root / "future_data_violation_audit.csv"
    raw_manifest_path = root / "raw_source_archive_manifest.csv"
    return {
        "exists": True,
        "path": str(root),
        "readiness": readiness,
        "manifest": manifest,
        "sample_path": str(sample_path),
        "shard_manifest_path": str(shard_manifest_path),
        "attempts_path": str(attempts_path),
        "coverage_market_path": str(coverage_market_path),
        "coverage_quarter_path": str(coverage_quarter_path),
        "missing_path": str(missing_path),
        "blocked_path": str(blocked_path),
        "future_audit_path": str(future_audit_path),
        "raw_manifest_path": str(raw_manifest_path),
        "sample_rows": int(len(_read_csv_if_exists(sample_path))),
        "shard_manifest_rows": int(len(_read_csv_if_exists(shard_manifest_path))),
        "attempt_rows": int(len(_read_csv_if_exists(attempts_path))),
        "coverage_market_rows": int(len(_read_csv_if_exists(coverage_market_path))),
        "coverage_quarter_rows": int(len(_read_csv_if_exists(coverage_quarter_path))),
        "missing_rows": int(len(_read_csv_if_exists(missing_path))),
        "blocked_rows": int(len(_read_csv_if_exists(blocked_path))),
        "future_audit_rows": int(len(_read_csv_if_exists(future_audit_path))),
        "raw_manifest_rows": int(len(_read_csv_if_exists(raw_manifest_path))),
    }


def _load_market_cap_pit(market_cap_output: str | Path | None) -> dict[str, Any]:
    if not market_cap_output:
        return {"exists": False, "path": ""}
    root = Path(market_cap_output)
    if not root.exists():
        return {"exists": False, "path": str(root), "missing_reason": "market cap output path does not exist"}
    readiness = _load_json(root / "readiness_for_core.json")
    manifest = _load_json(root / "manifest.json")
    accepted_path = root / "accepted_market_cap_rows.csv"
    proxy_path = root / "proxy_market_cap_rows.csv"
    accepted_manifest_path = root / "accepted_market_cap_rows_manifest.csv"
    twse_capital_stock_path = root / "accepted_twse_issued_shares_sample_rows.csv"
    twse_capital_stock_full_path = root / "accepted_twse_capital_stock_rows.csv"
    twse_capital_stock_manifest_path = root / "accepted_twse_capital_stock_rows_manifest.csv"
    twse_market_cap_path = root / "accepted_twse_market_cap_sample_rows.csv"
    twse_proxy_market_cap_path = root / "sample_proxy_market_cap_rows.csv"
    twse_proxy_contract_path = root / "proxy_market_cap_contract.csv"
    coverage_market_path = root / "coverage_by_market.csv"
    coverage_year_path = root / "coverage_by_year.csv"
    attempts_path = root / "source_probe_attempts.csv"
    route_attempts_path = root / "route_probe_attempts.csv"
    blocked_path = root / "rejected_or_blocked_rows.csv"
    blocked_source_path = root / "blocked_source_rows.csv"
    future_audit_path = root / "future_data_violation_audit.csv"
    return {
        "exists": True,
        "path": str(root),
        "readiness": readiness,
        "manifest": manifest,
        "accepted_path": str(accepted_path),
        "proxy_path": str(proxy_path),
        "accepted_manifest_path": str(accepted_manifest_path),
        "twse_capital_stock_path": str(twse_capital_stock_path),
        "twse_capital_stock_full_path": str(twse_capital_stock_full_path),
        "twse_capital_stock_manifest_path": str(twse_capital_stock_manifest_path),
        "twse_market_cap_path": str(twse_market_cap_path),
        "twse_proxy_market_cap_path": str(twse_proxy_market_cap_path),
        "twse_proxy_contract_path": str(twse_proxy_contract_path),
        "coverage_market_path": str(coverage_market_path),
        "coverage_year_path": str(coverage_year_path),
        "attempts_path": str(attempts_path),
        "route_attempts_path": str(route_attempts_path),
        "blocked_path": str(blocked_path),
        "blocked_source_path": str(blocked_source_path),
        "future_audit_path": str(future_audit_path),
        "accepted_rows_file_count": int(len(_read_csv_if_exists(accepted_path))),
        "proxy_rows_file_count": int(len(_read_csv_if_exists(proxy_path))),
        "accepted_manifest_rows": int(len(_read_csv_if_exists(accepted_manifest_path))),
        "twse_capital_stock_rows_file_count": int(
            len(_read_csv_if_exists(twse_capital_stock_path))
            or len(_read_csv_if_exists(twse_capital_stock_full_path))
        ),
        "twse_capital_stock_manifest_rows": int(len(_read_csv_if_exists(twse_capital_stock_manifest_path))),
        "twse_market_cap_rows_file_count": int(len(_read_csv_if_exists(twse_market_cap_path))),
        "twse_proxy_market_cap_rows_file_count": int(len(_read_csv_if_exists(twse_proxy_market_cap_path))),
        "twse_proxy_contract_rows_file_count": int(len(_read_csv_if_exists(twse_proxy_contract_path))),
        "coverage_market_rows": int(len(_read_csv_if_exists(coverage_market_path))),
        "coverage_year_rows": int(len(_read_csv_if_exists(coverage_year_path))),
        "attempt_rows": int(len(_read_csv_if_exists(attempts_path)) or len(_read_csv_if_exists(route_attempts_path))),
        "blocked_rows": int(len(_read_csv_if_exists(blocked_path)) or len(_read_csv_if_exists(blocked_source_path))),
        "future_audit_rows": int(len(_read_csv_if_exists(future_audit_path))),
    }


def _price_cache_coverage(price_cache_dir: Path, price_source_registry: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if price_cache_dir.exists():
        for path in sorted(price_cache_dir.glob("*.csv")):
            ticker = _ticker_from_price_filename(path.name)
            columns = _csv_columns(path)
            first_date, last_date = _date_range_from_csv(path, columns)
            adj_ready = _adj_close_ready(path, columns)
            rows.append(
                {
                    "ticker": ticker,
                    "source": "local_price_cache",
                    "source_path": str(path),
                    "first_date": first_date,
                    "last_date": last_date,
                    "row_count": _count_csv_rows(path),
                    "adjusted_close_available": adj_ready,
                    "source_type": "price_cache",
                    "price_source_ready": bool(first_date and last_date),
                    "strategy_ready": adj_ready,
                    "synthetic_used": False,
                    "diagnostic_only": not adj_ready,
                }
            )
    if price_source_registry.exists():
        registry = pd.read_csv(price_source_registry).fillna("")
        for item in registry.to_dict(orient="records"):
            rows.append(
                {
                    "ticker": str(item.get("ticker") or ""),
                    "source": "price_source_registry",
                    "source_path": str(item.get("source_path") or ""),
                    "first_date": str(item.get("first_date") or ""),
                    "last_date": str(item.get("last_date") or ""),
                    "row_count": "",
                    "adjusted_close_available": False if "unadjusted" in str(item.get("source_type") or "") else "",
                    "source_type": str(item.get("source_type") or ""),
                    "price_source_ready": _bool_like(item.get("price_source_ready")),
                    "strategy_ready": _bool_like(item.get("strategy_ready")),
                    "synthetic_used": _bool_like(item.get("synthetic_used")),
                    "diagnostic_only": not _bool_like(item.get("strategy_ready")),
                }
            )
    if not rows:
        return pd.DataFrame(columns=_price_columns())
    frame = pd.DataFrame(rows)
    return frame[_price_columns()]


def _build_contract_tables(
    source_inventory: list[dict[str, Any]],
    price_coverage: pd.DataFrame,
    liquidity_sweep: dict[str, Any],
    monthly_revenue: dict[str, Any],
    quarterly_fundamentals: dict[str, Any],
    market_cap: dict[str, Any],
) -> dict[str, pd.DataFrame]:
    return {
        "all_listed_liquid_universe_pit_daily": _all_listed_liquid_contract(
            source_inventory,
            price_coverage,
            liquidity_sweep,
        ),
        "monthly_revenue_pit": _monthly_revenue_contract_table(monthly_revenue),
        "quarterly_fundamentals_pit": _quarterly_fundamentals_contract_table(quarterly_fundamentals),
        "market_cap_pit": _market_cap_contract_table(market_cap, source_inventory),
        "sector_membership_pit": _source_contract_table(
            "sector_membership_pit",
            source_inventory,
            "blocked",
            "sector maps are current/generated; cannot be used as 2015 PIT membership",
        ),
        "sector_breadth_pit_daily": _source_contract_table(
            "sector_breadth_pit_daily",
            source_inventory,
            "blocked",
            "sector metrics are current/generated and lack accepted date-aware membership",
        ),
    }


def _all_listed_liquid_contract(
    source_inventory: list[dict[str, Any]],
    price_coverage: pd.DataFrame,
    liquidity_sweep: dict[str, Any],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    readiness = liquidity_sweep.get("readiness", {})
    if liquidity_sweep.get("exists") and readiness.get("all_listed_liquid_universe_pit_daily_full_range_ready"):
        rows.append(
            {
                "record_type": "full_sweep_contract_status",
                "date": "",
                "ticker": "",
                "name": "",
                "listing_status": "daily_presence_ready_master_metadata_missing",
                "delisting_or_suspension_status": "metadata_not_ready",
                "liquidity_20d_twd": "source_rows_available_in_local_shards",
                "liquidity_60d_twd": "source_rows_available_in_local_shards",
                "market_cap_twd": "",
                "is_liquid": "available_in_shards",
                "source_date": readiness.get("covered_date_range", {}).get("end", ""),
                "release_date": "",
                "effective_date": readiness.get("covered_date_range", {}).get("start", ""),
                "source": "radar_dynamic_pool1_all_listed_liquid_universe_full_sweep",
                "source_type": "official_daily_liquidity_presence_sweep",
                "readiness_status": "partial_daily_liquidity_pit_ready",
                "diagnostic_only": False,
                "accepted_for_formal": False,
                "blocked_reason": (
                    "daily liquidity/trading-table presence ready; listing/delisting/suspension master metadata "
                    "still blocks strategy replay"
                ),
            }
        )
    if not price_coverage.empty:
        for item in price_coverage.sort_values(["ticker", "source"]).to_dict(orient="records"):
            rows.append(
                {
                    "record_type": "known_price_source_coverage",
                    "date": "",
                    "ticker": item.get("ticker", ""),
                    "name": "",
                    "listing_status": "unknown",
                    "delisting_or_suspension_status": "unknown",
                    "liquidity_20d_twd": "",
                    "liquidity_60d_twd": "",
                    "market_cap_twd": "",
                    "is_liquid": "",
                    "source_date": item.get("last_date", ""),
                    "release_date": "",
                    "effective_date": item.get("first_date", ""),
                    "source": item.get("source", ""),
                    "source_type": item.get("source_type", ""),
                    "readiness_status": "diagnostic_price_coverage_only",
                    "diagnostic_only": True,
                    "accepted_for_formal": False,
                    "blocked_reason": "price coverage is not an all-listed liquidity/listing PIT universe",
                }
            )
    rows.extend(
        _source_status_rows(
            "all_listed_liquid_universe_pit_daily",
            source_inventory,
            "blocked",
            "no accepted all-listed listing/liquidity/suspension PIT ledger",
        )
    )
    return pd.DataFrame(rows, columns=_all_listed_columns())


def _source_contract_table(
    data_area: str,
    source_inventory: list[dict[str, Any]],
    fallback_status: str,
    fallback_reason: str,
) -> pd.DataFrame:
    rows = _source_status_rows(data_area, source_inventory, fallback_status, fallback_reason)
    if not rows:
        rows = [_generic_status_row(data_area, fallback_status, fallback_reason)]
    return pd.DataFrame(rows, columns=_contract_columns())


def _monthly_revenue_contract_table(monthly_revenue: dict[str, Any]) -> pd.DataFrame:
    summary = _monthly_revenue_summary(monthly_revenue)
    if not summary["exists"]:
        return _blocked_contract_table(
            "monthly_revenue_pit",
            "blocked",
            "no accepted monthly revenue PIT source with release_date in repo",
        )
    row = {
        "record_type": "source_candidate_status",
        "data_area": "monthly_revenue_pit",
        "ticker": "",
        "name": "",
        "value": f"accepted_rows={summary['accepted_rows']}; symbol_count={summary['symbol_count']}",
        "source_date": summary["period_end"],
        "release_date": "conservative_next_month_day_10_weekday_adjusted",
        "effective_date": summary["period_start"],
        "source": "radar_dynamic_pool1_mops_monthly_revenue_full_universe_pit",
        "source_path": summary["path"],
        "source_type": "mops_static_t21sc03_conservative_available_date",
        "row_count": summary["accepted_rows"],
        "readiness_status": summary["status"],
        "diagnostic_only": False,
        "accepted_for_formal": False,
        "blocked_reason": summary["remaining_blocker"],
    }
    return pd.DataFrame([row], columns=_contract_columns())


def _quarterly_fundamentals_contract_table(quarterly_fundamentals: dict[str, Any]) -> pd.DataFrame:
    summary = _quarterly_fundamentals_summary(quarterly_fundamentals)
    if not summary["exists"]:
        return _blocked_contract_table(
            "quarterly_fundamentals_pit",
            "blocked",
            "no accepted quarterly fundamentals PIT source with announcement/release dates in repo",
        )
    row = {
        "record_type": "route_unlock_status",
        "data_area": "quarterly_fundamentals_pit",
        "ticker": "",
        "name": "",
        "value": (
            f"accepted_rows={summary['accepted_rows']}; symbol_count={summary['symbol_count']}; "
            f"sample_rows={summary['sample_rows']}"
        ),
        "source_date": "",
        "release_date": summary["available_date_policy"],
        "effective_date": ",".join(summary["tested_periods"]) if isinstance(summary["tested_periods"], list) else "",
        "source": "radar_dynamic_pool1_quarterly_fundamentals_route_unlock",
        "source_path": summary["path"],
        "source_type": summary["source_type"],
        "row_count": summary["accepted_rows"] or summary["sample_rows"],
        "readiness_status": summary["status"],
        "diagnostic_only": True,
        "accepted_for_formal": False,
        "blocked_reason": summary["remaining_blocker"],
    }
    return pd.DataFrame([row], columns=_contract_columns())


def _market_cap_contract_table(market_cap: dict[str, Any], source_inventory: list[dict[str, Any]]) -> pd.DataFrame:
    summary = _market_cap_summary(market_cap)
    if summary["tpex_partial_ready"] or summary["twse_quarterly_capital_stock_route_partial"]:
        row = {
            "record_type": "partial_source_candidate_status",
            "data_area": "market_cap_pit",
            "ticker": "",
            "name": "",
            "value": (
                f"accepted_rows={summary['accepted_rows']}; "
                f"accepted_markets={','.join(summary['accepted_markets'])}; "
                f"twse_capital_stock_sample_rows={summary['twse_capital_stock_sample_rows']}"
            ),
            "source_date": "",
            "release_date": "",
            "effective_date": "",
            "source": "radar_dynamic_pool1_market_cap_pit",
            "source_path": summary["path"],
            "source_type": summary["source_type"],
            "row_count": summary["accepted_rows"],
            "readiness_status": summary["status"],
            "diagnostic_only": True,
            "accepted_for_formal": False,
            "blocked_reason": summary["remaining_blocker"],
        }
        return pd.DataFrame([row], columns=_contract_columns())
    return _source_contract_table(
        "market_cap_pit",
        source_inventory,
        "partial",
        "available sources are current/latest snapshots or date-filterable observation helpers, not a historical PIT panel",
    )


def _blocked_contract_table(data_area: str, status: str, reason: str) -> pd.DataFrame:
    return pd.DataFrame([_generic_status_row(data_area, status, reason)], columns=_contract_columns())


def _source_status_rows(
    data_area: str,
    source_inventory: list[dict[str, Any]],
    fallback_status: str,
    fallback_reason: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for source in source_inventory:
        if source["data_area"] != data_area:
            continue
        rows.append(
            {
                "record_type": "source_status",
                "data_area": data_area,
                "ticker": "",
                "name": "",
                "value": "",
                "source_date": source.get("max_date", ""),
                "release_date": "",
                "effective_date": source.get("min_date", ""),
                "source": source.get("source_name", ""),
                "source_path": source.get("path", ""),
                "source_type": source.get("source_family", ""),
                "row_count": source.get("row_count", 0),
                "readiness_status": source.get("pit_acceptance") or fallback_status,
                "diagnostic_only": bool(source.get("diagnostic_only", True)),
                "accepted_for_formal": False,
                "blocked_reason": source.get("blocked_reason") or fallback_reason,
            }
        )
    return rows


def _generic_status_row(data_area: str, status: str, reason: str) -> dict[str, Any]:
    return {
        "record_type": "contract_status",
        "data_area": data_area,
        "ticker": "",
        "name": "",
        "value": "",
        "source_date": "",
        "release_date": "",
        "effective_date": "",
        "source": "",
        "source_path": "",
        "source_type": "",
        "row_count": 0,
        "readiness_status": status,
        "diagnostic_only": True,
        "accepted_for_formal": False,
        "blocked_reason": reason,
    }


def _candidate_data_readiness_by_date(
    source_inventory: list[dict[str, Any]],
    price_coverage: pd.DataFrame,
    liquidity_sweep: dict[str, Any],
    monthly_revenue: dict[str, Any],
    quarterly_fundamentals: dict[str, Any],
    market_cap: dict[str, Any],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    price_status = _price_readiness_status(price_coverage)
    for bucket, start, end in YEAR_BUCKETS:
        for table in TABLE_SPECS:
            status, reason = _readiness_for_table(
                table,
                price_status,
                source_inventory,
                liquidity_sweep,
                monthly_revenue,
                quarterly_fundamentals,
                market_cap,
            )
            rows.append(
                {
                    "year_bucket": bucket,
                    "start_date": start,
                    "end_date": end,
                    "data_table": table,
                    "readiness_status": status,
                    "accepted_for_formal": False,
                    "diagnostic_only": status in {"diagnostic_only", "partial", "source_candidate_ready"}
                    and table != "monthly_revenue_pit",
                    "future_data_violation_count": 0,
                    "blocked_reason": reason,
                }
            )
    return pd.DataFrame(rows)


def _future_data_violation_audit(source_inventory: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for source in source_inventory:
        rows.append(
            {
                "data_area": source["data_area"],
                "source": source["source_name"],
                "source_path": source["path"],
                "source_type": source["source_family"],
                "accepted_for_formal": False,
                "diagnostic_only": bool(source["diagnostic_only"]),
                "current_snapshot_used_as_historical": False,
                "future_data_violation": False,
                "future_data_violation_count": 0,
                "audit_reason": "source is not accepted as historical formal PIT; current/proxy rows are blocked or diagnostic-only",
            }
        )
    return pd.DataFrame(rows)


def _source_manifest(source_inventory: list[dict[str, Any]], price_coverage: pd.DataFrame) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "task_id": TASK_ID,
        "generated_at": pd.Timestamp.now(tz="Asia/Taipei").isoformat(),
        "source_count": len(source_inventory),
        "price_coverage_rows": int(len(price_coverage)),
        "formal_model_changed": False,
        "trade_decision_changed": False,
        "active_in_trade_decision": False,
        "sources": source_inventory,
        "price_coverage_summary": {
            "tickers": int(price_coverage["ticker"].nunique()) if not price_coverage.empty else 0,
            "adjusted_ready_rows": int(price_coverage["adjusted_close_available"].astype(bool).sum())
            if not price_coverage.empty
            else 0,
            "strategy_ready_rows": int(price_coverage["strategy_ready"].astype(bool).sum()) if not price_coverage.empty else 0,
        },
    }


def _readiness_json(
    source_inventory: list[dict[str, Any]],
    price_coverage: pd.DataFrame,
    readiness_by_date: pd.DataFrame,
    violation_audit: pd.DataFrame,
    liquidity_sweep: dict[str, Any],
    listing_metadata: dict[str, Any],
    tpex_status: dict[str, Any],
    tpex_transition: dict[str, Any],
    monthly_revenue: dict[str, Any],
    quarterly_fundamentals: dict[str, Any],
    market_cap: dict[str, Any],
) -> dict[str, Any]:
    price_status = _price_readiness_status(price_coverage)
    table_status = {
        table: {
            "status": _readiness_for_table(
                table,
                price_status,
                source_inventory,
                liquidity_sweep,
                monthly_revenue,
                quarterly_fundamentals,
                market_cap,
            )[0],
            "reason": _readiness_for_table(
                table,
                price_status,
                source_inventory,
                liquidity_sweep,
                monthly_revenue,
                quarterly_fundamentals,
                market_cap,
            )[1],
            "accepted_for_formal": False,
        }
        for table in TABLE_SPECS
    }
    table_status["listing_delisting_suspension_metadata"] = _listing_metadata_status(listing_metadata)
    table_status["tpex_historical_listing_status"] = _tpex_status_blocker_status(tpex_status)
    future_data_violation_count = int(violation_audit["future_data_violation_count"].sum()) if not violation_audit.empty else 0
    return {
        "schema_version": 1,
        "task_id": TASK_ID,
        "status": "completed_readiness_contract",
        "dynamic_pool1_shadow_challenger_ready": False,
        "partial_ready_for_source_contract": True,
        "ready_for_strategy_replay": False,
        "formal_model_changed": False,
        "trade_decision_changed": False,
        "active_in_trade_decision": False,
        "uses_forward_return": False,
        "future_data_violation_count": future_data_violation_count,
        "liquidity_full_sweep": _liquidity_sweep_summary(liquidity_sweep),
        "listing_metadata": _listing_metadata_summary(listing_metadata),
        "tpex_historical_listing_status": _tpex_status_blocker_summary(tpex_status),
        "tpex_transition_candidates": _tpex_transition_candidate_summary(tpex_transition),
        "monthly_revenue": _monthly_revenue_summary(monthly_revenue),
        "quarterly_fundamentals": _quarterly_fundamentals_summary(quarterly_fundamentals),
        "market_cap": _market_cap_summary(market_cap),
        "coverage_by_year": {
            bucket: {
                row["data_table"]: {
                    "readiness_status": row["readiness_status"],
                    "accepted_for_formal": bool(row["accepted_for_formal"]),
                    "diagnostic_only": bool(row["diagnostic_only"]),
                    "blocked_reason": row["blocked_reason"],
                }
                for row in readiness_by_date[readiness_by_date["year_bucket"].eq(bucket)].to_dict(orient="records")
            }
            for bucket, _, _ in YEAR_BUCKETS
        },
        "table_status": table_status,
        "source_summary": {
            "inventory_sources": len(source_inventory),
            "price_coverage_rows": int(len(price_coverage)),
            "accepted_formal_source_count": 0,
            "diagnostic_or_blocked_source_count": len(source_inventory),
        },
        "next_blockers": [
            "Core/Research policy decision on whether TPEx inferred transition candidates are sufficient for universe integrity, or broader announcement verification for official transition events",
            "quarterly fundamentals full 2015-latest sweep and coverage audit using unlocked MOPS route",
            "exact per-company quarterly filing_date crawler if formal_exact is required beyond conservative statutory deadline",
            "TWSE historical issued shares/capital changes or direct official market cap route",
            "TPEx market cap full 2015-latest sweep using sample-verified dailyQuotes route",
            "free-float shares/free-float market cap route",
            "sector/mainline membership PIT and sector breadth daily panel",
            "exact per-company monthly revenue filing timestamp if formal_exact is required beyond conservative available_date",
        ],
    }


def _manifest(readiness: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "task_id": TASK_ID,
        "status": readiness["status"],
        "dynamic_pool1_shadow_challenger_ready": readiness["dynamic_pool1_shadow_challenger_ready"],
        "ready_for_strategy_replay": readiness["ready_for_strategy_replay"],
        "formal_model_changed": False,
        "trade_decision_changed": False,
        "active_in_trade_decision": False,
        "future_data_violation_count": readiness["future_data_violation_count"],
        "outputs": {
            "all_listed_liquid_universe_pit_daily": "all_listed_liquid_universe_pit_daily.csv",
            "monthly_revenue_pit": "monthly_revenue_pit.csv",
            "quarterly_fundamentals_pit": "quarterly_fundamentals_pit.csv",
            "market_cap_pit": "market_cap_pit.csv",
            "sector_membership_pit": "sector_membership_pit.csv",
            "sector_breadth_pit_daily": "sector_breadth_pit_daily.csv",
            "candidate_data_readiness_by_date": "candidate_data_readiness_by_date.csv",
            "dataset_readiness_summary": "dataset_readiness_summary.csv",
            "blocker_delta_after_liquidity_full_sweep": "blocker_delta_after_liquidity_full_sweep.csv",
            "blocker_delta_after_listing_metadata": "blocker_delta_after_listing_metadata.csv",
            "blocker_delta_after_listing_master_completion": "blocker_delta_after_listing_master_completion.csv",
            "blocker_delta_after_tpex_blocker_evidence": "blocker_delta_after_tpex_blocker_evidence.csv",
            "blocker_delta_after_tpex_full_route_coverage": "blocker_delta_after_tpex_full_route_coverage.csv",
            "blocker_delta_after_tpex_transition_candidates": "blocker_delta_after_tpex_transition_candidates.csv",
            "blocker_delta_after_mops_monthly_revenue": "blocker_delta_after_mops_monthly_revenue.csv",
            "blocker_delta_after_quarterly_fundamentals_route_unlock": "blocker_delta_after_quarterly_fundamentals_route_unlock.csv",
            "blocker_delta_after_quarterly_fundamentals_full_sweep": "blocker_delta_after_quarterly_fundamentals_full_sweep.csv",
            "blocker_delta_after_market_cap_partial": "blocker_delta_after_market_cap_partial.csv",
            "blocker_delta_after_tpex_market_cap_full_sweep": "blocker_delta_after_tpex_market_cap_full_sweep.csv",
            "blocker_delta_after_twse_capital_stock_route_partial": "blocker_delta_after_twse_capital_stock_route_partial.csv",
            "blocker_delta_after_twse_capital_stock_full_sweep_proxy_contract": "blocker_delta_after_twse_capital_stock_full_sweep_proxy_contract.csv",
            "future_data_violation_audit": "future_data_violation_audit.csv",
            "source_manifest": "source_manifest.json",
            "readiness": "readiness.json",
        },
    }


def _next_step_handoff(readiness: dict[str, Any]) -> str:
    blockers = "\n".join(f"- {item}" for item in readiness["next_blockers"])
    return (
        "# Dynamic Pool1 PIT readiness handoff\n\n"
        "Status: completed readiness contract, not ready for strategy replay.\n\n"
        "Next Data/Radar tasks:\n"
        f"{blockers}\n\n"
        "Experiments should not run `TASK-BACKTEST-EXPERIMENTS-DYNAMIC-POOL1-SHADOW-CHALLENGER-001` "
        "until at least the universe, liquidity, release-date fundamentals, market cap, and sector PIT layers "
        "are accepted or explicitly scoped as diagnostic-only sensitivity.\n"
    )


def _final_summary(readiness: dict[str, Any]) -> str:
    liquidity = readiness.get("liquidity_full_sweep", {})
    listing = readiness.get("listing_metadata", {})
    tpex = readiness.get("tpex_historical_listing_status", {})
    transition = readiness.get("tpex_transition_candidates", {})
    monthly = readiness.get("monthly_revenue", {})
    quarterly = readiness.get("quarterly_fundamentals", {})
    market_cap = readiness.get("market_cap", {})
    liquidity_line = (
        f"- all-listed liquid universe：partial，Radar full sweep covered "
        f"{liquidity.get('covered_start', '')}～{liquidity.get('covered_end', '')}，"
        f"accepted_liquidity_rows={liquidity.get('accepted_liquidity_rows', 0)}；"
        "但 listing/delisting/suspension master 仍未 ready。\n"
        if liquidity.get("full_range_ready")
        else "- all-listed liquid universe：blocked，缺上市/下市/停牌/流動性 PIT ledger。\n"
    )
    listing_line = (
        f"- listing/delisting/suspension metadata：{listing.get('readiness_level', 'partial')}，"
        f"accepted event rows={listing.get('accepted_event_rows', 0)}"
        f"（delta={listing.get('delta_vs_previous', 0)}）；但 complete cross-market master 仍未 ready。\n"
        if listing.get("partial_event_rows_available")
        else "- listing/delisting/suspension metadata：blocked，缺完整歷史 master。\n"
    )
    return (
        "# Dynamic Pool1 PIT readiness contract\n\n"
        "結論：目前只完成資料契約與來源盤點，尚不能交 Experiments 跑 dynamic Pool1 shadow challenger。\n\n"
        f"{liquidity_line}"
        f"{listing_line}"
        f"- TPEx historical listing/status：{tpex.get('status', 'blocked')}，"
        f"accepted historical rows={tpex.get('accepted_historical_rows', 0)}，"
        f"route_attempts={tpex.get('route_probe_attempts') or tpex.get('source_probe_attempts', 0)}；"
        "daily status snapshot 可做 as-of 判斷，但不是 explicit transition event ledger。\n"
        f"- TPEx transition candidates：{transition.get('status', 'missing')}，"
        f"inferred candidates={transition.get('transition_candidate_count', 0)}，"
        f"announcement verified events={transition.get('announcement_verified_event_count', 0)}；"
        "可作 universe integrity 診斷證據，但未升級為 official explicit event ledger。\n"
        f"- monthly revenue：{monthly.get('status', 'blocked')}，"
        f"accepted rows={monthly.get('accepted_rows', 0)}，"
        f"symbols={monthly.get('symbol_count', 0)}，"
        f"period={monthly.get('period_start', '')}～{monthly.get('period_end', '')}；"
        "available_date 採保守次月 10 日規則，formal_exact=false。\n"
        f"- quarterly fundamentals：{quarterly.get('status', 'blocked')}，"
        f"sample rows={quarterly.get('sample_rows', 0)}，"
        f"accepted rows={quarterly.get('accepted_rows', 0)}，"
        f"tested periods={','.join(quarterly.get('tested_periods', [])) if isinstance(quarterly.get('tested_periods', []), list) else quarterly.get('tested_periods', '')}；"
        "t163sb04 損益表彙總已可作 source candidate；filing_date 仍不是逐公司 exact。\n"
        f"- market cap：{market_cap.get('status', 'partial')}，"
        f"accepted rows={market_cap.get('accepted_rows', 0)}，"
        f"accepted markets={','.join(market_cap.get('accepted_markets', [])) if isinstance(market_cap.get('accepted_markets', []), list) else market_cap.get('accepted_markets', '')}；"
        "TPEx total market cap 可作 source candidate，TWSE 與 free-float 仍 blocked。\n"
        "- sector membership / breadth：blocked，目前 current/generated map 不可回推 2015。\n"
        f"- future_data_violation_count={readiness['future_data_violation_count']}，因未把 current/proxy source 當正式歷史資料使用。\n\n"
        "formal_model_changed=false；trade_decision_changed=false；active_in_trade_decision=false。\n"
    )


def _readiness_for_table(
    table: str,
    price_status: tuple[str, str],
    source_inventory: list[dict[str, Any]],
    liquidity_sweep: dict[str, Any],
    monthly_revenue: dict[str, Any],
    quarterly_fundamentals: dict[str, Any],
    market_cap: dict[str, Any],
) -> tuple[str, str]:
    if table == "all_listed_liquid_universe_pit_daily":
        readiness = liquidity_sweep.get("readiness", {})
        if readiness.get("all_listed_liquid_universe_pit_daily_full_range_ready"):
            return (
                "partial",
                "daily liquidity/trading-table presence full sweep is ready, but listing/delisting/suspension master metadata is missing",
            )
        return (
            "blocked",
            "local price coverage exists but no all-listed listing/liquidity/suspension PIT universe is accepted",
        )
    if table == "monthly_revenue_pit":
        summary = _monthly_revenue_summary(monthly_revenue)
        if summary["source_candidate_ready"]:
            return (
                "source_candidate_ready",
                "MOPS monthly revenue full-universe PIT has source_date/release_date/available_date; formal_exact=false because available_date is conservative, not per-company exact filing timestamp",
            )
        return "blocked", "no monthly revenue PIT with source_date/release_date/effective_date"
    if table == "quarterly_fundamentals_pit":
        summary = _quarterly_fundamentals_summary(quarterly_fundamentals)
        if summary["full_sweep_ready"]:
            return (
                "full_sweep_source_candidate_ready",
                "MOPS t163sb04 quarterly income-statement summary full sweep is ready as source candidate; formal_exact=false because per-company filing_date is not exact and balance sheet/cash flow expansions are separate",
            )
        if summary["route_unlocked"]:
            return (
                "route_unlocked_source_candidate_partial",
                "MOPS quarterly fundamentals route is unlocked with sample rows; full 2015-latest sweep and exact per-company filing_date are not ready",
            )
        return "blocked", "no quarterly fundamentals PIT with source_date/release_date/effective_date"
    if table == "market_cap_pit":
        summary = _market_cap_summary(market_cap)
        if summary["tpex_partial_ready"]:
            return (
                summary["status"],
                "TPEx official daily close and shares can derive total market cap source candidate; TWSE historical shares/direct market cap and free-float routes remain blocked",
            )
        if summary["twse_quarterly_capital_stock_route_partial"]:
            return (
                summary["status"],
                "TWSE MOPS quarterly capital stock route is unlocked as diagnostic/proxy source candidate; it is not daily issued shares or direct daily market cap, and free-float remains blocked",
            )
        exists = any(source["data_area"] == table and source["exists"] for source in source_inventory)
        return (
            "partial" if exists else "blocked",
            "current/latest market cap source candidates exist, but historical PIT market cap panel is not accepted"
            if exists
            else "no market cap source candidates found",
        )
    if table == "sector_membership_pit":
        return "blocked", "sector maps are current/generated and not accepted as historical PIT membership"
    if table == "sector_breadth_pit_daily":
        return "blocked", "sector breadth requires accepted PIT sector membership; current metrics are diagnostic only"
    return price_status


def _dataset_readiness_summary(readiness: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for table, status in readiness["table_status"].items():
        rows.append(
            {
                "dataset": table,
                "readiness_status": status["status"],
                "accepted_for_formal": bool(status["accepted_for_formal"]),
                "active_in_trade_decision": False,
                "reason": status["reason"],
            }
        )
    rows.append(
        {
            "dataset": "dynamic_pool1_shadow_challenger",
            "readiness_status": "blocked",
            "accepted_for_formal": False,
            "active_in_trade_decision": False,
            "reason": "monthly revenue, quarterly fundamentals, and TPEx partial market cap source candidates are available, but TWSE/free-float market cap, sector PIT, and universe-integrity policy remain incomplete",
        }
    )
    return pd.DataFrame(rows)


def _blocker_delta_after_liquidity_full_sweep(liquidity_sweep: dict[str, Any]) -> pd.DataFrame:
    readiness = liquidity_sweep.get("readiness", {})
    full_ready = bool(readiness.get("all_listed_liquid_universe_pit_daily_full_range_ready"))
    return pd.DataFrame(
        [
            {
                "blocker": "all_listed_liquid_universe_pit_daily",
                "before_status": "blocked",
                "after_status": "partial" if full_ready else "blocked",
                "delta": "downgraded_from_blocked_to_partial_daily_liquidity_presence"
                if full_ready
                else "unchanged_blocked",
                "remaining_blocker": "listing_delisting_suspension_metadata_ready=false"
                if full_ready
                else "full liquidity sweep not available",
                "ready_for_strategy_replay": bool(readiness.get("ready_for_strategy_replay", False)),
            },
            {
                "blocker": "monthly_revenue_pit",
                "before_status": "blocked",
                "after_status": "blocked",
                "delta": "unchanged",
                "remaining_blocker": "MOPS monthly revenue full universe PIT with release_date is still missing",
                "ready_for_strategy_replay": False,
            },
            {
                "blocker": "quarterly_fundamentals_pit",
                "before_status": "blocked",
                "after_status": "blocked",
                "delta": "unchanged",
                "remaining_blocker": "quarterly fundamentals PIT with announcement/release dates is still missing",
                "ready_for_strategy_replay": False,
            },
            {
                "blocker": "market_cap_pit",
                "before_status": "partial",
                "after_status": "partial",
                "delta": "unchanged",
                "remaining_blocker": "historical market cap/free-float market cap PIT is still missing",
                "ready_for_strategy_replay": False,
            },
            {
                "blocker": "sector_membership_pit",
                "before_status": "blocked",
                "after_status": "blocked",
                "delta": "unchanged",
                "remaining_blocker": "date-aware sector/mainline membership PIT is still missing",
                "ready_for_strategy_replay": False,
            },
        ]
    )


def _blocker_delta_after_listing_metadata(listing_metadata: dict[str, Any]) -> pd.DataFrame:
    summary = _listing_metadata_summary(listing_metadata)
    partial = bool(summary.get("partial_event_rows_available"))
    return pd.DataFrame(
        [
            {
                "blocker": "listing_delisting_suspension_metadata",
                "before_status": "blocked",
                "after_status": "partial" if partial else "blocked",
                "delta": "downgraded_from_blocked_to_partial_event_sources" if partial else "unchanged_blocked",
                "accepted_listing_metadata_rows": int(summary.get("accepted_listing_metadata_rows", 0)),
                "accepted_suspension_event_rows": int(summary.get("accepted_suspension_event_rows", 0)),
                "proxy_source_rows": int(summary.get("proxy_source_rows", 0)),
                "blocked_source_rows": int(summary.get("blocked_source_rows", 0)),
                "remaining_blocker": "complete historical listing/delisting/suspension/name-change/transfer master is missing",
                "ready_for_strategy_replay": bool(summary.get("ready_for_strategy_replay", False)),
            },
            {
                "blocker": "monthly_revenue_pit",
                "before_status": "blocked",
                "after_status": "blocked",
                "delta": "unchanged",
                "accepted_listing_metadata_rows": 0,
                "accepted_suspension_event_rows": 0,
                "proxy_source_rows": 0,
                "blocked_source_rows": 0,
                "remaining_blocker": "MOPS monthly revenue full universe PIT with release_date is still missing",
                "ready_for_strategy_replay": False,
            },
            {
                "blocker": "quarterly_fundamentals_pit",
                "before_status": "blocked",
                "after_status": "blocked",
                "delta": "unchanged",
                "accepted_listing_metadata_rows": 0,
                "accepted_suspension_event_rows": 0,
                "proxy_source_rows": 0,
                "blocked_source_rows": 0,
                "remaining_blocker": "quarterly fundamentals PIT with announcement/release dates is still missing",
                "ready_for_strategy_replay": False,
            },
            {
                "blocker": "market_cap_pit",
                "before_status": "partial",
                "after_status": "partial",
                "delta": "unchanged",
                "accepted_listing_metadata_rows": 0,
                "accepted_suspension_event_rows": 0,
                "proxy_source_rows": 0,
                "blocked_source_rows": 0,
                "remaining_blocker": "historical market cap/free-float market cap PIT is still missing",
                "ready_for_strategy_replay": False,
            },
        ]
    )


def _blocker_delta_after_listing_master_completion(listing_metadata: dict[str, Any]) -> pd.DataFrame:
    summary = _listing_metadata_summary(listing_metadata)
    stronger = summary.get("readiness_level") == "stronger_partial"
    return pd.DataFrame(
        [
            {
                "blocker": "listing_delisting_suspension_metadata",
                "before_status": "partial",
                "after_status": "stronger_partial" if stronger else summary.get("readiness_level", "partial"),
                "delta": "accepted_event_rows_increased" if summary.get("delta_vs_previous", 0) else "unchanged_or_unknown_delta",
                "previous_accepted_event_rows": int(summary.get("previous_accepted_event_rows", 0)),
                "accepted_event_rows_total": int(summary.get("accepted_event_rows", 0)),
                "delta_vs_previous": int(summary.get("delta_vs_previous", 0)),
                "accepted_listing_metadata_rows": int(summary.get("accepted_listing_metadata_rows", 0)),
                "accepted_suspension_event_rows": int(summary.get("accepted_suspension_event_rows", 0)),
                "accepted_code_name_change_rows": int(summary.get("accepted_code_name_change_rows", 0)),
                "accepted_transfer_listing_rows": int(summary.get("accepted_transfer_listing_rows", 0)),
                "twse_improved_partial_status_coverage": bool(summary.get("twse_improved_partial_status_coverage", False)),
                "tpex_2015_2025_blocked": bool(summary.get("tpex_2015_2025_blocked", True)),
                "cross_market_strategy_replay_ready": bool(summary.get("ready_for_strategy_replay", False)),
                "twse_only_diagnostic_possible": bool(summary.get("twse_only_diagnostic_possible", False)),
                "twse_only_diagnostic_recommendation": summary.get("twse_only_diagnostic_recommendation", ""),
                "remaining_blocker": summary.get("remaining_blocker", ""),
            },
            {
                "blocker": "monthly_revenue_pit",
                "before_status": "blocked",
                "after_status": "blocked",
                "delta": "unchanged",
                "previous_accepted_event_rows": 0,
                "accepted_event_rows_total": 0,
                "delta_vs_previous": 0,
                "accepted_listing_metadata_rows": 0,
                "accepted_suspension_event_rows": 0,
                "accepted_code_name_change_rows": 0,
                "accepted_transfer_listing_rows": 0,
                "twse_improved_partial_status_coverage": False,
                "tpex_2015_2025_blocked": False,
                "cross_market_strategy_replay_ready": False,
                "twse_only_diagnostic_possible": False,
                "twse_only_diagnostic_recommendation": "",
                "remaining_blocker": "MOPS monthly revenue full universe PIT with release_date is still missing",
            },
            {
                "blocker": "quarterly_fundamentals_pit",
                "before_status": "blocked",
                "after_status": "blocked",
                "delta": "unchanged",
                "previous_accepted_event_rows": 0,
                "accepted_event_rows_total": 0,
                "delta_vs_previous": 0,
                "accepted_listing_metadata_rows": 0,
                "accepted_suspension_event_rows": 0,
                "accepted_code_name_change_rows": 0,
                "accepted_transfer_listing_rows": 0,
                "twse_improved_partial_status_coverage": False,
                "tpex_2015_2025_blocked": False,
                "cross_market_strategy_replay_ready": False,
                "twse_only_diagnostic_possible": False,
                "twse_only_diagnostic_recommendation": "",
                "remaining_blocker": "quarterly fundamentals PIT with announcement/release dates is still missing",
            },
        ]
    )


def _blocker_delta_after_tpex_blocker_evidence(tpex_status: dict[str, Any]) -> pd.DataFrame:
    summary = _tpex_status_blocker_summary(tpex_status)
    has_historical_rows = bool(summary["accepted_historical_rows"] > 0)
    return pd.DataFrame(
        [
            {
                "blocker": "tpex_historical_listing_status",
                "before_status": "blocked",
                "after_status": summary["status"],
                "delta": (
                    "partial_historical_rows_accepted"
                    if has_historical_rows
                    else (
                        "attempt_evidence_added_no_historical_rows_accepted"
                        if summary["exists"]
                        else "unchanged_missing_package"
                    )
                ),
                "accepted_historical_rows": int(summary["accepted_historical_rows"]),
                "accepted_listing_metadata_rows": int(summary["accepted_listing_metadata_rows"]),
                "accepted_status_snapshot_rows": int(summary["accepted_status_snapshot_rows"]),
                "accepted_suspension_event_rows": int(summary["accepted_suspension_event_rows"]),
                "accepted_current_or_carried_tpex_rows": int(summary["accepted_current_or_carried_tpex_rows"]),
                "current_or_carried_rows_used_as_historical": False,
                "source_probe_attempts": int(summary["source_probe_attempts"] or summary["route_probe_attempts"]),
                "route_probe_attempts": int(summary["route_probe_attempts"]),
                "blocked_source_rows": int(summary["blocked_source_rows"]),
                "ready_for_strategy_replay": bool(summary["ready_for_strategy_replay"]),
                "full_2015_2025_master_ready": bool(summary["tpex_full_2015_2025_master_ready"]),
                "remaining_blocker": summary["remaining_blocker"],
            },
            {
                "blocker": "full_cross_market_listing_master",
                "before_status": "blocked",
                "after_status": "blocked",
                "delta": "still_blocked_partial_tpex_sample_only" if has_historical_rows else "unchanged_due_to_tpex_historical_blocker",
                "accepted_historical_rows": int(summary["accepted_historical_rows"]),
                "accepted_listing_metadata_rows": int(summary["accepted_listing_metadata_rows"]),
                "accepted_status_snapshot_rows": int(summary["accepted_status_snapshot_rows"]),
                "accepted_suspension_event_rows": int(summary["accepted_suspension_event_rows"]),
                "accepted_current_or_carried_tpex_rows": int(summary["accepted_current_or_carried_tpex_rows"]),
                "current_or_carried_rows_used_as_historical": False,
                "source_probe_attempts": int(summary["source_probe_attempts"]),
                "route_probe_attempts": int(summary["route_probe_attempts"]),
                "blocked_source_rows": int(summary["blocked_source_rows"]),
                "ready_for_strategy_replay": False,
                "full_2015_2025_master_ready": bool(summary["tpex_full_2015_2025_master_ready"]),
                "remaining_blocker": (
                    "TPEx route contract has sample accepted rows, but full 2015-2025 sweep is not complete"
                    if has_historical_rows
                    else "TPEx 2015-2025 historical listing/status rows remain 0 accepted"
                ),
            },
            {
                "blocker": "monthly_revenue_pit",
                "before_status": "blocked",
                "after_status": "blocked",
                "delta": "unchanged",
                "accepted_historical_rows": 0,
                "accepted_listing_metadata_rows": 0,
                "accepted_status_snapshot_rows": 0,
                "accepted_suspension_event_rows": 0,
                "accepted_current_or_carried_tpex_rows": 0,
                "current_or_carried_rows_used_as_historical": False,
                "source_probe_attempts": 0,
                "route_probe_attempts": 0,
                "blocked_source_rows": 0,
                "ready_for_strategy_replay": False,
                "full_2015_2025_master_ready": False,
                "remaining_blocker": "MOPS monthly revenue full universe PIT with release_date is still missing",
            },
        ]
    )


def _blocker_delta_after_tpex_full_route_coverage(tpex_status: dict[str, Any]) -> pd.DataFrame:
    summary = _tpex_status_blocker_summary(tpex_status)
    return pd.DataFrame(
        [
            {
                "blocker": "tpex_historical_listing_status",
                "before_status": "partial_with_accepted_historical_rows",
                "after_status": summary["status"],
                "delta": "full_route_coverage_ready_but_transition_events_blocked"
                if summary["full_tpex_2015_2025_route_coverage_ready"]
                else "full_route_coverage_not_ready",
                "covered_start": summary["covered_start"],
                "covered_end": summary["covered_end"],
                "route_request_attempts": int(summary["route_probe_attempts"]),
                "failed_attempts": int(summary["failed_attempts"]),
                "accepted_historical_rows": int(summary["accepted_historical_rows"]),
                "accepted_listing_metadata_rows": int(summary["accepted_listing_metadata_rows"]),
                "accepted_delisting_metadata_rows": int(summary["accepted_delisting_metadata_rows"]),
                "accepted_status_snapshot_rows": int(summary["accepted_status_snapshot_rows"]),
                "accepted_suspension_event_rows": int(summary["accepted_suspension_event_rows"]),
                "daily_status_snapshot_asof_ready": bool(summary["daily_status_snapshot_asof_ready"]),
                "explicit_transition_event_ledger_ready": bool(summary["explicit_transition_event_ledger_ready"]),
                "ready_for_strategy_replay": bool(summary["ready_for_strategy_replay"]),
                "remaining_blocker": summary["remaining_blocker"],
            },
            {
                "blocker": "full_cross_market_listing_master",
                "before_status": "blocked",
                "after_status": "blocked",
                "delta": "still_blocked_by_transition_event_policy_or_event_ledger",
                "covered_start": summary["covered_start"],
                "covered_end": summary["covered_end"],
                "route_request_attempts": int(summary["route_probe_attempts"]),
                "failed_attempts": int(summary["failed_attempts"]),
                "accepted_historical_rows": int(summary["accepted_historical_rows"]),
                "accepted_listing_metadata_rows": int(summary["accepted_listing_metadata_rows"]),
                "accepted_delisting_metadata_rows": int(summary["accepted_delisting_metadata_rows"]),
                "accepted_status_snapshot_rows": int(summary["accepted_status_snapshot_rows"]),
                "accepted_suspension_event_rows": int(summary["accepted_suspension_event_rows"]),
                "daily_status_snapshot_asof_ready": bool(summary["daily_status_snapshot_asof_ready"]),
                "explicit_transition_event_ledger_ready": bool(summary["explicit_transition_event_ledger_ready"]),
                "ready_for_strategy_replay": False,
                "remaining_blocker": "Core/Research must decide whether daily as-of status snapshots are sufficient or require explicit suspension/resumption transition events",
            },
            {
                "blocker": "monthly_revenue_pit",
                "before_status": "blocked",
                "after_status": "blocked",
                "delta": "unchanged",
                "covered_start": "",
                "covered_end": "",
                "route_request_attempts": 0,
                "failed_attempts": 0,
                "accepted_historical_rows": 0,
                "accepted_listing_metadata_rows": 0,
                "accepted_delisting_metadata_rows": 0,
                "accepted_status_snapshot_rows": 0,
                "accepted_suspension_event_rows": 0,
                "daily_status_snapshot_asof_ready": False,
                "explicit_transition_event_ledger_ready": False,
                "ready_for_strategy_replay": False,
                "remaining_blocker": "MOPS monthly revenue full universe PIT with release_date is still missing",
            },
        ]
    )


def _blocker_delta_after_tpex_transition_candidates(
    tpex_status: dict[str, Any], tpex_transition: dict[str, Any]
) -> pd.DataFrame:
    transition_summary = _tpex_transition_candidate_summary(tpex_transition)
    has_candidates = transition_summary["transition_candidate_count"] > 0
    has_verified = transition_summary["announcement_verified_event_count"] > 0
    return pd.DataFrame(
        [
            {
                "blocker": "tpex_explicit_transition_event_ledger",
                "before_status": "blocked_after_daily_status_snapshot",
                "after_status": transition_summary["status"],
                "delta": (
                    "inferred_transition_candidates_available_but_unverified"
                    if has_candidates and not has_verified
                    else "announcement_verified_transition_events_available"
                    if has_verified
                    else "unchanged_no_transition_candidates"
                ),
                "transition_candidate_count": int(transition_summary["transition_candidate_count"]),
                "announcement_verification_attempts": int(transition_summary["announcement_verification_attempts"]),
                "announcement_verified_event_count": int(transition_summary["announcement_verified_event_count"]),
                "unverified_transition_candidate_count": int(
                    transition_summary["unverified_transition_candidate_count"]
                ),
                "inferred_candidates_used_as_official_events": False,
                "official_explicit_transition_event_ledger_ready": bool(
                    transition_summary["official_explicit_transition_event_ledger_ready"]
                ),
                "ready_for_strategy_replay": bool(transition_summary["ready_for_strategy_replay"]),
                "remaining_blocker": transition_summary["remaining_blocker"],
            },
            {
                "blocker": "full_cross_market_listing_master",
                "before_status": "blocked_by_transition_event_policy_or_event_ledger",
                "after_status": "partial_candidate_evidence_not_formal_ready" if has_candidates else "blocked",
                "delta": (
                    "daily_snapshot_diff_candidates_reduce_uncertainty_but_require_policy_or_verification"
                    if has_candidates
                    else "unchanged"
                ),
                "transition_candidate_count": int(transition_summary["transition_candidate_count"]),
                "announcement_verification_attempts": int(transition_summary["announcement_verification_attempts"]),
                "announcement_verified_event_count": int(transition_summary["announcement_verified_event_count"]),
                "unverified_transition_candidate_count": int(
                    transition_summary["unverified_transition_candidate_count"]
                ),
                "inferred_candidates_used_as_official_events": False,
                "official_explicit_transition_event_ledger_ready": False,
                "ready_for_strategy_replay": False,
                "remaining_blocker": (
                    "Core/Research must decide whether daily status snapshot diff candidates are sufficient for "
                    "universe integrity, or Radar/Data must broaden official announcement verification"
                ),
            },
            {
                "blocker": "monthly_revenue_pit",
                "before_status": "blocked",
                "after_status": "blocked",
                "delta": "unchanged_highest_value_next_data_layer",
                "transition_candidate_count": int(transition_summary["transition_candidate_count"]),
                "announcement_verification_attempts": int(transition_summary["announcement_verification_attempts"]),
                "announcement_verified_event_count": int(transition_summary["announcement_verified_event_count"]),
                "unverified_transition_candidate_count": int(
                    transition_summary["unverified_transition_candidate_count"]
                ),
                "inferred_candidates_used_as_official_events": False,
                "official_explicit_transition_event_ledger_ready": False,
                "ready_for_strategy_replay": False,
                "remaining_blocker": "MOPS monthly revenue full universe PIT with release_date remains the highest-value next blocker",
            },
        ]
    )


def _blocker_delta_after_mops_monthly_revenue(monthly_revenue: dict[str, Any]) -> pd.DataFrame:
    summary = _monthly_revenue_summary(monthly_revenue)
    return pd.DataFrame(
        [
            {
                "blocker": "monthly_revenue_pit",
                "before_status": "blocked",
                "after_status": summary["status"],
                "delta": "full_universe_source_candidate_ready_formal_exact_false"
                if summary["source_candidate_ready"]
                else "unchanged_blocked",
                "period_start": summary["period_start"],
                "period_end": summary["period_end"],
                "route_request_attempts": int(summary["route_request_attempts"]),
                "failed_attempts": int(summary["failed_attempts"]),
                "accepted_rows": int(summary["accepted_rows"]),
                "symbol_count": int(summary["symbol_count"]),
                "accepted_month_market": int(summary["accepted_month_market"]),
                "expected_month_market": int(summary["expected_month_market"]),
                "coverage_ratio_month_market": summary["coverage_ratio_month_market"],
                "formal_exact": bool(summary["formal_exact"]),
                "available_date_policy": summary["available_date_policy"],
                "ready_for_strategy_replay": bool(summary["ready_for_strategy_replay"]),
                "remaining_blocker": summary["remaining_blocker"],
            },
            {
                "blocker": "dynamic_pool1_shadow_challenger",
                "before_status": "blocked",
                "after_status": "blocked",
                "delta": "monthly_revenue_blocker_cleared_but_other_layers_missing"
                if summary["source_candidate_ready"]
                else "unchanged",
                "period_start": summary["period_start"],
                "period_end": summary["period_end"],
                "route_request_attempts": int(summary["route_request_attempts"]),
                "failed_attempts": int(summary["failed_attempts"]),
                "accepted_rows": int(summary["accepted_rows"]),
                "symbol_count": int(summary["symbol_count"]),
                "accepted_month_market": int(summary["accepted_month_market"]),
                "expected_month_market": int(summary["expected_month_market"]),
                "coverage_ratio_month_market": summary["coverage_ratio_month_market"],
                "formal_exact": bool(summary["formal_exact"]),
                "available_date_policy": summary["available_date_policy"],
                "ready_for_strategy_replay": False,
                "remaining_blocker": "quarterly fundamentals PIT, market cap PIT, sector/mainline PIT, and universe-integrity policy remain incomplete",
            },
            {
                "blocker": "quarterly_fundamentals_pit",
                "before_status": "blocked",
                "after_status": "blocked",
                "delta": "unchanged_highest_value_next_data_layer",
                "period_start": "",
                "period_end": "",
                "route_request_attempts": 0,
                "failed_attempts": 0,
                "accepted_rows": 0,
                "symbol_count": 0,
                "accepted_month_market": 0,
                "expected_month_market": 0,
                "coverage_ratio_month_market": "",
                "formal_exact": False,
                "available_date_policy": "",
                "ready_for_strategy_replay": False,
                "remaining_blocker": "quarterly fundamentals PIT with announcement/release dates is the next highest-value data blocker",
            },
        ]
    )


def _blocker_delta_after_quarterly_fundamentals_route_unlock(quarterly_fundamentals: dict[str, Any]) -> pd.DataFrame:
    summary = _quarterly_fundamentals_summary(quarterly_fundamentals)
    return pd.DataFrame(
        [
            {
                "blocker": "quarterly_fundamentals_pit",
                "before_status": "blocked",
                "after_status": summary["status"],
                "delta": (
                    "full_sweep_source_candidate_ready_formal_exact_false"
                    if summary["full_sweep_ready"]
                    else "route_unlocked_sample_rows_available_full_sweep_still_missing"
                    if summary["route_unlocked"]
                    else "unchanged_blocked"
                ),
                "sample_rows": int(summary["sample_rows"]),
                "accepted_rows": int(summary["accepted_rows"]),
                "symbol_count": int(summary["symbol_count"]),
                "covered_start": summary["covered_start"],
                "covered_end": summary["covered_end"],
                "failed_or_missing_period_markets": int(summary["failed_or_missing_period_markets"]),
                "tested_periods": ",".join(summary["tested_periods"]),
                "tested_markets": ",".join(summary["tested_markets"]),
                "formal_exact": bool(summary["formal_exact"]),
                "filing_date_available": bool(summary["filing_date_available"]),
                "ready_for_strategy_replay": bool(summary["ready_for_strategy_replay"]),
                "remaining_blocker": summary["remaining_blocker"],
            },
            {
                "blocker": "dynamic_pool1_shadow_challenger",
                "before_status": "blocked",
                "after_status": "blocked",
                "delta": "quarterly_route_unlocked_but_full_sweep_and_other_layers_missing"
                if summary["route_unlocked"] and not summary["full_sweep_ready"]
                else "quarterly_full_sweep_ready_but_other_layers_missing"
                if summary["full_sweep_ready"]
                else "unchanged",
                "sample_rows": int(summary["sample_rows"]),
                "accepted_rows": int(summary["accepted_rows"]),
                "symbol_count": int(summary["symbol_count"]),
                "covered_start": summary["covered_start"],
                "covered_end": summary["covered_end"],
                "failed_or_missing_period_markets": int(summary["failed_or_missing_period_markets"]),
                "tested_periods": ",".join(summary["tested_periods"]),
                "tested_markets": ",".join(summary["tested_markets"]),
                "formal_exact": bool(summary["formal_exact"]),
                "filing_date_available": bool(summary["filing_date_available"]),
                "ready_for_strategy_replay": False,
                "remaining_blocker": "quarterly fundamentals full sweep, market cap PIT, sector/mainline PIT, and universe-integrity policy remain incomplete",
            },
            {
                "blocker": "market_cap_pit",
                "before_status": "partial",
                "after_status": "partial",
                "delta": "unchanged_next_data_layer_after_quarterly_full_sweep",
                "sample_rows": 0,
                "accepted_rows": 0,
                "symbol_count": 0,
                "covered_start": "",
                "covered_end": "",
                "failed_or_missing_period_markets": 0,
                "tested_periods": "",
                "tested_markets": "",
                "formal_exact": False,
                "filing_date_available": False,
                "ready_for_strategy_replay": False,
                "remaining_blocker": "historical market cap/free-float market cap PIT remains missing",
            },
        ]
    )


def _blocker_delta_after_market_cap_partial(market_cap: dict[str, Any]) -> pd.DataFrame:
    summary = _market_cap_summary(market_cap)
    return pd.DataFrame(
        [
            {
                "blocker": "market_cap_pit",
                "before_status": "partial_current_snapshot_only",
                "after_status": summary["status"],
                "delta": (
                    "tpex_full_total_market_cap_source_candidate_ready_twse_free_float_blocked"
                    if summary["tpex_full_sweep_completed"]
                    else "twse_capital_stock_full_sweep_proxy_contract_ready_direct_daily_market_cap_blocked"
                    if summary["twse_capital_stock_full_sweep_ready"] and summary["proxy_contract_ready"]
                    else "twse_quarterly_capital_stock_route_partial_proxy_candidate_direct_market_cap_blocked"
                    if summary["twse_quarterly_capital_stock_route_partial"]
                    else "tpex_total_market_cap_source_candidate_available_twse_free_float_blocked"
                    if summary["tpex_partial_ready"]
                    else "unchanged"
                ),
                "accepted_rows": int(summary["accepted_rows"]),
                "tpex_completed_dates": int(summary["tpex_completed_dates"]),
                "tpex_expected_weekday_dates": int(summary["tpex_expected_weekday_dates"]),
                "tpex_missing_dates": int(summary["tpex_missing_dates"]),
                "accepted_markets": ",".join(summary["accepted_markets"]),
                "blocked_markets": ",".join(summary["blocked_markets"]),
                "source_type": summary["source_type"],
                "twse_capital_stock_sample_rows": int(summary["twse_capital_stock_sample_rows"]),
                "twse_market_cap_sample_rows": int(summary["twse_market_cap_sample_rows"]),
                "twse_quarterly_capital_stock_route_partial": bool(
                    summary["twse_quarterly_capital_stock_route_partial"]
                ),
                "twse_capital_stock_full_sweep_ready": bool(summary["twse_capital_stock_full_sweep_ready"]),
                "proxy_contract_ready": bool(summary["proxy_contract_ready"]),
                "covered_periods": int(summary["covered_periods"]),
                "expected_periods": int(summary["expected_periods"]),
                "missing_or_failed_periods": int(summary["missing_or_failed_periods"]),
                "symbols": int(summary["symbols"]),
                "formal_exact": bool(summary["formal_exact"]),
                "free_float_market_cap_ready": bool(summary["free_float_market_cap_ready"]),
                "ready_for_strategy_replay": bool(summary["ready_for_strategy_replay"]),
                "remaining_blocker": summary["remaining_blocker"],
            },
            {
                "blocker": "dynamic_pool1_shadow_challenger",
                "before_status": "blocked",
                "after_status": "blocked",
                "delta": "market_cap_partial_available_but_required_layers_missing"
                if summary["tpex_partial_ready"] and not summary["tpex_full_sweep_completed"]
                else "twse_capital_stock_full_sweep_proxy_contract_available_but_policy_free_float_sector_missing"
                if summary["twse_capital_stock_full_sweep_ready"] and summary["proxy_contract_ready"]
                else "twse_capital_stock_proxy_candidate_available_but_daily_market_cap_free_float_sector_missing"
                if summary["twse_quarterly_capital_stock_route_partial"]
                else "tpex_market_cap_full_sweep_available_but_twse_free_float_sector_missing"
                if summary["tpex_full_sweep_completed"]
                else "unchanged",
                "accepted_rows": int(summary["accepted_rows"]),
                "tpex_completed_dates": int(summary["tpex_completed_dates"]),
                "tpex_expected_weekday_dates": int(summary["tpex_expected_weekday_dates"]),
                "tpex_missing_dates": int(summary["tpex_missing_dates"]),
                "accepted_markets": ",".join(summary["accepted_markets"]),
                "blocked_markets": ",".join(summary["blocked_markets"]),
                "source_type": summary["source_type"],
                "twse_capital_stock_sample_rows": int(summary["twse_capital_stock_sample_rows"]),
                "twse_market_cap_sample_rows": int(summary["twse_market_cap_sample_rows"]),
                "twse_quarterly_capital_stock_route_partial": bool(
                    summary["twse_quarterly_capital_stock_route_partial"]
                ),
                "twse_capital_stock_full_sweep_ready": bool(summary["twse_capital_stock_full_sweep_ready"]),
                "proxy_contract_ready": bool(summary["proxy_contract_ready"]),
                "covered_periods": int(summary["covered_periods"]),
                "expected_periods": int(summary["expected_periods"]),
                "missing_or_failed_periods": int(summary["missing_or_failed_periods"]),
                "symbols": int(summary["symbols"]),
                "formal_exact": bool(summary["formal_exact"]),
                "free_float_market_cap_ready": bool(summary["free_float_market_cap_ready"]),
                "ready_for_strategy_replay": False,
                "remaining_blocker": "TWSE historical market cap, free-float market cap, sector/mainline PIT, and universe-integrity policy remain incomplete",
            },
            {
                "blocker": "sector_membership_pit",
                "before_status": "blocked",
                "after_status": "blocked",
                "delta": "unchanged_next_data_layer_after_market_cap",
                "accepted_rows": 0,
                "tpex_completed_dates": 0,
                "tpex_expected_weekday_dates": 0,
                "tpex_missing_dates": 0,
                "accepted_markets": "",
                "blocked_markets": "",
                "source_type": "",
                "twse_capital_stock_sample_rows": 0,
                "twse_market_cap_sample_rows": 0,
                "twse_quarterly_capital_stock_route_partial": False,
                "twse_capital_stock_full_sweep_ready": False,
                "proxy_contract_ready": False,
                "covered_periods": 0,
                "expected_periods": 0,
                "missing_or_failed_periods": 0,
                "symbols": 0,
                "formal_exact": False,
                "free_float_market_cap_ready": False,
                "ready_for_strategy_replay": False,
                "remaining_blocker": "date-aware sector/mainline membership PIT remains missing",
            },
        ]
    )


def _liquidity_sweep_summary(liquidity_sweep: dict[str, Any]) -> dict[str, Any]:
    readiness = liquidity_sweep.get("readiness", {})
    covered = readiness.get("covered_date_range", {})
    return {
        "exists": bool(liquidity_sweep.get("exists")),
        "path": liquidity_sweep.get("path", ""),
        "full_range_ready": bool(readiness.get("all_listed_liquid_universe_pit_daily_full_range_ready", False)),
        "listing_delisting_suspension_metadata_ready": bool(
            readiness.get("listing_delisting_suspension_metadata_ready", False)
        ),
        "ready_for_core_rerun": bool(readiness.get("ready_for_core_rerun", False)),
        "ready_for_strategy_replay": bool(readiness.get("ready_for_strategy_replay", False)),
        "dynamic_pool1_shadow_challenger_ready": bool(readiness.get("dynamic_pool1_shadow_challenger_ready", False)),
        "covered_start": covered.get("start", ""),
        "covered_end": covered.get("end", ""),
        "accepted_liquidity_rows": int(readiness.get("accepted_liquidity_rows", 0) or 0),
        "accepted_shard_count": int(readiness.get("accepted_shard_count", 0) or 0),
        "failed_attempts": int(readiness.get("failed_attempts", 0) or 0),
        "missing_attempts": int(readiness.get("missing_attempts", 0) or 0),
    }


def _tpex_status_blocker_status(tpex_status: dict[str, Any]) -> dict[str, Any]:
    summary = _tpex_status_blocker_summary(tpex_status)
    if summary["exists"]:
        if summary["accepted_historical_rows"] > 0:
            if summary["full_tpex_2015_2025_route_coverage_ready"]:
                reason = (
                    "TPEx 2015-2025 full route coverage is available for listing/delisting and daily as-of "
                    "status snapshots, but explicit suspension/resumption transition events remain unavailable"
                )
            else:
                reason = (
                    "TPEx official historical route contract produced bounded accepted sample rows, but full "
                    "2015-2025 sweep and suspension/resumption event history are not complete"
                )
            return {
                "status": summary["status"],
                "reason": reason,
                "accepted_for_formal": False,
            }
        return {
            "status": summary["status"],
            "reason": (
                "bounded TPEx probes completed with zero accepted 2015-2025 historical rows; "
                "current/carried 2026 rows are excluded from historical PIT"
            ),
            "accepted_for_formal": False,
        }
    return {
        "status": "blocked",
        "reason": "TPEx historical listing/status blocker evidence package not available",
        "accepted_for_formal": False,
    }


def _tpex_status_blocker_summary(tpex_status: dict[str, Any]) -> dict[str, Any]:
    readiness = tpex_status.get("readiness", {})
    accepted_listing = int(readiness.get("accepted_listing_metadata_rows", 0) or 0)
    accepted_status_snapshot = int(readiness.get("accepted_status_snapshot_rows", 0) or 0)
    accepted_suspension = int(readiness.get("accepted_suspension_event_rows", 0) or 0)
    accepted_historical = int(
        readiness.get("accepted_historical_rows", accepted_listing + accepted_status_snapshot + accepted_suspension) or 0
    )
    current_or_carried = int(
        readiness.get("accepted_current_or_carried_tpex_rows", tpex_status.get("current_or_carried_rows", 0)) or 0
    )
    route_probe_attempts = int(
        readiness.get("route_probe_attempts", readiness.get("route_request_attempts", 0)) or 0
    )
    source_probe_attempts = int(readiness.get("source_probe_attempts", tpex_status.get("attempt_rows", 0)) or 0)
    status = readiness.get("status", "blocked_with_attempt_evidence" if tpex_status.get("exists") else "blocked")
    if accepted_historical > 0 and readiness.get("full_tpex_2015_2025_route_coverage_ready", False):
        status = "route_coverage_ready_status_snapshot_partial"
    elif accepted_historical > 0:
        status = "partial_with_accepted_historical_rows"
    return {
        "exists": bool(tpex_status.get("exists")),
        "path": tpex_status.get("path", ""),
        "status": status,
        "accepted_historical_rows": accepted_historical,
        "accepted_listing_metadata_rows": accepted_listing,
        "accepted_suspension_event_rows": accepted_suspension,
        "accepted_status_snapshot_rows": accepted_status_snapshot,
        "accepted_current_or_carried_tpex_rows": current_or_carried,
        "current_or_carried_rows_used_as_historical": False,
        "source_probe_attempts": source_probe_attempts,
        "route_probe_attempts": route_probe_attempts,
        "failed_attempts": int(readiness.get("failed_attempts", 0) or 0),
        "blocked_source_rows": int(readiness.get("blocked_source_rows", tpex_status.get("blocked_rows", 0)) or 0),
        "future_data_violation_count": int(readiness.get("future_data_violation_count", 0) or 0),
        "covered_start": readiness.get("covered_start", ""),
        "covered_end": readiness.get("covered_end", ""),
        "tpex_2015_2025_historical_listing_status_ready": bool(
            readiness.get("tpex_2015_2025_historical_listing_status_ready", False)
        ),
        "full_tpex_2015_2025_route_coverage_ready": bool(
            readiness.get("full_tpex_2015_2025_route_coverage_ready", False)
        ),
        "tpex_full_2015_2025_master_ready": bool(
            readiness.get("tpex_full_2015_2025_master_ready", False)
            or readiness.get("full_2015_2025_master_ready", False)
        ),
        "accepted_delisting_metadata_rows": int(readiness.get("accepted_delisting_metadata_rows", 0) or 0),
        "daily_status_snapshot_asof_ready": bool(
            readiness.get("full_tpex_2015_2025_route_coverage_ready", False) and accepted_status_snapshot > 0
        ),
        "explicit_transition_event_ledger_ready": bool(accepted_suspension > 0),
        "full_cross_market_listing_master_ready": bool(readiness.get("full_cross_market_listing_master_ready", False)),
        "ready_for_core_rerun": bool(readiness.get("ready_for_core_rerun", False)),
        "ready_for_strategy_replay": bool(readiness.get("ready_for_strategy_replay", False)),
        "dynamic_pool1_shadow_challenger_ready": bool(readiness.get("dynamic_pool1_shadow_challenger_ready", False)),
        "readiness_decision": readiness.get("readiness_decision", ""),
        "remaining_blocker": (
            "TPEx full route coverage is ready for listing/delisting and daily as-of status snapshots, but explicit "
            "suspension/resumption transition event ledger remains 0 accepted rows"
            if readiness.get("full_tpex_2015_2025_route_coverage_ready", False)
            else "TPEx historical route contract has sample accepted rows, but full 2015-2025 sweep and suspension/"
            "resumption event history are not complete"
            if accepted_historical > 0
            else "TPEx 2015-2025 historical listing/status route remains unresolved; current/carried 2026 rows are not "
            "historical PIT"
        ),
    }


def _tpex_transition_candidate_summary(tpex_transition: dict[str, Any]) -> dict[str, Any]:
    readiness = tpex_transition.get("readiness", {})
    candidate_count = int(
        readiness.get("transition_candidate_count", tpex_transition.get("candidate_rows", 0)) or 0
    )
    verified_count = int(
        readiness.get("announcement_verified_event_count", tpex_transition.get("verified_rows", 0)) or 0
    )
    unverified_count = int(
        readiness.get("unverified_transition_candidate_count", tpex_transition.get("unverified_rows", 0)) or 0
    )
    verification_attempts = int(
        readiness.get("announcement_verification_attempts", tpex_transition.get("attempt_rows", 0)) or 0
    )
    exists = bool(tpex_transition.get("exists"))
    if verified_count > 0:
        status = "partial_with_announcement_verified_transition_events"
    elif candidate_count > 0:
        status = "partial_unverified_inferred_transition_candidates"
    elif exists:
        status = "blocked_no_transition_candidates"
    else:
        status = "missing"
    return {
        "exists": exists,
        "path": tpex_transition.get("path", ""),
        "status": status,
        "transition_candidate_count": candidate_count,
        "announcement_verification_attempts": verification_attempts,
        "announcement_verified_event_count": verified_count,
        "unverified_transition_candidate_count": unverified_count,
        "future_data_violation_count": int(readiness.get("future_data_violation_count", 0) or 0),
        "ready_for_core_rerun": bool(readiness.get("ready_for_core_rerun", False)),
        "ready_for_strategy_replay": False,
        "dynamic_pool1_shadow_challenger_ready": False,
        "inferred_candidates_used_as_official_events": False,
        "official_explicit_transition_event_ledger_ready": bool(verified_count > 0),
        "source_type_boundary": readiness.get(
            "source_type_boundary",
            "inferred candidates are not official explicit events unless source_type=announcement_verified",
        ),
        "remaining_blocker": (
            "transition candidates are inferred from daily status snapshot diffs and remain unverified; "
            "announcement verified events are 0"
            if candidate_count > 0 and verified_count == 0
            else "TPEx transition candidate package is missing or has zero accepted transition candidates"
            if candidate_count == 0
            else "official verification is partial; Core/Research must decide whether coverage is sufficient"
        ),
    }


def _monthly_revenue_summary(monthly_revenue: dict[str, Any]) -> dict[str, Any]:
    readiness = monthly_revenue.get("readiness", {})
    exists = bool(monthly_revenue.get("exists"))
    source_candidate_ready = bool(readiness.get("monthly_revenue_pit_full_universe_ready", False))
    return {
        "exists": exists,
        "path": monthly_revenue.get("path", ""),
        "status": "source_candidate_ready" if source_candidate_ready else "blocked",
        "source_candidate_ready": source_candidate_ready,
        "monthly_revenue_pit_partial_ready": bool(readiness.get("monthly_revenue_pit_partial_ready", False)),
        "period_start": readiness.get("period_start", ""),
        "period_end": readiness.get("period_end", ""),
        "route_request_attempts": int(readiness.get("route_request_attempts", 0) or 0),
        "failed_attempts": int(readiness.get("failed_attempts", 0) or 0),
        "accepted_rows": int(readiness.get("accepted_rows", monthly_revenue.get("sample_rows", 0)) or 0),
        "symbol_count": int(readiness.get("symbol_count", 0) or 0),
        "accepted_month_market": int(readiness.get("accepted_month_market", 0) or 0),
        "expected_month_market": int(readiness.get("expected_month_market", 0) or 0),
        "coverage_ratio_month_market": readiness.get("coverage_ratio_month_market", ""),
        "future_data_violation_count": int(readiness.get("future_data_violation_count", 0) or 0),
        "ready_for_core_rerun": bool(readiness.get("ready_for_core_rerun", False)),
        "ready_for_strategy_replay": False,
        "dynamic_pool1_shadow_challenger_ready": False,
        "formal_exact": False,
        "available_date_policy": "conservative_next_month_day_10_weekday_adjusted",
        "source_type_boundary": readiness.get(
            "source_type_boundary",
            "release/source dates are conservative available dates, not per-company actual filing timestamps; formal_exact=false",
        ),
        "shard_manifest_rows": int(monthly_revenue.get("shard_manifest_rows", 0) or 0),
        "sample_rows": int(monthly_revenue.get("sample_rows", 0) or 0),
        "remaining_blocker": (
            "monthly revenue PIT source candidate is ready for Dynamic Pool1 diagnostics/basic fundamentals using "
            "conservative available_date; exact per-company filing timestamp remains non-exact"
            if source_candidate_ready
            else "MOPS monthly revenue full-universe PIT with source/release/available dates is missing"
        ),
    }


def _quarterly_fundamentals_summary(quarterly_fundamentals: dict[str, Any]) -> dict[str, Any]:
    readiness = quarterly_fundamentals.get("readiness", {})
    exists = bool(quarterly_fundamentals.get("exists"))
    tested_periods = list(readiness.get("tested_periods", []) or [])
    tested_markets = list(readiness.get("tested_markets", []) or [])
    route_unlocked = bool(readiness.get("quarterly_fundamentals_route_unlocked", False))
    full_ready = bool(
        readiness.get("quarterly_fundamentals_pit_full_universe_ready", False)
        or readiness.get("quarterly_fundamentals_full_sweep_ready", False)
    )
    sample_rows = int(readiness.get("sample_rows", quarterly_fundamentals.get("sample_rows", 0)) or 0)
    accepted_rows = int(readiness.get("accepted_rows", sample_rows) or 0)
    if full_ready and not route_unlocked:
        route_unlocked = True
    return {
        "exists": exists,
        "path": quarterly_fundamentals.get("path", ""),
        "status": "full_sweep_source_candidate_ready"
        if full_ready
        else "route_unlocked_source_candidate_partial"
        if route_unlocked
        else "blocked",
        "route_unlocked": route_unlocked,
        "full_sweep_ready": full_ready,
        "quarterly_fundamentals_pit_full_universe_ready": full_ready,
        "sample_rows": sample_rows,
        "accepted_rows": accepted_rows,
        "symbol_count": int(readiness.get("symbol_count", 0) or 0),
        "covered_start": readiness.get("covered_start", ""),
        "covered_end": readiness.get("covered_end", ""),
        "failed_or_missing_period_markets": int(readiness.get("failed_or_missing_period_markets", 0) or 0),
        "route_request_attempts": int(readiness.get("route_request_attempts", 0) or 0),
        "full_source_rows_observed": int(readiness.get("full_source_rows_observed", accepted_rows) or 0),
        "tested_periods": tested_periods,
        "tested_markets": tested_markets,
        "source_type": readiness.get("source_type", "source_candidate_no_exact_filing_date"),
        "formal_exact": bool(readiness.get("formal_exact", False)),
        "filing_date_available": bool(readiness.get("filing_date_available", False)),
        "available_date_policy": readiness.get(
            "available_date_policy",
            "conservative statutory deadline, not exact filing timestamp",
        ),
        "future_data_violation_count": int(readiness.get("future_data_violation_count", 0) or 0),
        "ready_for_core_rerun": bool(readiness.get("ready_for_core_rerun", False)),
        "ready_for_strategy_replay": False,
        "dynamic_pool1_shadow_challenger_ready": False,
        "remaining_blocker": (
            "quarterly fundamentals t163sb04 income-statement summary full sweep is ready as source candidate; "
            "per-company exact filing_date and balance sheet/cash flow/ratio expansion remain separate blockers"
            if full_ready
            else
            "quarterly fundamentals route is unlocked with sample rows, but full 2015-latest sweep and exact "
            "per-company filing_date are not ready"
            if route_unlocked
            else "quarterly fundamentals route/source candidate is missing"
        ),
    }


def _market_cap_summary(market_cap: dict[str, Any]) -> dict[str, Any]:
    readiness = market_cap.get("readiness", {})
    exists = bool(market_cap.get("exists"))
    accepted_markets = list(readiness.get("accepted_markets", []) or [])
    blocked_markets = list(readiness.get("blocked_markets", []) or [])
    tpex_full_sweep_completed = bool(readiness.get("tpex_full_sweep_completed", False))
    twse_capital_stock_full_sweep_ready = bool(
        readiness.get("twse_capital_stock_full_sweep_ready", False)
        or market_cap.get("twse_capital_stock_manifest_rows", 0)
    )
    twse_proxy_contract_ready = bool(
        readiness.get("proxy_contract_ready", False)
        or market_cap.get("twse_proxy_contract_rows_file_count", 0)
    )
    twse_capital_stock_partial = bool(
        twse_capital_stock_full_sweep_ready
        or readiness.get("twse_capital_stock_route_partial", False)
        or readiness.get("twse_quarterly_capital_stock_route_unlocked", False)
        or readiness.get("twse_sample_ready", False)
        or market_cap.get("twse_capital_stock_rows_file_count", 0)
    )
    partial_ready = bool(readiness.get("market_cap_pit_partial_ready", False))
    full_ready = bool(readiness.get("market_cap_pit_ready", False))
    if tpex_full_sweep_completed and "TPEx" not in accepted_markets:
        accepted_markets = ["TPEx", *accepted_markets]
    if twse_capital_stock_partial and "TWSE" not in accepted_markets:
        accepted_markets = [*accepted_markets, "TWSE"]
    if tpex_full_sweep_completed and "TWSE" not in blocked_markets and not readiness.get("twse_route_unlocked", False):
        blocked_markets = [*blocked_markets, "TWSE"]
    if twse_capital_stock_partial and "TWSE" not in blocked_markets and not readiness.get("twse_market_cap_route_unlocked", False):
        blocked_markets = [*blocked_markets, "TWSE"]
    twse_capital_stock_sample_rows = int(
        readiness.get(
            "accepted_twse_capital_stock_rows",
            readiness.get(
                "accepted_twse_capital_stock_sample_rows",
                market_cap.get("twse_capital_stock_rows_file_count", 0),
            ),
        )
        or 0
    )
    twse_capital_stock_sample_only_rows = int(
        readiness.get(
            "accepted_twse_capital_stock_sample_rows",
            market_cap.get("twse_capital_stock_rows_file_count", 0),
        )
        or 0
    )
    twse_market_cap_sample_rows = int(
        readiness.get(
            "accepted_twse_market_cap_sample_rows",
            market_cap.get("twse_market_cap_rows_file_count", 0),
        )
        or 0
    )
    accepted_rows = int(
        readiness.get(
            "accepted_rows",
            readiness.get(
                "tpex_accepted_rows",
                twse_capital_stock_sample_rows
                or market_cap.get("accepted_rows_file_count", 0)
                or market_cap.get("twse_capital_stock_rows_file_count", 0),
            ),
        )
        or 0
    )
    return {
        "exists": exists,
        "path": market_cap.get("path", ""),
        "status": "market_cap_pit_ready"
        if full_ready
        else "tpex_full_total_market_cap_source_candidate"
        if tpex_full_sweep_completed
        else "twse_capital_stock_full_sweep_proxy_contract_ready"
        if twse_capital_stock_full_sweep_ready and twse_proxy_contract_ready
        else "twse_quarterly_capital_stock_route_partial"
        if twse_capital_stock_partial
        else "tpex_partial_total_market_cap_source_candidate"
        if partial_ready
        else "partial_current_snapshot_only"
        if exists
        else "blocked",
        "market_cap_pit_ready": full_ready,
        "market_cap_pit_partial_ready": partial_ready,
        "tpex_partial_ready": bool((partial_ready or tpex_full_sweep_completed) and "TPEx" in accepted_markets),
        "tpex_full_sweep_completed": tpex_full_sweep_completed,
        "tpex_completed_dates": int(readiness.get("tpex_completed_dates", 0) or 0),
        "tpex_expected_weekday_dates": int(readiness.get("tpex_expected_weekday_dates", 0) or 0),
        "tpex_missing_dates": int(readiness.get("tpex_missing_dates", 0) or 0),
        "twse_quarterly_capital_stock_route_partial": twse_capital_stock_partial,
        "twse_capital_stock_full_sweep_ready": twse_capital_stock_full_sweep_ready,
        "proxy_contract_ready": twse_proxy_contract_ready,
        "twse_market_cap_route_unlocked": bool(readiness.get("twse_market_cap_route_unlocked", False)),
        "twse_issued_shares_route_unlocked": bool(readiness.get("twse_issued_shares_route_unlocked", False)),
        "twse_still_blocked": bool(readiness.get("twse_still_blocked", False) or twse_capital_stock_partial),
        "twse_capital_stock_sample_rows": twse_capital_stock_sample_rows,
        "twse_market_cap_sample_rows": twse_market_cap_sample_rows,
        "covered_periods": int(readiness.get("covered_periods", 0) or 0),
        "expected_periods": int(readiness.get("expected_periods", 0) or 0),
        "missing_or_failed_periods": int(readiness.get("missing_or_failed_periods", 0) or 0),
        "accepted_periods": list(readiness.get("accepted_periods", []) or []),
        "accepted_symbols": int(readiness.get("accepted_symbols", 0) or 0),
        "symbols": int(readiness.get("symbols", readiness.get("accepted_symbols", 0)) or 0),
        "accepted_rows": accepted_rows,
        "accepted_markets": accepted_markets,
        "blocked_markets": blocked_markets,
        "source_type": readiness.get(
            "source_type",
            "quarterly_capital_stock_source_candidate_no_exact_filing_date"
            if twse_capital_stock_partial
            else "shares_derived_official_daily_candidate",
        ),
        "formal_exact": bool(readiness.get("formal_exact", False)),
        "free_float_market_cap_ready": bool(readiness.get("free_float_market_cap_ready", False)),
        "future_data_violation_count": int(readiness.get("future_data_violation_count", 0) or 0),
        "ready_for_core_rerun": bool(readiness.get("ready_for_core_rerun", False)),
        "ready_for_strategy_replay": False,
        "dynamic_pool1_shadow_challenger_ready": False,
        "remaining_blocker": (
            "TPEx total market cap full sweep is ready as source candidate from official daily close and shares; "
            "TWSE historical issued shares/direct market cap route and free-float market cap remain blocked"
            if tpex_full_sweep_completed
            else
            "TWSE quarterly capital stock full sweep and proxy market-cap contract are ready for diagnostic/shadow "
            "candidate use, but they are not direct daily market cap, not daily exact issued shares, and not "
            "free-float market cap; capital-stock-to-shares normalization policy remains required"
            if twse_capital_stock_full_sweep_ready and twse_proxy_contract_ready
            else
            "TWSE quarterly capital stock route is unlocked as a diagnostic/proxy source candidate, but it is not "
            "daily issued shares or direct daily market cap; full quarter sweep, daily close proxy contract, "
            "capital-change effective-date handling, and free-float market cap remain blocked"
            if twse_capital_stock_partial
            else
            "TPEx total market cap source candidate is sample-verified from official daily close and shares; TWSE "
            "historical issued shares/direct market cap route and free-float market cap remain blocked"
            if partial_ready
            else "historical market cap/free-float market cap PIT is missing or current snapshot only"
        ),
    }


def _listing_metadata_status(listing_metadata: dict[str, Any]) -> dict[str, Any]:
    summary = _listing_metadata_summary(listing_metadata)
    if summary["partial_event_rows_available"]:
        return {
            "status": summary["readiness_level"],
            "reason": (
                "improved TWSE status coverage and partial official event rows are available, but complete "
                "cross-market historical master remains missing"
                if summary["readiness_level"] == "stronger_partial"
                else "partial official event rows are available, but complete historical master remains missing"
            ),
            "accepted_for_formal": False,
        }
    return {
        "status": "blocked",
        "reason": "no accepted listing/delisting/suspension metadata package available",
        "accepted_for_formal": False,
    }


def _listing_metadata_summary(listing_metadata: dict[str, Any]) -> dict[str, Any]:
    readiness = listing_metadata.get("readiness", {})
    accepted_listing = int(readiness.get("accepted_listing_metadata_rows", listing_metadata.get("accepted_listing_rows", 0)) or 0)
    accepted_suspension = int(
        readiness.get("accepted_suspension_event_rows", listing_metadata.get("accepted_suspension_rows", 0)) or 0
    )
    accepted_code_name_change = int(readiness.get("accepted_code_name_change_rows", 0) or 0)
    accepted_transfer = int(readiness.get("accepted_transfer_listing_rows", 0) or 0)
    accepted_total = int(
        readiness.get(
            "accepted_event_rows_total",
            accepted_listing + accepted_suspension + accepted_code_name_change + accepted_transfer,
        )
        or 0
    )
    previous_total = int(readiness.get("previous_accepted_event_rows", 0) or 0)
    delta_vs_previous = int(readiness.get("new_or_carried_forward_event_rows_delta_vs_previous", 0) or 0)
    twse_improved = bool(
        readiness.get("twse_suspension_resumption_range_sweep_candidate")
        or readiness.get("twse_altered_trading_monthly_anchor_candidate")
    )
    tpex_blocked = not bool(
        readiness.get("tpex_historical_listing_delisting_master_ready")
        and readiness.get("tpex_historical_suspension_resumption_master_ready")
    )
    stronger_partial = bool(accepted_total >= 1000 and twse_improved and tpex_blocked)
    return {
        "exists": bool(listing_metadata.get("exists")),
        "path": listing_metadata.get("path", ""),
        "status": readiness.get("status", ""),
        "readiness_level": "stronger_partial" if stronger_partial else ("partial" if accepted_total else "blocked"),
        "partial_event_rows_available": bool(accepted_listing or accepted_suspension),
        "accepted_listing_metadata_rows": accepted_listing,
        "accepted_suspension_event_rows": accepted_suspension,
        "accepted_code_name_change_rows": accepted_code_name_change,
        "accepted_transfer_listing_rows": accepted_transfer,
        "accepted_event_rows": accepted_total,
        "previous_accepted_event_rows": previous_total,
        "delta_vs_previous": delta_vs_previous,
        "proxy_source_rows": int(readiness.get("proxy_source_rows", 0) or 0),
        "blocked_source_rows": int(readiness.get("blocked_source_rows", listing_metadata.get("blocked_rows", 0)) or 0),
        "twse_improved_partial_status_coverage": twse_improved,
        "tpex_2015_2025_blocked": tpex_blocked,
        "listing_delisting_suspension_metadata_ready": bool(
            readiness.get("listing_delisting_suspension_metadata_ready", False)
        ),
        "ready_for_core_rerun": bool(readiness.get("ready_for_core_rerun", False)),
        "ready_for_strategy_replay": bool(readiness.get("ready_for_strategy_replay", False)),
        "dynamic_pool1_shadow_challenger_ready": bool(readiness.get("dynamic_pool1_shadow_challenger_ready", False)),
        "future_data_violation_count": int(readiness.get("future_data_violation_count", 0) or 0),
        "readiness_decision": readiness.get("readiness_decision", ""),
        "twse_only_diagnostic_possible": stronger_partial,
        "twse_only_diagnostic_recommendation": (
            "possible only as TWSE-only diagnostic/sensitivity; not recommended as formal cross-market challenger "
            "because TPEx 2015-2025 and full master metadata remain blocked"
            if stronger_partial
            else "not enough listing metadata coverage for even TWSE-only diagnostic"
        ),
        "remaining_blocker": (
            "TPEx 2015-2025 historical listing/status routes, MOPS material-information date-range crawler, "
            "and full daily TWSE altered-trading status panel"
            if stronger_partial
            else "complete historical listing/delisting/suspension master is missing"
        ),
    }


def _price_readiness_status(price_coverage: pd.DataFrame) -> tuple[str, str]:
    if price_coverage.empty:
        return "blocked", "no price coverage found"
    adjusted_rows = int(price_coverage["adjusted_close_available"].astype(bool).sum())
    if adjusted_rows == len(price_coverage):
        return "partial", "price coverage exists but does not define all-listed universe membership"
    return "partial", "some price sources are unadjusted-only or not strategy-ready"


def _csv_columns(path: Path) -> list[str]:
    try:
        return list(pd.read_csv(path, nrows=0).columns)
    except Exception:
        return []


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _read_csv_if_exists(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path).fillna("")
    except Exception:
        return pd.DataFrame()


def _count_csv_rows(path: Path) -> int:
    try:
        with path.open("r", encoding="utf-8-sig", errors="ignore") as handle:
            return max(sum(1 for _ in handle) - 1, 0)
    except Exception:
        return 0


def _date_range_from_csv(path: Path, columns: list[str]) -> tuple[str, str]:
    date_column = next(
        (column for column in ("date", "source_date", "effective_date", "report_date", "source_updated_at") if column in columns),
        "",
    )
    if not date_column:
        return "", ""
    try:
        frame = pd.read_csv(path, usecols=[date_column])
    except Exception:
        return "", ""
    dates = pd.to_datetime(frame[date_column], errors="coerce").dropna()
    if dates.empty:
        return "", ""
    return dates.min().strftime("%Y-%m-%d"), dates.max().strftime("%Y-%m-%d")


def _adj_close_ready(path: Path, columns: list[str]) -> bool:
    if "adj_close" not in columns:
        return False
    try:
        values = pd.read_csv(path, usecols=["adj_close"])["adj_close"]
    except Exception:
        return False
    numeric = pd.to_numeric(values, errors="coerce")
    return bool(numeric.notna().any())


def _ticker_from_price_filename(name: str) -> str:
    stem = Path(name).stem
    if stem.endswith("_TW"):
        return f"{stem[:-3]}.TW"
    return stem.replace("_", ".")


def _bool_like(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _price_columns() -> list[str]:
    return [
        "ticker",
        "source",
        "source_path",
        "first_date",
        "last_date",
        "row_count",
        "adjusted_close_available",
        "source_type",
        "price_source_ready",
        "strategy_ready",
        "synthetic_used",
        "diagnostic_only",
    ]


def _contract_columns() -> list[str]:
    return [
        "record_type",
        "data_area",
        "ticker",
        "name",
        "value",
        "source_date",
        "release_date",
        "effective_date",
        "source",
        "source_path",
        "source_type",
        "row_count",
        "readiness_status",
        "diagnostic_only",
        "accepted_for_formal",
        "blocked_reason",
    ]


def _all_listed_columns() -> list[str]:
    return [
        "record_type",
        "date",
        "ticker",
        "name",
        "listing_status",
        "delisting_or_suspension_status",
        "liquidity_20d_twd",
        "liquidity_60d_twd",
        "market_cap_twd",
        "is_liquid",
        "source_date",
        "release_date",
        "effective_date",
        "source",
        "source_type",
        "readiness_status",
        "diagnostic_only",
        "accepted_for_formal",
        "blocked_reason",
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build Dynamic Pool1 PIT readiness contract package.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--price-cache-dir", default=DEFAULT_PRICE_CACHE_DIR)
    parser.add_argument("--price-source-registry", default=DEFAULT_PRICE_SOURCE_REGISTRY)
    parser.add_argument("--tw50-constituents", default=DEFAULT_TW50_CONSTITUENTS)
    parser.add_argument("--ai-theme-candidates", default=DEFAULT_AI_THEME_CANDIDATES)
    parser.add_argument("--radar-data-dir", default=DEFAULT_RADAR_DATA_DIR)
    parser.add_argument("--liquidity-sweep-output", default=DEFAULT_LIQUIDITY_SWEEP_OUTPUT)
    parser.add_argument("--listing-metadata-output", default=DEFAULT_LISTING_METADATA_OUTPUT)
    parser.add_argument("--tpex-status-output", default=DEFAULT_TPEX_STATUS_OUTPUT)
    parser.add_argument("--tpex-transition-output", default=DEFAULT_TPEX_TRANSITION_OUTPUT)
    parser.add_argument("--monthly-revenue-output", default=DEFAULT_MONTHLY_REVENUE_OUTPUT)
    parser.add_argument("--quarterly-fundamentals-output", default=DEFAULT_QUARTERLY_FUNDAMENTALS_OUTPUT)
    parser.add_argument("--market-cap-output", default=DEFAULT_MARKET_CAP_OUTPUT)
    args = parser.parse_args(argv)
    run_dynamic_pool1_pit_readiness_contract(
        output_dir=args.output_dir,
        price_cache_dir=args.price_cache_dir,
        price_source_registry=args.price_source_registry,
        tw50_constituents_path=args.tw50_constituents,
        ai_theme_candidates_path=args.ai_theme_candidates,
        radar_data_dir=args.radar_data_dir,
        liquidity_sweep_output=args.liquidity_sweep_output,
        listing_metadata_output=args.listing_metadata_output,
        tpex_status_output=args.tpex_status_output,
        tpex_transition_output=args.tpex_transition_output,
        monthly_revenue_output=args.monthly_revenue_output,
        quarterly_fundamentals_output=args.quarterly_fundamentals_output,
        market_cap_output=args.market_cap_output,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
