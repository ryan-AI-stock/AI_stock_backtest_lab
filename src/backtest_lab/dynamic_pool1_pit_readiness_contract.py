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
        source_inventory = _source_inventory(
            price_cache_dir=Path(price_cache_dir),
            price_source_registry=Path(price_source_registry),
            tw50_constituents_path=Path(tw50_constituents_path),
            ai_theme_candidates_path=Path(ai_theme_candidates_path),
            radar_data_dir=Path(radar_data_dir),
            liquidity_sweep=liquidity_sweep,
            listing_metadata=listing_metadata,
            tpex_status=tpex_status,
        )
        price_coverage = _price_cache_coverage(Path(price_cache_dir), Path(price_source_registry))

        log("build_contract_tables", "started", "")
        tables = _build_contract_tables(source_inventory, price_coverage, liquidity_sweep)
        readiness_by_date = _candidate_data_readiness_by_date(source_inventory, price_coverage, liquidity_sweep)
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
        )
        dataset_summary = _dataset_readiness_summary(readiness)
        blocker_delta = _blocker_delta_after_liquidity_full_sweep(liquidity_sweep)
        listing_delta = _blocker_delta_after_listing_metadata(listing_metadata)
        listing_completion_delta = _blocker_delta_after_listing_master_completion(listing_metadata)
        tpex_delta = _blocker_delta_after_tpex_blocker_evidence(tpex_status)

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
) -> dict[str, pd.DataFrame]:
    return {
        "all_listed_liquid_universe_pit_daily": _all_listed_liquid_contract(
            source_inventory,
            price_coverage,
            liquidity_sweep,
        ),
        "monthly_revenue_pit": _blocked_contract_table(
            "monthly_revenue_pit",
            "blocked",
            "no accepted monthly revenue PIT source with release_date in repo",
        ),
        "quarterly_fundamentals_pit": _blocked_contract_table(
            "quarterly_fundamentals_pit",
            "blocked",
            "no accepted quarterly fundamentals PIT source with announcement/release dates in repo",
        ),
        "market_cap_pit": _source_contract_table(
            "market_cap_pit",
            source_inventory,
            "partial",
            "available sources are current/latest snapshots or date-filterable observation helpers, not a historical PIT panel",
        ),
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
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    price_status = _price_readiness_status(price_coverage)
    for bucket, start, end in YEAR_BUCKETS:
        for table in TABLE_SPECS:
            status, reason = _readiness_for_table(table, price_status, source_inventory, liquidity_sweep)
            rows.append(
                {
                    "year_bucket": bucket,
                    "start_date": start,
                    "end_date": end,
                    "data_table": table,
                    "readiness_status": status,
                    "accepted_for_formal": False,
                    "diagnostic_only": status in {"diagnostic_only", "partial"} and table != "monthly_revenue_pit",
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
) -> dict[str, Any]:
    price_status = _price_readiness_status(price_coverage)
    table_status = {
        table: {
            "status": _readiness_for_table(table, price_status, source_inventory, liquidity_sweep)[0],
            "reason": _readiness_for_table(table, price_status, source_inventory, liquidity_sweep)[1],
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
            "TPEx 2015-2025 historical status/listing metadata and full cross-market listing master",
            "monthly revenue PIT with announcement/release dates",
            "quarterly fundamentals PIT with announcement/release dates",
            "historical market cap/free-float market cap PIT",
            "sector/mainline membership PIT and sector breadth daily panel",
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
        "full 2015-2025 sweep 尚未完成。\n"
        "- monthly revenue：blocked，缺帶 release_date 的月營收 PIT。\n"
        "- quarterly fundamentals：blocked，缺帶公告日的季財報 PIT。\n"
        "- market cap：partial/current snapshot only，不可回推 2015。\n"
        "- sector membership / breadth：blocked，目前 current/generated map 不可回推 2015。\n"
        f"- future_data_violation_count={readiness['future_data_violation_count']}，因未把 current/proxy source 當正式歷史資料使用。\n\n"
        "formal_model_changed=false；trade_decision_changed=false；active_in_trade_decision=false。\n"
    )


def _readiness_for_table(
    table: str,
    price_status: tuple[str, str],
    source_inventory: list[dict[str, Any]],
    liquidity_sweep: dict[str, Any],
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
        return "blocked", "no monthly revenue PIT with source_date/release_date/effective_date"
    if table == "quarterly_fundamentals_pit":
        return "blocked", "no quarterly fundamentals PIT with source_date/release_date/effective_date"
    if table == "market_cap_pit":
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
            "reason": "monthly revenue, quarterly fundamentals, market cap PIT, sector PIT, and listing master remain incomplete",
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
            return {
                "status": summary["status"],
                "reason": (
                    "TPEx official historical route contract produced bounded accepted sample rows, but full "
                    "2015-2025 sweep and suspension/resumption event history are not complete"
                ),
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
    route_probe_attempts = int(readiness.get("route_probe_attempts", 0) or 0)
    source_probe_attempts = int(readiness.get("source_probe_attempts", tpex_status.get("attempt_rows", 0)) or 0)
    status = readiness.get("status", "blocked_with_attempt_evidence" if tpex_status.get("exists") else "blocked")
    if accepted_historical > 0:
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
        "blocked_source_rows": int(readiness.get("blocked_source_rows", tpex_status.get("blocked_rows", 0)) or 0),
        "future_data_violation_count": int(readiness.get("future_data_violation_count", 0) or 0),
        "tpex_2015_2025_historical_listing_status_ready": bool(
            readiness.get("tpex_2015_2025_historical_listing_status_ready", False)
        ),
        "tpex_full_2015_2025_master_ready": bool(
            readiness.get("tpex_full_2015_2025_master_ready", False)
            or readiness.get("full_2015_2025_master_ready", False)
        ),
        "full_cross_market_listing_master_ready": bool(readiness.get("full_cross_market_listing_master_ready", False)),
        "ready_for_core_rerun": bool(readiness.get("ready_for_core_rerun", False)),
        "ready_for_strategy_replay": bool(readiness.get("ready_for_strategy_replay", False)),
        "dynamic_pool1_shadow_challenger_ready": bool(readiness.get("dynamic_pool1_shadow_challenger_ready", False)),
        "readiness_decision": readiness.get("readiness_decision", ""),
        "remaining_blocker": (
            "TPEx historical route contract has sample accepted rows, but full 2015-2025 sweep and suspension/"
            "resumption event history are not complete"
            if accepted_historical > 0
            else "TPEx 2015-2025 historical listing/status route remains unresolved; current/carried 2026 rows are not "
            "historical PIT"
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
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
