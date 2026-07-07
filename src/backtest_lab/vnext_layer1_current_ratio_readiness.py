"""Build current-ratio-only Layer 1 PIT contract design/readiness.

This uses Radar/Data bounded parser samples for MOPS t163sb05. It intentionally
does not claim full-universe ingest readiness. The output is a sample-backed
contract design and PIT timing audit for Research judgment.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from backtest_lab.vnext_funnel_layer1_refreshed_readiness import (
    DEFAULT_MATERIALIZATION_DIR,
    _read_json,
    _read_optional_csv,
    _weekly_universe,
    _write_csv,
)


TASK_ID = "TASK-BACKTEST-CORE-VNEXT-LAYER1-CURRENT-RATIO-ONLY-PIT-CONTRACT-READINESS-001"
DEFAULT_RADAR_SAMPLE_DIR = Path(
    r"C:\Users\zergv\Documents\Codex\2026-05-23\ai-stock-rotation-radar-https-docs\outputs"
    r"\radar_vnext_layer1_remaining_fields_bounded_parser_sample_unlock_20260707"
)
DEFAULT_OUTPUT_DIR = Path("outputs/vnext_layer1_current_ratio_only_pit_contract_20260707")

PERIODS = [
    ("P1", "2015-01-02", "2022-12-29"),
    ("P2", "2023-01-02", "2026-06-30"),
    ("2024-latest", "2024-01-02", "2026-06-30"),
    ("2026YTD", "2026-01-02", "2026-06-30"),
]


def build_current_ratio_readiness(
    *,
    materialization_dir: str | Path = DEFAULT_MATERIALIZATION_DIR,
    radar_sample_dir: str | Path = DEFAULT_RADAR_SAMPLE_DIR,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    materialization = Path(materialization_dir)
    radar_dir = Path(radar_sample_dir)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    radar_readiness = _read_json(radar_dir / "readiness_for_core_layer1_remaining_parser_ingest.json")
    sample_contract = _read_optional_csv(radar_dir / "t163sb05_balance_sheet_sample_parse_contract.csv")
    mapping = _read_optional_csv(radar_dir / "derived_ratio_field_mapping.csv")
    blocked = _read_optional_csv(radar_dir / "parser_blocked_fields_ledger.csv")
    timing = _read_optional_csv(radar_dir / "sample_pit_timing_audit.csv")

    universe = _weekly_universe(materialization / "vnext_weekly_candidate_snapshot.csv")
    current_ratio = _normalize_current_ratio_samples(sample_contract)
    joined = _candidate_join_contract(universe, current_ratio)
    source_quality = _source_quality_matrix(joined, radar_readiness, mapping, blocked)
    missingness = _missingness_by_period(joined)
    future_audit = _future_data_audit(joined, radar_readiness, timing)
    readiness = _readiness_json(joined, current_ratio, source_quality, missingness, future_audit, radar_readiness)

    _write_csv(current_ratio, output / "layer1_current_ratio_pit_contract.csv")
    _write_csv(joined, output / "layer1_current_ratio_candidate_join_contract.csv")
    _write_csv(source_quality, output / "layer1_current_ratio_source_quality_matrix.csv")
    _write_csv(missingness, output / "layer1_current_ratio_missingness_by_period.csv")
    _write_csv(future_audit, output / "layer1_current_ratio_future_data_audit.csv")
    (output / "readiness_for_layer1_current_ratio_diagnostic.json").write_text(
        json.dumps(readiness, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    manifest = {
        "task_id": TASK_ID,
        "status": readiness["status"],
        "output_dir": str(output.resolve()),
        "input_materialization_dir": str(materialization.resolve()),
        "radar_sample_dir": str(radar_dir.resolve()),
        "radar_sample_commit": "4ff1ec9",
        "output_files": [
            "layer1_current_ratio_pit_contract.csv",
            "layer1_current_ratio_candidate_join_contract.csv",
            "layer1_current_ratio_source_quality_matrix.csv",
            "layer1_current_ratio_missingness_by_period.csv",
            "layer1_current_ratio_future_data_audit.csv",
            "readiness_for_layer1_current_ratio_diagnostic.json",
            "manifest.json",
            "final_summary_zh.md",
        ],
        "formal_model_changed": False,
        "trade_decision_changed": False,
        "active_in_trade_decision": False,
        "report_changed": False,
        "portfolio_replay_executed": False,
        "ready_for_strategy_replay": False,
        "not_live_rule": True,
        "forward_returns_live_rule_usage": False,
        "diagnostic_only": True,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "final_summary_zh.md").write_text(_summary(readiness, source_quality), encoding="utf-8")
    return manifest


def _normalize_current_ratio_samples(sample_contract: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for source in sample_contract.itertuples(index=False):
        fiscal_period = str(getattr(source, "fiscal_period"))
        available_date = _conservative_available_date(fiscal_period)
        sample_rows = json.loads(getattr(source, "sample_rows_json"))
        for row in sample_rows:
            current_assets = pd.to_numeric(row.get("current_assets"), errors="coerce")
            current_liabilities = pd.to_numeric(row.get("current_liabilities"), errors="coerce")
            derived_ratio = current_assets / current_liabilities if pd.notna(current_liabilities) and current_liabilities != 0 else pd.NA
            rows.append(
                {
                    "ticker": str(row.get("ticker")),
                    "name": row.get("name"),
                    "market": getattr(source, "market"),
                    "fiscal_period": fiscal_period,
                    "source_period": getattr(source, "source_period"),
                    "table_index": getattr(source, "table_index"),
                    "source_file": getattr(source, "source_file"),
                    "available_date": available_date,
                    "current_assets": current_assets,
                    "current_liabilities": current_liabilities,
                    "current_ratio": derived_ratio,
                    "current_ratio_sample_from_radar": pd.to_numeric(row.get("current_ratio_sample"), errors="coerce"),
                    "source_quality": "sample_unlocked_proxy",
                    "parser_status": getattr(source, "status"),
                    "pit_timing_policy": "conservative_statutory_quarter_available_date_sample_only",
                    "full_universe_materialized": False,
                    "diagnostic_only": True,
                    "not_live_rule": True,
                }
            )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["available_date"] = pd.to_datetime(out["available_date"])
    out = out.sort_values(["ticker", "available_date", "fiscal_period"]).drop_duplicates(
        ["ticker", "fiscal_period"], keep="last"
    )
    return out


def _conservative_available_date(fiscal_period: str) -> str:
    year = int(fiscal_period[:4])
    quarter = int(fiscal_period[-1])
    if quarter == 1:
        return f"{year}-05-15"
    if quarter == 2:
        return f"{year}-08-14"
    if quarter == 3:
        return f"{year}-11-14"
    return f"{year + 1}-03-31"


def _candidate_join_contract(universe: pd.DataFrame, current_ratio: pd.DataFrame) -> pd.DataFrame:
    keys = universe[["signal_date", "ticker"]].drop_duplicates().sort_values(["ticker", "signal_date"])
    latest = _latest_asof(keys, current_ratio, "available_date")
    joined = universe.merge(latest, on=["signal_date", "ticker"], how="left")
    joined["current_ratio_available"] = joined["current_ratio"].notna()
    joined["current_ratio_contract_scope"] = "sample_only_not_full_universe"
    joined["inventory_risk_available"] = False
    joined["receivable_risk_available"] = False
    joined["operating_cash_flow_quality_available"] = False
    joined["free_cash_flow_quality_available"] = False
    joined["free_float_market_cap_available"] = False
    joined["exact_market_cap_available"] = False
    joined["full_sector_pit_available"] = False
    joined["forward_return_as_rule"] = False
    joined["diagnostic_only"] = True
    joined["not_live_rule"] = True
    cols = [
        "signal_date",
        "ticker",
        "name_x",
        "theme_id",
        "theme_name",
        "valid_universe",
        "fundamental_pass",
        "market_attention_member",
        "eligible_pool_member",
        "fiscal_period",
        "available_date",
        "current_assets",
        "current_liabilities",
        "current_ratio",
        "source_quality",
        "parser_status",
        "pit_timing_policy",
        "full_universe_materialized",
        "current_ratio_available",
        "current_ratio_contract_scope",
        "inventory_risk_available",
        "receivable_risk_available",
        "operating_cash_flow_quality_available",
        "free_cash_flow_quality_available",
        "free_float_market_cap_available",
        "exact_market_cap_available",
        "full_sector_pit_available",
        "forward_return_as_rule",
        "diagnostic_only",
        "not_live_rule",
    ]
    return joined.reindex(columns=cols).rename(columns={"name_x": "name"})


def _latest_asof(keys: pd.DataFrame, source: pd.DataFrame, date_col: str) -> pd.DataFrame:
    if source.empty:
        return keys.copy()
    source = source.copy()
    source["ticker"] = source["ticker"].astype(str)
    source[date_col] = pd.to_datetime(source[date_col], errors="coerce")
    source = source.dropna(subset=[date_col]).sort_values(["ticker", date_col])
    parts = []
    for ticker, group in keys.groupby("ticker", sort=False):
        source_group = source[source["ticker"].eq(ticker)]
        if source_group.empty:
            parts.append(group.copy())
            continue
        parts.append(
            pd.merge_asof(
                group.sort_values("signal_date"),
                source_group.sort_values(date_col),
                left_on="signal_date",
                right_on=date_col,
                by="ticker",
                direction="backward",
            )
        )
    return pd.concat(parts, ignore_index=True)


def _source_quality_matrix(
    joined: pd.DataFrame,
    radar_readiness: dict[str, Any],
    mapping: pd.DataFrame,
    blocked: pd.DataFrame,
) -> pd.DataFrame:
    current_ratio_rows = int(joined["current_ratio_available"].sum())
    rows = [
        ("current_assets", "sample_unlocked_proxy", current_ratio_rows, "parsed from bounded t163sb05 sample; full sweep not materialized"),
        ("current_liabilities", "sample_unlocked_proxy", current_ratio_rows, "parsed from bounded t163sb05 sample; full sweep not materialized"),
        ("current_ratio", "sample_unlocked_proxy", current_ratio_rows, "derived as current_assets/current_liabilities; sample-only"),
        ("inventory_risk", "blocked", 0, "not in standard t163sb05 summary sample"),
        ("receivable_risk", "blocked", 0, "profile-specific or missing; no universal parser"),
        ("operating_cash_flow_quality", "blocked", 0, "cash-flow route sample missing"),
        ("free_cash_flow_quality", "blocked", 0, "depends on operating cash flow and capex fields"),
        ("free_float_market_cap", "blocked", 0, "outside parser scope; no local official free-float route"),
        ("exact_market_cap", "blocked", 0, "TWSE exact daily market cap still blocked"),
        ("full_sector_pit", "blocked", 0, "TPEx all-stock sector PIT remains blocked"),
        ("forward_return_as_rule", "prohibited", 0, "forward returns prohibited"),
    ]
    out = pd.DataFrame(
        [
            {
                "field": field,
                "source_quality": quality,
                "available_rows": available,
                "source_quality_reason": reason,
                "usable_for_current_ratio_contract_design": quality == "sample_unlocked_proxy",
                "usable_for_full_layer1_diagnostic": False,
                "usable_for_formal": False,
                "diagnostic_only": True,
            }
            for field, quality, available, reason in rows
        ]
    )
    out["radar_ready_for_core_contract_design_current_ratio_only"] = bool(
        radar_readiness.get("ready_for_core_contract_design_current_ratio_only", False)
    )
    out["radar_ready_for_core_layer1_remaining_parser_ingest"] = bool(
        radar_readiness.get("ready_for_core_layer1_remaining_parser_ingest", False)
    )
    out["mapping_rows"] = int(len(mapping))
    out["radar_blocked_ledger_rows"] = int(len(blocked))
    return out


def _missingness_by_period(joined: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for period, start, end in PERIODS:
        subset = joined[(joined["signal_date"] >= pd.Timestamp(start)) & (joined["signal_date"] <= pd.Timestamp(end))]
        for field in ["current_assets", "current_liabilities", "current_ratio", "current_ratio_available"]:
            available = subset[field].astype(bool) if field.endswith("_available") else subset[field].notna()
            rows.append(
                {
                    "period": period,
                    "requested_start": start,
                    "requested_end": end,
                    "actual_start": subset["signal_date"].min() if not subset.empty else pd.NaT,
                    "actual_end": subset["signal_date"].max() if not subset.empty else pd.NaT,
                    "field": field,
                    "rows": int(len(subset)),
                    "available_rows": int(available.sum()) if len(subset) else 0,
                    "missing_rows": int(len(subset) - available.sum()) if len(subset) else 0,
                    "available_share": float(available.mean()) if len(subset) else 0.0,
                    "diagnostic_only": True,
                }
            )
    return pd.DataFrame(rows)


def _future_data_audit(joined: pd.DataFrame, radar_readiness: dict[str, Any], timing: pd.DataFrame) -> pd.DataFrame:
    dates = pd.to_datetime(joined["available_date"], errors="coerce")
    bad = int((dates.notna() & (dates > joined["signal_date"])).sum())
    return pd.DataFrame(
        [
            {
                "audit_item": "current_ratio_available_date_lte_signal_date",
                "status": "passed" if bad == 0 else "failed",
                "future_data_violation_count": bad,
                "note": "sample rows joined by backward as-of conservative statutory quarter date",
            },
            {
                "audit_item": "radar_sample_future_data_audit",
                "status": "passed" if int(radar_readiness.get("future_data_violation_count", 0)) == 0 else "failed",
                "future_data_violation_count": int(radar_readiness.get("future_data_violation_count", 0)),
                "note": f"Radar sample timing audit rows={len(timing)}",
            },
            {
                "audit_item": "forward_return_as_rule",
                "status": "passed",
                "future_data_violation_count": 0,
                "note": "no forward return fields included",
            },
        ]
    )


def _readiness_json(
    joined: pd.DataFrame,
    current_ratio: pd.DataFrame,
    source_quality: pd.DataFrame,
    missingness: pd.DataFrame,
    future_audit: pd.DataFrame,
    radar_readiness: dict[str, Any],
) -> dict[str, Any]:
    future_count = int(future_audit["future_data_violation_count"].sum())
    design_ready = (
        bool(radar_readiness.get("ready_for_core_contract_design_current_ratio_only", False))
        and not current_ratio.empty
        and future_count == 0
    )
    return {
        "date": "2026-07-07",
        "task_id": TASK_ID,
        "owner": "BACKTEST_LAB Core/Data",
        "status": "sample_ready_current_ratio_only_contract_design_not_full_ingest" if design_ready else "blocked_current_ratio_contract_design",
        "ready_for_current_ratio_contract_design_review": bool(design_ready),
        "ready_for_layer1_current_ratio_full_diagnostic": False,
        "ready_for_layer1_remaining_parser_ingest": False,
        "ready_for_merge_with_layer1_fuller_interim_diagnostic": False,
        "recommendation": "research_judgment_required_merge_now_vs_wait_for_cashflow_inventory_receivable",
        "current_ratio_contract_scope": "sample_only_not_full_universe",
        "sample_contract_rows": int(len(current_ratio)),
        "candidate_join_rows": int(len(joined)),
        "candidate_join_current_ratio_available_rows": int(joined["current_ratio_available"].sum()),
        "source_quality_rows": int(len(source_quality)),
        "missingness_rows": int(len(missingness)),
        "future_data_violation_count": future_count,
        "blocked_fields": source_quality[source_quality["source_quality"].eq("blocked")]["field"].tolist(),
        "prohibited_fields": source_quality[source_quality["source_quality"].eq("prohibited")]["field"].tolist(),
        "ready_for_portfolio_like_diagnostic": False,
        "ready_for_strategy_replay": False,
        "ready_for_formal": False,
        "formal_model_changed": False,
        "trade_decision_changed": False,
        "active_in_trade_decision": False,
        "report_changed": False,
        "portfolio_replay_executed": False,
        "not_live_rule": True,
        "forward_returns_live_rule_usage": False,
    }


def _summary(readiness: dict[str, Any], source_quality: pd.DataFrame) -> str:
    rows = [
        "# Layer1 Current Ratio Only PIT Contract Readiness",
        "",
        f"Status: {readiness['status']}",
        "",
        "Boundary: current_ratio-only sample-backed contract design; no full Layer1 ingest, no Experiments, no replay, no formal/report/trade change.",
        "",
        "Readiness:",
        f"- ready_for_current_ratio_contract_design_review={str(readiness['ready_for_current_ratio_contract_design_review']).lower()}",
        "- ready_for_layer1_current_ratio_full_diagnostic=false",
        "- ready_for_layer1_remaining_parser_ingest=false",
        "- ready_for_merge_with_layer1_fuller_interim_diagnostic=false",
        f"- current_ratio_contract_scope={readiness['current_ratio_contract_scope']}",
        f"- sample_contract_rows={readiness['sample_contract_rows']}",
        f"- candidate_join_current_ratio_available_rows={readiness['candidate_join_current_ratio_available_rows']}",
        f"- future_data_violation_count={readiness['future_data_violation_count']}",
        "",
        "Blocked fields kept blocked:",
    ]
    rows.extend(
        f"- {row.field}: {row.source_quality}; {row.source_quality_reason}"
        for row in source_quality[source_quality["source_quality"].isin(["blocked", "prohibited"])].itertuples()
    )
    rows.extend(
        [
            "",
            "Next handoff:",
            "- vNext Research should decide whether to merge this current_ratio-only design with Layer1 fuller interim diagnostic, or wait for cash-flow/inventory/receivable unlock.",
            "",
            "Flags:",
            "- formal_model_changed=false",
            "- trade_decision_changed=false",
            "- active_in_trade_decision=false",
            "- report_changed=false",
            "- portfolio_replay_executed=false",
            "- ready_for_strategy_replay=false",
            "- not_live_rule=true",
            "- forward_returns_live_rule_usage=false",
        ]
    )
    return "\n".join(rows)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--materialization-dir", type=Path, default=DEFAULT_MATERIALIZATION_DIR)
    parser.add_argument("--radar-sample-dir", type=Path, default=DEFAULT_RADAR_SAMPLE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args(argv)
    manifest = build_current_ratio_readiness(
        materialization_dir=args.materialization_dir,
        radar_sample_dir=args.radar_sample_dir,
        output_dir=args.output_dir,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
