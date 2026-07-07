"""Materialize bounded broader Layer1 t164 source table from Core contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


TASK_ID = "TASK-BACKTEST-CORE-VNEXT-LAYER1-T164-BOUNDED-BROADER-MATERIALIZATION-001"
DEFAULT_CONTRACT_DIR = Path("outputs/vnext_layer1_t164_bounded_broader_ingest_contract_20260707")
DEFAULT_OUTPUT_DIR = Path("outputs/vnext_layer1_t164_bounded_broader_materialization_20260707")


def build_materialization(
    *, contract_dir: str | Path = DEFAULT_CONTRACT_DIR, output_dir: str | Path = DEFAULT_OUTPUT_DIR
) -> dict[str, Any]:
    contract_path = Path(contract_dir)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    readiness_in = _read_json(contract_path / "readiness_for_layer1_t164_bounded_broader_ingest_contract.json")
    contract = _read_csv(contract_path / "layer1_t164_bounded_broader_ingest_contract.csv", dtype={"ticker": str})
    field_policy = _read_csv(contract_path / "layer1_t164_field_policy.csv")
    blockers_in = _read_csv(contract_path / "layer1_t164_bounded_contract_blockers.csv")

    materialized = _materialized_table(contract)
    availability = _field_availability(materialized)
    asof_audit = _asof_eligibility_audit(materialized)
    label_ledger = _blocked_proxy_label_ledger(field_policy, blockers_in)
    future_audit = _future_data_audit(materialized)
    coverage = _coverage_summary(materialized)
    readiness = _readiness(readiness_in, materialized, coverage, future_audit)

    _write_csv(materialized, output / "layer1_t164_bounded_materialized_source_table.csv")
    _write_csv(availability, output / "layer1_t164_bounded_field_availability_by_ticker_market_period.csv")
    _write_csv(asof_audit, output / "layer1_t164_bounded_official_asof_eligibility_audit.csv")
    _write_csv(label_ledger, output / "layer1_t164_bounded_blocked_proxy_human_review_ledger.csv")
    _write_csv(future_audit, output / "layer1_t164_bounded_future_data_audit.csv")
    _write_csv(coverage, output / "layer1_t164_bounded_coverage_summary.csv")
    (output / "readiness_for_layer1_t164_bounded_materialization.json").write_text(
        json.dumps(readiness, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    manifest = {
        "task_id": TASK_ID,
        "status": readiness["status"],
        "output_dir": str(output.resolve()),
        "contract_input_dir": str(contract_path.resolve()),
        "output_files": [
            "layer1_t164_bounded_materialized_source_table.csv",
            "layer1_t164_bounded_field_availability_by_ticker_market_period.csv",
            "layer1_t164_bounded_official_asof_eligibility_audit.csv",
            "layer1_t164_bounded_blocked_proxy_human_review_ledger.csv",
            "layer1_t164_bounded_future_data_audit.csv",
            "layer1_t164_bounded_coverage_summary.csv",
            "readiness_for_layer1_t164_bounded_materialization.json",
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
    (output / "final_summary_zh.md").write_text(_summary(readiness), encoding="utf-8")
    return manifest


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _read_csv(path: Path, **kwargs: Any) -> pd.DataFrame:
    if not path.exists() or path.read_text(encoding="utf-8").strip() == "empty":
        return pd.DataFrame()
    return pd.read_csv(path, **kwargs)


def _materialized_table(contract: pd.DataFrame) -> pd.DataFrame:
    out = contract.copy()
    out["materialization_scope"] = "bounded_broader_pruning_v2_seed"
    out["source_contract_status"] = "materialized_from_core_bounded_contract"
    out["layer1_source_candidate"] = True
    out["quality_floor_input_candidate"] = True
    out["formal_ready"] = False
    out["ready_for_experiments"] = False
    out["diagnostic_only"] = True
    out["not_live_rule"] = True
    out["forward_returns_live_rule_usage"] = False
    return out


def _field_availability(materialized: pd.DataFrame) -> pd.DataFrame:
    fields = [
        "operating_cash_flow",
        "investing_cash_flow",
        "capex_proxy",
        "inventory",
        "receivables_trade",
        "current_assets",
        "current_liabilities",
        "current_ratio",
    ]
    rows = []
    for row in materialized.to_dict("records"):
        for field in fields:
            rows.append(
                {
                    "ticker": row["ticker"],
                    "market": row["market"],
                    "report_period": row["report_period"],
                    "field": field,
                    "available": pd.notna(row.get(field)),
                    "source_quality": row.get(f"{field}_source_quality")
                    or _default_source_quality(field),
                    "diagnostic_only": True,
                }
            )
    return pd.DataFrame(rows)


def _default_source_quality(field: str) -> str:
    if field in {"capex_proxy", "receivables_trade"}:
        return "human_review_proxy_label_required"
    if field == "current_ratio":
        return "derived_pit_after_official_asof_join"
    return "exact_pit_after_official_asof_join"


def _asof_eligibility_audit(materialized: pd.DataFrame) -> pd.DataFrame:
    return materialized[
        [
            "ticker",
            "market",
            "report_period",
            "official_asof_match_status",
            "match_status",
            "official_market_available_at",
            "official_market_available_at_iso",
            "signal_eligible_date",
            "signal_eligible_date_policy",
            "after_close_policy_applies",
            "blocked_reason",
            "quarter_end_date_used",
            "query_response_datetime_used",
            "conservative_deadline_proxy_used",
            "diagnostic_only",
        ]
    ].copy()


def _blocked_proxy_label_ledger(field_policy: pd.DataFrame, blockers: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if not field_policy.empty:
        for row in field_policy.to_dict("records"):
            status = row.get("policy_status")
            if "proxy" in str(status) or "human_review" in str(status):
                rows.append(
                    {
                        "item": row.get("field"),
                        "status": status,
                        "source_quality": row.get("source_quality"),
                        "reason": "human-review proxy label required; not formal-ready",
                        "diagnostic_only": True,
                    }
                )
    if not blockers.empty:
        for row in blockers.to_dict("records"):
            rows.append(
                {
                    "item": row.get("blocker"),
                    "status": row.get("status"),
                    "source_quality": "blocked",
                    "reason": row.get("detail"),
                    "diagnostic_only": True,
                }
            )
    return pd.DataFrame(rows)


def _future_data_audit(materialized: pd.DataFrame) -> pd.DataFrame:
    prohibited = int(
        materialized[["quarter_end_date_used", "query_response_datetime_used", "conservative_deadline_proxy_used"]]
        .astype(bool)
        .any(axis=1)
        .sum()
    )
    return pd.DataFrame(
        [
            {
                "audit_item": "quarter_end_query_response_deadline_proxy_not_used",
                "status": "passed" if prohibited == 0 else "failed",
                "future_data_violation_count": prohibited,
                "note": "official route does not use prohibited available_at sources",
            },
            {
                "audit_item": "forward_return_as_rule",
                "status": "passed" if not materialized["forward_returns_live_rule_usage"].astype(bool).any() else "failed",
                "future_data_violation_count": int(materialized["forward_returns_live_rule_usage"].astype(bool).sum()),
                "note": "no forward return rule inputs",
            },
            {
                "audit_item": "official_asof_match_all_rows",
                "status": "passed" if materialized["match_status"].eq("accepted").all() else "blocked",
                "future_data_violation_count": 0,
                "note": "all bounded rows use official t05st01/t05st01_detail timestamp",
            },
        ]
    )


def _coverage_summary(materialized: pd.DataFrame) -> pd.DataFrame:
    rows = [
        {
            "coverage_axis": "overall",
            "group": "all",
            "sample_rows": len(materialized),
            "ticker_count": int(materialized["ticker"].nunique()),
            "period_count": int(materialized["report_period"].nunique()),
            "statement_success_rows": int(materialized["t164sb05_status"].astype(str).str.startswith("code=200").sum()),
            "official_asof_matched_rows": int(materialized["match_status"].eq("accepted").sum()),
            "blocked_rows": int(materialized["blocked_reason"].notna().sum()),
            "diagnostic_only": True,
        }
    ]
    for axis, column in [("market", "market"), ("period", "report_period")]:
        for group, subset in materialized.groupby(column, dropna=False):
            rows.append(
                {
                    "coverage_axis": axis,
                    "group": group,
                    "sample_rows": len(subset),
                    "ticker_count": int(subset["ticker"].nunique()),
                    "period_count": int(subset["report_period"].nunique()),
                    "statement_success_rows": int(subset["t164sb05_status"].astype(str).str.startswith("code=200").sum()),
                    "official_asof_matched_rows": int(subset["match_status"].eq("accepted").sum()),
                    "blocked_rows": int(subset["blocked_reason"].notna().sum()),
                    "diagnostic_only": True,
                }
            )
    return pd.DataFrame(rows)


def _readiness(readiness_in: dict[str, Any], materialized: pd.DataFrame, coverage: pd.DataFrame, future_audit: pd.DataFrame) -> dict[str, Any]:
    future_count = int(future_audit["future_data_violation_count"].sum())
    ready = (
        len(materialized) == int(readiness_in.get("sample_rows", len(materialized)))
        and materialized["match_status"].eq("accepted").all()
        and future_count == 0
    )
    return {
        "date": "2026-07-07",
        "task_id": TASK_ID,
        "owner": "BACKTEST_LAB Core/Data",
        "status": "bounded_broader_materialized_not_experiments_ready",
        "diagnostic_only": True,
        "sample_rows": len(materialized),
        "ticker_count": int(materialized["ticker"].nunique()),
        "period_count": int(materialized["report_period"].nunique()),
        "markets": sorted(materialized["market"].dropna().unique().tolist()),
        "statement_success_rows": int(materialized["t164sb05_status"].astype(str).str.startswith("code=200").sum()),
        "official_asof_matched_rows": int(materialized["match_status"].eq("accepted").sum()),
        "blocked_rows": int(materialized["blocked_reason"].notna().sum()),
        "ready_for_layer1_t164_bounded_interim_diagnostic_planning": bool(ready),
        "ready_for_core_t164_broader_or_full_ingest_next": False,
        "ready_for_experiments": False,
        "ready_for_formal": False,
        "ready_for_strategy_replay": False,
        "ready_for_full_universe": False,
        "retained_caveats": [
            "bounded materialization only, not full universe",
            "TPEx all-stock proof not complete",
            "full period range not complete",
            "capex_proxy / receivables_trade human-review proxy policy required",
        ],
        "future_data_violation_count": future_count,
        "not_live_rule": True,
        "forward_returns_live_rule_usage": False,
        "formal_model_changed": False,
        "trade_decision_changed": False,
        "active_in_trade_decision": False,
        "report_changed": False,
        "portfolio_replay_executed": False,
    }


def _summary(readiness: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Layer1 t164 Bounded Broader Materialization",
            "",
            f"Status: {readiness['status']}",
            "",
            "Conclusion: Core materialized the bounded broader t164 source table from the approved contract. This remains source materialization only, not full universe, not Experiments-ready, and not formal-ready.",
            "",
            "Readiness:",
            f"- ready_for_layer1_t164_bounded_interim_diagnostic_planning={str(readiness['ready_for_layer1_t164_bounded_interim_diagnostic_planning']).lower()}",
            "- ready_for_core_t164_broader_or_full_ingest_next=false",
            "- ready_for_experiments=false",
            "- ready_for_formal=false",
            "- ready_for_strategy_replay=false",
            f"- sample_rows={readiness['sample_rows']}",
            f"- ticker_count={readiness['ticker_count']}",
            f"- period_count={readiness['period_count']}",
            f"- statement_success_rows={readiness['statement_success_rows']}",
            f"- official_asof_matched_rows={readiness['official_asof_matched_rows']}",
            f"- blocked_rows={readiness['blocked_rows']}",
            f"- future_data_violation_count={readiness['future_data_violation_count']}",
            "",
            "Retained caveats:",
            *[f"- {item}" for item in readiness["retained_caveats"]],
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


def _write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract-dir", type=Path, default=DEFAULT_CONTRACT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args(argv)
    manifest = build_materialization(contract_dir=args.contract_dir, output_dir=args.output_dir)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
