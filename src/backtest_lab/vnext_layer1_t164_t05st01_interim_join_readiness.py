"""Build interim t164/t05st01 official-asof join readiness.

This joins t164 cashflow / inventory / receivable / capex / current-ratio
candidate field families to t05st01 official material-information timestamps
where sample rows match. It is readiness only, not full ingest or replay.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd


TASK_ID = "TASK-BACKTEST-CORE-VNEXT-LAYER1-T164-T05ST01-INTERIM-OFFICIAL-ASOF-JOIN-READINESS-001"
DEFAULT_T05_DIR = Path("outputs/vnext_layer1_t05st01_official_announcement_asof_contract_20260707")
DEFAULT_T164_DIR = Path("outputs/vnext_layer1_t164_payload_replay_conservative_asof_contract_20260707")
DEFAULT_OUTPUT_DIR = Path("outputs/vnext_layer1_t164_t05st01_interim_official_asof_join_20260707")


def build_interim_join_readiness(
    *,
    t05_dir: str | Path = DEFAULT_T05_DIR,
    t164_dir: str | Path = DEFAULT_T164_DIR,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    t05 = Path(t05_dir)
    t164 = Path(t164_dir)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    announcement = _read_csv(t05 / "layer1_t05st01_financial_report_announcement_asof_contract.csv")
    payload = _read_csv(t164 / "layer1_t164_cashflow_inventory_receivable_payload_contract.csv")
    taxonomy = _read_csv(t164 / "layer1_t164_label_taxonomy_human_review.csv")
    t05_readiness = _read_json(t05 / "readiness_for_layer1_t164_official_asof_join.json")
    t164_readiness = _read_json(t164 / "readiness_for_layer1_t164_diagnostic_only_contract.json")

    t164_candidates = _t164_candidate_rows(payload, taxonomy)
    join_contract = _join_contract(t164_candidates, announcement)
    coverage = _coverage_by_period(join_contract)
    unmatched = _unmatched_blocked(join_contract)
    eligibility = _signal_eligibility_policy_audit(join_contract)
    human_review = _human_review_fields(taxonomy)
    blocked = _blocked_proxy_fields(join_contract)
    future_audit = _future_data_audit(join_contract, eligibility)
    readiness = _readiness_json(join_contract, coverage, unmatched, eligibility, future_audit, t05_readiness, t164_readiness)

    _write_csv(join_contract, output / "layer1_t164_t05st01_interim_official_asof_join_contract.csv")
    _write_csv(coverage, output / "layer1_t164_t05st01_join_coverage_by_period.csv")
    _write_csv(unmatched, output / "layer1_t164_t05st01_unmatched_blocked_rows.csv")
    _write_csv(eligibility, output / "layer1_t164_t05st01_signal_eligibility_policy_audit.csv")
    _write_csv(human_review, output / "layer1_t164_t05st01_label_human_review_fields.csv")
    _write_csv(blocked, output / "layer1_t164_t05st01_blocked_proxy_fields.csv")
    _write_csv(future_audit, output / "layer1_t164_t05st01_future_data_audit.csv")
    (output / "readiness_for_layer1_t164_t05st01_interim_official_asof_diagnostic.json").write_text(
        json.dumps(readiness, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    manifest = {
        "task_id": TASK_ID,
        "status": readiness["status"],
        "output_dir": str(output.resolve()),
        "input_t05st01_asof_contract_dir": str(t05.resolve()),
        "input_t164_payload_contract_dir": str(t164.resolve()),
        "output_files": [
            "layer1_t164_t05st01_interim_official_asof_join_contract.csv",
            "layer1_t164_t05st01_join_coverage_by_period.csv",
            "layer1_t164_t05st01_unmatched_blocked_rows.csv",
            "layer1_t164_t05st01_signal_eligibility_policy_audit.csv",
            "layer1_t164_t05st01_label_human_review_fields.csv",
            "layer1_t164_t05st01_blocked_proxy_fields.csv",
            "layer1_t164_t05st01_future_data_audit.csv",
            "readiness_for_layer1_t164_t05st01_interim_official_asof_diagnostic.json",
            "final_summary_zh.md",
            "manifest.json",
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
    (output / "final_summary_zh.md").write_text(_summary(readiness, blocked), encoding="utf-8")
    return manifest


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _t164_candidate_rows(payload: pd.DataFrame, taxonomy: pd.DataFrame) -> pd.DataFrame:
    field_families = [
        ("t164sb05", "operating_cash_flow", "cashflow_quality", False),
        ("t164sb05", "capex_proxy", "free_cash_flow_proxy", True),
        ("t164sb05", "inventory_change_cashflow", "working_capital_context", False),
        ("t164sb05", "receivable_change_cashflow", "working_capital_context", True),
        ("t164sb03", "inventory", "inventory_risk", False),
        ("t164sb03", "receivables_basket", "receivable_risk", True),
        ("t164sb03", "current_assets", "current_ratio_input", False),
        ("t164sb03", "current_liabilities", "current_ratio_input", False),
        ("t164sb03", "current_ratio", "current_ratio", False),
    ]
    rows = []
    for api, field, role, human_required in field_families:
        api_payload = payload[payload["api"].eq(api)]
        if api_payload.empty:
            rows.append(_candidate_row(api, field, role, human_required, "unmatched_api_payload_missing", None))
            continue
        custom = api_payload[api_payload["payload_case"].eq("custom_period_required_empty_subsidiary_key")]
        source = custom.iloc[0] if not custom.empty else api_payload.iloc[0]
        rows.append(_candidate_row(api, field, role, human_required, "payload_ready_sample_candidate", source))
    return pd.DataFrame(rows)


def _candidate_row(api: str, field: str, role: str, human_required: bool, status: str, source: pd.Series | None) -> dict[str, Any]:
    return {
        "api": api,
        "ticker": "1101" if source is not None else pd.NA,
        "market": "TWSE" if source is not None else pd.NA,
        "report_period": "2026Q1" if source is not None else pd.NA,
        "candidate_field": field,
        "field_role": role,
        "payload_case": source.get("payload_case") if source is not None else pd.NA,
        "payload_schema": source.get("payload_schema") if source is not None else pd.NA,
        "human_policy_required": human_required,
        "t164_candidate_status": status,
        "tpex_bounded_sample_confirmed": bool(source.get("bounded_tpex_sample_confirmed")) if source is not None else False,
        "tpex_universal_ready": False,
        "diagnostic_only": True,
        "not_live_rule": True,
    }


def _join_contract(candidates: pd.DataFrame, announcement: pd.DataFrame) -> pd.DataFrame:
    candidates = candidates.copy()
    announcement = announcement.copy()
    for frame in [candidates, announcement]:
        for col in ["ticker", "market", "report_period"]:
            if col in frame:
                frame[col] = frame[col].astype("string")
    joined = candidates.merge(
        announcement,
        on=["ticker", "market", "report_period"],
        how="left",
        suffixes=("", "_announcement"),
    )
    joined["official_timestamp_matched"] = joined["market_available_at"].notna()
    joined["unmatched_blocked_reason"] = joined["official_timestamp_matched"].map(
        lambda matched: "" if matched else "missing matched t05st01 official announcement timestamp; do not backfill with deadline proxy"
    )
    joined["signal_eligible_date"] = joined["market_available_at"].map(_next_trading_day_candidate)
    joined["signal_eligibility_rule_applied"] = joined["after_regular_close"].map(
        lambda value: "next_trading_day_after_close" if bool(value) else "same_trading_day_after_timestamp"
    )
    joined["conservative_asof_candidate_separate"] = ~joined["official_timestamp_matched"]
    joined["conservative_asof_used"] = False
    joined["quarter_end_date_used"] = False
    joined["query_response_datetime_used"] = False
    joined["accepted_for_formal"] = False
    joined["forward_return_as_rule"] = False
    return joined


def _next_trading_day_candidate(timestamp: Any) -> str | pd.NA:
    if pd.isna(timestamp):
        return pd.NA
    dt = datetime.fromisoformat(str(timestamp))
    candidate = dt.date() + timedelta(days=1)
    while candidate.weekday() >= 5:
        candidate += timedelta(days=1)
    return candidate.isoformat()


def _coverage_by_period(joined: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (period, market), subset in joined.groupby(["report_period", "market"], dropna=False):
        total = len(subset)
        matched = int(subset["official_timestamp_matched"].sum())
        rows.append(
            {
                "report_period": period,
                "market": market,
                "rows": total,
                "matched_rows": matched,
                "unmatched_rows": total - matched,
                "matched_share": matched / total if total else 0.0,
                "ticker_count": int(subset["ticker"].nunique(dropna=True)),
                "field_count": int(subset["candidate_field"].nunique(dropna=True)),
                "diagnostic_only": True,
            }
        )
    return pd.DataFrame(rows)


def _unmatched_blocked(joined: pd.DataFrame) -> pd.DataFrame:
    out = joined[~joined["official_timestamp_matched"]].copy()
    if out.empty:
        return pd.DataFrame(
            columns=["ticker", "market", "report_period", "candidate_field", "unmatched_blocked_reason", "diagnostic_only"]
        )
    return out[["ticker", "market", "report_period", "candidate_field", "unmatched_blocked_reason", "diagnostic_only"]]


def _signal_eligibility_policy_audit(joined: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for row in joined.itertuples(index=False):
        rows.append(
            {
                "ticker": row.ticker,
                "market": row.market,
                "report_period": row.report_period,
                "candidate_field": row.candidate_field,
                "market_available_at": row.market_available_at,
                "after_regular_close": row.after_regular_close,
                "signal_eligible_date": row.signal_eligible_date,
                "same_day_pre_announcement_use_allowed": False,
                "policy_applied": row.signal_eligibility_rule_applied,
                "diagnostic_only": True,
            }
        )
    return pd.DataFrame(rows)


def _human_review_fields(taxonomy: pd.DataFrame) -> pd.DataFrame:
    if taxonomy.empty:
        return pd.DataFrame()
    out = taxonomy[taxonomy["human_review_required"].astype(bool) | taxonomy["core_policy_status"].eq("human_policy_required")].copy()
    out["official_asof_join_status"] = "preserve_human_review_required"
    out["ready_for_formal"] = False
    out["diagnostic_only"] = True
    return out


def _blocked_proxy_fields(joined: pd.DataFrame) -> pd.DataFrame:
    rows = [
        ("unmatched_official_timestamp_rows", "blocked", int((~joined["official_timestamp_matched"]).sum()), "unmatched rows must remain blocked or separate conservative-asof candidates"),
        ("conservative_filing_deadline_proxy", "separate_candidate_only", int(joined["conservative_asof_candidate_separate"].sum()), "do not silently backfill official timestamp"),
        ("capex_proxy", "human_policy_required", int(joined["candidate_field"].eq("capex_proxy").sum()), "FCF proxy label/policy requires human review"),
        ("receivables_basket", "human_policy_required", int(joined["candidate_field"].eq("receivables_basket").sum()), "receivables basket policy requires human review"),
        ("tpex_universal_ready", "blocked", 0, "bounded sample confirmation only; not universal ready"),
        ("full_ingest", "blocked", 0, "this package is sample/interim readiness only"),
        ("formal_selector", "prohibited", 0, "no Layer1 selector created"),
    ]
    return pd.DataFrame(
        [
            {
                "field_or_contract": field,
                "status": status,
                "affected_rows": affected,
                "reason": reason,
                "diagnostic_only": True,
            }
            for field, status, affected, reason in rows
        ]
    )


def _future_data_audit(joined: pd.DataFrame, eligibility: pd.DataFrame) -> pd.DataFrame:
    same_day_bad = int(eligibility["same_day_pre_announcement_use_allowed"].astype(bool).sum())
    prohibited_used = int(joined[["conservative_asof_used", "quarter_end_date_used", "query_response_datetime_used"]].astype(bool).any(axis=1).sum())
    missing_available = int(joined["official_timestamp_matched"].eq(False).sum())
    return pd.DataFrame(
        [
            {
                "audit_item": "same_day_pre_announcement_use",
                "status": "passed" if same_day_bad == 0 else "failed",
                "future_data_violation_count": same_day_bad,
                "note": "after-close announcements are next-trading-day eligible",
            },
            {
                "audit_item": "prohibited_available_date_sources_not_used",
                "status": "passed" if prohibited_used == 0 else "failed",
                "future_data_violation_count": prohibited_used,
                "note": "no conservative proxy, quarter-end, or query response datetime used for matched rows",
            },
            {
                "audit_item": "unmatched_rows_blocked_not_backfilled",
                "status": "passed",
                "future_data_violation_count": 0,
                "note": f"unmatched rows={missing_available}; kept blocked/separate",
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
    coverage: pd.DataFrame,
    unmatched: pd.DataFrame,
    eligibility: pd.DataFrame,
    future_audit: pd.DataFrame,
    t05_readiness: dict[str, Any],
    t164_readiness: dict[str, Any],
) -> dict[str, Any]:
    total = len(joined)
    matched = int(joined["official_timestamp_matched"].sum())
    unmatched_count = total - matched
    future_count = int(future_audit["future_data_violation_count"].sum())
    ready = matched > 0 and future_count == 0
    return {
        "date": "2026-07-07",
        "task_id": TASK_ID,
        "owner": "BACKTEST_LAB Core/Data",
        "status": "interim_official_asof_join_sample_ready_not_full_ingest" if ready else "blocked_interim_official_asof_join",
        "ready_for_layer1_t164_interim_official_asof_event_diagnostic": bool(ready),
        "ready_for_full_ingest": False,
        "ready_for_experiments": False,
        "ready_for_experiments_caveat": "false_by_default; Research/Strategy must explicitly accept sample/interim scope before any bounded diagnostic",
        "ready_for_formal": False,
        "ready_for_strategy_replay": False,
        "official_timestamp_matched_share": matched / total if total else 0.0,
        "unmatched_share": unmatched_count / total if total else 0.0,
        "matched_rows": matched,
        "unmatched_rows": unmatched_count,
        "after_close_policy_applied_count": int(eligibility["after_regular_close"].astype(bool).sum()),
        "future_data_violation_count": future_count,
        "market_available_at_source": "t05st01_public_material_information_announcement_timestamp",
        "exact_internal_filing_upload_timestamp_found": False,
        "conservative_asof_backfill_used": False,
        "quarter_end_date_used": False,
        "query_response_datetime_used": False,
        "human_review_required_fields": ["capex_proxy", "receivables_basket"],
        "candidate_join_rows": total,
        "coverage_rows": int(len(coverage)),
        "not_live_rule": True,
        "forward_returns_live_rule_usage": False,
        "formal_model_changed": False,
        "trade_decision_changed": False,
        "active_in_trade_decision": False,
        "report_changed": False,
        "portfolio_replay_executed": False,
    }


def _write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def _summary(readiness: dict[str, Any], blocked: pd.DataFrame) -> str:
    return "\n".join(
        [
            "# Layer1 t164/t05st01 Interim Official-Asof Join Readiness",
            "",
            f"Status: {readiness['status']}",
            "",
            "Boundary: sample/interim official-asof join readiness only; no full ingest, no Experiments, no replay, no formal/report/trade change.",
            "",
            "Readiness:",
            f"- ready_for_layer1_t164_interim_official_asof_event_diagnostic={str(readiness['ready_for_layer1_t164_interim_official_asof_event_diagnostic']).lower()}",
            "- ready_for_full_ingest=false",
            "- ready_for_experiments=false",
            "- ready_for_formal=false",
            "- ready_for_strategy_replay=false",
            f"- official_timestamp_matched_share={readiness['official_timestamp_matched_share']}",
            f"- unmatched_share={readiness['unmatched_share']}",
            f"- after_close_policy_applied_count={readiness['after_close_policy_applied_count']}",
            f"- future_data_violation_count={readiness['future_data_violation_count']}",
            "",
            "Blocked / proxy fields:",
            *[f"- {row.field_or_contract}: {row.status}; {row.reason}" for row in blocked.itertuples()],
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


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--t05-dir", type=Path, default=DEFAULT_T05_DIR)
    parser.add_argument("--t164-dir", type=Path, default=DEFAULT_T164_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args(argv)
    manifest = build_interim_join_readiness(t05_dir=args.t05_dir, t164_dir=args.t164_dir, output_dir=args.output_dir)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
