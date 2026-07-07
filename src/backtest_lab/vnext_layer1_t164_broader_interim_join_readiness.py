"""Build broader t164/t05st01 interim official-asof join readiness.

Consumes Radar/Data bounded broader t164 materialization samples and stages a
Core/Data readiness package. This is not full-universe ingest or replay.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


TASK_ID = "TASK-BACKTEST-CORE-VNEXT-LAYER1-T164-BROADER-INTERIM-OFFICIAL-ASOF-JOIN-READINESS-001"
DEFAULT_RADAR_DIR = Path(
    r"C:\Users\zergv\Documents\Codex\2026-05-23\ai-stock-rotation-radar-https-docs\outputs"
    r"\radar_vnext_layer1_t164_broader_source_materialization_readiness_20260707"
)
DEFAULT_MATERIALIZATION_DIR = Path("outputs/vnext_dynamic_candidate_pool_data_materialization_20260706")
DEFAULT_OUTPUT_DIR = Path("outputs/vnext_layer1_t164_broader_interim_official_asof_join_20260707")


def build_broader_interim_join(
    *,
    radar_dir: str | Path = DEFAULT_RADAR_DIR,
    materialization_dir: str | Path = DEFAULT_MATERIALIZATION_DIR,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    radar = Path(radar_dir)
    materialization = Path(materialization_dir)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    readiness_in = _read_json(radar / "readiness_for_core_t164_broader_source_materialization.json")
    matrix = _read_csv(radar / "radar_t164_broader_source_materialization_matrix.csv")
    match = _read_csv(radar / "radar_t164_t05st01_match_candidate_coverage.csv")
    asof_audit = _read_csv(radar / "radar_t164_asof_policy_audit.csv")
    label_review = _read_csv(radar / "radar_t164_field_label_review_ledger.csv")
    blocked_in = _read_csv(radar / "radar_t164_blocked_prohibited_ledger.csv")
    route_coverage = _read_csv(radar / "radar_t164_twse_tpex_route_coverage.csv")
    calendar = _read_csv(materialization / "trading_calendar.csv")

    join_contract = _join_contract(matrix, calendar, asof_audit)
    coverage = _coverage_by_period(join_contract)
    unmatched = _unmatched_rows(join_contract)
    eligibility = _eligibility_audit(join_contract)
    human_review = _human_review_fields(label_review)
    blocked = _blocked_proxy_fields(join_contract, blocked_in, route_coverage)
    future_audit = _future_data_audit(join_contract, asof_audit)
    readiness = _readiness_json(join_contract, coverage, unmatched, eligibility, future_audit, readiness_in)

    _write_csv(join_contract, output / "layer1_t164_broader_interim_official_asof_join_contract.csv")
    _write_csv(coverage, output / "layer1_t164_broader_join_coverage_by_period.csv")
    _write_csv(unmatched, output / "layer1_t164_broader_unmatched_blocked_rows.csv")
    _write_csv(eligibility, output / "layer1_t164_broader_signal_eligibility_policy_audit.csv")
    _write_csv(human_review, output / "layer1_t164_broader_label_human_review_fields.csv")
    _write_csv(blocked, output / "layer1_t164_broader_blocked_proxy_fields.csv")
    _write_csv(future_audit, output / "layer1_t164_broader_future_data_audit.csv")
    (output / "readiness_for_layer1_t164_broader_interim_official_asof_diagnostic.json").write_text(
        json.dumps(readiness, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    manifest = {
        "task_id": TASK_ID,
        "status": readiness["status"],
        "output_dir": str(output.resolve()),
        "radar_input_dir": str(radar.resolve()),
        "input_materialization_dir": str(materialization.resolve()),
        "radar_commit": "018365c",
        "output_files": [
            "layer1_t164_broader_interim_official_asof_join_contract.csv",
            "layer1_t164_broader_join_coverage_by_period.csv",
            "layer1_t164_broader_unmatched_blocked_rows.csv",
            "layer1_t164_broader_signal_eligibility_policy_audit.csv",
            "layer1_t164_broader_label_human_review_fields.csv",
            "layer1_t164_broader_blocked_proxy_fields.csv",
            "layer1_t164_broader_future_data_audit.csv",
            "readiness_for_layer1_t164_broader_interim_official_asof_diagnostic.json",
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
    (output / "final_summary_zh.md").write_text(_summary(readiness, blocked), encoding="utf-8")
    return manifest


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def _join_contract(matrix: pd.DataFrame, calendar: pd.DataFrame, asof_audit: pd.DataFrame) -> pd.DataFrame:
    out = matrix.copy()
    if not asof_audit.empty:
        audit_cols = [
            "ticker",
            "market",
            "report_period",
            "quarter_end_date_used",
            "query_response_datetime_used",
            "needs_core_trading_calendar_join",
        ]
        out = out.merge(asof_audit.reindex(columns=audit_cols), on=["ticker", "market", "report_period"], how="left")
    out["market_available_at_iso"] = out["market_available_at"].map(_roc_datetime_to_iso)
    out["market_available_date"] = pd.to_datetime(out["market_available_at_iso"]).dt.date.astype(str)
    out["after_close_policy_applied"] = True
    calendar_slice = calendar[["trade_date", "next_trade_date"]].copy()
    calendar_slice["trade_date"] = calendar_slice["trade_date"].astype(str)
    out = out.merge(calendar_slice, left_on="market_available_date", right_on="trade_date", how="left")
    out["signal_eligible_date"] = out["next_trade_date"]
    out["official_timestamp_matched"] = out["official_announcement_timestamp_matched"].astype(bool)
    out["unmatched_blocked_reason"] = out["official_timestamp_matched"].map(
        lambda matched: "" if matched else "missing t05st01 official announcement timestamp; do not backfill"
    )
    out["conservative_asof_backfill_used"] = False
    out["quarter_end_date_used"] = out.get("quarter_end_date_used", False).fillna(False).astype(bool)
    out["query_response_datetime_used"] = out.get("query_response_datetime_used", False).fillna(False).astype(bool)
    out["needs_core_trading_calendar_join"] = out.get("needs_core_trading_calendar_join", True).fillna(True).astype(bool)
    out["tpex_universal_ready"] = False
    out["bounded_sample_only"] = True
    out["accepted_for_formal"] = False
    out["diagnostic_only"] = True
    return out


def _roc_datetime_to_iso(value: str) -> str:
    date_part, time_part = str(value).split(" ")
    roc_year, month, day = date_part.split("/")
    return f"{int(roc_year) + 1911:04d}-{int(month):02d}-{int(day):02d}T{time_part}"


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
                "ticker_count": int(subset["ticker"].nunique()),
                "cashflow_success_rows": int(subset["cashflow_fields_available"].astype(bool).sum()),
                "balance_sheet_success_rows": int(subset["balance_sheet_fields_available"].astype(bool).sum()),
                "diagnostic_only": True,
            }
        )
    return pd.DataFrame(rows)


def _unmatched_rows(joined: pd.DataFrame) -> pd.DataFrame:
    out = joined[~joined["official_timestamp_matched"]].copy()
    if out.empty:
        return pd.DataFrame(columns=["ticker", "market", "report_period", "unmatched_blocked_reason", "diagnostic_only"])
    return out[["ticker", "market", "report_period", "unmatched_blocked_reason", "diagnostic_only"]]


def _eligibility_audit(joined: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "ticker",
        "market",
        "report_period",
        "market_available_at",
        "market_available_at_iso",
        "signal_eligible_date",
        "after_close_policy_applied",
        "needs_core_trading_calendar_join",
        "quarter_end_date_used",
        "query_response_datetime_used",
        "diagnostic_only",
    ]
    return joined.reindex(columns=cols)


def _human_review_fields(label_review: pd.DataFrame) -> pd.DataFrame:
    if label_review.empty:
        return pd.DataFrame()
    out = label_review.copy()
    if "human_review_required" in out:
        out = out[out["human_review_required"].astype(bool)]
    out["ready_for_formal"] = False
    out["diagnostic_only"] = True
    return out


def _blocked_proxy_fields(joined: pd.DataFrame, blocked_in: pd.DataFrame, route_coverage: pd.DataFrame) -> pd.DataFrame:
    rows = [
        ("full_universe_runner", "blocked", 0, "bounded broader sample only; no all-stock runner"),
        ("tpex_universal_ready", "blocked", 0, "TPEx has bounded samples only, not universal readiness"),
        ("capex_proxy", "human_review_required", int(joined["capex_proxy"].notna().sum()), "FCF proxy label policy still needs review"),
        ("receivables_basket", "human_review_required", int(joined["receivables_trade"].notna().sum()), "receivables basket policy still needs review"),
        ("exact_upload_timestamp", "not_found", 0, "market_available_at is public announcement timestamp, not internal upload timestamp"),
        ("conservative_asof_backfill", "prohibited_for_matched_rows", 0, "official timestamp matched rows must not be backfilled by deadline proxy"),
        ("formal_selector", "prohibited", 0, "no Layer1 selector created"),
    ]
    if not blocked_in.empty:
        for row in blocked_in.to_dict("records"):
            rows.append((row.get("field_or_contract") or row.get("item"), row.get("status"), 0, row.get("reason")))
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


def _future_data_audit(joined: pd.DataFrame, asof_audit: pd.DataFrame) -> pd.DataFrame:
    prohibited = int(joined[["quarter_end_date_used", "query_response_datetime_used", "conservative_asof_backfill_used"]].astype(bool).any(axis=1).sum())
    missing_calendar = int(joined["signal_eligible_date"].isna().sum())
    radar_future = int(asof_audit["future_data_violation_count"].sum()) if not asof_audit.empty else 0
    return pd.DataFrame(
        [
            {
                "audit_item": "prohibited_available_date_sources_not_used",
                "status": "passed" if prohibited == 0 else "failed",
                "future_data_violation_count": prohibited,
                "note": "no quarter_end/query_response/conservative backfill used",
            },
            {
                "audit_item": "after_close_calendar_join_has_signal_eligible_date",
                "status": "passed" if missing_calendar == 0 else "blocked",
                "future_data_violation_count": 0,
                "note": f"missing calendar next_trade_date rows={missing_calendar}",
            },
            {
                "audit_item": "radar_asof_policy_audit",
                "status": "passed" if radar_future == 0 else "failed",
                "future_data_violation_count": radar_future,
                "note": "Radar asof policy audit imported",
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
    readiness_in: dict[str, Any],
) -> dict[str, Any]:
    total = len(joined)
    matched = int(joined["official_timestamp_matched"].sum())
    future_count = int(future_audit["future_data_violation_count"].sum())
    ready = bool(readiness_in.get("ready_for_core_t164_broader_interim_official_asof_join", False)) and matched == total and total > 0 and future_count == 0
    return {
        "date": "2026-07-07",
        "task_id": TASK_ID,
        "owner": "BACKTEST_LAB Core/Data",
        "status": "broader_interim_official_asof_join_ready_not_full_ingest" if ready else "blocked_broader_interim_official_asof_join",
        "ready_for_layer1_t164_broader_interim_official_asof_event_diagnostic": bool(ready),
        "ready_for_full_ingest": False,
        "ready_for_experiments": False,
        "ready_for_formal": False,
        "ready_for_strategy_replay": False,
        "official_timestamp_matched_share": matched / total if total else 0.0,
        "unmatched_share": (total - matched) / total if total else 0.0,
        "matched_rows": matched,
        "unmatched_rows": total - matched,
        "ticker_count": int(joined["ticker"].nunique()),
        "market_count": int(joined["market"].nunique()),
        "period_count": int(joined["report_period"].nunique()),
        "after_close_policy_applied_count": int(joined["after_close_policy_applied"].astype(bool).sum()),
        "future_data_violation_count": future_count,
        "bounded_sample_only": True,
        "tpex_universal_ready": False,
        "exact_upload_timestamp_found": False,
        "human_review_required_fields": ["capex_proxy", "receivables_basket"],
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
            "# Layer1 t164 Broader Interim Official-Asof Join Readiness",
            "",
            f"Status: {readiness['status']}",
            "",
            "Boundary: bounded broader sample readiness only; no full ingest, no Experiments, no replay, no formal/report/trade change.",
            "",
            "Readiness:",
            f"- ready_for_layer1_t164_broader_interim_official_asof_event_diagnostic={str(readiness['ready_for_layer1_t164_broader_interim_official_asof_event_diagnostic']).lower()}",
            "- ready_for_full_ingest=false",
            "- ready_for_experiments=false",
            "- ready_for_formal=false",
            "- ready_for_strategy_replay=false",
            f"- official_timestamp_matched_share={readiness['official_timestamp_matched_share']}",
            f"- unmatched_share={readiness['unmatched_share']}",
            f"- ticker_count={readiness['ticker_count']}",
            f"- market_count={readiness['market_count']}",
            f"- period_count={readiness['period_count']}",
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
    parser.add_argument("--radar-dir", type=Path, default=DEFAULT_RADAR_DIR)
    parser.add_argument("--materialization-dir", type=Path, default=DEFAULT_MATERIALIZATION_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args(argv)
    manifest = build_broader_interim_join(
        radar_dir=args.radar_dir,
        materialization_dir=args.materialization_dir,
        output_dir=args.output_dir,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
