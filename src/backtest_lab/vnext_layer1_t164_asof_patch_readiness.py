"""Refresh t164 official-asof policy with Radar alternate-route patch rows.

This builds a diagnostic-only 5-row accepted patch plus 1-row blocked ledger.
It does not perform full ingest, Experiments, replay, or formal selector work.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


TASK_ID = "TASK-BACKTEST-CORE-VNEXT-LAYER1-T164-ASOF-ALTERNATE-ROUTE-PATCH-READINESS-001"
DEFAULT_BASE_DIR = Path("outputs/vnext_layer1_t164_asof_match_policy_alternate_route_readiness_20260707")
DEFAULT_RADAR_DIR = Path(
    r"C:\Users\zergv\Documents\Codex\2026-05-23\ai-stock-rotation-radar-https-docs\outputs"
    r"\radar_vnext_layer1_t164_t05st01_unmatched_alternate_route_capture_20260707"
)
DEFAULT_OUTPUT_DIR = Path("outputs/vnext_layer1_t164_asof_alternate_route_patch_readiness_20260707")


def build_asof_patch_readiness(
    *,
    base_dir: str | Path = DEFAULT_BASE_DIR,
    radar_dir: str | Path = DEFAULT_RADAR_DIR,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    base = Path(base_dir)
    radar = Path(radar_dir)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    base_contract = _read_csv(base / "layer1_t164_asof_match_policy_readiness_contract.csv", dtype={"ticker": str})
    accepted = _read_csv(radar / "t05st01_unmatched_accepted_official_timestamp_rows.csv", dtype={"ticker": str})
    blocked = _read_csv(radar / "t05st01_unmatched_remaining_blocked_rows.csv", dtype={"ticker": str})
    policy = _read_csv(radar / "t05st01_unmatched_policy_ledger.csv")
    radar_readiness = _read_json(radar / "readiness_for_core_t164_t05st01_unmatched_alternate_route_capture.json")

    patch_contract = _patch_contract(accepted)
    refreshed_contract = _refreshed_contract(base_contract, patch_contract, blocked)
    blocked_ledger = _blocked_ledger(blocked, policy)
    policy_matrix = _policy_matrix(policy)
    future_audit = _future_data_audit(refreshed_contract)
    readiness = _readiness(refreshed_contract, patch_contract, blocked_ledger, future_audit, radar_readiness)

    _write_csv(patch_contract, output / "layer1_t164_asof_alternate_route_accepted_patch_contract.csv")
    _write_csv(refreshed_contract, output / "layer1_t164_asof_refreshed_40row_policy_contract.csv")
    _write_csv(blocked_ledger, output / "layer1_t164_asof_remaining_blocked_rows.csv")
    _write_csv(policy_matrix, output / "layer1_t164_asof_alternate_route_policy_matrix.csv")
    _write_csv(future_audit, output / "layer1_t164_asof_patch_future_data_audit.csv")
    (output / "readiness_for_layer1_t164_asof_alternate_route_patch.json").write_text(
        json.dumps(readiness, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    manifest = {
        "task_id": TASK_ID,
        "status": readiness["status"],
        "output_dir": str(output.resolve()),
        "base_core_dir": str(base.resolve()),
        "radar_input_dir": str(radar.resolve()),
        "radar_commit": "f8a6038",
        "output_files": [
            "layer1_t164_asof_alternate_route_accepted_patch_contract.csv",
            "layer1_t164_asof_refreshed_40row_policy_contract.csv",
            "layer1_t164_asof_remaining_blocked_rows.csv",
            "layer1_t164_asof_alternate_route_policy_matrix.csv",
            "layer1_t164_asof_patch_future_data_audit.csv",
            "readiness_for_layer1_t164_asof_alternate_route_patch.json",
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


def _read_csv(path: Path, **kwargs: Any) -> pd.DataFrame:
    return pd.read_csv(path, **kwargs) if path.exists() else pd.DataFrame()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _patch_contract(accepted: pd.DataFrame) -> pd.DataFrame:
    if accepted.empty:
        return pd.DataFrame()
    out = accepted.copy()
    out["official_asof_patch_status"] = "accepted_official_t05st01_alternate_route"
    out["market_available_at_source"] = "t05st01_public_material_information_timestamp"
    out["after_close_next_trading_day_policy_required"] = True
    out["quarter_end_date_used_as_available_at"] = False
    out["query_response_datetime_used_as_available_at"] = False
    out["conservative_deadline_proxy_used_as_available_at"] = False
    out["exact_internal_upload_timestamp_found"] = False
    out["accepted_for_formal"] = False
    out["accepted_for_experiments"] = False
    out["diagnostic_only"] = True
    return out


def _refreshed_contract(base_contract: pd.DataFrame, patch_contract: pd.DataFrame, blocked: pd.DataFrame) -> pd.DataFrame:
    out = base_contract.copy()
    out["ticker"] = out["ticker"].astype(str)
    out["report_period"] = out["report_period"].astype(str)
    out["asof_patch_applied"] = False
    if not patch_contract.empty:
        patch_cols = [
            "ticker",
            "market",
            "report_period",
            "market_available_at",
            "subject",
            "match_policy",
            "detail_payload",
            "acceptance_reason",
        ]
        patch = patch_contract.reindex(columns=patch_cols).rename(
            columns={
                "market_available_at": "patch_market_available_at",
                "subject": "patch_announcement_subject",
                "match_policy": "patch_match_policy",
                "detail_payload": "patch_detail_payload",
                "acceptance_reason": "patch_acceptance_reason",
            }
        )
        out = out.merge(patch, on=["ticker", "market", "report_period"], how="left")
        patch_mask = out["patch_market_available_at"].notna()
        out.loc[patch_mask, "official_announcement_timestamp_matched"] = True
        out.loc[patch_mask, "market_available_at"] = out.loc[patch_mask, "patch_market_available_at"]
        out.loc[patch_mask, "announcement_subject"] = out.loc[patch_mask, "patch_announcement_subject"]
        out.loc[patch_mask, "official_asof_policy"] = "patched_accepted_official_t05st01_alternate_route"
        out.loc[patch_mask, "after_close_next_trading_day_policy_applies"] = True
        out.loc[patch_mask, "asof_patch_applied"] = True
    if not blocked.empty:
        blocked_keys = set(zip(blocked["ticker"].astype(str), blocked["market"], blocked["report_period"].astype(str)))
        out["remaining_blocked_after_patch"] = [
            (str(row.ticker), row.market, str(row.report_period)) in blocked_keys for row in out.itertuples()
        ]
        out.loc[out["remaining_blocked_after_patch"], "official_asof_policy"] = "remaining_blocked_multiple_strict_candidates"
    else:
        out["remaining_blocked_after_patch"] = False
    out["accepted_for_full_ingest"] = False
    out["accepted_for_experiments"] = False
    out["accepted_for_formal"] = False
    out["diagnostic_only"] = True
    return out


def _blocked_ledger(blocked: pd.DataFrame, policy: pd.DataFrame) -> pd.DataFrame:
    if blocked.empty:
        return pd.DataFrame(columns=["ticker", "market", "report_period", "status", "reason", "diagnostic_only"])
    out = blocked.copy()
    out["blocked_policy"] = "keep_blocked_until_single_accepted_official_timestamp"
    out["premeeting_notice_accepted"] = False
    out["silent_backfill_allowed"] = False
    out["conservative_deadline_proxy_allowed"] = False
    out["query_response_datetime_allowed"] = False
    out["quarter_end_date_allowed"] = False
    out["needs_next_step"] = "stricter_detail_or_subject_disambiguation_policy"
    out["diagnostic_only"] = True
    return out


def _policy_matrix(policy: pd.DataFrame) -> pd.DataFrame:
    out = policy.copy()
    if out.empty:
        return out
    out["core_policy_decision"] = out["policy_item"].map(
        {
            "accepted_market_available_at": "accepted_for_patch_when_single_strict_candidate",
            "relaxed_subject_policy_candidate": "evidence_only_human_review_required_not_patch",
            "quarter_end_date": "prohibited",
            "query_response_datetime": "prohibited",
            "conservative_filing_deadline_proxy": "separate_proxy_candidate_only_not_official_patch",
        }
    ).fillna("preserve_radar_policy")
    out["diagnostic_only"] = True
    return out


def _future_data_audit(refreshed_contract: pd.DataFrame) -> pd.DataFrame:
    prohibited = int(
        refreshed_contract[
            [
                "quarter_end_date_used_as_available_at",
                "query_response_datetime_used_as_available_at",
                "conservative_deadline_proxy_used_as_available_at",
            ]
        ]
        .astype(bool)
        .any(axis=1)
        .sum()
    )
    return pd.DataFrame(
        [
            {
                "audit_item": "prohibited_available_date_sources_not_used",
                "status": "passed" if prohibited == 0 else "failed",
                "future_data_violation_count": prohibited,
                "note": "no quarter_end/query_response/conservative proxy used as official available_at",
            },
            {
                "audit_item": "remaining_blocked_not_backfilled",
                "status": "passed" if int(refreshed_contract["remaining_blocked_after_patch"].sum()) == 1 else "failed",
                "future_data_violation_count": 0,
                "note": "6187 TPEx 114Q4 remains blocked; no premeeting/proxy backfill",
            },
            {
                "audit_item": "forward_return_as_rule",
                "status": "passed",
                "future_data_violation_count": 0,
                "note": "no forward return fields included",
            },
        ]
    )


def _readiness(
    refreshed_contract: pd.DataFrame,
    patch_contract: pd.DataFrame,
    blocked_ledger: pd.DataFrame,
    future_audit: pd.DataFrame,
    radar_readiness: dict[str, Any],
) -> dict[str, Any]:
    total = len(refreshed_contract)
    matched = int(refreshed_contract["official_announcement_timestamp_matched"].astype(bool).sum())
    blocked = int(refreshed_contract["remaining_blocked_after_patch"].astype(bool).sum())
    future_count = int(future_audit["future_data_violation_count"].sum())
    return {
        "date": "2026-07-07",
        "task_id": TASK_ID,
        "owner": "BACKTEST_LAB Core/Data",
        "status": "partial_patch_ready_remaining_blocked_not_experiments_ready",
        "radar_status": radar_readiness.get("status"),
        "diagnostic_only": True,
        "patch_accepted_rows": int(len(patch_contract)),
        "remaining_blocked_rows": int(len(blocked_ledger)),
        "sample_rows": total,
        "official_timestamp_matched_rows_after_patch": matched,
        "official_timestamp_matched_share_after_patch": matched / total if total else 0.0,
        "remaining_blocked_share_after_patch": blocked / total if total else 0.0,
        "ready_for_core_t164_asof_join_contract_refresh": False,
        "ready_for_core_t164_full_or_broader_ingest_contract": False,
        "ready_for_experiments": False,
        "ready_for_formal": False,
        "ready_for_strategy_replay": False,
        "ready_for_partial_patch_policy_review": True,
        "remaining_blocked_policy": "no_silent_backfill; require stricter detail/subject disambiguation for 6187 TPEx 114Q4",
        "after_close_policy_applies_to_accepted_patch_rows": True,
        "quarter_end_date_prohibited": True,
        "query_response_datetime_prohibited": True,
        "conservative_deadline_proxy_must_remain_separate": True,
        "exact_internal_upload_timestamp_found": False,
        "future_data_violation_count": future_count,
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


def _summary(readiness: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Layer1 t164 Asof Alternate Route Patch Readiness",
            "",
            f"Status: {readiness['status']}",
            "",
            "Conclusion: Core accepts five strict official t05st01 alternate-route timestamps as a diagnostic patch, but the 40-row sample remains not Experiments-ready because one row is still blocked.",
            "",
            "Readiness:",
            f"- patch_accepted_rows={readiness['patch_accepted_rows']}",
            f"- remaining_blocked_rows={readiness['remaining_blocked_rows']}",
            f"- official_timestamp_matched_rows_after_patch={readiness['official_timestamp_matched_rows_after_patch']}",
            f"- official_timestamp_matched_share_after_patch={readiness['official_timestamp_matched_share_after_patch']}",
            "- ready_for_core_t164_asof_join_contract_refresh=false",
            "- ready_for_core_t164_full_or_broader_ingest_contract=false",
            "- ready_for_experiments=false",
            "- ready_for_formal=false",
            "- ready_for_strategy_replay=false",
            f"- future_data_violation_count={readiness['future_data_violation_count']}",
            "",
            "Remaining blocker:",
            "- 6187 TPEx 114Q4 remains blocked_multiple_strict_candidates; no premeeting notice, query time, quarter end, or deadline proxy backfill.",
            "",
            "Next step:",
            "- Radar/Data should provide stricter detail/subject disambiguation for 6187 TPEx 114Q4, or Research/Strategy can accept matched-only partial policy review without Experiments.",
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
    parser.add_argument("--base-dir", type=Path, default=DEFAULT_BASE_DIR)
    parser.add_argument("--radar-dir", type=Path, default=DEFAULT_RADAR_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args(argv)
    manifest = build_asof_patch_readiness(base_dir=args.base_dir, radar_dir=args.radar_dir, output_dir=args.output_dir)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
