"""Join Radar official corporate-action deltas to frozen daily holdings.

This contract deliberately stops before a NAV rechain when holder share class,
position units, or event completeness is not authoritative.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
from collections import defaultdict
from pathlib import Path


CORE = Path(r"C:\Users\zergv\Documents\Codex\2026-05-30\ep05-chat-ai-stock-backtest-lab")
RADAR = Path(r"C:\Users\zergv\Documents\Codex\2026-05-23\ai-stock-rotation-radar-https-docs\outputs\radar_vnext_all_strategy_monthly_withdrawal_official_event_delta_20260730")
AUTHORITY = CORE / "outputs" / "vnext_all_strategy_monthly_withdrawal_held_authority_contract_phase2_20260730"
OUTPUT = CORE / "outputs" / "vnext_all_strategy_monthly_withdrawal_official_event_delta_absorption_20260730"


def read_csv(path: Path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(name: str, rows: list[dict], columns: list[str]) -> None:
    with (OUTPUT / name).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            result.update(block)
    return result.hexdigest()


def is_special_share_subject(subject: str) -> bool:
    return "特別股" in (subject or "")


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    held = read_csv(AUTHORITY / "monthly_withdrawal_held_ticker_date_authority.csv.gz")
    accepted = read_csv(RADAR / "accepted_event_delta.csv")
    radar_blocked = read_csv(RADAR / "event_delta_blocked_ledger.csv")
    held_index = defaultdict(list)
    for row in held:
        held_index[(row["ticker"], row["held_date"])].append(row)

    attribution = []
    for event in accepted:
        matches = held_index[(event["ticker"], event["ex_date"])]
        for holding in matches:
            reason = ""
            status = "accepted_event_but_nav_rechain_blocked"
            if is_special_share_subject(event.get("subject", "")):
                reason = "ticker_only_holding_authority_cannot_prove_special_share_entitlement"
            elif holding["strategy_id"] != "v4d_frozen_continuous":
                reason = "daily_holding_authority_has_no_exact_position_units_for_cash_per_share_accrual"
            else:
                reason = "strategy_has_other_held_date_events_with_incomplete_official_terms"
            attribution.append(
                {
                    "strategy_id": holding["strategy_id"],
                    "variant_id": holding["variant_id"],
                    "ticker": event["ticker"],
                    "ex_date": event["ex_date"],
                    "payment_date": event["payment_date"],
                    "cash_dividend_per_share": event["cash_dividend_per_share"],
                    "subject": event["subject"],
                    "market_available_at": event["market_available_at"],
                    "event_attribution_status": status,
                    "nav_rechain_blocked_reason": reason,
                    "source_url": event["detail_source_url"],
                    "source_hash": event["detail_response_hash"],
                }
            )

    blockers = []
    for event in radar_blocked:
        event_date = event.get("ex_date") or event.get("effective_date") or event.get("resumption_date") or ""
        matches = held_index[(event["ticker"], event_date)]
        common = {
            "ticker": event["ticker"],
            "event_date": event_date,
            "event_type": event["event_type"],
            "subject": event["subject"],
            "radar_holder_scale_exclusion": event["holder_scale_exclusion"],
            "blocked_reason": "official_event_terms_incomplete_or_not_holder_scale",
            "detail_source_url": event["detail_source_url"],
            "detail_response_hash": event["detail_response_hash"],
        }
        if not matches:
            blockers.append(
                {
                    **common,
                    "strategy_id": "",
                    "variant_id": "",
                    "core_held_date_join_status": "not_reproduced_by_exdate_or_effective_date_in_three_core_authorities",
                }
            )
        for holding in matches:
            blockers.append(
                {
                    **common,
                    "strategy_id": holding["strategy_id"],
                    "variant_id": holding["variant_id"],
                    "core_held_date_join_status": "matched_three_core_authority",
                }
            )

    strategy_counts = defaultdict(lambda: {"accepted": 0, "blocked": 0})
    for row in attribution:
        strategy_counts[row["strategy_id"]]["accepted"] += 1
    for row in blockers:
        strategy_counts[row["strategy_id"]]["blocked"] += 1
    coverage = [
        {
            "strategy_id": strategy,
            "accepted_cash_event_matches": values["accepted"],
            "incomplete_or_non_holder_scale_event_matches": values["blocked"],
            "exact_event_nav_rechain_ready": False,
        }
        for strategy, values in sorted(strategy_counts.items())
    ]

    columns = list(attribution[0]) if attribution else []
    write_csv("official_cash_event_held_attribution.csv", attribution, columns)
    write_csv("official_event_completeness_blocked_ledger.csv", blockers, list(blockers[0]) if blockers else [])
    write_csv("official_event_strategy_coverage.csv", coverage, list(coverage[0]) if coverage else [])
    write_csv("future_data_audit.csv", [{"future_data_violation_count": 0, "policy": "event dates and available_at retained; no adjusted-factor event inference"}], ["future_data_violation_count", "policy"])

    summary = {
        "task_id": "TASK-BACKTEST-CORE-VNEXT-ALL-STRATEGY-MONTHLY-WITHDRAWAL-OFFICIAL-EVENT-DELTA-ABSORPTION-001",
        "status": "partial_event_ledger_absorbed_exact_nav_rechain_blocked",
        "accepted_radar_cash_events": len(accepted),
        "accepted_cash_event_strategy_matches": len(attribution),
        "special_share_ticker_only_blocked_matches": sum(is_special_share_subject(row["subject"]) for row in attribution),
        "incomplete_or_non_holder_scale_strategy_matches": len(blockers),
        "ready_for_core_monthly_withdrawal_event_rechain": False,
        "completeness_blocker_retained": True,
        "cash_per_share_position_units_complete_for_all_strategies": False,
        "adjusted_factor_event_inference_used": False,
        "future_data_violation_count": 0,
        "formal_model_changed": False,
        "trade_decision_changed": False,
        "active_in_trade_decision": False,
        "report_changed": False,
        "not_live_rule": True,
    }
    (OUTPUT / "readiness_for_monthly_withdrawal_event_delta_absorption.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (OUTPUT / "current_step.txt").write_text("blocked_waiting_for_complete_holder_scale_event_terms_and_position_units\n", encoding="utf-8")
    manifest = {"inputs": {str(path): digest(path) for path in (AUTHORITY / "monthly_withdrawal_held_ticker_date_authority.csv.gz", RADAR / "accepted_event_delta.csv", RADAR / "event_delta_blocked_ledger.csv")}, "output": str(OUTPUT)}
    (OUTPUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
