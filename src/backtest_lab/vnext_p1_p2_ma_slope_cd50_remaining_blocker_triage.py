from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "outputs/vnext_p1_p2_primary80_MA_slope_CD50_official_raw_close_diagnostic_20260716"
OUT = ROOT / "outputs/vnext_p1_p2_MA_slope_CD50_remaining_blocker_triage_20260716"
RADAR_OUTPUTS = Path(r"C:\Users\zergv\Documents\Codex\2026-05-23\ai-stock-rotation-radar-https-docs\outputs")
PATH_INDEPENDENT = RADAR_OUTPUTS / "radar_vnext_p1_p2_primary80_path_independent_raw_close_bulk_fill_20260716"
PATH_BLOCKED = PATH_INDEPENDENT / "path_independent_final_blocked.csv.gz"
PATH_NO_TRADE = PATH_INDEPENDENT / "path_independent_final_official_no_trade_termination.csv.gz"
EVENT_DIR = RADAR_OUTPUTS / "radar_vnext_p1_full_lifecycle_minimum_data_acquisition_20260710/compact/trusted_corporate_action_events"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read(path: Path, **kwargs: object) -> pd.DataFrame:
    return pd.read_csv(path, dtype={"ticker": str}, **kwargs)


def _explode_periods(source: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for row in source.itertuples(index=False):
        data = row._asdict()
        for period in str(data.get("periods", "")).split("|"):
            item = dict(data)
            item["period"] = period
            rows.append(item)
    frame = pd.DataFrame(rows)
    if len(frame):
        frame["ticker"] = frame.ticker.astype(str).str.zfill(4)
        frame["date"] = pd.to_datetime(frame.date)
    return frame


def _load_event_authority() -> pd.DataFrame:
    files = sorted(EVENT_DIR.glob("*.csv.gz"))
    if not files:
        return pd.DataFrame(columns=["ticker", "event_date", "event_type", "source_quality", "human_review_required"])
    events = pd.concat([_read(path) for path in files], ignore_index=True)
    events["ticker"] = events.ticker.astype(str).str.zfill(4)
    events["event_date"] = pd.to_datetime(events.event_date)
    events["accepted_explicit_event_authority"] = events.source_quality.astype(str).ne("") & events.event_type.astype(str).ne("")
    return events.drop_duplicates(["ticker", "event_date", "event_type"])


def _next_raw_date(raw_dates: pd.DataFrame, period: str, ticker: str, requested: object) -> pd.Timestamp | pd.NaT:
    if pd.isna(requested):
        return pd.NaT
    requested_ts = pd.Timestamp(requested)
    dates = raw_dates.loc[(raw_dates.period.eq(period)) & (raw_dates.ticker.eq(ticker)) & (raw_dates.date.gt(requested_ts)), "date"]
    return dates.min() if len(dates) else pd.NaT


def run() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    readiness = json.loads((INPUT / "readiness_for_raw_close_diagnostic.json").read_text(encoding="utf-8"))
    blocked = _read(INPUT / "raw_close_execution_blocked_ledger.csv")
    blocked["ticker"] = blocked.ticker.astype(str).str.zfill(4)
    blocked["decision_date"] = pd.to_datetime(blocked.decision_date)
    blocked["requested_execution_date"] = pd.to_datetime(blocked.requested_execution_date)
    variant = _read(INPUT / "raw_close_CD50_per_variant_readiness.csv")
    no_obs = _read(INPUT / "raw_close_incumbent_no_observation_ledger.csv.gz")
    no_obs["incumbent"] = no_obs.incumbent.astype(str).str.zfill(4)
    no_obs["decision_date"] = pd.to_datetime(no_obs.decision_date)
    guard = _read(INPUT / "raw_close_corporate_action_guard_held_path_ledger.csv.gz")
    guard["ticker"] = guard.ticker.astype(str).str.zfill(4)
    guard["date"] = pd.to_datetime(guard.date)

    per_variant_class = (
        blocked.groupby(["variant_id", "reason", "precise_source_class"], dropna=False)
        .size()
        .reset_index(name="blocked_rows")
        .sort_values(["variant_id", "reason", "precise_source_class"])
    )
    per_variant = variant.merge(
        per_variant_class.groupby("variant_id", as_index=False).blocked_rows.sum().rename(columns={"blocked_rows": "execution_blocked_rows_recomputed"}),
        on="variant_id",
        how="left",
    )
    per_variant["any_variant_ready_without_contract_relaxation"] = False
    per_variant["minimal_blocking_reason"] = per_variant.apply(
        lambda r: "close_path_blocked;corporate_action_guard_incomplete"
        if not bool(r.close_path_ready)
        else "corporate_action_guard_incomplete",
        axis=1,
    )
    per_variant.to_csv(OUT / "per_variant_minimal_blocker_summary.csv", index=False, encoding="utf-8-sig")
    per_variant_class.to_csv(OUT / "per_variant_blocker_class_counts.csv", index=False, encoding="utf-8-sig")

    conflict = _explode_periods(_read(PATH_BLOCKED, low_memory=False))
    conflict_cols = [
        "period",
        "ticker",
        "date",
        "name",
        "market",
        "market_required",
        "close",
        "market_source",
        "source_quality",
        "source_component",
        "source_url",
        "source_hash",
        "classification",
        "classification_reason",
    ]
    conflict = conflict[[c for c in conflict_cols if c in conflict.columns]].drop_duplicates(["period", "ticker", "date"])
    local_conflict = blocked.loc[
        blocked.reason.eq("missing_official_raw_execution_close") & blocked.precise_source_class.eq("local_source_conflict")
    ].merge(
        conflict,
        left_on=["period", "ticker", "requested_execution_date"],
        right_on=["period", "ticker", "date"],
        how="left",
    )
    local_conflict["triage_class"] = "local_source_conflict_requires_authority_reconciliation_or_strategy_center_bounded_exact_source_authorization"
    local_conflict["source_download_authority"] = False
    local_conflict.to_csv(OUT / "local_source_conflict_40_rows_triage.csv", index=False, encoding="utf-8-sig")

    raw_dates = _read(INPUT / "raw_close_MA_slope_feature_guard_compact.csv.gz", usecols=["period", "ticker", "date"])
    raw_dates["ticker"] = raw_dates.ticker.astype(str).str.zfill(4)
    raw_dates["date"] = pd.to_datetime(raw_dates.date)
    raw_dates = raw_dates.drop_duplicates(["period", "ticker", "date"]).sort_values(["period", "ticker", "date"])
    no_trade_source = _explode_periods(_read(PATH_NO_TRADE, low_memory=False))
    no_trade_source = no_trade_source[
        ["period", "ticker", "date", "market", "classification", "classification_reason", "source_url", "source_hash"]
    ].drop_duplicates(["period", "ticker", "date"])
    atomic = blocked.loc[blocked.reason.eq("official_no_trade_atomic_transition_not_executed")].copy()
    atomic["next_available_ticker_trading_date"] = [
        _next_raw_date(raw_dates, row.period, row.ticker, row.requested_execution_date)
        for row in atomic.itertuples(index=False)
    ]
    atomic = atomic.merge(
        no_trade_source,
        left_on=["period", "ticker", "requested_execution_date"],
        right_on=["period", "ticker", "date"],
        how="left",
        suffixes=("", "_no_trade"),
    )
    atomic["policy_requires_strategy_center_choice"] = True
    atomic["mechanical_policy_choices"] = (
        "A_defer_atomic_switch_until_all_legs_same_tradable_date|"
        "B_cancel_transition_and_rejudge_next_decision|"
        "C_hold_pending_existing_incumbent_until_exit_leg_tradable|"
        "D_allow_partial_execution_forbidden_without_new_authorization"
    )
    atomic.to_csv(OUT / "atomic_no_trade_70_rows_policy_triage.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(
        [
            {
                "choice_id": "A",
                "policy_choice": "defer_atomic_switch_until_all_legs_same_tradable_date",
                "effect": "changes execution date and path; requires Strategy Center approval",
                "core_selected": False,
            },
            {
                "choice_id": "B",
                "policy_choice": "cancel_transition_and_rejudge_next_decision",
                "effect": "keeps next-day-only semantics but may miss intended transition",
                "core_selected": False,
            },
            {
                "choice_id": "C",
                "policy_choice": "hold_pending_existing_incumbent_until_exit_leg_tradable",
                "effect": "adds pending state; changes path semantics",
                "core_selected": False,
            },
            {
                "choice_id": "D",
                "policy_choice": "allow_partial_execution",
                "effect": "explicitly forbidden unless newly authorized",
                "core_selected": False,
            },
        ]
    ).to_csv(OUT / "atomic_no_trade_policy_choices_for_strategy_center.csv", index=False, encoding="utf-8-sig")

    events = _load_event_authority()
    event_hits = guard.merge(
        events,
        left_on=["ticker", "date"],
        right_on=["ticker", "event_date"],
        how="left",
    )
    accepted_mask = event_hits.accepted_explicit_event_authority.fillna(False).astype(bool)
    accepted_hits = event_hits.loc[accepted_mask].copy()
    missing_or_legacy = event_hits.loc[~accepted_mask].copy()
    accepted_hits.to_csv(OUT / "corporate_action_explicit_event_hits.csv", index=False, encoding="utf-8-sig")
    missing_or_legacy.to_csv(OUT / "corporate_action_guard_legacy_warning_or_missing_authority.csv", index=False, encoding="utf-8-sig")
    ranges = (
        missing_or_legacy.groupby(["variant_id", "period", "ticker"], dropna=False)
        .agg(
            first_guard_date=("date", "min"),
            last_guard_date=("date", "max"),
            guard_rows=("date", "size"),
            explicit_event_rows=("explicit_corporate_action_event", "sum"),
            large_raw_close_move_warning_rows=("large_raw_close_move_warning", "sum"),
        )
        .reset_index()
    )
    ranges["triage_class"] = "legacy_warning_or_missing_accepted_event_authority_not_strategy_verdict"
    ranges.to_csv(OUT / "corporate_action_guard_missing_authority_ranges.csv", index=False, encoding="utf-8-sig")

    summary = {
        "task": "TASK-BACKTEST-CORE-VNEXT-P1-P2-MA-SLOPE-CD50-REMAINING-BLOCKER-TRIAGE-001",
        "input_output": str(INPUT),
        "ready_variant_count": int(readiness["ready_variant_count"]),
        "any_variant_ready_without_relaxing_data_or_event_contract": False,
        "execution_blocked_rows": int(len(blocked)),
        "execution_hard_source_blocked_rows": int(readiness["execution_hard_source_blocked_rows"]),
        "local_source_conflict_rows": int(len(local_conflict)),
        "atomic_no_trade_rows": int(len(atomic)),
        "corporate_action_guard_held_rows": int(len(guard)),
        "corporate_action_explicit_event_hit_rows": int(len(accepted_hits)),
        "corporate_action_legacy_warning_or_missing_authority_rows": int(len(missing_or_legacy)),
        "incumbent_no_close_observation_rows": int(len(no_obs)),
        "incumbent_hard_close_blocked_rows": int(readiness["incumbent_hard_close_blocked_rows"]),
        "ticker_4739_transition_gap_rows": int(readiness["ticker_4739_cross_venue_remaining_local_scope_gap_rows"]),
        "strategy_verdict_allowed": False,
        "may_be_used_to_reject_strategy": False,
        "data_readiness_blocked_only": True,
        "ready_for_experiments": False,
        "radar_download_authorized": False,
        "formal_model_changed": False,
        "trade_decision_changed": False,
        "active_in_trade_decision": False,
        "report_changed": False,
        "not_live_rule": True,
        "future_data_violation_count": 0,
        "next_step": "Strategy Center must choose atomic no-trade semantics and decide whether local source conflicts need bounded official authority reconciliation; no Experiments handoff.",
    }
    (OUT / "readiness_for_strategy_center_triage.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "final_summary_zh.md").write_text(
        "# MA-slope CD50 remaining blocker triage\n\n"
        f"- ready variants: {summary['ready_variant_count']}/50\n"
        f"- local source conflict rows: {summary['local_source_conflict_rows']}\n"
        f"- atomic no-trade rows: {summary['atomic_no_trade_rows']}\n"
        f"- corporate-action guard held rows: {summary['corporate_action_guard_held_rows']}\n"
        f"- explicit event authority hit rows: {summary['corporate_action_explicit_event_hit_rows']}\n"
        "- strategy_verdict_allowed=false; may_be_used_to_reject_strategy=false.\n"
        "- no Experiments handoff and no Radar download authorized.\n",
        encoding="utf-8",
    )
    (OUT / "current_step.txt").write_text("blocked_strategy_center_policy_triage\n", encoding="utf-8")
    files = sorted(path for path in OUT.iterdir() if path.is_file() and path.name != "manifest.json")
    (OUT / "manifest.json").write_text(
        json.dumps(
            {
                "task": summary["task"],
                "input_output": str(INPUT),
                "files": [{"name": path.name, "sha256": _sha(path), "bytes": path.stat().st_size} for path in files],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    run()
