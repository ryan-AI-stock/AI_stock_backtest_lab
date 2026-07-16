from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from backtest_lab.vnext_p1_p2_primary80_ma_slope_cd50_action_legs import (
    PATH_INDEPENDENT,
    SHIFTED_RAW_BLOCKED,
    active_candidate_panel,
    feature_panel,
    load_prices,
    ranked_candidates,
    simulate_actions,
)
from backtest_lab.vnext_p1_p2_primary80_ma_slope_cd50_contract import PERIODS, TASK, parameter_matrix


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/vnext_p1_p2_primary80_MA_slope_CD50_official_raw_close_diagnostic_20260716"
PATH_INDEPENDENT_RAW = PATH_INDEPENDENT / "path_independent_primary80_official_raw_close_compact.csv.gz"
PATH_INDEPENDENT_BLOCKED = PATH_INDEPENDENT / "path_independent_final_blocked.csv.gz"
PATH_INDEPENDENT_NO_TRADE = PATH_INDEPENDENT / "path_independent_final_official_no_trade_termination.csv.gz"
EVENT_DIR = Path(r"C:\Users\zergv\Documents\Codex\2026-05-23\ai-stock-rotation-radar-https-docs\outputs\radar_vnext_p1_full_lifecycle_minimum_data_acquisition_20260710\compact\trusted_corporate_action_events")
TRANSFER_TERMINATION = Path(r"C:\Users\zergv\Documents\Codex\2026-05-23\ai-stock-rotation-radar-https-docs\outputs\radar_vnext_4739_market_transfer_3474_termination_source_package_20260716")
TRANSFER_CLOSE_PATCH = TRANSFER_TERMINATION / "ticker_4739_post_transition_exact_close_patch.csv.gz"
TRANSFER_REMAINING = TRANSFER_TERMINATION / "ticker_4739_post_transition_remaining_local_gap.csv"
TERMINATION_LEDGER = TRANSFER_TERMINATION / "ticker_3474_termination_event_ledger.csv"
TRANSFER_4739_FILL = Path(r"C:\Users\zergv\Documents\Codex\2026-05-23\ai-stock-rotation-radar-https-docs\outputs\radar_vnext_4739_post_transfer_twse_close_fill_20260716")
TRANSFER_4739_FILL_PATCH = TRANSFER_4739_FILL / "ticker_4739_exact_twse_close_patch.csv.gz"
TRANSFER_4739_FILL_NO_TRADE = TRANSFER_4739_FILL / "ticker_4739_exact_twse_official_no_trade.csv"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_path_independent_raw() -> pd.DataFrame:
    source = pd.read_csv(PATH_INDEPENDENT_RAW, dtype={"ticker": str})
    source["ticker"] = source.ticker.str.zfill(4)
    source["date"] = pd.to_datetime(source.date)
    source = source.rename(columns={"close": "value"})
    transfer = pd.read_csv(TRANSFER_CLOSE_PATCH, dtype={"ticker": str}).rename(columns={"close": "value"})
    transfer["ticker"] = transfer.ticker.str.zfill(4)
    transfer["date"] = pd.to_datetime(transfer.date)
    source = pd.concat(
        [source, transfer[["ticker", "date", "market", "value", "source_quality"]]],
        ignore_index=True,
    ).drop_duplicates(["ticker", "date"], keep="last")
    transfer_fill = pd.read_csv(TRANSFER_4739_FILL_PATCH, dtype={"ticker": str}).rename(columns={"close": "value"})
    transfer_fill["ticker"] = transfer_fill.ticker.str.zfill(4)
    transfer_fill["date"] = pd.to_datetime(transfer_fill.date)
    source = pd.concat(
        [source, transfer_fill[["ticker", "date", "market", "value", "source_quality"]]],
        ignore_index=True,
    ).drop_duplicates(["ticker", "date"], keep="last")
    pieces = []
    for period, (start, end) in PERIODS.items():
        warmup_start = start - pd.Timedelta(days=100)
        part = source.loc[source.date.between(warmup_start, end)].copy()
        part["period"] = period
        pieces.append(part[["period", "ticker", "date", "value", "source_quality"]])
    return pd.concat(pieces, ignore_index=True).drop_duplicates(["period", "ticker", "date"], keep="last")


def load_termination_events() -> dict[tuple[str, str], dict[str, object]]:
    ledger = pd.read_csv(TERMINATION_LEDGER, dtype={"ticker": str})
    closing = ledger.loc[ledger.event_type.eq("cash_share_swap_closing_date_confirmation")].iloc[0]
    return {
        ("P1", str(closing.ticker).zfill(4)): {
            "effective_date": pd.Timestamp(closing.termination_effective_date),
            "cash_per_share": 30.0,
            "source_quality": "PIT_official_cash_share_swap_holder_treatment",
        }
    }


def load_explicit_events() -> pd.DataFrame:
    files = sorted(EVENT_DIR.glob("*.csv.gz"))
    if not files:
        return pd.DataFrame(columns=["ticker", "event_date", "event_type", "source_quality"])
    events = pd.concat([pd.read_csv(path, dtype={"ticker": str}) for path in files], ignore_index=True)
    events["ticker"] = events.ticker.str.zfill(4)
    events["event_date"] = pd.to_datetime(events.event_date)
    return events.drop_duplicates(["ticker", "event_date", "event_type"])


def expand_period_keys(path: Path, classification: str) -> pd.DataFrame:
    source = pd.read_csv(path, dtype={"ticker": str}, low_memory=False)
    rows = []
    for row in source.itertuples(index=False):
        for period in str(row.periods).split("|"):
            rows.append({"period": period, "ticker": str(row.ticker).zfill(4), "date": pd.Timestamp(row.date), "path_independent_classification": classification})
    return pd.DataFrame(rows).drop_duplicates(["period", "ticker", "date"])


def unresolved_4739_transition_gap_rows() -> int:
    remaining = pd.read_csv(TRANSFER_REMAINING, dtype={"ticker": str})
    if not len(remaining):
        return 0
    remaining["ticker"] = remaining.ticker.str.zfill(4)
    remaining["date"] = pd.to_datetime(remaining.date)
    filled = pd.read_csv(TRANSFER_4739_FILL_PATCH, usecols=["ticker", "date"], dtype={"ticker": str})
    filled["ticker"] = filled.ticker.str.zfill(4)
    filled["date"] = pd.to_datetime(filled.date)
    no_trade = pd.read_csv(TRANSFER_4739_FILL_NO_TRADE, usecols=["ticker", "date"], dtype={"ticker": str})
    no_trade["ticker"] = no_trade.ticker.str.zfill(4)
    no_trade["date"] = pd.to_datetime(no_trade.date)
    resolved = pd.concat([filled, no_trade], ignore_index=True).drop_duplicates(["ticker", "date"])
    unresolved = remaining.merge(resolved.assign(_resolved=True), on=["ticker", "date"], how="left")
    return int(unresolved._resolved.isna().sum())


def raw_close_features(raw: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    analysis = raw.copy()
    analysis["source_quality"] = analysis.source_quality.astype(str) + ";official_raw_close_intentional_diagnostic"
    features = feature_panel(analysis)
    features["raw_close_return"] = features.groupby(["period", "ticker"], sort=False).value.pct_change(fill_method=None)
    features["large_raw_close_move_warning"] = features.raw_close_return.abs().gt(0.15)
    features["large_raw_close_move_warning_60obs"] = (
        features.groupby(["period", "ticker"], sort=False).large_raw_close_move_warning
        .transform(lambda s: s.rolling(60, min_periods=1).max().astype(bool))
    )
    event_keys = set(events[["ticker", "event_date"]].itertuples(index=False, name=None))
    features["explicit_corporate_action_event"] = [
        (ticker, date) in event_keys for ticker, date in features[["ticker", "date"]].itertuples(index=False, name=None)
    ]
    features["corporate_action_guard_60obs"] = (
        features.groupby(["period", "ticker"], sort=False).explicit_corporate_action_event
        .transform(lambda s: s.rolling(60, min_periods=1).max().astype(bool))
    )
    return features


def run() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "current_step.txt").write_text("materialize_raw_close_diagnostic\n", encoding="utf-8")
    raw = load_path_independent_raw()
    events = load_explicit_events()
    features = raw_close_features(raw, events)
    panel = active_candidate_panel(features)
    candidates = ranked_candidates(panel)
    actions, requirements, blocked = simulate_actions(features, candidates, raw, termination_events=load_termination_events())
    no_trade_keys = expand_period_keys(PATH_INDEPENDENT_NO_TRADE, "official_no_trade_or_termination")
    transfer_no_trade = pd.read_csv(TRANSFER_4739_FILL_NO_TRADE, dtype={"ticker": str})
    transfer_no_trade["ticker"] = transfer_no_trade.ticker.str.zfill(4)
    transfer_no_trade["date"] = pd.to_datetime(transfer_no_trade.date)
    transfer_no_trade_keys = []
    for period in PERIODS:
        transfer_no_trade_keys.append(
            transfer_no_trade.assign(period=period, path_independent_classification="official_no_trade_or_termination")[
                ["period", "ticker", "date", "path_independent_classification"]
            ]
        )
    if transfer_no_trade_keys:
        no_trade_keys = pd.concat([no_trade_keys, *transfer_no_trade_keys], ignore_index=True).drop_duplicates(
            ["period", "ticker", "date"]
        )
    conflict_keys = expand_period_keys(PATH_INDEPENDENT_BLOCKED, "local_source_conflict")

    guard = features.loc[
        features.corporate_action_guard_60obs,
        ["period", "ticker", "date", "raw_close_return", "explicit_corporate_action_event", "large_raw_close_move_warning"],
    ]
    held = actions.loc[actions.incumbent.notna(), ["variant_id", "period", "decision_date", "incumbent"]].rename(
        columns={"decision_date": "date", "incumbent": "ticker"}
    )
    held_guard = held.merge(guard, on=["period", "ticker", "date"], how="inner").drop_duplicates()
    no_observation = actions.loc[actions.action.eq("hold_no_ticker_observation")].copy()
    no_observation = no_observation.merge(
        pd.concat([no_trade_keys, conflict_keys], ignore_index=True),
        left_on=["period", "incumbent", "decision_date"],
        right_on=["period", "ticker", "date"],
        how="left",
    )
    no_observation["path_independent_classification"] = no_observation.path_independent_classification.fillna(
        "exact_raw_close_absent_from_path_independent_partition"
    )
    pending_3474 = (
        no_observation.incumbent.eq("3474")
        & no_observation.decision_date.between(pd.Timestamp("2016-11-30"), pd.Timestamp("2016-12-05"))
    )
    no_observation.loc[pending_3474, "path_independent_classification"] = "PIT_known_termination_pending_holder_treatment_last_mark_carry"
    last_raw = raw.groupby(["period", "ticker"], as_index=False).date.max().rename(columns={"date": "last_official_raw_close_date"})
    no_observation = no_observation.merge(last_raw, left_on=["period", "incumbent"], right_on=["period", "ticker"], how="left", suffixes=("", "_last"))
    post_last_trade = (
        no_observation.path_independent_classification.eq("exact_raw_close_absent_from_path_independent_partition")
        & no_observation.decision_date.gt(no_observation.last_official_raw_close_date)
    )
    no_observation.loc[post_last_trade, "path_independent_classification"] = "post_last_official_trade_event_or_venue_transition_contract_missing"

    source_class = pd.concat([no_trade_keys, conflict_keys], ignore_index=True).rename(columns={"date": "requested_execution_date"})
    if len(blocked):
        blocked = blocked.merge(source_class, on=["period", "ticker", "requested_execution_date"], how="left")
        blocked["precise_source_class"] = blocked.path_independent_classification.fillna(
            "exact_raw_close_absent_after_local_only_close_basis_rechain"
        )
    else:
        blocked["precise_source_class"] = pd.Series(dtype=str)

    rows = []
    for variant in parameter_matrix().variant_id:
        b = blocked.loc[blocked.variant_id.eq(variant)]
        g = held_guard.loc[held_guard.variant_id.eq(variant)]
        n = no_observation.loc[no_observation.variant_id.eq(variant)]
        n_hard = n.loc[~n.path_independent_classification.isin([
            "official_no_trade_or_termination",
            "PIT_known_termination_pending_holder_treatment_last_mark_carry",
        ])]
        b_hard = b.loc[~b.precise_source_class.eq("official_no_trade_or_termination")]
        req = requirements.loc[requirements.variant_id.eq(variant)]
        close_ready = len(b_hard) == 0 and len(n_hard) == 0
        event_guard_ready = False
        ready = close_ready and event_guard_ready
        rows.append(
            {
                "variant_id": variant,
                "execution_legs_ready": len(req),
                "execution_blocked_rows": len(b),
                "execution_hard_source_blocked_rows": len(b_hard),
                "corporate_action_guard_held_rows": len(g),
                "incumbent_no_close_observation_rows": len(n),
                "incumbent_hard_close_blocked_rows": len(n_hard),
                "close_path_ready": close_ready,
                "corporate_action_guard_ready": event_guard_ready,
                "exact_path_coverage_pass": ready,
                "ready_for_experiments": ready,
            }
        )
    readiness_table = pd.DataFrame(rows)
    ready_variants = readiness_table.loc[readiness_table.ready_for_experiments, "variant_id"].tolist()

    features.to_csv(OUT / "raw_close_MA_slope_feature_guard_compact.csv.gz", index=False, compression="gzip")
    candidates.to_csv(OUT / "raw_close_candidate_rank_compact.csv.gz", index=False, compression="gzip")
    actions.to_csv(OUT / "raw_close_CD50_action_trace.csv.gz", index=False, compression="gzip")
    requirements.to_csv(OUT / "raw_close_execution_requirement_ledger.csv", index=False, encoding="utf-8-sig")
    blocked.to_csv(OUT / "raw_close_execution_blocked_ledger.csv", index=False, encoding="utf-8-sig")
    held_guard.to_csv(OUT / "raw_close_corporate_action_guard_held_path_ledger.csv.gz", index=False, compression="gzip")
    no_observation.to_csv(OUT / "raw_close_incumbent_no_observation_ledger.csv.gz", index=False, compression="gzip")
    readiness_table.to_csv(OUT / "raw_close_CD50_per_variant_readiness.csv", index=False, encoding="utf-8-sig")
    policy = {
        "analysis_price_basis": "official_raw_close_intentional_diagnostic",
        "raw_as_adjusted_fallback": False,
        "total_return_basis": False,
        "formal_basis": False,
        "non_close_family_used": False,
        "new_radar_download_authorized": False,
        "large_raw_close_move_warning": "abs_same_ticker_raw_close_return_gt_15pct_audit_only_not_a_hard_guard",
        "corporate_action_hard_guard": "actual_event_ledger_or_accepted_explicit_event_evidence_only; current local ledger is P1 diagnostic and not complete P1/P2 authority",
        "neighbor_price_substitution": False,
        "fixed_variants": 50,
    }
    (OUT / "raw_close_diagnostic_policy.json").write_text(json.dumps(policy, ensure_ascii=False, indent=2), encoding="utf-8")
    unresolved_4739_rows = unresolved_4739_transition_gap_rows()
    readiness = {
        "task_id": TASK,
        "status": "ready_for_experiments" if ready_variants else "data_readiness_blocked_local_source_conflicts_or_corporate_action_guard",
        "analysis_price_basis": "official_raw_close_intentional_diagnostic",
        "fixed_variants": 50,
        "ready_variant_count": len(ready_variants),
        "close_path_ready_variant_count": int(readiness_table.close_path_ready.sum()),
        "corporate_action_guard_ready_variant_count": int(readiness_table.corporate_action_guard_ready.sum()),
        "ready_variants": ready_variants,
        "execution_blocked_rows": len(blocked),
        "execution_hard_source_blocked_rows": int(readiness_table.execution_hard_source_blocked_rows.sum()),
        "execution_blocked_unique_keys": len(blocked.drop_duplicates(["period", "ticker", "role", "requested_execution_date"])) if len(blocked) else 0,
        "corporate_action_guard_held_rows": len(held_guard),
        "incumbent_no_close_observation_rows": len(no_observation),
        "incumbent_hard_close_blocked_rows": int(readiness_table.incumbent_hard_close_blocked_rows.sum()),
        "path_independent_official_raw_close_rows": len(pd.read_csv(PATH_INDEPENDENT_RAW, usecols=["ticker"])),
        "path_independent_local_source_conflict_rows": len(pd.read_csv(PATH_INDEPENDENT_BLOCKED, usecols=["ticker"])),
        "ticker_4739_cross_venue_exact_close_patch_rows": len(pd.read_csv(TRANSFER_CLOSE_PATCH, usecols=["ticker"])),
        "ticker_4739_post_transfer_exact_close_fill_rows": len(pd.read_csv(TRANSFER_4739_FILL_PATCH, usecols=["ticker"])),
        "ticker_4739_post_transfer_official_no_trade_rows": len(pd.read_csv(TRANSFER_4739_FILL_NO_TRADE, usecols=["ticker"])),
        "ticker_4739_cross_venue_original_remaining_local_scope_gap_rows": len(pd.read_csv(TRANSFER_REMAINING, usecols=["ticker"])),
        "ticker_4739_cross_venue_remaining_local_scope_gap_rows": unresolved_4739_rows,
        "ticker_4739_post_transfer_fill_absorbed": unresolved_4739_rows == 0,
        "ticker_3474_holder_treatment_materialized": True,
        "ticker_3474_holder_treatment_strategy_sell": False,
        "ticker_3474_holder_treatment_market_sell_slippage": 0.0,
        "ticker_3474_holder_treatment_transition_cost": 0.0,
        "ticker_3474_holder_treatment_tax_fee_status": "diagnostic_caveat_no_accepted_contract_no_cost_invented",
        "explicit_event_rows_loaded": len(events),
        "corporate_action_event_authority_complete_for_P1_P2": False,
        "ready_for_experiments": bool(ready_variants),
        "data_readiness_blocked_only": not bool(ready_variants),
        "may_be_used_to_reject_strategy": False,
        "path_independent_close_authority_absorbed": True,
        "path_dependent_close_authority_not_sufficient_for_final_readiness": False,
        "market_transfer_forced_exit_prohibited": True,
        "same_ticker_cross_venue_close_continuity_required": True,
        "termination_requires_PIT_official_announcement_authority": True,
        "forced_exit_holder_treatment_requires_announcement_last_trade_effective_consideration_authority": True,
        "termination_event_and_forced_exit_contract_ready": True,
        "further_radar_probe_authorized": False,
        "formal_model_changed": False,
        "trade_decision_changed": False,
        "active_in_trade_decision": False,
        "report_changed": False,
        "not_live_rule": True,
        "future_data_violation_count": 0,
    }
    (OUT / "readiness_for_raw_close_diagnostic.json").write_text(json.dumps(readiness, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "final_summary_zh.md").write_text(
        "# MA-slope CD50 official raw-close diagnostic\n\n"
        f"- ready variants: {len(ready_variants)}/50\n"
        f"- execution blocked rows: {len(blocked)}\n"
        f"- corporate-action guarded held rows: {len(held_guard)}\n"
        f"- incumbent no-close rows: {len(no_observation)}\n"
        "- intentional raw-close diagnostic; not adjusted, total-return, or formal.\n"
        "- no further Radar download authorized.\n",
        encoding="utf-8",
    )
    (OUT / "current_step.txt").write_text("ready_for_experiments_handoff\n" if ready_variants else "blocked_no_further_download\n", encoding="utf-8")
    files = sorted(p for p in OUT.iterdir() if p.is_file() and p.name != "manifest.json")
    (OUT / "manifest.json").write_text(
        json.dumps({"task_id": TASK, "files": [{"name": p.name, "sha256": _sha(p), "bytes": p.stat().st_size} for p in files]}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    run()
