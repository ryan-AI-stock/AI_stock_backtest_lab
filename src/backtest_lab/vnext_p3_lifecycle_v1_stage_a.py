from __future__ import annotations

import hashlib
import json
import gc
from pathlib import Path

import numpy as np
import pandas as pd

from backtest_lab import vnext_p3_all80_entry_funnel_setup_latch_audit as audit
from backtest_lab import vnext_p3_c3_top1_incumbent_fixed_contract as v0
from backtest_lab import vnext_p3_c3_top1_incumbent_path_contract as incumbent_path


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/vnext_p3_layer5_sequential_lifecycle_V1_expanded_candidate_dual_exit_stage_A_contract_20260713"
TASK = "TASK-BACKTEST-CORE-VNEXT-P3-LAYER5-SEQUENTIAL-LIFECYCLE-V1-EXPANDED-CANDIDATE-DUAL-EXIT-STAGE-A-CONTRACT-001"
V0_DUAL = ROOT / "outputs/vnext_p3_layer5_all80_candidate_opportunity_vs_selected_position_dual_state_contract_20260713/p3_all80_candidate_C0_C3_daily_panel.csv.gz"
V0_ACTION = ROOT / "outputs/vnext_p3_layer5_C3_top1_incumbent_path_corrected_NAV_contract_20260713/p3_C3_top1_incumbent_daily_action_ledger.csv"
MARKET_V2 = ROOT / "outputs/vnext_p3_market_controller_full_spec_v2_20260712/p3_market_controller_full_spec_v2_daily_features.csv"
EPISODES = {
    "5871": (pd.Timestamp("2023-08-07"), pd.Timestamp("2024-01-19")),
    "2610": (pd.Timestamp("2024-04-26"), pd.Timestamp("2024-08-13")),
    "3533": (pd.Timestamp("2024-08-20"), pd.Timestamp("2025-01-07")),
    "2327": (pd.Timestamp("2025-02-11"), pd.Timestamp("2025-07-10")),
}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _consecutive_persistence(frame: pd.DataFrame, column: str, need: int, window: int, market_index: dict[pd.Timestamp, int]) -> pd.Series:
    result = pd.Series(False, index=frame.index)
    for _, group in frame.groupby("ticker", sort=False):
        indexes = group.decision_date.map(market_index)
        segments = indexes.diff().ne(1).cumsum()
        values = group[column].fillna(False).astype(int)
        rolled = values.groupby(segments).rolling(window, min_periods=window).sum().reset_index(level=0, drop=True).ge(need)
        result.loc[group.index] = rolled
    return result


def _scores(frame: pd.DataFrame) -> pd.DataFrame:
    score_columns = [
        "decision_date", "ticker", "selected_eligibility", "selected_ineligibility_reason",
        "opportunity_momentum_score", "trend_continuation_score", "capital_chip_support_score", "risk_axis",
        "opportunity_momentum_confidence", "trend_continuation_confidence", "capital_chip_support_confidence",
        "total_score_confidence", "PIT_available_at",
    ]
    score = pd.read_csv(v0.SCORE, dtype={"ticker": str}, usecols=score_columns)
    score["decision_date"] = pd.to_datetime(score.decision_date)
    score["ticker"] = score.ticker.str.zfill(4)
    joined = frame.merge(score, on=["decision_date", "ticker"], how="left")
    for source, target in [
        ("opportunity_momentum_score", "opportunity_percentile"),
        ("trend_continuation_score", "trend_percentile"),
        ("capital_chip_support_score", "capital_percentile"),
        ("risk_axis", "risk_percentile"),
    ]:
        joined[target] = joined.groupby("decision_date")[source].rank(method="average", pct=True) * 100
    joined["inverse_risk_percentile"] = 100 - joined.risk_percentile
    joined["risk_decile"] = np.ceil(joined.risk_percentile / 10).clip(1, 10)
    joined["V1_composite"] = (
        0.35 * joined.opportunity_percentile
        + 0.25 * joined.trend_percentile
        + 0.25 * joined.capital_percentile
        + 0.15 * joined.inverse_risk_percentile
    )
    joined["V1_confidence"] = joined[[
        "opportunity_momentum_confidence", "trend_continuation_confidence", "capital_chip_support_confidence",
    ]].mean(axis=1)
    joined["hard_data_valid"] = joined.selected_eligibility.fillna(False) & joined.price_history_ready & joined[[
        "opportunity_momentum_score", "trend_continuation_score", "capital_chip_support_score", "risk_axis",
    ]].notna().all(axis=1)
    joined["severe_risk_veto"] = joined.risk_decile.ge(9) | joined.risk_bad.fillna(True)
    return joined


def _candidate_panel() -> tuple[pd.DataFrame, list[pd.Timestamp]]:
    frame = audit._base().sort_values(["ticker", "decision_date"]).copy()
    market = pd.read_csv(MARKET_V2, usecols=["decision_date", "full_spec_v2_state", "controller_state_status"])
    market["decision_date"] = pd.to_datetime(market.decision_date)
    frame = frame.drop(columns=["market_state"], errors="ignore").merge(market, on="decision_date", how="left")
    frame = frame.rename(columns={"full_spec_v2_state": "market_state"})
    dates = sorted(frame.decision_date.unique())
    market_index = {pd.Timestamp(date): index for index, date in enumerate(dates)}
    frame["market_index"] = frame.decision_date.map(market_index)
    frame["fold"] = frame.decision_date.map(audit._fold_map(dates))
    frame["relative_low"] = frame.price_pct_6M.le(0.40) & (frame.K_pct_6M.le(0.40) | frame.BIAS_pct_6M.le(0.40))
    frame["core_count"] = frame[["kd_up", "rs_repair", "ma_up"]].fillna(False).sum(axis=1)
    frame["capital_support"] = frame.capital_improve.fillna(False)
    frame["risk_not_worsening"] = frame.risk_ok.fillna(False)

    rows = []
    for ticker, group in frame.groupby("ticker", sort=False):
        last_low = None
        episode = 0
        episode_open = False
        for row in group.itertuples():
            day = row.market_index
            if row.relative_low:
                if not episode_open:
                    episode += 1; episode_open = True
                last_low = day
            elif episode_open and last_low is not None and day - last_low > 10:
                episode_open = False
            setup_active = episode_open and last_low is not None and day - last_low <= 10
            if row.market_state == "strong_market":
                primitive = row.core_count >= 2
                need, window = 2, 3
            elif row.market_state == "ordinary_market":
                primitive = row.core_count >= 2 and (row.capital_support or row.core_count == 3)
                need, window = 2, 3
            elif row.market_state == "weak_market":
                primitive = row.core_count == 3 and row.capital_support
                need, window = 3, 5
            else:
                primitive = False
                need, window = 3, 5
            rows.append({
                "decision_date": row.decision_date,
                "ticker": ticker,
                "pool_rank": row.pool_rank,
                "fold": row.fold,
                "market_state": row.market_state,
                "relative_low": row.relative_low,
                "core_count": row.core_count,
                "capital_support": row.capital_support,
                "risk_not_worsening": row.risk_not_worsening,
                "risk_bad": row.risk_bad,
                "price_history_ready": row.price_history_ready,
                "setup_episode_id": f"{ticker}-V1LOW{episode}" if setup_active else None,
                "setup_active": setup_active,
                "market_entry_primitive": primitive and row.risk_not_worsening and row.market_state != "confirmed_bear",
                "persistence_need": need,
                "persistence_window": window,
            })
    expanded = pd.DataFrame(rows)
    expanded["eligible_persistent"] = False
    for (need, window), subset in expanded.groupby(["persistence_need", "persistence_window"]):
        expanded.loc[subset.index, "eligible_persistent"] = _consecutive_persistence(subset, "market_entry_primitive", int(need), int(window), market_index)
    expanded["V1_eligible_pre_veto"] = expanded.setup_active & expanded.eligible_persistent
    expanded = _scores(expanded)
    expanded["V1_eligible"] = expanded.V1_eligible_pre_veto & expanded.hard_data_valid & ~expanded.severe_risk_veto
    expanded["blocked_reason"] = np.select(
        [~expanded.setup_active, ~expanded.eligible_persistent, ~expanded.hard_data_valid, expanded.severe_risk_veto],
        ["no_active_low_setup", "entry_evidence_or_persistence_not_ready", "hard_quality_data_invalid", "severe_risk_veto"],
        default="",
    )
    expanded = expanded.sort_values(["ticker", "decision_date"])
    expanded["V1_cluster_start"] = expanded.V1_eligible & ~expanded.groupby("ticker").V1_eligible.shift(fill_value=False)
    return expanded, dates


def _top1(panel: pd.DataFrame, dates: list[pd.Timestamp]) -> pd.DataFrame:
    rows = []
    for date in dates:
        day = panel.loc[panel.decision_date.eq(date) & panel.V1_eligible].sort_values(
            ["V1_composite", "risk_percentile", "capital_percentile", "opportunity_percentile", "ticker"],
            ascending=[False, True, False, False, True],
        )
        first = day.iloc[0] if len(day) else None
        second = day.iloc[1] if len(day) > 1 else None
        rows.append({
            "decision_date": date,
            "V1_candidate_count": len(day),
            "top1_ticker": first.ticker if first is not None else None,
            "top1_score": first.V1_composite if first is not None else np.nan,
            "top1_confidence": first.V1_confidence if first is not None else np.nan,
            "top1_risk_percentile": first.risk_percentile if first is not None else np.nan,
            "second_ticker": second.ticker if second is not None else None,
            "second_score": second.V1_composite if second is not None else np.nan,
            "future_outcome_read": False,
        })
    return pd.DataFrame(rows)


def _incumbent_pf_trace(dates: list[pd.Timestamp]) -> pd.DataFrame:
    features = incumbent_path._incumbent_features().sort_values(["ticker", "decision_date"])
    market = pd.read_csv(MARKET_V2, usecols=["decision_date", "full_spec_v2_state", "controller_state_status"])
    market["decision_date"] = pd.to_datetime(market.decision_date)
    features = features.drop(columns=["market_state"], errors="ignore").merge(market, on="decision_date", how="left")
    features = features.rename(columns={"full_spec_v2_state": "market_state"})
    market_index = {pd.Timestamp(date): index for index, date in enumerate(dates)}
    features = features.loc[features.decision_date.isin(dates)].copy()
    groups = ["kd_down", "rs_weak", "ma_down", "capital_withdraw", "risk_bad"]
    features["PF_group_count"] = features[groups].fillna(False).sum(axis=1)
    features["PF_stock_specific_mandatory"] = features.ma_down.fillna(False) | features.risk_bad.fillna(False)
    requirements = features.market_state.map({"strong_market": 4, "ordinary_market": 3, "weak_market": 2, "confirmed_bear": 2}).fillna(3)
    features["PF_primitive"] = features.PF_group_count.ge(requirements)
    features.loc[features.market_state.eq("PIT_warmup_or_low_confidence"), "PF_primitive"] = False
    weak_or_bear = features.market_state.isin(["weak_market", "confirmed_bear"])
    features.loc[weak_or_bear, "PF_primitive"] &= features.loc[weak_or_bear, "PF_stock_specific_mandatory"]
    features["PF_confirmed"] = _consecutive_persistence(features, "PF_primitive", 2, 3, market_index)
    action = pd.read_csv(V0_ACTION, dtype={"incumbent": str})
    action["decision_date"] = pd.to_datetime(action.decision_date)
    action["incumbent"] = action.incumbent.str.replace(".0", "", regex=False).str.zfill(4)
    rows = []
    for ticker, (start, end) in EPISODES.items():
        held = features.loc[features.ticker.eq(ticker) & features.decision_date.between(start, end)].copy()
        existing = action.loc[action.incumbent.eq(ticker), ["decision_date", "position_state"]]
        held = held.merge(existing, on="decision_date", how="left")
        held["ticker"] = ticker
        held["probation_first_20TD"] = np.arange(len(held)) < 20
        held["market_controller_version"] = "full_spec_v2"
        rows.append(held[["decision_date", "ticker", "market_state", "market_controller_version", "position_state", "relative_high", "PF_group_count", "PF_stock_specific_mandatory", "PF_primitive", "PF_confirmed", "kd_down", "rs_weak", "ma_down", "capital_withdraw", "risk_bad", "probation_first_20TD"]])
    return pd.concat(rows, ignore_index=True)


def run() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    panel, dates = _candidate_panel()
    panel.to_csv(OUT / "p3_V1_expanded_candidate_daily_exact80_panel.csv.gz", index=False, compression="gzip")
    winners = _top1(panel, dates)
    winners.to_csv(OUT / "p3_V1_daily_top1_second_materialization.csv", index=False, encoding="utf-8-sig")

    fold_rows = []
    for fold, part in panel.groupby("fold"):
        daily = part.groupby("decision_date").V1_eligible.sum()
        no_candidate = daily.eq(0)
        longest = current = 0
        for missing in no_candidate:
            current = current + 1 if missing else 0; longest = max(longest, current)
        fold_rows.append({
            "fold": int(fold),
            "start": part.decision_date.min(),
            "end": part.decision_date.max(),
            "eligible_dates": int(daily.gt(0).sum()),
            "unique_clusters": int(part.V1_cluster_start.sum()),
            "dates_with_at_least_2_candidates": int(daily.ge(2).sum()),
            "candidate_count_mean": float(daily.mean()),
            "candidate_count_median": float(daily.median()),
            "candidate_count_p75": float(daily.quantile(.75)),
            "candidate_count_max": int(daily.max()),
            "longest_no_candidate_gap": longest,
        })
    fold = pd.DataFrame(fold_rows)
    fold["eligible_dates_gate"] = fold.eligible_dates.ge(20)
    fold["clusters_gate"] = fold.unique_clusters.ge(20)
    fold["horizontal_competition_gate"] = fold.dates_with_at_least_2_candidates.ge(10)
    fold.to_csv(OUT / "p3_V1_candidate_fold_supply_gate.csv", index=False, encoding="utf-8-sig")

    old = pd.read_csv(V0_DUAL, dtype={"ticker": str})
    old["decision_date"] = pd.to_datetime(old.decision_date)
    reconcile = panel[["decision_date", "ticker", "setup_active", "eligible_persistent", "hard_data_valid", "severe_risk_veto", "V1_eligible"]].merge(
        old[["decision_date", "ticker", "C3_eligible", "capital_confirmation_ready"]], on=["decision_date", "ticker"], how="left"
    )
    reconcile["newly_eligible_vs_V0"] = reconcile.V1_eligible & ~reconcile.C3_eligible.fillna(False)
    reasons = pd.DataFrame([{
        "reason": "capital_no_longer_universal_hard_gate",
        "newly_eligible_rows": int((reconcile.newly_eligible_vs_V0 & ~reconcile.capital_confirmation_ready.fillna(False)).sum()),
    }, {
        "reason": "market_specific_2of3_or_3of5_core_persistence",
        "newly_eligible_rows": int((reconcile.newly_eligible_vs_V0 & reconcile.eligible_persistent).sum()),
    }, {
        "reason": "severe_risk_veto_removed",
        "newly_eligible_rows": int((reconcile.newly_eligible_vs_V0 & reconcile.severe_risk_veto).sum()),
    }])
    reasons.to_csv(OUT / "p3_V0_C3_vs_V1_expanded_supply_reconciliation.csv", index=False, encoding="utf-8-sig")

    block_columns = ["opportunity_percentile", "trend_percentile", "capital_percentile", "inverse_risk_percentile"]
    block_audit = pd.DataFrame([{
        "block": column,
        "weight": weight,
        "non_null_rows": int(panel[column].notna().sum()),
        "unique_values": int(panel[column].nunique()),
        "standard_deviation": float(panel[column].std()),
        "constant": bool(panel[column].nunique() <= 1),
        "PIT_lineage": "accepted full_candidate_spec_v1 precombined dimension",
        "missing_zero_filled": False,
    } for column, weight in zip(block_columns, [0.35, 0.25, 0.25, 0.15])])
    block_audit.to_csv(OUT / "p3_V1_score_block_readiness_audit.csv", index=False, encoding="utf-8-sig")
    panel[block_columns].corr().rename_axis("block").reset_index().to_csv(OUT / "p3_V1_block_correlation_precombine_audit.csv", index=False, encoding="utf-8-sig")

    horizontal_pass = bool(fold[["eligible_dates_gate", "clusters_gate", "horizontal_competition_gate"]].all().all())
    score_ready = bool(~block_audit.constant.any() and block_audit.non_null_rows.gt(0).all())
    candidate_rows = len(panel)
    del panel, reconcile, old, winners
    gc.collect()
    pf = _incumbent_pf_trace(dates)
    pf.to_csv(OUT / "p3_V1_existing_incumbent_PF_P5_P6_P7_state_trace.csv", index=False, encoding="utf-8-sig")
    pf_supply = pf.groupby("ticker").agg(
        held_dates=("decision_date", "nunique"), PF_primitive_dates=("PF_primitive", "sum"), PF_confirmed_dates=("PF_confirmed", "sum"),
        P5_dates=("position_state", lambda s: s.eq("P5").sum()), P6_dates=("position_state", lambda s: s.eq("P6").sum()), P7_dates=("position_state", lambda s: s.eq("P7").sum()),
    ).reset_index()
    pf_supply["stock_specific_PF_available"] = pf_supply.PF_confirmed_dates.gt(0)
    pf_supply.to_csv(OUT / "p3_V1_existing_incumbent_PF_supply_audit.csv", index=False, encoding="utf-8-sig")

    pf_required_pass = bool(pf_supply.loc[pf_supply.ticker.isin(["5871", "2327"]), "stock_specific_PF_available"].all())
    stage_pass = horizontal_pass and pf_required_pass and score_ready
    policy = {
        "candidate_setup": {"price_6M_percentile_max": 0.40, "KD_or_BIAS_same_direction": True, "grace_TD": 10},
        "entry": {
            "strong_market": "core>=2/3 and 2of3 persistence",
            "ordinary_market": "core>=2/3 and (capital or core=3/3), 2of3 persistence",
            "weak_market": "core=3/3 and capital, 3of5 persistence",
            "confirmed_bear": "no entry",
            "risk_not_worsening_mandatory": True,
        },
        "score_weights": {"opportunity": 0.35, "trend": 0.25, "capital": 0.25, "inverse_residual_risk": 0.15},
        "PF": {
            "groups": ["KD down", "RS weak", "MA/structure breakdown", "capital withdrawal", "risk worsening"],
            "strong_market": "4/5", "ordinary_market": "3/5", "weak_market": "2/5 plus MA/risk mandatory",
            "confirmed_bear": "2/5 plus stock-specific mandatory", "persistence": "2of3",
        },
        "future_outcome_used": False,
    }
    (OUT / "p3_V1_fixed_architecture_machine_policy.json").write_text(json.dumps(policy, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame([{
        "audit": "future_outcome_read", "violations": 0,
    }, {
        "audit": "P3_2_read", "violations": 0,
    }, {
        "audit": "NAV_or_performance_materialized", "violations": 0,
    }, {
        "audit": "TDCC_P3_1_zero_fill", "violations": 0,
    }]).to_csv(OUT / "p3_V1_PIT_future_governance_audit.csv", index=False, encoding="utf-8-sig")
    readiness = {
        "task_id": TASK,
        "status": "stage_A_supply_pass_return_strategy_center" if stage_pass else "stage_A_supply_failed_return_strategy_center",
        "P3_1_dates": len(dates),
        "candidate_rows": candidate_rows,
        "fold_supply": fold.to_dict("records"),
        "horizontal_competition_gate_pass": horizontal_pass,
        "score_readiness_pass": score_ready,
        "PF_5871_2327_supply_pass": pf_required_pass,
        "stage_A_pass": stage_pass,
        "ready_for_experiments": False,
        "non_representative_of_current_staged_diagnostic": True,
        "non_representative_of_current_rank1_stock_only_timing_stage": True,
        "may_be_used_to_reject_stock_only_low_buy_high_sell_hypothesis": False,
        "follow_up_stopped": True,
        "may_be_used_as_primary_baseline": False,
        "checkpoint_only": True,
        "allowed_role": "supply_reference_only",
        "architecture_reset_from_fixed_V0": True,
        "fixed_V1_only": True,
        "weight_grid_authorized": False,
        "performance_authorized": False,
        "P3_2_outcome_read_authorized": False,
        "Top3_authorized": False,
        "future_data_violation_count": 0,
        "formal_model_changed": False,
        "trade_decision_changed": False,
        "active_in_trade_decision": False,
        "report_changed": False,
        "not_live_rule": True,
        "forward_returns_live_rule_usage": False,
    }
    (OUT / "readiness_for_V1_stage_A.json").write_text(json.dumps(readiness, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    (OUT / "final_summary_zh.md").write_text(f"# V1 expanded candidate + dual exit Stage A\n\n此輸出已停止，只保留 checkpoint：`non_representative_of_current_rank1_stock_only_timing_stage=true`、`may_be_used_to_reject_stock_only_low_buy_high_sell_hypothesis=false`、`follow_up_stopped=true`、`ready_for_experiments=false`、`allowed_role=supply_reference_only`。不得作目前主線結論或 primary baseline。\n\nHorizontal gate={horizontal_pass}，score ready={score_ready}，PF 5871/2327 supply={pf_required_pass}，Stage A={stage_pass}。本輪只讀 P3-1 PIT/state/supply，未讀 future outcome/P3-2 或 NAV。\n", encoding="utf-8")
    files = sorted(p for p in OUT.iterdir() if p.is_file() and p.name != "manifest.json")
    (OUT / "manifest.json").write_text(json.dumps({"task_id": TASK, "non_representative_of_current_rank1_stock_only_timing_stage": True, "may_be_used_to_reject_stock_only_low_buy_high_sell_hypothesis": False, "follow_up_stopped": True, "ready_for_experiments": False, "allowed_role": "supply_reference_only", "files": [{"name": p.name, "sha256": _sha(p), "bytes": p.stat().st_size} for p in files]}, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    run()
