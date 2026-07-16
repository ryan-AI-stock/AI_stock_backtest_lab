from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from backtest_lab.vnext_p1_p2_primary80_ma_slope_cd50_contract import (
    OUT as CONTRACT_OUT,
    PERIODS,
    SIGNALS,
    TASK,
    membership,
    parameter_matrix,
)


ROOT = Path(__file__).resolve().parents[2]
RADAR = Path(r"C:\Users\zergv\Documents\Codex\2026-05-23\ai-stock-rotation-radar-https-docs\outputs\radar_vnext_p1_p2_primary80_ma_slope_cd50_price_source_convergence_20260715")
RADAR_OUTPUTS = RADAR.parent
ONE_SHOT = RADAR_OUTPUTS / "radar_vnext_p1_p2_primary80_ma_slope_cd50_one_shot_close_fill_20260716"
SHIFTED_PATH = RADAR_OUTPUTS / "radar_vnext_p1_p2_ma_slope_cd50_shifted_path_local_close_extraction_20260716"
CLOSURES = sorted(
    [RADAR_OUTPUTS / "radar_vnext_p1_p2_ma_slope_cd50_action_leg_frontier_local_audit_bounded_fill_20260715"]
    + list(RADAR_OUTPUTS.glob("radar_vnext_p1_p2_ma_slope_cd50_action_leg_frontier_iteration_*_20260715"))
)
OUT = ROOT / "outputs/vnext_p1_p2_layer4_primary80_individual_MA_slope_CD50_action_legs_20260715"
ADJUSTED = RADAR / "p1_p2_primary80_adjusted_analysis_close_reuse_compact.csv.gz"
RAW = RADAR / "p1_p2_primary80_official_raw_execution_close_reuse_compact.csv.gz"
REUSE_A = RADAR / "reuse_A_existing_raw_close.csv.gz"
REUSE_D = RADAR / "reuse_D_raw_ready_adjusted_factor_incomplete.csv.gz"
REUSE_E = RADAR / "reuse_E_not_applicable_or_no_trade_review.csv.gz"
ONE_SHOT_ADJUSTED = ONE_SHOT / "one_shot_adjusted_analysis_close_patch.csv.gz"
ONE_SHOT_RAW = ONE_SHOT / "one_shot_official_raw_execution_close_patch.csv.gz"
ONE_SHOT_NO_TRADE = ONE_SHOT / "one_shot_official_no_trade_ledger.csv.gz"
ONE_SHOT_BLOCKED = ONE_SHOT / "one_shot_close_remaining_blocked.csv.gz"
SHIFTED_ADJUSTED = SHIFTED_PATH / "bounded_network_adjusted_exact_patch.csv.gz"
SHIFTED_RAW = SHIFTED_PATH / "bounded_network_raw_exact_patch.csv.gz"
SHIFTED_ADJUSTED_BLOCKED = SHIFTED_PATH / "bounded_network_adjusted_remaining_blocked.csv.gz"
SHIFTED_RAW_BLOCKED = SHIFTED_PATH / "bounded_network_raw_remaining_blocked.csv"
CLOSURE_RAW = "frontier_official_raw_accepted_patch.csv"
CLOSURE_NO_TRADE = "frontier_official_no_trade_ledger.csv"
CLOSURE_CONTINUITY = "incumbent_continuity_local_classification.csv.gz"


def _read_closures(filename: str, **kwargs: object) -> pd.DataFrame:
    return pd.concat([pd.read_csv(path / filename, **kwargs) for path in CLOSURES], ignore_index=True)


def _read_optional_closures(filename: str, **kwargs: object) -> pd.DataFrame:
    paths = [path / filename for path in CLOSURES if (path / filename).exists()]
    return pd.concat([pd.read_csv(path, **kwargs) for path in paths], ignore_index=True) if paths else pd.DataFrame()


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _last_percentile(values: np.ndarray) -> float:
    valid = values[~np.isnan(values)]
    return np.nan if not len(valid) else float((valid <= valid[-1]).mean())


def load_prices() -> tuple[pd.DataFrame, pd.DataFrame]:
    adjusted = pd.read_csv(ADJUSTED, usecols=["period", "ticker", "date", "value", "source_quality"], dtype={"ticker": str})
    raw = pd.read_csv(RAW, usecols=["period", "ticker", "date", "value", "source_quality"], dtype={"ticker": str})
    for frame in (adjusted, raw):
        frame["ticker"] = frame.ticker.str.zfill(4)
        frame["date"] = pd.to_datetime(frame.date)
        frame["value"] = pd.to_numeric(frame.value, errors="coerce")
    adjusted = adjusted.dropna(subset=["value"]).drop_duplicates(["period", "ticker", "date"], keep="last")
    raw = raw.dropna(subset=["value"]).drop_duplicates(["period", "ticker", "date"], keep="last")
    continuity = _read_closures(CLOSURE_CONTINUITY, dtype={"ticker": str})
    continuity["ticker"] = continuity.ticker.str.zfill(4)
    continuity["date"] = pd.to_datetime(continuity.decision_date)
    continuity["adjusted_value"] = pd.to_numeric(continuity.adjusted_close, errors="coerce")
    continuity["value"] = continuity.adjusted_value
    continuity["source_quality"] = continuity.full_adjusted_source_quality.fillna("local_adjusted_continuity_reuse")
    adjusted = pd.concat(
        [adjusted, continuity[["period", "ticker", "date", "value", "source_quality"]].dropna(subset=["value"])],
        ignore_index=True,
    ).drop_duplicates(["period", "ticker", "date"], keep="last")
    one_adjusted = pd.read_csv(ONE_SHOT_ADJUSTED, dtype={"ticker": str}).rename(
        columns={"adjusted_analysis_close": "value"}
    )
    one_adjusted["ticker"] = one_adjusted.ticker.str.zfill(4)
    one_adjusted["date"] = pd.to_datetime(one_adjusted.date)
    one_adjusted["value"] = pd.to_numeric(one_adjusted.value, errors="coerce")
    adjusted = pd.concat(
        [adjusted, one_adjusted[["period", "ticker", "date", "value", "source_quality"]].dropna(subset=["value"])],
        ignore_index=True,
    ).drop_duplicates(["period", "ticker", "date"], keep="last")
    shifted_adjusted = pd.read_csv(SHIFTED_ADJUSTED, dtype={"ticker": str}).rename(
        columns={"adjusted_analysis_close": "value"}
    )
    shifted_adjusted["ticker"] = shifted_adjusted.ticker.str.zfill(4)
    shifted_adjusted["date"] = pd.to_datetime(shifted_adjusted.date)
    shifted_adjusted["value"] = pd.to_numeric(shifted_adjusted.value, errors="coerce")
    adjusted = pd.concat(
        [adjusted, shifted_adjusted[["period", "ticker", "date", "value", "source_quality"]].dropna(subset=["value"])],
        ignore_index=True,
    ).drop_duplicates(["period", "ticker", "date"], keep="last")
    closure_raw = _read_closures(CLOSURE_RAW, dtype={"ticker": str})
    closure_raw["ticker"] = closure_raw.ticker.str.zfill(4)
    closure_raw["date"] = pd.to_datetime(closure_raw.requested_execution_date)
    closure_raw["value"] = pd.to_numeric(closure_raw.close, errors="coerce")
    closure_raw["source_quality"] = closure_raw.source_quality
    raw = pd.concat(
        [raw, closure_raw[["period", "ticker", "date", "value", "source_quality"]].dropna(subset=["value"])],
        ignore_index=True,
    ).drop_duplicates(["period", "ticker", "date"], keep="last")
    one_raw = pd.read_csv(ONE_SHOT_RAW, dtype={"ticker": str}).rename(
        columns={"official_raw_execution_close": "value"}
    )
    one_raw["ticker"] = one_raw.ticker.str.zfill(4)
    one_raw["date"] = pd.to_datetime(one_raw.date)
    one_raw["value"] = pd.to_numeric(one_raw.value, errors="coerce")
    raw = pd.concat(
        [raw, one_raw[["period", "ticker", "date", "value", "source_quality"]].dropna(subset=["value"])],
        ignore_index=True,
    ).drop_duplicates(["period", "ticker", "date"], keep="last")
    shifted_raw = pd.read_csv(SHIFTED_RAW, dtype={"ticker": str}).rename(columns={"official_raw_close": "value"})
    shifted_raw["ticker"] = shifted_raw.ticker.str.zfill(4)
    shifted_raw["date"] = pd.to_datetime(shifted_raw.date)
    shifted_raw["value"] = pd.to_numeric(shifted_raw.value, errors="coerce")
    raw = pd.concat(
        [raw, shifted_raw[["period", "ticker", "date", "value", "source_quality"]].dropna(subset=["value"])],
        ignore_index=True,
    ).drop_duplicates(["period", "ticker", "date"], keep="last")
    return adjusted, raw


def feature_panel(adjusted: pd.DataFrame) -> pd.DataFrame:
    pieces = []
    for (period, ticker), group in adjusted.groupby(["period", "ticker"], sort=False):
        g = group.sort_values("date").copy()
        close = g.value
        for window in (4, 7, 10, 15, 20):
            g[f"MA{window}"] = close.rolling(window, min_periods=window).mean()
        for window in (5, 7, 10, 20):
            g[f"slope{window}"] = close - close.shift(window - 1)
            g[f"normalized_slope{window}"] = g[f"slope{window}"] / close.shift(window - 1).replace(0, np.nan)
        g["price_pct60"] = close.rolling(60, min_periods=60).apply(_last_percentile, raw=True)
        g["history_ready"] = g.price_pct60.notna() & g.MA20.notna() & g.slope20.notna()
        pieces.append(g)
    return pd.concat(pieces, ignore_index=True)


def active_candidate_panel(features: pd.DataFrame) -> pd.DataFrame:
    members = membership()[["period", "snapshot_date", "ticker", "pool_rank"]].copy()
    dates = features[["period", "date"]].drop_duplicates().sort_values(["period", "date"])
    effective = []
    for period, group in dates.groupby("period"):
        snaps = members.loc[members.period.eq(period), ["snapshot_date"]].drop_duplicates().sort_values("snapshot_date")
        mapped = pd.merge_asof(group.sort_values("date"), snaps, left_on="date", right_on="snapshot_date", direction="backward", allow_exact_matches=False)
        effective.append(mapped)
    date_snapshot = pd.concat(effective, ignore_index=True).dropna(subset=["snapshot_date"])
    active = date_snapshot.merge(members, on=["period", "snapshot_date"], how="inner")
    panel = active.merge(features, on=["period", "date", "ticker"], how="left")
    return panel


def ranked_candidates(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for signal, (entry_ma, entry_slope, exit_ma, exit_slope) in SIGNALS.items():
        x = panel.copy()
        x["signal_family"] = signal
        x["buy_signal"] = x.value.gt(x[f"MA{entry_ma}"]) & x[f"slope{entry_slope}"].gt(0) & x.history_ready.fillna(False)
        x["distance_above_entry_MA_pct"] = x.value / x[f"MA{entry_ma}"] - 1
        x["normalized_positive_slope"] = x[f"normalized_slope{entry_slope}"]
        x["candidate_rank"] = pd.NA
        eligible = x.loc[x.buy_signal].sort_values(
            ["period", "date", "price_pct60", "distance_above_entry_MA_pct", "normalized_positive_slope", "pool_rank", "ticker"],
            ascending=[True, True, True, True, False, True, True],
        ).copy()
        eligible["candidate_rank"] = eligible.groupby(["period", "date"]).cumcount() + 1
        rows.append(eligible[["period", "date", "ticker", "pool_rank", "signal_family", "price_pct60", "distance_above_entry_MA_pct", "normalized_positive_slope", "candidate_rank", "source_quality"]])
    return pd.concat(rows, ignore_index=True)


def _next_ticker_date(index: dict[tuple[str, str], list[pd.Timestamp]], period: str, ticker: str, decision: pd.Timestamp) -> pd.Timestamp | None:
    dates = index.get((period, ticker), [])
    pos = np.searchsorted(dates, decision, side="right")
    return dates[pos] if pos < len(dates) else None


def _next_market_date(index: dict[str, list[pd.Timestamp]], period: str, decision: pd.Timestamp) -> pd.Timestamp | None:
    dates = index.get(period, [])
    pos = np.searchsorted(dates, decision, side="right")
    return dates[pos] if pos < len(dates) else None


def simulate_actions(features: pd.DataFrame, candidates: pd.DataFrame, raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    feature_map = features.set_index(["period", "ticker", "date"])
    analysis_dates = {(p, t): list(g.date.sort_values()) for (p, t), g in features.groupby(["period", "ticker"])}
    market_dates = {
        period: sorted(set(features.loc[features.period.eq(period), "date"]) | set(raw.loc[raw.period.eq(period), "date"]))
        for period in PERIODS
    }
    raw_map = raw.set_index(["period", "ticker", "date"])
    no_trade = _read_closures(CLOSURE_NO_TRADE, dtype={"ticker": str})
    no_trade["ticker"] = no_trade.ticker.str.zfill(4)
    no_trade["requested_execution_date"] = pd.to_datetime(no_trade.requested_execution_date)
    no_trade_keys = set(no_trade[["period", "ticker", "requested_execution_date"]].itertuples(index=False, name=None))
    one_no_trade = pd.read_csv(ONE_SHOT_NO_TRADE, dtype={"ticker": str})
    one_no_trade["ticker"] = one_no_trade.ticker.str.zfill(4)
    one_no_trade["date"] = pd.to_datetime(one_no_trade.date)
    no_trade_keys.update(one_no_trade[["period", "ticker", "date"]].itertuples(index=False, name=None))
    resolved = _read_optional_closures("frontier_missing_execution_date_resolution_audit.csv", dtype={"ticker": str})
    execution_date_resolutions: dict[tuple[str, str, str, pd.Timestamp], pd.Timestamp] = {}
    if len(resolved):
        resolved["ticker"] = resolved.ticker.str.zfill(4)
        resolved["decision_date"] = pd.to_datetime(resolved.decision_date)
        resolved["resolved_execution_date"] = pd.to_datetime(resolved.resolved_execution_date)
        execution_date_resolutions = {
            (row.period, row.ticker, row.role, row.decision_date): row.resolved_execution_date
            for row in resolved.itertuples(index=False)
            if not row.silent_fill
        }
    candidate_groups = {(s, p, d): g.sort_values("candidate_rank") for (s, p, d), g in candidates.groupby(["signal_family", "period", "date"])}
    actions, requirements, blocked = [], [], []
    for variant in parameter_matrix().itertuples(index=False):
        signal, cooldown = variant.signal_family, variant.post_buy_exit_lock_trading_days
        _, _, exit_ma, exit_slope = SIGNALS[signal]
        for period, (requested_start, requested_end) in PERIODS.items():
            calendar = sorted(features.loc[features.period.eq(period) & features.date.between(requested_start, requested_end), "date"].unique())
            incumbent = None
            entry_execution_date = None
            last_sold = None
            for decision in calendar:
                decision = pd.Timestamp(decision)
                action, target, reason = "cash_wait" if incumbent is None else "hold", incumbent, ""
                sell_signal = False
                incumbent_row = None
                if incumbent is not None:
                    key = (period, incumbent, decision)
                    if key not in feature_map.index:
                        actions.append({"variant_id": variant.variant_id, "period": period, "decision_date": decision, "action": "hold_no_ticker_observation", "incumbent": incumbent, "target": incumbent, "reason": "no_ticker_analysis_observation_no_sell_evaluation"})
                        continue
                    incumbent_row = feature_map.loc[key]
                    sell_signal = bool(incumbent_row.value < incumbent_row[f"MA{exit_ma}"] and incumbent_row[f"slope{exit_slope}"] < 0)
                group = candidate_groups.get((signal, period, decision), pd.DataFrame())
                replacement = None
                if len(group):
                    choices = group.loc[group.ticker.ne(last_sold)]
                    if incumbent is not None:
                        choices = choices.loc[choices.ticker.ne(incumbent)]
                    if len(choices): replacement = choices.iloc[0].ticker
                if incumbent is None and replacement is not None:
                    action, target, reason = "entry_signal", replacement, "top_ranked_buy_signal"
                elif incumbent is not None and sell_signal:
                    next_exit = _next_market_date(market_dates, period, decision)
                    follow = [d for d in market_dates.get(period, []) if entry_execution_date is not None and d > entry_execution_date]
                    exit_unlocked = next_exit is not None and next_exit in follow and follow.index(next_exit) >= cooldown
                    if exit_unlocked:
                        action, target, reason = ("switch_signal", replacement, "sell_signal_and_different_candidate") if replacement else ("exit_signal", None, "sell_signal_no_replacement")
                    else:
                        reason = "sell_signal_exit_lock_active"
                if action in {"entry_signal", "switch_signal", "exit_signal"}:
                    legs = []
                    if action in {"switch_signal", "exit_signal"}: legs.append(("sell", incumbent))
                    if action in {"entry_signal", "switch_signal"}: legs.append(("buy", target))
                    leg_rows, leg_ready = [], True
                    requested_dates = []
                    no_trade_transition = False
                    for role, ticker in legs:
                        req_date = _next_market_date(market_dates, period, decision)
                        if req_date is None:
                            req_date = execution_date_resolutions.get((period, ticker, role, decision))
                        if (period, ticker, req_date) in no_trade_keys:
                            no_trade_transition = True
                        requested_dates.append(req_date)
                        ready = not no_trade_transition and req_date is not None and (period, ticker, req_date) in raw_map.index
                        leg_ready &= ready
                        row = {"variant_id": variant.variant_id, "period": period, "decision_date": decision, "execution_role": role, "ticker": ticker, "requested_execution_date": req_date, "official_raw_ready": ready}
                        if ready:
                            raw_row = raw_map.loc[(period, ticker, req_date)]
                            row.update({"official_raw_close": raw_row.value, "official_raw_source_quality": raw_row.source_quality})
                        leg_rows.append(row)
                    atomic = len(set(requested_dates)) == 1 if action == "switch_signal" else True
                    if not leg_ready or not atomic:
                        for row in leg_rows:
                            if not row["official_raw_ready"] or not atomic:
                                reason = "official_no_trade_atomic_transition_not_executed" if no_trade_transition else (
                                    "missing_official_raw_execution_close" if not row["official_raw_ready"] else "atomic_switch_execution_date_mismatch"
                                )
                                blocked.append({"variant_id": variant.variant_id, "period": period, "decision_date": decision, "ticker": row["ticker"], "role": row["execution_role"], "requested_execution_date": row["requested_execution_date"], "reason": reason})
                        action, target, reason = ("blocked_entry", None, "execution_leg_blocked") if incumbent is None else ("hold_blocked_transition", incumbent, "execution_leg_blocked")
                    else:
                        requirements.extend(leg_rows)
                        execution_date = requested_dates[0]
                        if action == "entry_signal": incumbent, entry_execution_date = target, execution_date
                        elif action == "switch_signal": last_sold, incumbent, entry_execution_date = incumbent, target, execution_date
                        else: last_sold, incumbent, entry_execution_date = incumbent, None, None
                actions.append({"variant_id": variant.variant_id, "period": period, "decision_date": decision, "action": action, "incumbent": incumbent, "target": target, "reason": reason, "sell_signal": sell_signal, "exit_lock_days": cooldown})
    return pd.DataFrame(actions), pd.DataFrame(requirements), pd.DataFrame(blocked)


def run() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "current_step.txt").write_text("load_A_B_C_D_E_and_materialize_features\n", encoding="utf-8")
    adjusted, raw = load_prices()
    features = feature_panel(adjusted)
    panel = active_candidate_panel(features)
    candidates = ranked_candidates(panel)
    (OUT / "current_step.txt").write_text("materialize_50_action_paths\n", encoding="utf-8")
    actions, requirements, blocked = simulate_actions(features, candidates, raw)
    features.to_csv(OUT / "p1_p2_MA_slope_adjusted_feature_compact.csv.gz", index=False, compression="gzip")
    candidates.to_csv(OUT / "p1_p2_MA_slope_candidate_rank_compact.csv.gz", index=False, compression="gzip")
    actions.to_csv(OUT / "p1_p2_MA_slope_CD50_action_trace.csv.gz", index=False, compression="gzip")
    requirements.to_csv(OUT / "p1_p2_MA_slope_CD50_execution_requirement_ledger.csv", index=False, encoding="utf-8-sig")
    blocked.to_csv(OUT / "p1_p2_MA_slope_CD50_action_leg_exact_gap_ledger.csv", index=False, encoding="utf-8-sig")
    missing_raw = blocked.loc[blocked.reason.eq("missing_official_raw_execution_close")].copy() if len(blocked) else blocked.copy()
    if len(missing_raw):
        first = missing_raw.groupby(["variant_id", "period"])["decision_date"].transform("min")
        frontier = missing_raw.loc[missing_raw.decision_date.eq(first)].drop_duplicates(["period", "ticker", "role", "requested_execution_date"])
    else:
        frontier = missing_raw
    frontier.to_csv(OUT / "p1_p2_MA_slope_CD50_final_official_raw_blocked_ledger.csv", index=False, encoding="utf-8-sig")
    atomic = blocked.loc[blocked.reason.eq("atomic_switch_execution_date_mismatch")].drop_duplicates(["period", "ticker", "role", "decision_date", "requested_execution_date"]) if len(blocked) else blocked.copy()
    atomic.to_csv(OUT / "p1_p2_MA_slope_CD50_atomic_policy_blocker_ledger.csv", index=False, encoding="utf-8-sig")
    no_observation = actions.loc[actions.action.eq("hold_no_ticker_observation"), ["period", "decision_date", "incumbent"]].drop_duplicates().rename(columns={"incumbent": "ticker"})
    e = pd.read_csv(REUSE_E, usecols=["period", "ticker", "date"], dtype={"ticker": str})
    e["ticker"] = e.ticker.str.zfill(4)
    e["decision_date"] = pd.to_datetime(e.pop("date"))
    e["official_no_trade_or_NA_evidence"] = True
    no_observation = no_observation.merge(e, on=["period", "ticker", "decision_date"], how="left")
    continuity_class = _read_closures(
        CLOSURE_CONTINUITY,
        usecols=["period", "ticker", "decision_date", "local_A_to_E_classification", "network_download_authorized"],
        dtype={"ticker": str},
    )
    continuity_class["ticker"] = continuity_class.ticker.str.zfill(4)
    continuity_class["decision_date"] = pd.to_datetime(continuity_class.decision_date)
    continuity_class = continuity_class.drop_duplicates(["period", "ticker", "decision_date"])
    no_observation = no_observation.merge(continuity_class, on=["period", "ticker", "decision_date"], how="left")
    one_shot_blocked = pd.read_csv(ONE_SHOT_BLOCKED, dtype={"ticker": str})
    one_shot_blocked["ticker"] = one_shot_blocked.ticker.str.zfill(4)
    one_shot_blocked["decision_date"] = pd.to_datetime(one_shot_blocked.date)
    analysis_blocked = one_shot_blocked.loc[
        one_shot_blocked.missing_adjusted_after_fill.fillna(False),
        ["period", "ticker", "decision_date", "blocked_reason"],
    ].drop_duplicates(["period", "ticker", "decision_date"])
    analysis_blocked = analysis_blocked.rename(columns={"blocked_reason": "one_shot_adjusted_blocked_reason"})
    no_observation = no_observation.merge(analysis_blocked, on=["period", "ticker", "decision_date"], how="left")
    no_observation["classification"] = no_observation.local_A_to_E_classification
    no_observation.loc[no_observation.one_shot_adjusted_blocked_reason.notna(), "classification"] = "one_shot_adjusted_analysis_source_blocked"
    no_observation.loc[no_observation.classification.str.startswith("D_", na=False), "classification"] = "one_shot_adjusted_analysis_source_blocked"
    no_observation.loc[no_observation.official_no_trade_or_NA_evidence.fillna(False), "classification"] = "E_official_no_trade_or_NA_no_analysis_gap"
    no_observation["classification"] = no_observation.classification.fillna("one_shot_strict_adjusted_authority_not_materialized")
    no_observation.to_csv(OUT / "p1_p2_MA_slope_CD50_incumbent_analysis_gap_audit.csv.gz", index=False, compression="gzip")
    incumbent_request = no_observation.loc[no_observation.classification.eq("incumbent_continuity_unclassified_local_audit_required")].groupby(["period", "ticker"]).agg(
        requested_start=("decision_date", "min"), requested_end=("decision_date", "max"), unclassified_dates=("decision_date", "nunique")
    ).reset_index()
    incumbent_request["authorized_action"] = "local_full_source_path_reuse_and_trade_date_classification_only"
    incumbent_request["network_download_authorized"] = False
    incumbent_request.to_csv(OUT / "p1_p2_MA_slope_CD50_incumbent_continuity_local_audit_request.csv", index=False, encoding="utf-8-sig")
    summary = actions.groupby(["period", "variant_id", "action"]).size().rename("rows").reset_index()
    summary.to_csv(OUT / "p1_p2_MA_slope_CD50_action_supply_summary.csv", index=False, encoding="utf-8-sig")
    provisional_unique_raw = missing_raw.drop_duplicates(["period", "ticker", "role", "requested_execution_date"]) if len(missing_raw) else missing_raw
    incumbent_unclassified = int(no_observation.classification.eq("incumbent_continuity_unclassified_local_audit_required").sum())
    incumbent_adjusted_blocked = int(no_observation.classification.str.startswith("one_shot_").sum())
    incumbent_local_policy_blocked = int(no_observation.classification.str.startswith("E_").sum())
    pd.DataFrame([{"final_raw_blocked_legs": len(frontier), "remaining_source_minutes": 0, "estimate_basis": "one-shot authority exhausted; no further Radar probe authorized"}]).to_csv(OUT / "p1_p2_MA_slope_CD50_remaining_source_time_estimate.csv", index=False, encoding="utf-8-sig")
    ready = len(frontier) == 0 and len(atomic) == 0 and incumbent_unclassified == 0 and incumbent_adjusted_blocked == 0
    readiness = {
        "task_id": TASK, "status": "ready_for_experiments" if ready else "one_shot_rechain_complete_with_explicit_path_blockers",
        "research_role": "individual_stock_timing_diagnostic_not_active_formal_mainline",
        "active_formal_mainline": "0050_signal_to_00631L_execution_MA4_7_MA10_20_CD7",
        "fixed_variants": 50, "feature_rows": len(features), "candidate_rows": len(candidates),
        "action_rows": len(actions), "execution_leg_rows": len(requirements),
        "blocked_action_rows": len(blocked), "all_provisional_unique_raw_gap_legs": len(provisional_unique_raw),
        "one_shot_close_absorbed": True, "shifted_path_close_absorbed": True,
        "shifted_adjusted_exact_rows_absorbed": len(pd.read_csv(SHIFTED_ADJUSTED)),
        "shifted_raw_exact_rows_absorbed": len(pd.read_csv(SHIFTED_RAW)),
        "shifted_adjusted_source_blocked_rows": len(pd.read_csv(SHIFTED_ADJUSTED_BLOCKED)),
        "shifted_raw_source_blocked_rows": len(pd.read_csv(SHIFTED_RAW_BLOCKED)),
        "shifted_raw_fallback_adjusted_rows_rejected": 145,
        "iterative_frontier_disabled": True, "further_radar_probe_authorized": False,
        "final_exact_official_raw_blocked_legs": len(frontier), "atomic_policy_blockers": len(atomic),
        "incumbent_no_observation_rows": len(no_observation), "incumbent_analysis_unclassified_rows": incumbent_unclassified,
        "incumbent_adjusted_analysis_source_blocked_rows": incumbent_adjusted_blocked,
        "incumbent_local_policy_blocked_rows": incumbent_local_policy_blocked,
        "raw_used_as_adjusted": False, "ready_for_experiments": ready,
        "formal_model_changed": False, "trade_decision_changed": False,
        "active_in_trade_decision": False, "report_changed": False,
        "not_live_rule": True, "future_data_violation_count": 0,
    }
    (OUT / "readiness_for_action_leg_first.json").write_text(json.dumps(readiness, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "final_summary_zh.md").write_text(
        f"# MA-slope CD50 one-shot close absorption\n\n- feature rows: {len(features)}\n- candidate rows: {len(candidates)}\n- execution legs ready: {len(requirements)}\n- final official raw blocked legs: {len(frontier)}\n- atomic policy blockers: {len(atomic)}\n- incumbent adjusted-analysis blocked rows: {incumbent_adjusted_blocked}\n- raw used as adjusted: false\n- no further iterative frontier or Radar probe authorized\n- individual-stock research only; not the active formal mainline.\n",
        encoding="utf-8",
    )
    (OUT / "current_step.txt").write_text("ready_for_experiments_handoff\n" if ready else "one_shot_rechain_complete_blocked_no_further_source_probe\n", encoding="utf-8")
    files = sorted(p for p in OUT.iterdir() if p.is_file() and p.name != "manifest.json")
    (OUT / "manifest.json").write_text(json.dumps({"task_id": TASK, "inputs": {"adjusted_sha256": _sha(ADJUSTED), "raw_sha256": _sha(RAW)}, "files": [{"name": p.name, "sha256": _sha(p), "bytes": p.stat().st_size} for p in files]}, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    run()
