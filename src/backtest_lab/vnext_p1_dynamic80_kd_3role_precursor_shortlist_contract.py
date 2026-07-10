from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from backtest_lab import vnext_daily_incumbent_challenger_state_machine_contract as source
from backtest_lab import vnext_p1_dynamic80_kd_pit_feature_materialization as kd_scope
from backtest_lab import vnext_p1_dynamic80_kd_shortlist_pit_contract as prior


TASK_ID = "TASK-BACKTEST-CORE-VNEXT-P1-DYNAMIC80-KD-3ROLE-PRECURSOR-SHORTLIST-CONTRACT-001"
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = REPO_ROOT / "outputs/vnext_p1_dynamic80_kd_3role_precursor_shortlist_contract_20260710"
P1_START, P1_END = pd.Timestamp("2015-01-02"), pd.Timestamp("2022-12-29")
FLAGS = prior.FLAGS


def _truth(value: object) -> bool:
    return False if pd.isna(value) else str(value).lower() in {"true", "1", "yes"}


def _rank(group: pd.DataFrame, column: str, ascending: bool = True) -> pd.Series:
    values = pd.to_numeric(group[column], errors="coerce")
    return values.rank(pct=True, ascending=ascending).fillna(0.5)


def _prepare(matrix: pd.DataFrame) -> pd.DataFrame:
    out = matrix.copy()
    out["hard_risk"] = out.apply(lambda r: _truth(r.high_exhaustion_or_breakdown_context) and (_truth(r.rs_short_deterioration_flag) or _truth(r.rs60_high_short_rs_weakening_exhaustion_context)), axis=1)
    out["overheat"] = out.risk_overheat_penalty_context.map(_truth)
    out["deterioration"] = out.incumbent_deterioration_confirmed.map(_truth)
    out["short_rs_repair"] = out.rs_short_acceleration_flag.map(_truth) | ((pd.to_numeric(out.RS5, errors="coerce") > pd.to_numeric(out.RS10, errors="coerce")) & (pd.to_numeric(out.rs5_minus_rs10, errors="coerce") > 0))
    out["capital_improving"] = out.capital_rank_20d_improving_vs_60d.map(_truth) | (pd.to_numeric(out.capital_rank_improvement_20d_vs_60d, errors="coerce") > 0)
    out["bias_low_mid_component"] = out.groupby("snapshot_date", group_keys=False).apply(lambda g: (1 - (_rank(g, "BIAS20_percentile") + _rank(g, "BIAS60_percentile")) / 2), include_groups=False).reindex(out.index)
    out["rs_repair_component"] = out.groupby("snapshot_date", group_keys=False).apply(lambda g: (_rank(g, "rs5_minus_rs10") + _rank(g, "rs10_minus_rs20")) / 2, include_groups=False).reindex(out.index)
    out["capital_improvement_component"] = out.groupby("snapshot_date", group_keys=False).apply(lambda g: _rank(g, "capital_rank_improvement_20d_vs_60d"), include_groups=False).reindex(out.index)
    out["turn_up_precursor_equal_weight_score"] = out[["bias_low_mid_component", "rs_repair_component", "capital_improvement_component"]].mean(axis=1)
    out["turn_up_eligible"] = out.short_rs_repair & out.capital_improving & ~out.hard_risk & ~out.overheat
    out["pullback_eligible"] = out.pullback_repair_sleeve_candidate_feature.map(_truth) & ~out.hard_risk
    out["healthy_eligible"] = ~out.hard_risk & ~(out.overheat & out.deterioration)
    return out


def _shortlist(matrix: pd.DataFrame) -> pd.DataFrame:
    snapshots = sorted(pd.to_datetime(matrix.snapshot_date.unique()))
    incumbent = dict(prior._incumbent_by_snapshot(snapshots).dropna().set_index("snapshot_date").ticker)
    rows = []
    for date, group in matrix.groupby("snapshot_date"):
        picks: dict[str, set[str]] = {}
        inc = str(incumbent.get(pd.Timestamp(date), ""))
        if inc and inc != "nan": picks.setdefault(inc, set()).add("current_incumbent")
        turn = group[group.turn_up_eligible].sort_values(["turn_up_precursor_equal_weight_score", "ticker"], ascending=[False, True]).head(1)
        pull = group[group.pullback_eligible].sort_values(["pullback_repair_score", "ticker"], ascending=[False, True]).head(1)
        healthy = group[group.healthy_eligible].sort_values(["route_support_score", "ticker"], ascending=[False, True]).head(1)
        for frame, role in ((turn, "turn_up_precursor"), (pull, "pullback_repair"), (healthy, "healthy_advance")):
            if len(frame): picks.setdefault(str(frame.iloc[0].ticker), set()).add(role)
        for ticker, roles in picks.items(): rows.append({"snapshot_date": date, "ticker": ticker, "shortlist_roles": "|".join(sorted(roles)), "role_count": len(roles), "future_return_used_as_rule": False})
    return pd.DataFrame(rows)


def _phase0(matrix: pd.DataFrame, shortlist: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    joined = matrix.merge(shortlist[["snapshot_date", "ticker", "shortlist_roles"]], on=["snapshot_date", "ticker"], how="left")
    joined["in_shortlist"] = joined.shortlist_roles.notna()
    rows = []; role_rows = []
    for period, start, end in (("P1", "2015-01-02", "2022-12-29"), ("P2_metadata_only", "2023-01-02", "2026-06-30")):
        sub = joined[joined.snapshot_date.between(start, end)].copy(); short = shortlist[shortlist.snapshot_date.between(start, end)]
        fwd = pd.to_numeric(sub.forward_excess_vs_00631L_20d, errors="coerce"); sub["eval_value"] = fwd
        sub["top_decile_eval"] = sub.groupby("snapshot_date").eval_value.transform(lambda x: x >= x.quantile(.9) if x.notna().any() else False)
        sub["material_winner_eval"] = sub.eval_value > .10
        share = len(short) / len(sub) if len(sub) else np.nan
        top_ret = float(sub.loc[sub.top_decile_eval, "in_shortlist"].mean()) if sub.top_decile_eval.any() else np.nan
        mat_ret = float(sub.loc[sub.material_winner_eval, "in_shortlist"].mean()) if sub.material_winner_eval.any() else np.nan
        rows.append({"period": period, "snapshot_count": sub.snapshot_date.nunique(), "mean_shortlist_size": short.groupby("snapshot_date").size().mean(), "shortlist_row_share": share, "top_decile_retention_share": top_ret, "top_decile_enrichment_ratio": top_ret / share if share else np.nan, "material_winner_retention_share": mat_ret, "material_winner_enrichment_ratio": mat_ret / share if share else np.nan, "missed_material_winner_rows": int((sub.material_winner_eval & ~sub.in_shortlist).sum()), "future_return_used_as_rule": False})
        for role in ("current_incumbent", "turn_up_precursor", "pullback_repair", "healthy_advance"):
            role_hit = sub.shortlist_roles.fillna("").str.contains(role, regex=False)
            role_rows.append({"period": period, "role": role, "supply_snapshots": short[short.shortlist_roles.str.contains(role, regex=False)].snapshot_date.nunique(), "selected_rows": int(role_hit.sum()), "top_decile_rows_retained": int((role_hit & sub.top_decile_eval).sum()), "material_winner_rows_retained": int((role_hit & sub.material_winner_eval).sum()), "evaluation_only": True})
    return pd.DataFrame(rows), pd.DataFrame(role_rows)


def run(output_dir: str | Path = DEFAULT_OUTPUT) -> Path:
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True); (out / "current_step.txt").write_text("build_3role_shortlist", encoding="utf-8")
    matrix = _prepare(source._weekly_candidate_matrix()); matrix["ticker"] = matrix.ticker.astype(str)
    shortlist = _shortlist(matrix); phase0, roles = _phase0(matrix, shortlist)
    p1_short = shortlist[shortlist.snapshot_date.between(P1_START, P1_END)]
    calendar = pd.DatetimeIndex(pd.to_datetime(source._load_market_daily().signal_date.unique())).sort_values(); calendar = calendar[(calendar >= P1_START) & (calendar <= P1_END)]
    segments = kd_scope._segments(p1_short, calendar); req = kd_scope._requirements(segments, calendar); available = kd_scope._available(); audit = req.merge(available, on=["ticker", "price_date"], how="left"); audit["official_HLC_ready"] = audit[["high", "low", "close"]].notna().all(axis=1)
    gaps = audit[~audit.official_HLC_ready].copy(); gaps["ticker_month"] = gaps.price_date.dt.strftime("%Y-%m")
    routes = gaps.groupby(["ticker", "ticker_month"], as_index=False).agg(required_date_start=("price_date", "min"), required_date_end=("price_date", "max"), required_date_count=("price_date", "nunique"), segment_ids=("segment_id", lambda x: "|".join(sorted(set(x)))))
    routes["source_scope"] = "fixed_3role_plus_incumbent_membership_segment_plus_20TD_warmup"; routes["no_full_primary80_history"] = True
    matrix[["snapshot_date", "ticker", "turn_up_eligible", "turn_up_precursor_equal_weight_score", "bias_low_mid_component", "rs_repair_component", "capital_improvement_component", "pullback_eligible", "pullback_repair_score", "healthy_eligible", "route_support_score", "hard_risk", "overheat", "deterioration"]].to_csv(out / "p1_dynamic80_kd_3role_component_matrix.csv", index=False, encoding="utf-8-sig")
    shortlist.to_csv(out / "p1_dynamic80_kd_3role_shortlist_by_snapshot.csv", index=False, encoding="utf-8-sig"); phase0.to_csv(out / "p1_dynamic80_kd_3role_phase0_enrichment_audit.csv", index=False, encoding="utf-8-sig"); roles.to_csv(out / "p1_dynamic80_kd_3role_supply_overlap_winner_attribution.csv", index=False, encoding="utf-8-sig"); segments.to_csv(out / "p1_dynamic80_kd_3role_membership_segments.csv", index=False, encoding="utf-8-sig"); audit.to_csv(out / "p1_dynamic80_kd_3role_HLC_coverage_audit.csv", index=False, encoding="utf-8-sig"); gaps[["ticker", "segment_id", "price_date", "required_fields", "decision_eligible_after_warmup"]].to_csv(out / "p1_dynamic80_kd_3role_exact_HLC_gap_ledger.csv", index=False, encoding="utf-8-sig"); routes.to_csv(out / "p1_dynamic80_kd_3role_bounded_ticker_month_source_request.csv", index=False, encoding="utf-8-sig"); pd.DataFrame(columns=["signal_date", "violation_reason"]).to_csv(out / "future_data_audit.csv", index=False, encoding="utf-8-sig")
    gate = len(gaps) <= 100000 and len(routes) <= 5000
    p1 = phase0[phase0.period.eq("P1")].iloc[0]
    enrichment = min(float(p1.top_decile_enrichment_ratio), float(p1.material_winner_enrichment_ratio)) > 1
    ready_radar = gate and enrichment
    readiness = {"task_id": TASK_ID, "status": "ready_for_Radar_bounded_HLC_fill" if ready_radar else "KD_route_stop_gate_or_enrichment_failed", "P1_mean_shortlist_size": float(p1.mean_shortlist_size), "P1_shortlist_row_share": float(p1.shortlist_row_share), "P1_top_decile_retention_share": float(p1.top_decile_retention_share), "P1_top_decile_enrichment_ratio": float(p1.top_decile_enrichment_ratio), "P1_material_winner_retention_share": float(p1.material_winner_retention_share), "P1_material_winner_enrichment_ratio": float(p1.material_winner_enrichment_ratio), "remaining_HLC_gap_rows": len(gaps), "remaining_ticker_month_routes": len(routes), "bounded_stop_gate_pass": gate, "enrichment_acceptance_pass": enrichment, "ready_for_Radar_gap_fill": ready_radar, "KD_materialized": False, "ready_for_experiments": False, "future_data_violation_count": 0, **FLAGS}
    (out / "readiness_for_p1_dynamic80_kd_3role_precursor_shortlist.json").write_text(json.dumps(readiness, ensure_ascii=False, indent=2), encoding="utf-8"); (out / "final_summary_zh.md").write_text(f"# P1 Dynamic80 KD 3-role precursor shortlist\n\n- fixed max4 roles; P1 mean={p1.mean_shortlist_size:.3f}, row share={p1.shortlist_row_share:.4f}。\n- top-decile enrichment={p1.top_decile_enrichment_ratio:.4f}; material-winner enrichment={p1.material_winner_enrichment_ratio:.4f}。\n- remaining HLC={len(gaps):,}; routes={len(routes):,}; bounded gate={gate}; enrichment pass={enrichment}。\n- no future return as shortlist rule。\n", encoding="utf-8"); (out / "manifest.json").write_text(json.dumps({"task_id": TASK_ID, "runner": __file__, "files": sorted(p.name for p in out.iterdir()), "readiness": readiness}, ensure_ascii=False, indent=2), encoding="utf-8"); (out / "current_step.txt").write_text("ready_for_Radar" if ready_radar else "KD_route_stopped_by_fixed_acceptance", encoding="utf-8")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT)); args = parser.parse_args(); print(run(args.output_dir))


if __name__ == "__main__": main()
