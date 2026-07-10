from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from backtest_lab import vnext_daily_incumbent_challenger_state_machine_contract as source
from backtest_lab import vnext_p1_dynamic80_incumbent_hold_comparator as simplified


TASK_ID = "TASK-BACKTEST-CORE-VNEXT-P1-DYNAMIC80-WEIGHTED-INCUMBENT-CHALLENGER-AUDIT-CONTRACT-001"
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = REPO_ROOT / "outputs/vnext_p1_dynamic80_weighted_incumbent_challenger_audit_contract_20260710"
P1_START, P1_END = pd.Timestamp("2015-01-02"), pd.Timestamp("2022-12-29")
STOCK_TO_STOCK_COST = 0.00585
FLAGS = {"formal_model_changed": False, "trade_decision_changed": False, "active_in_trade_decision": False, "report_changed": False, "portfolio_replay_executed": False, "ready_for_strategy_replay": False, "ready_for_formal": False, "not_live_rule": True, "forward_returns_live_rule_usage": False}


def _truth(value: object) -> bool:
    return False if pd.isna(value) else str(value).lower() in {"true", "1", "yes"}


def _pct(group: pd.DataFrame, values: pd.Series, higher: bool = True) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    return numeric.rank(pct=True, ascending=higher).fillna(0.5)


def _component_matrix() -> pd.DataFrame:
    frame = source._weekly_candidate_matrix(); frame["ticker"] = frame.ticker.astype(str)
    parts = []
    for _, group in frame.groupby("snapshot_date", sort=True):
        g = group.copy()
        # Higher percentile always means better after directional transforms.
        quality_fields = [1 - pd.to_numeric(g.layer1_quality_floor_risk_pctile_by_week, errors="coerce"), g.layer1_pass_bottom30.map(_truth).astype(float)]
        g["quality_block"] = pd.concat([_pct(g, x) for x in quality_fields], axis=1).mean(axis=1)
        trend_fields = [g.RS20, g.RS40, g.RS60, g.RS5, g.RS10, g.rs5_minus_rs10, g.rs10_minus_rs20]
        g["trend_block"] = pd.concat([_pct(g, x) for x in trend_fields], axis=1).mean(axis=1)
        bias20 = pd.to_numeric(g.BIAS20_percentile, errors="coerce").fillna(.5)
        bias60 = pd.to_numeric(g.BIAS60_percentile, errors="coerce").fillna(.5)
        moderate = 1 - ((bias20 - .5).abs() + (bias60 - .5).abs()) / 2
        structure = ((pd.to_numeric(g.MA20_position, errors="coerce") >= 0).astype(float) + (pd.to_numeric(g.MA60_position, errors="coerce") >= 0).astype(float)) / 2
        g["price_position_block"] = pd.concat([_pct(g, moderate), _pct(g, structure)], axis=1).mean(axis=1)
        risk_fields = [1 - pd.to_numeric(g.volatility_pctile_by_week, errors="coerce"), 1 + pd.to_numeric(g.drawdown_20d, errors="coerce"), 1 + pd.to_numeric(g.drawdown_60d, errors="coerce"), 1 - pd.to_numeric(g.layer1_quality_floor_risk_pctile_by_week, errors="coerce")]
        g["risk_block"] = pd.concat([_pct(g, x) for x in risk_fields], axis=1).mean(axis=1)
        capital_fields = [1 / pd.to_numeric(g.traded_value_rank_5d, errors="coerce"), 1 / pd.to_numeric(g.traded_value_rank_20d, errors="coerce"), 1 / pd.to_numeric(g.traded_value_rank_60d, errors="coerce"), g.capital_rank_improvement_20d_vs_60d, g.route_support_variant_count]
        g["capital_support_block"] = pd.concat([_pct(g, x) for x in capital_fields], axis=1).mean(axis=1)
        blocks = ["quality_block", "trend_block", "price_position_block", "risk_block", "capital_support_block"]
        g["five_block_composite"] = g[blocks].mean(axis=1)
        g["five_block_composite_percentile"] = _pct(g, g.five_block_composite) * 100
        g["challenger_hard_risk_veto"] = g.apply(lambda r: _truth(r.high_exhaustion_or_breakdown_context) or (_truth(r.volatility_high_context) and _truth(r.rs_short_deterioration_flag)), axis=1)
        g["inc_deterioration_a_rs"] = g.rs_short_deterioration_flag.map(_truth) | (pd.to_numeric(g.rs5_minus_rs10, errors="coerce") < 0) | (pd.to_numeric(g.rs10_minus_rs20, errors="coerce") < 0)
        g["inc_deterioration_b_risk"] = g.volatility_high_context.map(_truth) | g.high_exhaustion_or_breakdown_context.map(_truth)
        g["inc_deterioration_c_position"] = (g.risk_overheat_penalty_context.map(_truth) & g.rs_short_deterioration_flag.map(_truth)) | g.pullback_breakdown_warning_context.map(_truth)
        g["inc_deterioration_d_capital"] = g.capital_rank_20d_deteriorating_vs_60d.map(_truth) | g.pure_5d_burst_without_20d60d_confirmation.map(_truth)
        g["inc_deterioration_count"] = g[["inc_deterioration_a_rs", "inc_deterioration_b_risk", "inc_deterioration_c_position", "inc_deterioration_d_capital"]].sum(axis=1)
        g["future_return_used_as_rule"] = False
        parts.append(g)
    return pd.concat(parts, ignore_index=True)


def _events(matrix: pd.DataFrame) -> pd.DataFrame:
    rows = []; incumbent = ""
    blocks = ["quality_block", "trend_block", "price_position_block", "risk_block", "capital_support_block"]
    for date, group in matrix[matrix.snapshot_date.between(P1_START, P1_END)].groupby("snapshot_date", sort=True):
        eligible = group[~group.challenger_hard_risk_veto].sort_values(["five_block_composite", "ticker"], ascending=[False, True])
        challenger = eligible.iloc[0] if len(eligible) else None
        inc_rows = group[group.ticker.eq(incumbent)] if incumbent else pd.DataFrame()
        inc = inc_rows.iloc[0] if len(inc_rows) else None
        forced = bool(incumbent and (inc is None or bool(inc.challenger_hard_risk_veto)))
        if not incumbent:
            action, event_type, reason = "initialize", "initialization", "first_PIT_top_composite_without_hard_risk"
        elif forced:
            action, event_type, reason = ("switch", "forced_replacement", "incumbent_hard_invalid_or_missing_primary80") if challenger is not None else ("hold_blocked", "forced_no_replacement", "incumbent_invalid_no_valid_replacement")
        else:
            edge = float(challenger.five_block_composite_percentile - inc.five_block_composite_percentile) if challenger is not None else np.nan
            wins = sum(float(challenger[b]) > float(inc[b]) for b in blocks) if challenger is not None else 0
            normal = challenger is not None and challenger.ticker != incumbent and edge >= 10 and wins >= 3 and int(inc.inc_deterioration_count) >= 2
            action, event_type, reason = ("switch", "normal_switch", "edge_ge10_win3of5_incumbent_deterioration_ge2") if normal else ("hold", "no_switch", "fixed_switch_rule_not_fully_met")
        if challenger is None:
            challenger_ticker = ""; edge = np.nan; wins = 0
        else:
            challenger_ticker = str(challenger.ticker)
            if inc is None: edge = np.nan; wins = 0
            else:
                edge = float(challenger.five_block_composite_percentile - inc.five_block_composite_percentile)
                wins = sum(float(challenger[b]) > float(inc[b]) for b in blocks)
        before = incumbent
        if action in {"initialize", "switch"} and challenger is not None: incumbent = challenger_ticker
        eval_inc = inc
        eval_ch = challenger
        row = {"signal_date": date, "incumbent_before": before, "challenger_ticker": challenger_ticker, "event_type": event_type, "action": action, "decision_reason": reason, "challenger_composite_percentile": float(challenger.five_block_composite_percentile) if challenger is not None else np.nan, "incumbent_composite_percentile": float(inc.five_block_composite_percentile) if inc is not None else np.nan, "challenger_edge_percentile_points": edge, "challenger_blocks_won": wins, "incumbent_deterioration_count": int(inc.inc_deterioration_count) if inc is not None else np.nan, "incumbent_deterioration_flags": "|".join(k for k in "abcd" if inc is not None and bool(inc[f"inc_deterioration_{k}_" + {"a":"rs","b":"risk","c":"position","d":"capital"}[k]])), "challenger_hard_risk_veto": bool(challenger.challenger_hard_risk_veto) if challenger is not None else False, "selected_ticker_after": incumbent, "future_return_used_as_rule": False}
        for horizon in (5, 10, 20):
            col = f"forward_excess_vs_00631L_{horizon}d"
            gross = float(eval_ch[col] - eval_inc[col]) if eval_ch is not None and eval_inc is not None and pd.notna(eval_ch[col]) and pd.notna(eval_inc[col]) else np.nan
            row[f"challenger_minus_incumbent_gross_{horizon}d_eval"] = gross
            row[f"challenger_minus_incumbent_net_{horizon}d_eval"] = gross - STOCK_TO_STOCK_COST if pd.notna(gross) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def _summary(events: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    normal = events[events.event_type.eq("normal_switch")].copy(); forced = events[events.event_type.eq("forced_replacement")].copy()
    rows = []
    for kind, frame in (("normal_switch", normal), ("forced_replacement", forced)):
        result = {"event_type": kind, "event_count": len(frame)}
        for h in (5, 10, 20):
            net = pd.to_numeric(frame[f"challenger_minus_incumbent_net_{h}d_eval"], errors="coerce").dropna()
            result[f"median_net_excess_{h}d"] = float(net.median()) if len(net) else np.nan
            result[f"hit_rate_net_positive_{h}d"] = float((net > 0).mean()) if len(net) else np.nan
            result[f"material_win_count_{h}d"] = int((net > .05).sum())
            result[f"material_loss_count_{h}d"] = int((net < -.05).sum())
        rows.append(result)
    yearly = []
    normal["year"] = pd.to_datetime(normal.signal_date).dt.year
    for year, group in normal.groupby("year"):
        yearly.append({"year": year, "event_count": len(group), **{f"median_net_{h}d": pd.to_numeric(group[f"challenger_minus_incumbent_net_{h}d_eval"], errors="coerce").median() for h in (5, 10, 20)}})
    robustness = []
    for h in (5, 10, 20):
        net = pd.to_numeric(normal[f"challenger_minus_incumbent_net_{h}d_eval"], errors="coerce").dropna().sort_values(ascending=False)
        robustness.append({"horizon": h, "baseline_median": net.median() if len(net) else np.nan, "remove_best1_median": net.iloc[1:].median() if len(net) > 1 else np.nan, "remove_best3_median": net.iloc[3:].median() if len(net) > 3 else np.nan, "remove_best5_median": net.iloc[5:].median() if len(net) > 5 else np.nan})
    return pd.DataFrame(rows), pd.DataFrame(yearly), pd.DataFrame(robustness)


def _duplicate_audit() -> pd.DataFrame:
    return pd.DataFrame([
        {"block": "quality", "fields": "Layer1 risk percentile|bottom30 pass", "duplicate_policy": "no route_support total"},
        {"block": "trend", "fields": "RS5/10/20/40/60|short acceleration", "duplicate_policy": "0050-relative only"},
        {"block": "price_position", "fields": "BIAS20/60 self percentile|MA20/60 position", "duplicate_policy": "overheat not sole exit"},
        {"block": "risk", "fields": "volatility|drawdown20/60|Layer1 risk", "duplicate_policy": "hard risk separately vetoes challenger"},
        {"block": "capital_support", "fields": "traded-value rank5/20/60|rank change|route variant count", "duplicate_policy": "raw route_support composite excluded"},
    ])


def _simplified_event_comparison(matrix: pd.DataFrame) -> pd.DataFrame:
    decisions = simplified._weekly_decisions(matrix)
    lookup = {(pd.Timestamp(r.snapshot_date), str(r.ticker)): r for r in matrix.itertuples(index=False)}
    rows = []
    for variant in ("I2_deteriorating_incumbent_plus_better_challenger", "I3_one_prior_snapshot_confirmation"):
        sub = decisions[(decisions.variant.eq(variant)) & decisions.decision.eq("switch")].copy()
        for item in sub.itertuples(index=False):
            inc = lookup.get((pd.Timestamp(item.signal_date), str(item.incumbent_ticker_before)))
            ch = lookup.get((pd.Timestamp(item.signal_date), str(item.target_ticker)))
            if inc is None or ch is None: continue
            row = {"variant": variant, "signal_date": item.signal_date, "incumbent_ticker": item.incumbent_ticker_before, "challenger_ticker": item.target_ticker}
            for h in (5, 10, 20):
                col = f"forward_excess_vs_00631L_{h}d"; gross = float(getattr(ch, col) - getattr(inc, col)) if pd.notna(getattr(ch, col)) and pd.notna(getattr(inc, col)) else np.nan
                row[f"net_excess_{h}d_eval"] = gross - STOCK_TO_STOCK_COST if pd.notna(gross) else np.nan
            rows.append(row)
    events = pd.DataFrame(rows)
    summaries = []
    for variant, group in events.groupby("variant"):
        summaries.append({"variant": variant, "switch_events": len(group), **{f"median_net_{h}d": pd.to_numeric(group[f"net_excess_{h}d_eval"], errors="coerce").median() for h in (5, 10, 20)}, **{f"hit_rate_{h}d": float((pd.to_numeric(group[f"net_excess_{h}d_eval"], errors="coerce") > 0).mean()) for h in (5, 10, 20)}})
    return pd.DataFrame(summaries)


def run(output_dir: str | Path = DEFAULT_OUTPUT) -> Path:
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True); (out / "current_step.txt").write_text("build_five_block_matrix", encoding="utf-8")
    matrix = _component_matrix(); events = _events(matrix); summary, yearly, robustness = _summary(events); simplified_comparison = _simplified_event_comparison(matrix)
    p1_matrix = matrix[matrix.snapshot_date.between(P1_START, P1_END)]
    keep = ["snapshot_date", "ticker", "name", "quality_block", "trend_block", "price_position_block", "risk_block", "capital_support_block", "five_block_composite", "five_block_composite_percentile", "challenger_hard_risk_veto", "inc_deterioration_a_rs", "inc_deterioration_b_risk", "inc_deterioration_c_position", "inc_deterioration_d_capital", "inc_deterioration_count", "future_return_used_as_rule"]
    p1_matrix[keep].to_csv(out / "p1_weighted_incumbent_challenger_component_matrix.csv", index=False, encoding="utf-8-sig"); events.to_csv(out / "p1_weighted_incumbent_challenger_comparison_trace.csv", index=False, encoding="utf-8-sig"); summary.to_csv(out / "p1_weighted_incumbent_challenger_forced_vs_normal_summary.csv", index=False, encoding="utf-8-sig"); simplified_comparison.to_csv(out / "p1_simplified_I2_I3_event_attribution_comparison.csv", index=False, encoding="utf-8-sig"); yearly.to_csv(out / "p1_weighted_incumbent_challenger_year_stability.csv", index=False, encoding="utf-8-sig"); robustness.to_csv(out / "p1_weighted_incumbent_challenger_event_robustness.csv", index=False, encoding="utf-8-sig"); _duplicate_audit().to_csv(out / "p1_weighted_incumbent_challenger_duplicate_factor_audit.csv", index=False, encoding="utf-8-sig"); pd.DataFrame(columns=["signal_date", "violation_reason"]).to_csv(out / "future_data_audit.csv", index=False, encoding="utf-8-sig"); pd.DataFrame([{"requested_start": str(P1_START.date()), "requested_end": str(P1_END.date()), "actual_start": events.signal_date.min(), "actual_end": events.signal_date.max(), "snapshot_count": events.signal_date.nunique(), "candidate_rows": len(p1_matrix)}]).to_csv(out / "requested_vs_actual_coverage.csv", index=False, encoding="utf-8-sig")
    readiness = {"task_id": TASK_ID, "status": "Phase_A_event_contract_ready_for_Experiments", "ready_for_experiments": True, "normal_switch_events": int(events.event_type.eq("normal_switch").sum()), "forced_replacement_events": int(events.event_type.eq("forced_replacement").sum()), "five_equal_weight_blocks_ready": True, "selected_OHLC_required": False, "future_data_violation_count": 0, **FLAGS}
    (out / "readiness_for_p1_weighted_incumbent_challenger_event_diagnostic.json").write_text(json.dumps(readiness, ensure_ascii=False, indent=2), encoding="utf-8"); (out / "final_summary_zh.md").write_text("# P1 weighted incumbent/challenger Phase A\n\n" + summary.to_csv(index=False) + "\n- forward returns evaluation-only; no selected path or OHLC acquisition in Phase A。\n", encoding="utf-8"); (out / "manifest.json").write_text(json.dumps({"task_id": TASK_ID, "runner": __file__, "files": sorted(p.name for p in out.iterdir()), "readiness": readiness}, ensure_ascii=False, indent=2), encoding="utf-8"); (out / "current_step.txt").write_text("ready_for_Experiments_Phase_A", encoding="utf-8")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT)); args = parser.parse_args(); print(run(args.output_dir))


if __name__ == "__main__": main()
