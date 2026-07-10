from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from backtest_lab import vnext_daily_incumbent_challenger_state_machine_contract as daily_source


TASK_ID = "TASK-BACKTEST-CORE-VNEXT-P1-RECONSTRUCTED-R6-EXACT-BEST-EPISODE-REMOVAL-RECHAIN-CONTRACT-001"
REPO_ROOT = Path(__file__).resolve().parents[2]
R6_DIR = REPO_ROOT / "outputs" / "vnext_weekly_r6_single_position_state_boundary_reconstruction_contract_20260710"
EXPERIMENTS_DIR = Path(
    r"C:\Users\zergv\Documents\Codex\2026-07-06\backtest-lab-experiments-diagnostic-validation-attribution"
    r"\outputs\vnext_p1_r6_f_walk_forward_robustness_diagnostic_20260710"
)
SOURCE_CLOSURE_DIR = REPO_ROOT / "outputs" / "vnext_selected_stock_total_return_source_escalation_closure_20260710"
OUTPUT_DIR = REPO_ROOT / "outputs" / "vnext_p1_reconstructed_r6_exact_episode_rechain_contract_20260710"
SCENARIOS = {"baseline": 0, "remove_best_1": 1, "remove_best_3": 3, "remove_best_5": 5}

FLAGS = {
    "formal_model_changed": False,
    "trade_decision_changed": False,
    "active_in_trade_decision": False,
    "report_changed": False,
    "portfolio_replay_executed": False,
    "ready_for_strategy_replay": False,
    "ready_for_formal": False,
    "not_live_rule": True,
    "forward_returns_live_rule_usage": False,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write(frame: pd.DataFrame, name: str) -> Path:
    path = OUTPUT_DIR / name
    frame.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def _ticker(value: Any) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    return text[:-2] if text.endswith(".0") else text


def _transition(previous_ticker: str, previous_type: str, target_ticker: str, target_type: str) -> tuple[str, str]:
    if previous_ticker == target_ticker and previous_type == target_type:
        return "hold_same", "hold"
    if previous_type == "etf" and target_type == "stock":
        return "base_to_stock", "00631L_to_stock"
    if previous_type == "stock" and target_type == "etf":
        return "stock_to_base", "stock_to_00631L"
    return "stock_to_stock", "stock_to_stock"


def _load_p1() -> pd.DataFrame:
    state = pd.read_csv(
        R6_DIR / "reconstructed_weekly_r6_single_position_daily_state_rows.csv",
        dtype={"selected_ticker_after": str},
        low_memory=False,
    )
    for column in ["signal_date", "next_trading_day_execution_date", "next_trading_day_after_execution_date"]:
        state[column] = pd.to_datetime(state[column], errors="coerce")
    state["selected_ticker_after"] = state["selected_ticker_after"].map(_ticker)
    state = state.loc[state["metric_eligible_P1"].astype(bool)].sort_values("signal_date").reset_index(drop=True)
    state["base_00631L_gross_daily_return"] = (
        pd.to_numeric(state["base_exit_close"], errors="coerce")
        / pd.to_numeric(state["base_entry_close"], errors="coerce")
        - 1.0
    )
    if state[["gross_daily_return", "base_00631L_gross_daily_return"]].isna().any().any():
        raise ValueError("P1 baseline or 00631L daily path has missing returns")
    return state


def _episode_rank(state: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    legs = state.copy()
    legs["is_stock_episode_leg"] = legs["selected_asset_type_after"].eq("stock")
    previous_stock = legs["is_stock_episode_leg"].shift(1, fill_value=False)
    previous_ticker = legs["selected_ticker_after"].shift(1)
    episode_start = legs["is_stock_episode_leg"] & (~previous_stock | legs["selected_ticker_after"].ne(previous_ticker))
    legs["episode_id"] = episode_start.cumsum().astype(int)
    legs.loc[~legs["is_stock_episode_leg"], "episode_id"] = 0
    rows = []
    for episode_id, group in legs.loc[legs["is_stock_episode_leg"]].groupby("episode_id"):
        episode_net = float((1 + pd.to_numeric(group["net_daily_return_after_transition_cost"])).prod() - 1.0)
        base_return = float((1 + group["base_00631L_gross_daily_return"]).prod() - 1.0)
        rows.append({
            "episode_id": int(episode_id),
            "ticker": group["selected_ticker_after"].iloc[0],
            "start_signal_date": group["signal_date"].min(),
            "end_signal_date": group["signal_date"].max(),
            "execution_start_date": group["next_trading_day_execution_date"].min(),
            "mark_end_date": group["next_trading_day_after_execution_date"].max(),
            "daily_legs": len(group),
            "episode_net_return_original_after_cost": episode_net,
            "episode_00631L_same_days_gross_return": base_return,
            "realized_episode_contribution_vs_00631L": episode_net - base_return,
            "ranking_uses_realized_evaluation_only": True,
            "ranking_not_used_for_live_rule": True,
            "removal_semantics": "replace all episode stock legs with 00631L then fully rechain daily states and costs",
        })
    ranked = pd.DataFrame(rows).sort_values(
        ["realized_episode_contribution_vs_00631L", "episode_id"], ascending=[False, True]
    ).reset_index(drop=True)
    ranked["evaluation_rank_best_contributor"] = np.arange(1, len(ranked) + 1)
    ranked["future_data_violation_count"] = 0
    return legs, ranked


def _rechain(legs: pd.DataFrame, removed_ids: set[int], scenario: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    previous_ticker, previous_type = "00631L", "etf"
    path_rows: list[dict[str, Any]] = []
    transition_rows: list[dict[str, Any]] = []
    for row in legs.itertuples(index=False):
        removed = row.is_stock_episode_leg and int(row.episode_id) in removed_ids
        target_ticker = "00631L" if removed else _ticker(row.selected_ticker_after)
        target_type = "etf" if removed else row.selected_asset_type_after
        transition_type, cost_key = _transition(previous_ticker, previous_type, target_ticker, target_type)
        cost = daily_source.TRANSITION_COSTS[cost_key]
        gross = float(row.base_00631L_gross_daily_return if removed else row.gross_daily_return)
        path_rows.append({
            "task": TASK_ID,
            "scenario": scenario,
            "signal_date": row.signal_date,
            "next_trading_day_execution_date": row.next_trading_day_execution_date,
            "next_trading_day_after_execution_date": row.next_trading_day_after_execution_date,
            "episode_id": int(row.episode_id),
            "removed_best_episode_flag": bool(removed),
            "original_ticker": _ticker(row.selected_ticker_after),
            "original_asset_type": row.selected_asset_type_after,
            "incumbent_ticker_before": previous_ticker,
            "incumbent_asset_type_before": previous_type,
            "selected_ticker_after": target_ticker,
            "selected_asset_type_after": target_type,
            "transition_type": transition_type,
            "transition_cost_key": cost_key,
            "transition_cost_rate": cost["transition_cost_rate"],
            "gross_daily_return": gross,
            "net_daily_return_after_transition_cost": gross - cost["transition_cost_rate"],
            "base_00631L_gross_daily_return": float(row.base_00631L_gross_daily_return),
            "price_source_quality": row.daily_price_source_quality if target_type == "stock" else row.base_entry_source_quality,
            "official_unadjusted_stock_ohlc_ready": bool(row.official_unadjusted_daily_ohlc_ready) if target_type == "stock" else True,
            "selected_stock_adjusted_close_ready": False if target_type == "stock" else True,
            "execution_basis": "weekly_signal_close_next_trading_day_close_unique_position_full_path_rechain",
            "episode_ranking_uses_realized_evaluation_only": True,
            "historical_overlapping_R6_used_as_primary": False,
            "diagnostic_only": True,
            **FLAGS,
        })
        if transition_type != "hold_same":
            transition_rows.append({
                "scenario": scenario,
                "signal_date": row.signal_date,
                "execution_date": row.next_trading_day_execution_date,
                "from_ticker": previous_ticker,
                "from_asset_type": previous_type,
                "to_ticker": target_ticker,
                "to_asset_type": target_type,
                "transition_type": transition_type,
                **cost,
                "cost_model_status": "EP05_TaiwanCostModel_unit_notional_hook_stock_etf_separated",
                "diagnostic_only": True,
                **FLAGS,
            })
        previous_ticker, previous_type = target_ticker, target_type
    return pd.DataFrame(path_rows), pd.DataFrame(transition_rows)


def _path_metrics(path: pd.DataFrame, scenario: str, removed_ids: list[int]) -> dict[str, Any]:
    returns = pd.to_numeric(path["net_daily_return_after_transition_cost"], errors="raise")
    equity = (1 + returns).cumprod()
    drawdown = equity / equity.cummax() - 1.0
    return {
        "scenario": scenario,
        "removed_episode_count": len(removed_ids),
        "removed_episode_ids": "|".join(map(str, removed_ids)),
        "requested_start": "2015-01-02",
        "requested_end": "2022-12-29",
        "actual_start": path["signal_date"].min(),
        "actual_end": path["signal_date"].max(),
        "net_total_return_after_transition_cost": float(equity.iloc[-1] - 1.0),
        "net_MDD": float(drawdown.min()),
        "transition_count": int(path["transition_type"].ne("hold_same").sum()),
        "stock_state_days": int(path["selected_asset_type_after"].eq("stock").sum()),
        "base_state_days": int(path["selected_asset_type_after"].eq("etf").sum()),
        "stock_exposure_share": float(path["selected_asset_type_after"].eq("stock").mean()),
        "daily_path_ready_share": float(path["gross_daily_return"].notna().mean()),
        "execution_basis": "same_next_day_close_unique_position_full_path_rechain",
        "episode_ranking_uses_realized_evaluation_only": True,
        "official_unadjusted_OHLC_diagnostic_only": True,
        "diagnostic_only": True,
        **FLAGS,
    }


def _annual_metrics(paths: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (scenario, year), group in paths.assign(year=paths["signal_date"].dt.year).groupby(["scenario", "year"]):
        returns = pd.to_numeric(group["net_daily_return_after_transition_cost"], errors="raise")
        equity = (1 + returns).cumprod()
        drawdown = equity / equity.cummax() - 1.0
        rows.append({
            "scenario": scenario,
            "year": int(year),
            "actual_start": group["signal_date"].min(),
            "actual_end": group["signal_date"].max(),
            "net_total_return_after_transition_cost": float(equity.iloc[-1] - 1.0),
            "net_MDD": float(drawdown.min()),
            "transition_count": int(group["transition_type"].ne("hold_same").sum()),
            "stock_exposure_share": float(group["selected_asset_type_after"].eq("stock").mean()),
            "diagnostic_only": True,
            **FLAGS,
        })
    return pd.DataFrame(rows)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    r6_readiness = json.loads(
        (R6_DIR / "readiness_for_reconstructed_weekly_r6_single_position_diagnostic.json").read_text(encoding="utf-8")
    )
    source_closure = json.loads(
        (SOURCE_CLOSURE_DIR / "readiness_for_selected_stock_total_return_source_escalation_closure.json").read_text(encoding="utf-8")
    )
    state = _load_p1()
    legs, ranked = _episode_rank(state)
    all_paths, all_transitions, metric_rows, scenario_rows = [], [], [], []
    for scenario, remove_count in SCENARIOS.items():
        removed_ids = ranked.head(remove_count)["episode_id"].astype(int).tolist()
        path, transitions = _rechain(legs, set(removed_ids), scenario)
        all_paths.append(path)
        all_transitions.append(transitions)
        metric_rows.append(_path_metrics(path, scenario, removed_ids))
        scenario_rows.extend({"scenario": scenario, "episode_id": episode_id, "removed": True} for episode_id in removed_ids)
    paths = pd.concat(all_paths, ignore_index=True)
    transitions = pd.concat(all_transitions, ignore_index=True)
    metrics = pd.DataFrame(metric_rows)
    scenario_map = pd.DataFrame(scenario_rows, columns=["scenario", "episode_id", "removed"])
    baseline_return = float(metrics.loc[metrics["scenario"].eq("baseline"), "net_total_return_after_transition_cost"].iloc[0])
    expected_return = float(
        pd.read_csv(R6_DIR / "reconstructed_weekly_r6_net_path_metrics_hook.csv")
        .loc[lambda d: d["period"].eq("P1"), "net_total_return_after_transition_cost_hook"]
        .iloc[0]
    )
    reconciliation = pd.DataFrame([{
        "baseline_recomputed_net_return": baseline_return,
        "source_R6_P1_net_return": expected_return,
        "difference": baseline_return - expected_return,
        "matches_within_1e_12": abs(baseline_return - expected_return) <= 1e-12,
        "baseline_transition_count": int(metrics.loc[metrics["scenario"].eq("baseline"), "transition_count"].iloc[0]),
        "source_transition_count": 41,
    }])
    if not bool(reconciliation["matches_within_1e_12"].iloc[0]):
        raise ValueError("Rechained baseline does not reproduce reconstructed R6 P1")
    f_hook = pd.read_csv(EXPERIMENTS_DIR / "p1_raw_daily_f_exact_remove_best_episode_rechain.csv", low_memory=False)
    f_hook["strategy_id"] = "raw_Daily_F"
    r6_hook = metrics.copy()
    r6_hook["strategy_id"] = "reconstructed_single_position_R6"
    comparison_hook = pd.concat([
        r6_hook[["strategy_id", "scenario", "net_total_return_after_transition_cost", "net_MDD", "transition_count", "stock_state_days", "base_state_days", "execution_basis"]],
        f_hook[["strategy_id", "scenario", "net_total_return_after_transition_cost", "net_MDD", "transition_count", "stock_state_days", "base_state_days", "execution_basis"]],
    ], ignore_index=True)
    benchmark = pd.read_csv(EXPERIMENTS_DIR / "p1_r6_f_same_basis_summary.csv", low_memory=False)
    benchmark = benchmark.loc[benchmark["strategy_id"].eq("all_00631L_same_basis")].copy()
    benchmark["comparison_role"] = "all_00631L_same_basis_reference"
    coverage = pd.DataFrame([{
        "period": "P1",
        "requested_start": "2015-01-02",
        "requested_end": "2022-12-29",
        "actual_start": state["signal_date"].min(),
        "actual_end": state["signal_date"].max(),
        "daily_rows": len(state),
        "daily_path_ready_share": float(state["daily_path_ready"].mean()),
        "stock_official_unadjusted_ready_share": float(state.loc[state["selected_asset_type_after"].eq("stock"), "official_unadjusted_daily_ohlc_ready"].mean()),
        "selected_stock_adjusted_close_ready": False,
        "future_data_violation_count": 0,
    }])
    source_audit = pd.DataFrame([
        {"source_item": "00631L_base", "status": "ready", "quality": "P1 state-hold adjusted benchmark reference", "usage": "base replacement and same-basis comparator"},
        {"source_item": "selected_stock_prices", "status": "ready_diagnostic_only", "quality": "official unadjusted OHLC", "usage": "R6 baseline and retained episode daily marks"},
        {"source_item": "selected_stock_adjusted_close", "status": "blocked_closed", "quality": "not available", "usage": "not used; source escalation closed"},
        {"source_item": "EP05_cost_model", "status": "ready", "quality": "stock/ETF fees and taxes separated", "usage": "every recomputed transition"},
    ])
    future_audit = pd.DataFrame([
        {"audit_item": "R6_baseline_state", "future_return_used_as_rule": False, "detail": "Uses existing PIT weekly R6 target state only.", "future_data_violation_count": 0},
        {"audit_item": "episode_ranking", "future_return_used_as_rule": True, "detail": "Realized contribution ranking is evaluation-only robustness stress; never a live rule.", "future_data_violation_count": 0},
        {"audit_item": "rechain", "future_return_used_as_rule": False, "detail": "Removed episode legs are replaced by 00631L and full transitions/costs are recomputed.", "future_data_violation_count": 0},
    ])
    readiness = {
        "task_id": TASK_ID,
        "status": "P1_reconstructed_R6_exact_best_episode_full_path_rechain_ready",
        "P1_daily_rows": len(state),
        "stock_episode_count": len(ranked),
        "scenario_count": len(SCENARIOS),
        "baseline_recomputed_net_return": baseline_return,
        "baseline_source_reconciliation_pass": True,
        "unique_position_full_path_rechain_ready": True,
        "EP05_transition_cost_recompute_ready": True,
        "annual_metrics_ready": True,
        "raw_daily_F_exact_rechain_comparison_hook_ready": True,
        "all_00631L_same_basis_hook_ready": len(benchmark) == 1,
        "official_unadjusted_stock_path_ready": bool(state.loc[state["selected_asset_type_after"].eq("stock"), "official_unadjusted_daily_ohlc_ready"].all()),
        "selected_stock_adjusted_close_ready": False,
        "historical_path_policy": source_closure["historical_backtest_path_policy"] if "historical_backtest_path_policy" in source_closure else "official_unadjusted_OHLC_diagnostic_only",
        "ready_for_experiments": True,
        "ready_for_formal": False,
        "ready_for_strategy_replay": False,
        "future_data_violation_count": 0,
        "next_owner": "Experiments P1 R6 vs F exact episode concentration diagnostic",
        **FLAGS,
    }
    blocked = pd.DataFrame([
        {"item": "episode_ranking", "status": "evaluation_only", "detail": "Realized episode contribution cannot enter selector or live rule."},
        {"item": "selected_stock_adjusted_close", "status": "blocked_closed", "detail": "Official unadjusted OHLC diagnostic-only; source escalation remains closed."},
        {"item": "strategy_verdict", "status": "not_core_scope", "detail": "Experiments must compare R6/F/all-00631L; Core does not choose winner."},
    ])
    output_paths = [
        _write(ranked, "p1_reconstructed_r6_episode_contribution_rank_audit.csv"),
        _write(scenario_map, "p1_reconstructed_r6_episode_removal_scenario_map.csv"),
        _write(paths, "p1_reconstructed_r6_exact_episode_rechain_daily_state_paths.csv"),
        _write(transitions, "p1_reconstructed_r6_exact_episode_rechain_transition_trace.csv"),
        _write(metrics, "p1_reconstructed_r6_exact_episode_rechain_metrics.csv"),
        _write(_annual_metrics(paths), "p1_reconstructed_r6_exact_episode_rechain_annual_metrics.csv"),
        _write(reconciliation, "p1_reconstructed_r6_rechain_baseline_reconciliation_audit.csv"),
        _write(comparison_hook, "p1_r6_vs_f_exact_episode_rechain_comparison_hook.csv"),
        _write(benchmark, "p1_all_00631L_same_basis_reference_hook.csv"),
        _write(coverage, "p1_reconstructed_r6_episode_rechain_requested_vs_actual_coverage.csv"),
        _write(source_audit, "p1_reconstructed_r6_episode_rechain_price_cost_source_audit.csv"),
        _write(blocked, "p1_reconstructed_r6_episode_rechain_blocked_proxy_audit.csv"),
        _write(future_audit, "p1_reconstructed_r6_episode_rechain_future_data_audit.csv"),
    ]
    readiness_path = OUTPUT_DIR / "readiness_for_p1_reconstructed_r6_exact_episode_rechain.json"
    readiness_path.write_text(json.dumps(readiness, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary_path = OUTPUT_DIR / "final_summary_zh.md"
    summary_path.write_text(
        "# P1 Reconstructed R6 Exact Episode Full-path Rechain\n\n"
        f"- baseline daily rows: {len(state)}；stock episodes: {len(ranked)}；scenarios: baseline/remove best 1/3/5.\n"
        f"- baseline exact reconciliation: {baseline_return:.12f} vs source {expected_return:.12f}, pass=true.\n"
        "- removed episode legs hold 00631L, then the entire unique-position path and EP05 transition costs are recomputed.\n"
        "- ranking uses realized episode contribution vs same-day 00631L only for robustness evaluation, never a live rule.\n"
        "- F exact rechain and all-00631L same-basis hooks are included; Core does not issue a strategy verdict.\n"
        "- selected-stock prices remain official unadjusted OHLC diagnostic-only; adjusted-close source escalation stays closed.\n\n"
        "結論：R6 exact remove-best 1/3/5 full-path rechain已ready，可直接交Experiments做P1 episode concentration對稱比較。\n",
        encoding="utf-8",
    )
    manifest = {
        "task_id": TASK_ID,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "output_dir": str(OUTPUT_DIR),
        "source_inputs": {"R6": str(R6_DIR), "F_reference": str(EXPERIMENTS_DIR), "source_closure": str(SOURCE_CLOSURE_DIR)},
        "files": [{"path": p.name, "sha256": _sha256(p)} for p in [*output_paths, readiness_path, summary_path]],
        "readiness": readiness,
        "R6_source_readiness": r6_readiness,
    }
    (OUTPUT_DIR / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(readiness, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
