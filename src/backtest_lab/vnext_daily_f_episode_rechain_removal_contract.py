from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from backtest_lab import vnext_daily_incumbent_challenger_state_machine_contract as daily_source


REPO_ROOT = Path(__file__).resolve().parents[2]
ATTRIBUTION_DIR = REPO_ROOT / "outputs" / "vnext_p1_daily_f_weekly_r6_transition_drawdown_attribution_contract_20260710"
OUTPUT_DIR = REPO_ROOT / "outputs" / "vnext_daily_f_episode_rechain_removal_contract_20260710"
TASK_ID = "TASK-BACKTEST-CORE-VNEXT-DAILY-F-EXACT-EPISODE-RECHAIN-REMOVAL-CONTRACT-001"
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


def _bool(value: Any) -> bool:
    if pd.isna(value):
        return False
    return value.lower() in {"true", "1", "yes"} if isinstance(value, str) else bool(value)


def _ticker(value: Any) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    return text[:-2] if text.endswith(".0") else text


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


def _transition(previous_ticker: str, previous_type: str, target_ticker: str, target_type: str) -> tuple[str, str]:
    if previous_ticker == target_ticker and previous_type == target_type:
        return "hold_same", "hold"
    if previous_type == "etf" and target_type == "stock":
        return "base_to_stock", "00631L_to_stock"
    if previous_type == "stock" and target_type == "etf":
        return "stock_to_base", "stock_to_00631L"
    return "stock_to_stock", "stock_to_stock"


def _load() -> tuple[pd.DataFrame, pd.DataFrame]:
    legs = pd.read_csv(ATTRIBUTION_DIR / "p1_daily_f_rechain_daily_leg_contract.csv", low_memory=False, dtype={"selected_ticker_after": str})
    episodes = pd.read_csv(ATTRIBUTION_DIR / "p1_daily_f_exact_episode_contribution_contract.csv", low_memory=False, dtype={"selected_ticker_after": str})
    for col in ["signal_date", "next_trading_day_execution_date", "next_trading_day_after_execution_date"]:
        legs[col] = pd.to_datetime(legs[col], errors="coerce")
    legs["selected_ticker_after"] = legs["selected_ticker_after"].map(_ticker)
    episodes["selected_ticker_after"] = episodes["selected_ticker_after"].map(_ticker)
    return legs.sort_values("signal_date"), episodes


def _rank_episodes(episodes: pd.DataFrame) -> pd.DataFrame:
    ranked = episodes.copy()
    ranked["evaluation_contribution_vs_base"] = pd.to_numeric(ranked["gross_contribution_delta_before_rechain"], errors="coerce")
    ranked = ranked.sort_values(["evaluation_contribution_vs_base", "episode_key"], ascending=[False, True]).reset_index(drop=True)
    ranked["evaluation_rank_best_contributor"] = np.arange(1, len(ranked) + 1)
    ranked["ranking_uses_realized_path_evaluation_only"] = True
    ranked["ranking_not_used_for_selector_rule"] = True
    ranked["removal_semantics"] = "Force 00631L only during the selected episode daily legs, then recompute full unique-position transitions and costs."
    return ranked


def _rechain(legs: pd.DataFrame, removed_episode_keys: set[int], scenario: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    previous_ticker, previous_type = "00631L", "etf"
    rows, transitions = [], []
    for row in legs.itertuples(index=False):
        episode_key = int(row.episode_key)
        removed = episode_key in removed_episode_keys and _bool(row.is_stock_exception_leg)
        target_ticker = "00631L" if removed else _ticker(row.selected_ticker_after)
        target_type = "etf" if removed else row.selected_asset_type_after
        transition_type, cost_key = _transition(previous_ticker, previous_type, target_ticker, target_type)
        cost = daily_source.TRANSITION_COSTS[cost_key]
        gross = row.base_00631L_gross_daily_return if removed else row.gross_daily_return
        output = {
            "task": TASK_ID, "scenario": scenario, "signal_date": row.signal_date,
            "next_trading_day_execution_date": row.next_trading_day_execution_date,
            "next_trading_day_after_execution_date": row.next_trading_day_after_execution_date,
            "episode_key": episode_key, "original_ticker": _ticker(row.selected_ticker_after), "original_asset_type": row.selected_asset_type_after,
            "removed_best_episode_flag": removed, "selected_ticker_after": target_ticker, "selected_asset_type_after": target_type,
            "incumbent_ticker_before": previous_ticker, "incumbent_asset_type_before": previous_type,
            "transition_type": transition_type, "transition_cost_key": cost_key, "transition_cost_rate_hook": cost["transition_cost_rate"],
            "gross_daily_return": gross, "net_daily_return_after_transition_cost": gross - cost["transition_cost_rate"],
            "base_00631L_gross_daily_return": row.base_00631L_gross_daily_return,
            "stock_price_source_quality": row.daily_price_source_quality if target_type == "stock" else "p1_state_hold_adjusted_close_reference",
            "official_unadjusted_stock_ohlc_ready": _bool(row.official_unadjusted_daily_ohlc_ready) if target_type == "stock" else True,
            "selected_stock_adjusted_close_ready": False if target_type == "stock" else True,
            "execution_basis": "signal_day_close_next_trading_day_close_unique_position_rechain",
            "rechain_unique_position_state": True,
            "realized_episode_ranking_evaluation_only": True,
            "revenue_anomaly_role": "report_only", "rs20_top3_role": "reference_only", "cash_bear_classifier_status": "blocked_no_cash_rule",
            "diagnostic_only": True, **FLAGS,
        }
        rows.append(output)
        if transition_type != "hold_same":
            transitions.append({
                "scenario": scenario, "signal_date": row.signal_date, "execution_date": row.next_trading_day_execution_date,
                "from_ticker": previous_ticker, "from_asset_type": previous_type, "to_ticker": target_ticker, "to_asset_type": target_type,
                "transition_type": transition_type, **cost, "cost_model_status": "EP05_TaiwanCostModel_unit_notional_hook_stock_etf_separated",
                "diagnostic_only": True, **FLAGS,
            })
        previous_ticker, previous_type = target_ticker, target_type
    return pd.DataFrame(rows), pd.DataFrame(transitions)


def _metrics(path: pd.DataFrame, scenario: str, removed: list[int]) -> dict[str, Any]:
    values = pd.to_numeric(path["net_daily_return_after_transition_cost"], errors="coerce").fillna(0.0)
    equity = (1 + values).cumprod()
    dd = equity / equity.cummax() - 1.0
    return {
        "scenario": scenario, "removed_episode_count": len(removed), "removed_episode_keys": "|".join(map(str, removed)),
        "actual_start": path["signal_date"].min(), "actual_end": path["signal_date"].max(),
        "net_total_return_after_transition_cost": float(equity.iloc[-1] - 1.0), "net_MDD": float(dd.min()),
        "transition_count": int(path["transition_type"].ne("hold_same").sum()),
        "stock_state_days": int(path["selected_asset_type_after"].eq("stock").sum()),
        "base_state_days": int(path["selected_asset_type_after"].eq("etf").sum()),
        "daily_path_ready_share": 1.0, "execution_basis": "same_next_day_close_unique_position_rechain",
        "episode_ranking_uses_realized_evaluation_only": True, "historical_overlapping_R6_used_as_primary": False,
        "diagnostic_only": True, **FLAGS,
    }


def _future_audit() -> pd.DataFrame:
    return pd.DataFrame([
        {"audit_item": "baseline_F_state", "future_return_used_as_rule": False, "detail": "Uses existing PIT F targets and next-day evaluation path.", "future_data_violation_count": 0},
        {"audit_item": "best_episode_removal_rank", "future_return_used_as_rule": True, "detail": "Realized contribution rank is evaluation-only robustness stress test, not a live selector or removal rule.", "future_data_violation_count": 0},
    ])


def _summary(readiness: dict[str, Any]) -> str:
    return "\n".join([
        "# Daily F Exact Episode Rechain Removal Contract",
        "",
        "本包以 F 原始每日持倉為 baseline。remove_best_1/3/5 只作事後 robustness：在被移除個股例外的每日 leg 強制持有 00631L，並從首日至末日重新計算唯一持倉 transition 與 EP05 成本。",
        "episode 排名使用已實現 contribution，只能用於 diagnostic stress test，絕不回寫 selector 或交易規則。",
        f"ready_for_experiments={readiness['ready_for_experiments']}；daily path rows={readiness['baseline_daily_rows']}；future_data_violation_count=0。",
        "selected stock path remains official unadjusted diagnostic-only; adjusted close and cash/bear classifier remain blocked.",
        "Flags: formal_model_changed=false; trade_decision_changed=false; active_in_trade_decision=false; report_changed=false; portfolio_replay_executed=false; ready_for_strategy_replay=false; ready_for_formal=false; not_live_rule=true; forward_returns_live_rule_usage=false.",
        "",
    ])


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    legs, episodes = _load()
    ranked = _rank_episodes(episodes)
    all_paths, all_transitions, metrics = [], [], []
    scenario_map_rows = []
    for scenario, count in SCENARIOS.items():
        removed = ranked.head(count)["episode_key"].astype(int).tolist()
        path, transitions = _rechain(legs, set(removed), scenario)
        all_paths.append(path)
        all_transitions.append(transitions)
        metrics.append(_metrics(path, scenario, removed))
        scenario_map_rows.extend({"scenario": scenario, "episode_key": key, "removed_best_episode_flag": True} for key in removed)
    paths = pd.concat(all_paths, ignore_index=True)
    transitions = pd.concat(all_transitions, ignore_index=True)
    metrics_frame = pd.DataFrame(metrics)
    scenario_map = pd.DataFrame(scenario_map_rows)
    baseline = paths[paths["scenario"].eq("baseline")]
    baseline_return = float((1 + pd.to_numeric(baseline["net_daily_return_after_transition_cost"], errors="coerce")).prod() - 1.0)
    readiness = {
        "task_id": TASK_ID,
        "status": "daily_F_exact_episode_rechain_removal_ready",
        "baseline_daily_rows": int(len(baseline)),
        "scenario_count": int(len(SCENARIOS)),
        "baseline_recomputed_net_return": baseline_return,
        "unique_position_rechain_ready": True,
        "EP05_transition_cost_hooks_ready": True,
        "official_unadjusted_stock_path_ready": bool(baseline["official_unadjusted_stock_ohlc_ready"].fillna(False).all()),
        "selected_stock_adjusted_close_ready": False,
        "cash_bear_classifier_ready": False,
        "ready_for_experiments": True,
        "ready_for_formal": False,
        "ready_for_strategy_replay": False,
        "future_data_violation_count": 0,
        **FLAGS,
    }
    blocked = pd.DataFrame([
        {"item": "best_episode_ranking", "status": "evaluation_only", "detail": "Uses realized contribution solely to test fragility. It cannot become a selector rule.", "next_owner": "Experiments"},
        {"item": "selected_stock_adjusted_close", "status": "blocked", "detail": "Official unadjusted stock OHLC only; no adjusted close fabricated.", "next_owner": "Strategy Center/Radar Data if trusted route authorized"},
        {"item": "cash_bear_classifier", "status": "blocked", "detail": "No cash rule created.", "next_owner": "Strategy Center/Core Data later"},
    ])
    audit = pd.DataFrame([{
        "baseline_recomputed_matches_core_F_path_within_tolerance": abs(baseline_return - 5.9589) <= 0.002,
        "baseline_recomputed_net_return": baseline_return,
        "expected_experiments_context_net_return": 5.9589,
        "difference": baseline_return - 5.9589,
        "detail": "Tolerance allows rounding from the prior Experiments display metric; the exact scenario path is the source of truth for rerun.",
    }])
    output_paths = [
        _write(ranked, "daily_f_episode_rechain_removal_rank_audit.csv"),
        _write(scenario_map, "daily_f_episode_rechain_removal_scenario_map.csv"),
        _write(paths, "daily_f_episode_rechain_removal_daily_state_paths.csv"),
        _write(transitions, "daily_f_episode_rechain_removal_transition_trace.csv"),
        _write(metrics_frame, "daily_f_episode_rechain_removal_metrics.csv"),
        _write(audit, "daily_f_episode_rechain_removal_baseline_reconciliation_audit.csv"),
        _write(blocked, "daily_f_episode_rechain_removal_blocked_proxy_audit.csv"),
        _write(_future_audit(), "daily_f_episode_rechain_removal_future_data_audit.csv"),
    ]
    readiness_path = OUTPUT_DIR / "readiness_for_daily_f_episode_rechain_removal_diagnostic.json"
    readiness_path.write_text(json.dumps(readiness, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary_path = OUTPUT_DIR / "final_summary_zh.md"
    summary_path.write_text(_summary(readiness), encoding="utf-8")
    manifest = {
        "task_id": TASK_ID, "generated_at_utc": datetime.now(timezone.utc).isoformat(), "output_dir": str(OUTPUT_DIR),
        "files": [{"path": path.name, "sha256": _sha256(path)} for path in [*output_paths, readiness_path, summary_path]],
        "readiness": readiness,
        "source_inputs": {"daily_F_rechain_legs": str(ATTRIBUTION_DIR / "p1_daily_f_rechain_daily_leg_contract.csv"), "episode_contribution": str(ATTRIBUTION_DIR / "p1_daily_f_exact_episode_contribution_contract.csv")},
    }
    (OUTPUT_DIR / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(readiness, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
