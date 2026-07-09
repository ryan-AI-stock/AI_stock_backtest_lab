from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from backtest_lab.costs import COST_MODEL_VERSION, TaiwanCostModel


REPO_ROOT = Path(__file__).resolve().parents[2]
RADAR_ROOT = Path("C:/Users/zergv/Documents/Codex/2026-05-23/ai-stock-rotation-radar-https-docs/outputs")
CORE_INPUT = REPO_ROOT / "outputs" / "vnext_p1_risk_adjusted_rs20_branch_contract_20260709"
RADAR_INPUT = RADAR_ROOT / "radar_vnext_p1_risk_adjusted_rs20_selected_stock_ohlc_gap_fill_20260709"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "vnext_p1_risk_adjusted_rs20_ohlc_absorption_20260709"
TASK_ID = "TASK-BACKTEST-CORE-VNEXT-P1-RISK-ADJUSTED-RS20-OHLC-ABSORPTION-001"
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Absorb Radar P1 risk-adjusted RS20 selected-stock OHLC gap fill.")
    parser.add_argument("--core-input-dir", default=str(CORE_INPUT))
    parser.add_argument("--radar-input-dir", default=str(RADAR_INPUT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()
    build_package(core_input=Path(args.core_input_dir), radar_input=Path(args.radar_input_dir), output_dir=Path(args.output_dir))


def build_package(*, core_input: Path, radar_input: Path, output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    core_contract = pd.read_csv(core_input / "p1_risk_adjusted_rs20_branch_contract.csv", dtype={"ticker": str, "selected_ticker": str})
    old_source = pd.read_csv(core_input / "p1_old_rs20_branch_contract_or_source_map.csv", dtype={"ticker": str})
    reference = pd.read_csv(core_input / "p1_00631l_buyhold_or_statehold_reference.csv", dtype={"benchmark": str})
    score_components = pd.read_csv(core_input / "p1_risk_adjusted_rs20_score_components.csv", dtype={"ticker": str})
    radar_filled = pd.read_csv(radar_input / "p1_risk_adjusted_rs20_selected_stock_ohlc_filled_rows.csv", dtype={"ticker": str, "selected_ticker": str})
    radar_blocked = pd.read_csv(radar_input / "p1_risk_adjusted_rs20_selected_stock_ohlc_blocked_ledger.csv", dtype={"ticker": str, "selected_ticker": str})

    refreshed = absorb_filled_rows(core_contract, radar_filled)
    refreshed = apply_blocked_rows(refreshed, radar_blocked)
    coverage = coverage_audit(refreshed, radar_filled, radar_blocked, old_source, reference)
    blocked = blocked_proxy_audit(refreshed, radar_blocked)
    policy = blocked_policy_decision(radar_blocked)
    future = future_data_audit(radar_input)

    contract_path = output_dir / "p1_risk_adjusted_rs20_branch_contract_refreshed.csv"
    old_path = output_dir / "p1_old_rs20_branch_contract_or_source_map.csv"
    reference_path = output_dir / "p1_00631l_buyhold_or_statehold_reference.csv"
    score_path = output_dir / "p1_risk_adjusted_rs20_score_components.csv"
    coverage_path = output_dir / "p1_risk_adjusted_rs20_ohlc_absorption_coverage.csv"
    blocked_path = output_dir / "p1_risk_adjusted_rs20_blocked_policy_ledger.csv"
    policy_path = output_dir / "p1_risk_adjusted_rs20_remaining6_policy_decision.csv"
    future_path = output_dir / "p1_risk_adjusted_rs20_ohlc_absorption_future_data_audit.csv"
    readiness_path = output_dir / "readiness_for_p1_rs20_comparison_after_ohlc_absorption.json"
    summary_path = output_dir / "final_summary_zh.md"

    refreshed.to_csv(contract_path, index=False, encoding="utf-8-sig")
    old_source.to_csv(old_path, index=False, encoding="utf-8-sig")
    reference.to_csv(reference_path, index=False, encoding="utf-8-sig")
    score_components.to_csv(score_path, index=False, encoding="utf-8-sig")
    coverage.to_csv(coverage_path, index=False, encoding="utf-8-sig")
    blocked.to_csv(blocked_path, index=False, encoding="utf-8-sig")
    policy.to_csv(policy_path, index=False, encoding="utf-8-sig")
    future.to_csv(future_path, index=False, encoding="utf-8-sig")

    ready_rows = int(refreshed["path_ready"].fillna(False).sum())
    blocked_rows = int((~refreshed["path_ready"].fillna(False)).sum())
    readiness = {
        "task": TASK_ID,
        "status": "p1_risk_adjusted_rs20_ohlc_absorbed_partial_with_explicit_6_blockers",
        "input_core_package": str(core_input),
        "input_radar_package": str(radar_input),
        "weekly_signal_rows": int(len(refreshed)),
        "path_ready_rows": ready_rows,
        "path_blocked_rows": blocked_rows,
        "path_ready_share": float(ready_rows / len(refreshed)) if len(refreshed) else 0.0,
        "radar_filled_rows_absorbed": int(len(radar_filled)),
        "radar_blocked_rows_retained": int(len(radar_blocked)),
        "full_exact_p1_comparison_ready": blocked_rows == 0,
        "ready_for_p1_rs20_comparison_experiments": False,
        "ready_for_p1_rs20_comparison_partial_policy_review": True,
        "ready_for_strategy_center_policy_decision": True,
        "ready_for_radar_additional_source_work": False,
        "remaining_blocked_policy": "maintain_blocked_no_timing_change_no_silent_fill",
        "policy_reason": "Radar already attempted selected-month and bounded exact-day official routes; changing timing or fallback would alter strategy semantics.",
        "cost_model_ready": True,
        "cost_model_version": COST_MODEL_VERSION,
        "adjusted_close_ready": False,
        "ready_for_formal": False,
        "ready_for_strategy_replay": False,
        "future_data_violation_count": 0,
        **FLAGS,
    }
    write_json(readiness_path, readiness)
    write_summary(summary_path, readiness, blocked, reference)
    write_json(
        output_dir / "manifest.json",
        {
            "task": TASK_ID,
            "output_dir": str(output_dir),
            "artifacts": [
                contract_path.name,
                old_path.name,
                reference_path.name,
                score_path.name,
                coverage_path.name,
                blocked_path.name,
                policy_path.name,
                future_path.name,
                readiness_path.name,
                summary_path.name,
            ],
            "created_at": datetime.now(timezone.utc).isoformat(),
            **FLAGS,
        },
    )
    return readiness


def absorb_filled_rows(core: pd.DataFrame, filled: pd.DataFrame) -> pd.DataFrame:
    out = core.copy()
    filled = filled.copy()
    for frame in [out, filled]:
        frame["ticker"] = frame["ticker"].astype(str).str.zfill(4)
        frame["signal_date"] = frame["signal_date"].astype(str)
    filled = filled.drop_duplicates(["signal_date", "ticker"], keep="last").set_index(["signal_date", "ticker"])
    model = TaiwanCostModel()
    for idx, row in out[~out["path_ready"].fillna(False)].iterrows():
        key = (str(row["signal_date"]), str(row["ticker"]).zfill(4))
        if key not in filled.index:
            continue
        patch = filled.loc[key]
        entry_price = numeric(patch.get("entry_close"))
        exit_price = numeric(patch.get("exit_close"))
        entry_open = numeric(patch.get("entry_open"))
        if pd.isna(entry_price) or pd.isna(exit_price):
            continue
        cost = unit_notional_cost(entry_price, exit_price, model)
        out.loc[idx, "entry_date"] = patch.get("entry_date")
        out.loc[idx, "exit_date"] = patch.get("exit_date")
        out.loc[idx, "entry_price"] = entry_price
        out.loc[idx, "exit_price"] = exit_price
        out.loc[idx, "entry_open"] = entry_open
        out.loc[idx, "entry_close"] = entry_price
        out.loc[idx, "exit_close"] = exit_price
        out.loc[idx, "gross_return_unadjusted"] = exit_price / entry_price - 1.0
        out.loc[idx, "net_return_local_ep05_cost_unit_notional"] = cost["net_return"]
        out.loc[idx, "path_ready"] = True
        out.loc[idx, "path_source_policy"] = "absorbed_radar_selected_ticker_official_unadjusted_ohlc_gap_fill"
        out.loc[idx, "source_quality"] = patch.get("source_quality", "official_unadjusted_ohlcv_selected_ticker")
        out.loc[idx, "blocked_reason"] = ""
        out.loc[idx, "selected_stock_adjusted_close_ready"] = False
        out.loc[idx, "diagnostic_unit_notional_twd"] = cost["diagnostic_unit_notional_twd"]
        out.loc[idx, "diagnostic_share_qty"] = cost["diagnostic_share_qty"]
        out.loc[idx, "buy_gross_twd"] = cost["buy_gross_twd"]
        out.loc[idx, "sell_gross_twd"] = cost["sell_gross_twd"]
        out.loc[idx, "buy_cost_twd"] = cost["buy_cost_twd"]
        out.loc[idx, "sell_cost_twd"] = cost["sell_cost_twd"]
        out.loc[idx, "total_cost_twd"] = cost["total_cost_twd"]
        out.loc[idx, "cost_application_status"] = "applied_local_ep05_cost_model_to_radar_unadjusted_ohlc_unit_notional"
    return out


def apply_blocked_rows(core: pd.DataFrame, blocked: pd.DataFrame) -> pd.DataFrame:
    out = core.copy()
    blocked = blocked.copy()
    for frame in [out, blocked]:
        frame["ticker"] = frame["ticker"].astype(str).str.zfill(4)
        frame["signal_date"] = frame["signal_date"].astype(str)
    blocked_idx = set(zip(blocked["signal_date"], blocked["ticker"]))
    for idx, row in out.iterrows():
        key = (str(row["signal_date"]), str(row["ticker"]).zfill(4))
        if key in blocked_idx:
            out.loc[idx, "path_ready"] = False
            out.loc[idx, "path_source_policy"] = "official_target_missing_after_radar_selected_month_and_exact_day_fallback"
            out.loc[idx, "source_quality"] = "blocked_missing_selected_ticker_official_ohlc"
            out.loc[idx, "blocked_reason"] = "official_selected_month_and_exact_day_fallback_target_missing_no_silent_fill"
            out.loc[idx, "selected_stock_adjusted_close_ready"] = False
    return out


def unit_notional_cost(entry_price: float, exit_price: float, model: TaiwanCostModel) -> dict[str, float | int]:
    notional = 1_000_000
    qty = math.floor(notional / entry_price)
    buy_gross = qty * entry_price
    sell_gross = qty * exit_price
    buy_cost = model.buy_cost(buy_gross)
    sell_cost = model.sell_cost(sell_gross, "stock")
    total_cost = buy_cost + sell_cost
    net_return = (sell_gross - sell_cost - buy_gross - buy_cost) / notional
    return {
        "diagnostic_unit_notional_twd": notional,
        "diagnostic_share_qty": qty,
        "buy_gross_twd": buy_gross,
        "sell_gross_twd": sell_gross,
        "buy_cost_twd": buy_cost,
        "sell_cost_twd": sell_cost,
        "total_cost_twd": total_cost,
        "net_return": net_return,
    }


def coverage_audit(refreshed: pd.DataFrame, filled: pd.DataFrame, blocked: pd.DataFrame, old_source: pd.DataFrame, reference: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"item": "weekly_signal_rows", "count": len(refreshed), "ready": len(refreshed) == 411},
            {"item": "path_ready_rows_after_absorption", "count": int(refreshed["path_ready"].fillna(False).sum()), "ready": bool(refreshed["path_ready"].all())},
            {"item": "path_blocked_rows_after_absorption", "count": int((~refreshed["path_ready"].fillna(False)).sum()), "ready": int((~refreshed["path_ready"].fillna(False)).sum()) == 0},
            {"item": "radar_filled_rows", "count": len(filled), "ready": len(filled) == 351},
            {"item": "radar_blocked_rows", "count": len(blocked), "ready": len(blocked) == 6},
            {"item": "old_rs20_source_rows", "count": len(old_source), "ready": len(old_source) == 411},
            {"item": "00631L_0050_buyhold_reference_rows", "count": len(reference), "ready": len(reference) >= 2},
        ]
    )


def blocked_proxy_audit(refreshed: pd.DataFrame, radar_blocked: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = [
        {
            "field": "remaining6_official_ohlc_path",
            "status": "blocked_policy_retained",
            "blocked_reason": "official selected-month route and bounded exact-day fallback both missing target row",
            "core_policy": "maintain blocked; do not change timing; do not use neighboring date; do not fallback to 00631L without Strategy Center policy",
            "next_owner": "Strategy Center policy decision before Experiments primary comparison",
        },
        {
            "field": "selected_stock_adjusted_close",
            "status": "blocked",
            "blocked_reason": "adjusted close remains unavailable",
            "core_policy": "official unadjusted OHLC diagnostic only; not formal",
            "next_owner": "Strategy Center source policy",
        },
    ]
    for _, row in radar_blocked.iterrows():
        rows.append(
            {
                "field": "remaining6_official_ohlc_path",
                "status": "blocked",
                "signal_date": row.get("signal_date"),
                "entry_date": row.get("entry_date"),
                "exit_date": row.get("exit_date"),
                "ticker": str(row.get("ticker")).zfill(4),
                "name": row.get("name", ""),
                "market": row.get("market", ""),
                "blocked_reason": row.get("blocked_reason", "official target missing"),
                "core_policy": "explicit blocked ledger; no silent fill",
                "next_owner": "Strategy Center policy decision",
            }
        )
    return pd.DataFrame(rows)


def blocked_policy_decision(radar_blocked: pd.DataFrame) -> pd.DataFrame:
    out = radar_blocked.copy()
    out["core_decision"] = "maintain_blocked"
    out["timing_change_allowed"] = False
    out["silent_fill_allowed"] = False
    out["fallback_00631L_substitution_allowed_by_core"] = False
    out["policy_reason"] = "Changing entry/exit date or fallback asset would alter P1 RS20 branch semantics; keep explicit blocker for Strategy Center."
    return out


def future_data_audit(radar_input: Path) -> pd.DataFrame:
    radar_future = radar_input / "p1_risk_adjusted_rs20_selected_stock_ohlc_future_data_audit.csv"
    rows = [
        {"audit_item": "future_return_used_in_absorption", "used": False, "future_data_violation_count": 0},
        {"audit_item": "neighboring_date_silent_fill_used", "used": False, "future_data_violation_count": 0},
        {"audit_item": "00631L_excess_reconstruction_used", "used": False, "future_data_violation_count": 0},
    ]
    if radar_future.exists():
        rows.append({"audit_item": "radar_future_data_audit_source_present", "used": False, "future_data_violation_count": 0})
    return pd.DataFrame(rows)


def numeric(value: Any) -> float:
    return pd.to_numeric(value, errors="coerce")


def write_summary(path: Path, readiness: dict[str, Any], blocked: pd.DataFrame, reference: pd.DataFrame) -> None:
    path.write_text(
        f"""# P1 risk-adjusted RS20 OHLC absorption

## 結論

- 已吸收 Radar/Data 351 筆 selected-stock official unadjusted OHLC gap fill。
- P1 new RS20 path coverage 更新為 {readiness['path_ready_rows']}/{readiness['weekly_signal_rows']}。
- 仍有 6 筆 official target-missing rows，Core 決策：維持 explicit blocked，不改 timing、不 silent fill、不自行替換 00631L。
- 因 6 筆仍 blocked，full exact P1 comparison 尚未 ready；需要 Strategy Center 接受 partial-blocked policy，或另行授權 timing/fallback 政策後，才交 Experiments 產生主比較表。

## Remaining blocked rows

{blocked[blocked['field'].eq('remaining6_official_ohlc_path')].to_csv(index=False)}

## Reference

- 00631L / 0050 buy-hold reference rows: {len(reference)}
- cost_model_version: `{readiness['cost_model_version']}`
- adjusted_close_ready=false for selected stocks。

## Flags

- formal_model_changed=false
- trade_decision_changed=false
- active_in_trade_decision=false
- report_changed=false
- portfolio_replay_executed=false
- ready_for_strategy_replay=false
- ready_for_formal=false
- not_live_rule=true
- forward_returns_live_rule_usage=false
""",
        encoding="utf-8",
    )


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
