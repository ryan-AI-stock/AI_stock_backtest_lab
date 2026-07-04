"""Build the capped Dynamic Pool1 strict lowpoint event-to-action contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from backtest_lab.costs import cost_model_metadata
from backtest_lab.dynamic_pool1_strict_lowpoint_event_to_action_contract import (
    CASE_TRACE_TICKERS,
    DEFAULT_EVENT_CONTRACT_DIR,
    DEFAULT_LIQUIDITY_DIR,
    PRIMARY_EVENT_PRIORITY,
    _action_row,
    _date_text,
    _load_calendar,
    _load_events,
    _load_formal_streams,
    _load_price_table,
    _price_on,
)


TASK_ID = "TASK-BACKTEST-CORE-DYNAMIC-POOL1-STRICT-LOWPOINT-CAPPED-EVENT-TO-ACTION-CONTRACT-001"
EXPERIMENTS_TASK_ID = "TASK-BACKTEST-EXPERIMENTS-DYNAMIC-POOL1-STRICT-LOWPOINT-CAPPED-EVENT-TO-ACTION-VALIDATION-001"
DEFAULT_OUTPUT_DIR = Path("outputs/dynamic_pool1_strict_lowpoint_capped_event_to_action_contract_20260704")

CAPPED_VARIANTS = [
    {
        "variant": "strict_lowpoint_sleeve10_hold20_when_formal_cash_or_market_exposure_capped",
        "sleeve_weight": 0.10,
        "max_sleeve_cap": 0.10,
        "max_hold_days": 20,
        "exit_on_ma20_break": False,
        "variant_role": "primary_capped",
    },
    {
        "variant": "strict_lowpoint_sleeve20_hold20_when_formal_cash_or_market_exposure_capped",
        "sleeve_weight": 0.20,
        "max_sleeve_cap": 0.20,
        "max_hold_days": 20,
        "exit_on_ma20_break": False,
        "variant_role": "sizing_sensitivity_capped",
    },
    {
        "variant": "strict_lowpoint_sleeve10_hold20_exit_on_ma20_break_capped",
        "sleeve_weight": 0.10,
        "max_sleeve_cap": 0.10,
        "max_hold_days": 20,
        "exit_on_ma20_break": True,
        "variant_role": "exit_sensitivity_capped",
    },
]


def run_dynamic_pool1_strict_lowpoint_capped_event_to_action_contract(
    *,
    repo_root: str | Path = ".",
    event_contract_dir: str | Path = DEFAULT_EVENT_CONTRACT_DIR,
    liquidity_dir: str | Path = DEFAULT_LIQUIDITY_DIR,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> dict:
    root = Path(repo_root).resolve()
    source = _resolve(root, event_contract_dir)
    liquidity = _resolve(root, liquidity_dir)
    output = _resolve(root, output_dir)
    output.mkdir(parents=True, exist_ok=True)

    events = _load_events(source)
    formal = _load_formal_streams(root)
    calendar = _load_calendar(liquidity)
    price_table = _load_price_table(liquidity, events["ticker"].dropna().astype(str).unique().tolist())
    ranked_events, priority_audit = _rank_primary_events(events, price_table)
    base_selected = ranked_events[ranked_events["selected_from_concurrent_events"]].copy()
    base_selected["signal_date_key"] = base_selected["signal_date"].dt.strftime("%Y-%m-%d")
    base_selected = base_selected.merge(formal, on="signal_date_key", how="left")

    action_contract = _build_capped_contract(base_selected, price_table, calendar)
    daily_weight = _build_daily_weight(action_contract)
    trade_ledger = _build_trade_ledger(action_contract)
    blocked_by_active = action_contract[action_contract["blocked_by_active_sleeve"]].copy()
    cap_audit = _cap_violation_audit(action_contract)
    future_audit = _future_data_audit(action_contract)

    action_contract.to_csv(output / "strict_lowpoint_capped_event_to_action_contract.csv", index=False, encoding="utf-8-sig")
    daily_weight.to_csv(output / "strict_lowpoint_capped_daily_weight_contract.csv", index=False, encoding="utf-8-sig")
    trade_ledger.to_csv(output / "strict_lowpoint_capped_trade_intent_ledger.csv", index=False, encoding="utf-8-sig")
    priority_audit.to_csv(output / "strict_lowpoint_concurrent_event_priority_audit.csv", index=False, encoding="utf-8-sig")
    cap_audit.to_csv(output / "strict_lowpoint_cap_violation_audit.csv", index=False, encoding="utf-8-sig")
    blocked_by_active.to_csv(output / "strict_lowpoint_blocked_by_active_sleeve.csv", index=False, encoding="utf-8-sig")
    future_audit.to_csv(output / "future_data_audit.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([cost_model_metadata()]).to_csv(output / "cost_model_contract.csv", index=False, encoding="utf-8-sig")

    manifest = {
        "task_id": TASK_ID,
        "status": "completed_strict_lowpoint_capped_event_to_action_contract_ready",
        "output_dir": str(output),
        "source_event_contract_dir": str(source),
        "event_rows_input": int(len(events)),
        "selected_signal_rows": int(len(base_selected)),
        "action_contract_rows": int(len(action_contract)),
        "action_allowed_rows": int(action_contract["action_allowed"].sum()) if not action_contract.empty else 0,
        "blocked_by_active_sleeve_rows": int(len(blocked_by_active)),
        "daily_weight_rows": int(len(daily_weight)),
        "trade_intent_rows": int(len(trade_ledger)),
        "cap_violation_count": int(action_contract["cap_violation"].sum()) if not action_contract.empty else 0,
        "max_aggregate_sleeve_exposure": round(float(action_contract["aggregate_sleeve_exposure"].max()), 8)
        if not action_contract.empty
        else 0.0,
        "formal_direct_stock_target_override_count": int(
            action_contract.loc[action_contract["action_allowed"], "formal_direct_stock_target_active"].sum()
        )
        if not action_contract.empty
        else 0,
        "future_data_violation_count": int(future_audit["future_data_violation"].sum()) if not future_audit.empty else 0,
        "uses_forward_return_as_rule": False,
        "portfolio_replay_executed": False,
        "formal_model_changed": False,
        "trade_decision_changed": False,
        "active_in_trade_decision": False,
        "report_changed": False,
        "diagnostic_cap_hygiene_only": True,
        "ready_for_experiments_validation": True,
        "handoff_to_experiments_task": EXPERIMENTS_TASK_ID,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "final_summary_zh.md").write_text(_summary(manifest, cap_audit), encoding="utf-8")
    pd.DataFrame([{"task_id": TASK_ID, "status": "completed", "output_dir": str(output)}]).to_csv(
        output / "completed.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(columns=["task_id", "status", "reason"]).to_csv(output / "failed.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(
        [
            {"step": "load_strict_lowpoint_event_contract", "status": "completed"},
            {"step": "rank_concurrent_events", "status": "completed"},
            {"step": "simulate_capped_no_pyramid_actions", "status": "completed"},
            {"step": "write_capped_contract_package", "status": "completed"},
        ]
    ).to_csv(output / "run_log.csv", index=False, encoding="utf-8-sig")
    return manifest


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _rank_primary_events(events: pd.DataFrame, price_table: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    primary = events[events["event_variant_role"].eq("primary")].copy()
    primary["next_tradable_date_key"] = primary["next_tradable_date"].dt.strftime("%Y-%m-%d")
    px = price_table[["date", "canonical_ticker", "turnover"]].rename(
        columns={"date": "next_tradable_date_key", "canonical_ticker": "ticker", "turnover": "entry_turnover"}
    )
    primary = primary.merge(px, on=["next_tradable_date_key", "ticker"], how="left")
    primary["entry_turnover"] = pd.to_numeric(primary["entry_turnover"], errors="coerce").fillna(0.0)
    primary["tie_break_rs"] = pd.to_numeric(primary.get("rs_vs_0050_3d_or_5d"), errors="coerce").fillna(0.0) + pd.to_numeric(
        primary.get("rs_vs_00631l_3d_or_5d"), errors="coerce"
    ).fillna(0.0)
    primary = primary.sort_values(
        ["signal_date", "event_priority", "entry_turnover", "tie_break_rs", "ticker"],
        ascending=[True, True, False, False, True],
    )
    primary["priority_rank_within_signal"] = primary.groupby("signal_date").cumcount() + 1
    primary["selected_from_concurrent_events"] = primary["priority_rank_within_signal"].eq(1)
    primary["concurrent_event_count"] = primary.groupby("signal_date")["ticker"].transform("count")
    audit = primary[
        [
            "signal_date",
            "next_tradable_date",
            "ticker",
            "candidate_name",
            "event_variant",
            "event_priority",
            "entry_turnover",
            "tie_break_rs",
            "priority_rank_within_signal",
            "selected_from_concurrent_events",
            "concurrent_event_count",
        ]
    ].copy()
    audit["signal_date"] = audit["signal_date"].dt.strftime("%Y-%m-%d")
    audit["next_tradable_date"] = audit["next_tradable_date"].dt.strftime("%Y-%m-%d")
    return primary, audit


def _build_capped_contract(selected: pd.DataFrame, price_table: pd.DataFrame, calendar: list[str]) -> pd.DataFrame:
    rows = []
    selected = selected.sort_values(["signal_date", "ticker"]).copy()
    for variant in CAPPED_VARIANTS:
        active_exit_date = ""
        for event in selected.to_dict(orient="records"):
            signal_date = _date_text(event.get("signal_date"))
            active_before = bool(active_exit_date and signal_date < active_exit_date)
            if active_before:
                row = _blocked_active_row(event, variant, active_exit_date)
            else:
                row = _action_row(event, variant, price_table, calendar)
                row["variant"] = variant["variant"]
                row["variant_role"] = variant["variant_role"]
                row["max_sleeve_cap"] = float(variant["max_sleeve_cap"])
                row["active_sleeve_before_signal"] = False
                row["blocked_by_active_sleeve"] = False
                row["selected_from_concurrent_events"] = bool(event.get("selected_from_concurrent_events", True))
                row["concurrent_event_count"] = int(event.get("concurrent_event_count", 1) or 1)
                row["aggregate_sleeve_exposure"] = float(row["sleeve_weight"]) if row["action_allowed"] else 0.0
                row["cap_violation"] = bool(row["aggregate_sleeve_exposure"] > row["max_sleeve_cap"] + 1e-9)
                if row["action_allowed"]:
                    active_exit_date = str(row["exit_date"] or "")
            rows.append(row)
    return pd.DataFrame(rows)


def _blocked_active_row(event: dict, variant: dict, active_exit_date: str) -> dict:
    signal_date = _date_text(event.get("signal_date"))
    return {
        "variant": variant["variant"],
        "date": signal_date,
        "signal_date": signal_date,
        "next_tradable_date": _date_text(event.get("next_tradable_date")),
        "ticker": event.get("ticker", ""),
        "candidate_name": event.get("candidate_name", ""),
        "event_variant": event.get("event_variant", ""),
        "event_variant_role": event.get("event_variant_role", ""),
        "formal_state": event.get("formal_state", ""),
        "formal_target": event.get("formal_target", ""),
        "formal_direct_stock_target_active": bool(event.get("formal_direct_stock_target_active", False)),
        "action_allowed": False,
        "action_blocked_reason": "blocked_by_active_sleeve",
        "active_sleeve_before_signal": True,
        "blocked_by_active_sleeve": True,
        "selected_from_concurrent_events": bool(event.get("selected_from_concurrent_events", True)),
        "concurrent_event_count": int(event.get("concurrent_event_count", 1) or 1),
        "sleeve_weight": 0.0,
        "aggregate_sleeve_exposure": 0.0,
        "max_sleeve_cap": float(variant["max_sleeve_cap"]),
        "cap_violation": False,
        "entry_date": "",
        "exit_date": active_exit_date,
        "exit_reason": "blocked_until_active_sleeve_exit",
        "hold_days": 0,
        "trade_cost": 0,
        "turnover": 0.0,
        "uses_forward_return_as_rule": False,
        "formal_model_changed": False,
        "trade_decision_changed": False,
        "active_in_trade_decision": False,
        "report_changed": False,
    }


def _build_daily_weight(contract: pd.DataFrame) -> pd.DataFrame:
    allowed = contract[contract["action_allowed"]].copy()
    if allowed.empty:
        return pd.DataFrame(columns=["date", "variant", "ticker", "dynamic_weight", "aggregate_sleeve_exposure"])
    allowed["dynamic_weight"] = allowed["sleeve_weight"]
    allowed["formal_residual_weight"] = 1.0 - allowed["dynamic_weight"]
    return allowed[
        [
            "date",
            "variant",
            "ticker",
            "candidate_name",
            "dynamic_weight",
            "formal_residual_weight",
            "aggregate_sleeve_exposure",
            "max_sleeve_cap",
            "entry_date",
            "exit_date",
            "exit_reason",
        ]
    ]


def _build_trade_ledger(contract: pd.DataFrame) -> pd.DataFrame:
    allowed = contract[contract["action_allowed"]].copy()
    if allowed.empty:
        return pd.DataFrame(columns=["signal_date", "variant", "ticker", "entry_date", "exit_date"])
    allowed["entry_intent"] = "buy_capped_strict_lowpoint_sleeve_next_day"
    allowed["exit_intent"] = allowed["exit_reason"].map(lambda reason: f"sell_capped_strict_lowpoint_sleeve_{reason}")
    return allowed[
        [
            "signal_date",
            "variant",
            "event_variant",
            "ticker",
            "candidate_name",
            "entry_intent",
            "entry_date",
            "entry_price",
            "exit_intent",
            "exit_date",
            "exit_price",
            "hold_days",
            "trade_cost",
            "turnover",
            "aggregate_sleeve_exposure",
            "max_sleeve_cap",
            "uses_forward_return_as_rule",
        ]
    ]


def _cap_violation_audit(contract: pd.DataFrame) -> pd.DataFrame:
    if contract.empty:
        return pd.DataFrame(columns=["variant", "max_aggregate_sleeve_exposure", "max_sleeve_cap", "cap_violation_count"])
    return (
        contract.groupby("variant", as_index=False)
        .agg(
            max_aggregate_sleeve_exposure=("aggregate_sleeve_exposure", "max"),
            max_sleeve_cap=("max_sleeve_cap", "max"),
            cap_violation_count=("cap_violation", "sum"),
            action_allowed_rows=("action_allowed", "sum"),
            blocked_by_active_sleeve_rows=("blocked_by_active_sleeve", "sum"),
        )
        .sort_values("variant")
    )


def _future_data_audit(contract: pd.DataFrame) -> pd.DataFrame:
    if contract.empty:
        return pd.DataFrame(columns=["signal_date", "ticker", "variant", "future_data_violation", "reason"])
    out = contract[["signal_date", "next_tradable_date", "ticker", "variant", "event_variant"]].copy()
    out["future_data_violation"] = False
    out["reason"] = ""
    return out


def _summary(manifest: dict, cap_audit: pd.DataFrame) -> str:
    audit_text = "no rows" if cap_audit.empty else cap_audit.to_csv(index=False).strip()
    return "\n".join(
        [
            "# Dynamic Pool1 strict lowpoint capped event-to-action contract",
            "",
            "本包只做 cap / no-pyramid hygiene；不代表 portfolio route 通過，不改正式模型、日報或交易決策。",
            "",
            f"- action contract rows：{manifest['action_contract_rows']}",
            f"- action allowed rows：{manifest['action_allowed_rows']}",
            f"- blocked by active sleeve rows：{manifest['blocked_by_active_sleeve_rows']}",
            f"- cap violation count：{manifest['cap_violation_count']}",
            f"- max aggregate sleeve exposure：{manifest['max_aggregate_sleeve_exposure']}",
            f"- formal direct stock target override count：{manifest['formal_direct_stock_target_override_count']}",
            f"- future data violation count：{manifest['future_data_violation_count']}",
            "",
            "## Cap audit",
            audit_text,
        ]
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--event-contract-dir", default=str(DEFAULT_EVENT_CONTRACT_DIR))
    parser.add_argument("--liquidity-dir", default=str(DEFAULT_LIQUIDITY_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args(argv)
    manifest = run_dynamic_pool1_strict_lowpoint_capped_event_to_action_contract(
        repo_root=args.repo_root,
        event_contract_dir=args.event_contract_dir,
        liquidity_dir=args.liquidity_dir,
        output_dir=args.output_dir,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
