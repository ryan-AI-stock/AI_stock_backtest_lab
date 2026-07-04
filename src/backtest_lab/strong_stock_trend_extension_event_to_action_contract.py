"""Build the bounded trend-extension event-to-action contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from backtest_lab.costs import TaiwanCostModel, cost_model_metadata
from backtest_lab.dynamic_pool1_strict_lowpoint_event_to_action_contract import (
    _date_text,
    _load_formal_streams,
)


TASK_ID = "TASK-BACKTEST-CORE-STRONG-STOCK-TREND-EXTENSION-EVENT-TO-ACTION-CONTRACT-001"
EXPERIMENTS_TASK_ID = "TASK-BACKTEST-EXPERIMENTS-STRONG-STOCK-TREND-EXTENSION-EVENT-TO-ACTION-VALIDATION-001"
DEFAULT_OUTCOME_PANEL = Path(
    "outputs/strong_stock_trend_extension_exact_outcome_panel_20260704/trend_extension_exact_event_outcome_panel.csv"
)
DEFAULT_OUTPUT_DIR = Path("outputs/strong_stock_trend_extension_event_to_action_contract_20260704")
DEFAULT_CANDIDATE_CONTEXT = Path("outputs/dynamic_pool1_candidate_panel_v0_20260704/candidate_pool_by_month.csv")

EVENT_ROUTES = [
    {
        "event_route": "trend_ext_slope_acceleration_primary",
        "event_route_role": "primary",
        "allowed_event_variants": ["trend_ext_slope_acceleration"],
    },
    {
        "event_route": "trend_ext_ma_stack_breakout_sensitivity",
        "event_route_role": "sensitivity",
        "allowed_event_variants": ["trend_ext_ma_stack_breakout"],
    },
    {
        "event_route": "trend_ext_slope_or_ma_stack_best_daily",
        "event_route_role": "combined_sensitivity",
        "allowed_event_variants": ["trend_ext_slope_acceleration", "trend_ext_ma_stack_breakout"],
    },
]
ACTION_DESIGNS = [
    {
        "action_design": "sleeve10_hold20_when_formal_market_exposure_or_cash",
        "design_role": "primary_action_design",
        "sleeve_weight": 0.10,
        "max_hold_days": 20,
    },
    {
        "action_design": "sleeve10_hold40_when_formal_market_exposure_or_cash",
        "design_role": "hold_sensitivity",
        "sleeve_weight": 0.10,
        "max_hold_days": 40,
    },
    {
        "action_design": "sleeve20_hold20_when_formal_market_exposure_or_cash",
        "design_role": "sizing_sensitivity",
        "sleeve_weight": 0.20,
        "max_hold_days": 20,
    },
]
REFERENCE_EVENT_VARIANT = "trend_ext_new_high_rs_confirm"


def run_strong_stock_trend_extension_event_to_action_contract(
    *,
    repo_root: str | Path = ".",
    outcome_panel: str | Path = DEFAULT_OUTCOME_PANEL,
    candidate_context: str | Path = DEFAULT_CANDIDATE_CONTEXT,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> dict:
    root = Path(repo_root).resolve()
    outcome_path = _resolve(root, outcome_panel)
    context_path = _resolve(root, candidate_context)
    output = _resolve(root, output_dir)
    output.mkdir(parents=True, exist_ok=True)

    events = _load_outcome_panel(outcome_path)
    formal = _load_formal_streams(root)
    calendar = sorted(events["next_tradable_date"].dropna().dt.strftime("%Y-%m-%d").unique().tolist())
    selected, reference_blocked = _select_route_events(events)
    market_lookup = _load_market_lookup(context_path)
    if not selected.empty:
        selected["ticker"] = selected["ticker"].map(_canonical_from_market_lookup(market_lookup))
    if not reference_blocked.empty:
        reference_blocked["ticker"] = reference_blocked["ticker"].map(_canonical_from_market_lookup(market_lookup))
    action_contract, conflict_blocked = _build_action_contract(selected, formal, calendar)
    daily_weight = _build_daily_weight_contract(action_contract)
    trade_ledger = _build_trade_intent_ledger(action_contract)
    caution_audit = _build_00631l_caution_audit(action_contract)
    future_audit = _future_data_audit(action_contract)
    conflict_rows = pd.concat([conflict_blocked, reference_blocked], ignore_index=True, sort=False)

    action_contract.to_csv(output / "trend_extension_event_to_action_contract.csv", index=False, encoding="utf-8-sig")
    daily_weight.to_csv(output / "trend_extension_daily_weight_contract.csv", index=False, encoding="utf-8-sig")
    trade_ledger.to_csv(output / "trend_extension_trade_intent_ledger.csv", index=False, encoding="utf-8-sig")
    conflict_rows.to_csv(output / "trend_extension_conflict_blocked_rows.csv", index=False, encoding="utf-8-sig")
    caution_audit.to_csv(output / "trend_extension_00631l_caution_audit.csv", index=False, encoding="utf-8-sig")
    future_audit.to_csv(output / "future_data_audit.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([cost_model_metadata()]).to_csv(output / "cost_model_contract.csv", index=False, encoding="utf-8-sig")

    action_allowed = action_contract[action_contract["action_allowed"]].copy()
    manifest = {
        "task_id": TASK_ID,
        "status": "completed_trend_extension_event_to_action_contract_ready",
        "output_dir": str(output),
        "source_outcome_panel": str(outcome_path),
        "candidate_context_source": str(context_path),
        "event_rows_input": int(len(events)),
        "selected_route_event_rows": int(len(selected)),
        "action_contract_rows": int(len(action_contract)),
        "action_allowed_rows": int(len(action_allowed)),
        "daily_weight_rows": int(len(daily_weight)),
        "trade_intent_rows": int(len(trade_ledger)),
        "conflict_blocked_rows": int(len(conflict_rows)),
        "reference_only_blocked_rows": int(len(reference_blocked)),
        "future_data_violation_count": int(future_audit["future_data_violation"].sum()) if not future_audit.empty else 0,
        "formal_direct_stock_target_override_count": int(
            action_contract.loc[action_contract["action_allowed"], "formal_direct_stock_target_active"].sum()
        )
        if not action_contract.empty
        else 0,
        "proxy_rows_in_action_contract": int(action_contract.get("proxy_row", pd.Series(dtype=bool)).fillna(False).sum())
        if not action_contract.empty
        else 0,
        "case_trace_rows_in_action_allowed": int(
            action_contract.loc[action_contract["action_allowed"], "case_trace_only"].sum()
        )
        if not action_contract.empty
        else 0,
        "missing_action_price_rows": int(
            action_contract.loc[action_contract["action_allowed"], ["entry_price", "exit_price"]].isna().any(axis=1).sum()
        )
        if not action_contract.empty
        else 0,
        "uses_forward_return_as_rule": False,
        "forward_return_used_as_evaluation_metadata": True,
        "portfolio_replay_executed": False,
        "formal_model_changed": False,
        "trade_decision_changed": False,
        "active_in_trade_decision": False,
        "report_changed": False,
        "diagnostic_action_contract_only": True,
        "ready_for_experiments_validation": True,
        "handoff_to_experiments_task": EXPERIMENTS_TASK_ID,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "final_summary_zh.md").write_text(_summary(manifest, action_contract, caution_audit), encoding="utf-8")
    pd.DataFrame([{"task_id": TASK_ID, "status": "completed", "output_dir": str(output)}]).to_csv(
        output / "completed.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(columns=["task_id", "status", "reason"]).to_csv(output / "failed.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(
        [
            {"step": "load_exact_outcome_panel", "status": "completed"},
            {"step": "select_bounded_daily_route_events", "status": "completed"},
            {"step": "join_formal_state_and_price_calendar", "status": "completed"},
            {"step": "write_event_to_action_contract", "status": "completed"},
        ]
    ).to_csv(output / "run_log.csv", index=False, encoding="utf-8-sig")
    return manifest


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _load_outcome_panel(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    events = pd.read_csv(path).fillna("")
    required = {
        "signal_date",
        "next_tradable_date",
        "ticker",
        "event_variant",
        "case_trace_only",
        "uses_forward_return_as_rule",
        "entry_price",
        "event_return_20d_pct",
        "event_return_40d_pct",
        "excess_vs_0050_60d_pct",
        "excess_vs_00631L_60d_pct",
    }
    missing = sorted(required - set(events.columns))
    if missing:
        raise ValueError(f"Missing exact outcome fields: {missing}")
    events["signal_date"] = pd.to_datetime(events["signal_date"], errors="coerce")
    events["next_tradable_date"] = pd.to_datetime(events["next_tradable_date"], errors="coerce")
    events = events.dropna(subset=["signal_date", "next_tradable_date"]).copy()
    events["ticker"] = events["ticker"].astype(str).map(_canonical_from_unknown)
    events["event_variant"] = events["event_variant"].astype(str)
    events["case_trace_only"] = events["case_trace_only"].map(_as_bool)
    events["uses_forward_return_as_rule"] = events["uses_forward_return_as_rule"].map(_as_bool)
    events["proxy_row"] = False
    for col in [
        "entry_price",
        "event_return_20d_pct",
        "event_return_40d_pct",
        "excess_vs_0050_60d_pct",
        "excess_vs_00631L_60d_pct",
        "turnover",
        "rs20_vs_0050",
        "rs20_vs_00631L",
    ]:
        if col in events.columns:
            events[col] = pd.to_numeric(events[col], errors="coerce")
    return events


def _select_route_events(events: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    action_source = events[
        ~events["case_trace_only"]
        & ~events["uses_forward_return_as_rule"]
        & events["event_variant"].isin({"trend_ext_slope_acceleration", "trend_ext_ma_stack_breakout"})
    ].copy()

    selected_parts = []
    for route in EVENT_ROUTES:
        part = action_source[action_source["event_variant"].isin(route["allowed_event_variants"])].copy()
        if part.empty:
            continue
        if route["event_route"] == "trend_ext_slope_or_ma_stack_best_daily":
            part["route_priority"] = part["event_variant"].map({"trend_ext_slope_acceleration": 1, "trend_ext_ma_stack_breakout": 2})
        else:
            part["route_priority"] = 1
        part = part.sort_values(["signal_date", "route_priority", "ticker"], ascending=[True, True, True]).drop_duplicates(
            "signal_date", keep="first"
        )
        part["event_route"] = route["event_route"]
        part["event_route_role"] = route["event_route_role"]
        selected_parts.append(part)
    selected = pd.concat(selected_parts, ignore_index=True, sort=False) if selected_parts else pd.DataFrame()

    reference = events[events["event_variant"].eq(REFERENCE_EVENT_VARIANT)].copy()
    if not reference.empty:
        reference["date"] = reference["signal_date"].dt.strftime("%Y-%m-%d")
        reference["signal_date"] = reference["signal_date"].dt.strftime("%Y-%m-%d")
        reference["next_tradable_date"] = reference["next_tradable_date"].dt.strftime("%Y-%m-%d")
        reference["action_allowed"] = False
        reference["action_blocked_reason"] = "reference_only_event_variant_not_allowed_for_action"
        reference["uses_forward_return_as_rule"] = False
        reference["formal_model_changed"] = False
        reference["trade_decision_changed"] = False
        reference["active_in_trade_decision"] = False
        reference["report_changed"] = False
    return selected, reference


def _build_action_contract(
    selected: pd.DataFrame,
    formal: pd.DataFrame,
    calendar: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if selected.empty:
        return pd.DataFrame(), pd.DataFrame()
    selected = selected.copy()
    selected["signal_date_key"] = selected["signal_date"].dt.strftime("%Y-%m-%d")
    selected = selected.merge(formal, on="signal_date_key", how="left")
    rows = []
    blocked = []
    for event in selected.to_dict(orient="records"):
        for design in ACTION_DESIGNS:
            row = _action_row(event, design, calendar)
            rows.append(row)
            if not row["action_allowed"]:
                blocked.append(row)
    return pd.DataFrame(rows), pd.DataFrame(blocked)


def _action_row(event: dict, design: dict, calendar: list[str]) -> dict:
    signal_date = _date_text(event.get("signal_date"))
    entry_date = _date_text(event.get("next_tradable_date"))
    ticker = str(event.get("ticker") or "")
    formal_state = str(event.get("formal_state") or "missing_formal_state")
    formal_target = str(event.get("formal_target") or "")
    direct_stock = bool(event.get("formal_direct_stock_target_active", False))
    allowed_state = formal_state in {"cash", "no_target", "defensive_market_exposure", "market_exposure"}
    action_variant = f"{event.get('event_route')}_{design['action_design']}"
    entry_price = pd.to_numeric(pd.Series([event.get("entry_price")]), errors="coerce").iloc[0]
    exit_date, exit_price, exit_reason, hold_days = _resolve_exit(
        event=event,
        calendar=calendar,
        ticker=ticker,
        entry_date=entry_date,
        max_hold_days=int(design["max_hold_days"]),
    )
    blocked_reason = ""
    if direct_stock:
        blocked_reason = "blocked_formal_direct_stock_target_active_no_override"
    elif not allowed_state:
        blocked_reason = f"blocked_formal_state_{formal_state}_not_activation_state"
    elif not entry_date:
        blocked_reason = "blocked_missing_next_tradable_date"
    elif bool(event.get("case_trace_only", False)):
        blocked_reason = "blocked_case_trace_only_excluded_from_action"
    elif bool(event.get("proxy_row", False)):
        blocked_reason = "blocked_proxy_row_not_allowed"
    elif pd.isna(entry_price) or pd.isna(exit_price):
        blocked_reason = "blocked_missing_entry_or_exit_price"
    action_allowed = blocked_reason == ""
    sleeve_weight = float(design["sleeve_weight"]) if action_allowed else 0.0
    notional = 1_000_000.0 * sleeve_weight
    cost_model = TaiwanCostModel()
    trade_cost = cost_model.buy_cost(notional) + cost_model.sell_cost(notional, "stock") if action_allowed else 0.0
    return {
        "date": signal_date,
        "signal_date": signal_date,
        "next_tradable_date": entry_date,
        "ticker": ticker,
        "candidate_name": event.get("candidate_name", ""),
        "candidate_source": event.get("candidate_source", ""),
        "candidate_layer": event.get("candidate_layer", ""),
        "event_variant": event.get("event_variant", ""),
        "event_route": event.get("event_route", ""),
        "event_route_role": event.get("event_route_role", ""),
        "action_variant": action_variant,
        "action_design": design["action_design"],
        "action_design_role": design["design_role"],
        "formal_target": formal_target,
        "formal_state": formal_state,
        "formal_direct_stock_target_active": direct_stock,
        "action_allowed": action_allowed,
        "action_blocked_reason": blocked_reason,
        "sleeve_weight_candidate": sleeve_weight,
        "entry_date": entry_date if action_allowed else "",
        "entry_price": entry_price if action_allowed else pd.NA,
        "exit_date": exit_date if action_allowed else "",
        "exit_price": exit_price if action_allowed else pd.NA,
        "hold_days": hold_days if action_allowed else 0,
        "exit_rule": exit_reason if action_allowed else "",
        "trade_cost": trade_cost,
        "turnover": round(float(sleeve_weight * 2.0), 8) if action_allowed else 0.0,
        "case_trace_only": bool(event.get("case_trace_only", False)),
        "proxy_row": bool(event.get("proxy_row", False)),
        "event_excess_vs_0050_60d_eval_only": event.get("excess_vs_0050_60d_pct", pd.NA),
        "event_excess_vs_00631l_60d_eval_only": event.get("excess_vs_00631L_60d_pct", pd.NA),
        "uses_forward_return_as_rule": False,
        "formal_model_changed": False,
        "trade_decision_changed": False,
        "active_in_trade_decision": False,
        "report_changed": False,
    }


def _resolve_exit(
    *,
    event: dict,
    calendar: list[str],
    ticker: str,
    entry_date: str,
    max_hold_days: int,
) -> tuple[str, float, str, int]:
    if not entry_date or max_hold_days <= 0:
        return "", pd.NA, "", 0
    future_dates = [date for date in calendar if date >= entry_date]
    if not future_dates:
        return "", pd.NA, "missing_exit_calendar", 0
    max_index = min(max_hold_days, len(future_dates) - 1)
    exit_date = future_dates[max_index]
    entry_price = pd.to_numeric(pd.Series([event.get("entry_price")]), errors="coerce").iloc[0]
    return_col = f"event_return_{max_hold_days}d_pct"
    event_return = pd.to_numeric(pd.Series([event.get(return_col)]), errors="coerce").iloc[0]
    if pd.isna(entry_price) or pd.isna(event_return):
        exit_price = pd.NA
    else:
        exit_price = float(entry_price) * (1.0 + float(event_return) / 100.0)
    return exit_date, exit_price, f"max_hold_{max_hold_days}_trading_days", int(max_index)


def _build_daily_weight_contract(contract: pd.DataFrame) -> pd.DataFrame:
    allowed = contract[contract["action_allowed"]].copy()
    if allowed.empty:
        return pd.DataFrame(columns=["date", "action_variant", "ticker", "dynamic_weight", "formal_residual_weight"])
    allowed["dynamic_weight"] = allowed["sleeve_weight_candidate"]
    allowed["formal_residual_weight"] = 1.0 - allowed["dynamic_weight"]
    return allowed[
        [
            "date",
            "signal_date",
            "next_tradable_date",
            "action_variant",
            "event_variant",
            "ticker",
            "candidate_name",
            "dynamic_weight",
            "formal_residual_weight",
            "entry_date",
            "exit_date",
            "exit_rule",
            "active_in_trade_decision",
        ]
    ]


def _build_trade_intent_ledger(contract: pd.DataFrame) -> pd.DataFrame:
    allowed = contract[contract["action_allowed"]].copy()
    if allowed.empty:
        return pd.DataFrame(columns=["signal_date", "action_variant", "ticker", "entry_date", "exit_date", "trade_cost"])
    allowed["entry_intent"] = "buy_trend_extension_sleeve_next_day"
    allowed["exit_intent"] = allowed["exit_rule"].map(lambda rule: f"sell_trend_extension_sleeve_{rule}")
    return allowed[
        [
            "signal_date",
            "next_tradable_date",
            "action_variant",
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
            "uses_forward_return_as_rule",
        ]
    ]


def _build_00631l_caution_audit(contract: pd.DataFrame) -> pd.DataFrame:
    if contract.empty:
        return pd.DataFrame(columns=["action_variant", "rows", "avg_excess_vs_00631l_60d_eval_only", "caution_state"])
    return (
        contract.groupby(["event_route", "action_design", "action_variant"], as_index=False)
        .agg(
            rows=("ticker", "count"),
            action_allowed_rows=("action_allowed", "sum"),
            avg_excess_vs_0050_60d_eval_only=("event_excess_vs_0050_60d_eval_only", "mean"),
            avg_excess_vs_00631l_60d_eval_only=("event_excess_vs_00631l_60d_eval_only", "mean"),
        )
        .assign(
            caution_state=lambda df: df["avg_excess_vs_00631l_60d_eval_only"].map(
                lambda value: "00631l_caution_negative_or_incomplete"
                if pd.isna(value) or float(value) < 0
                else "00631l_caution_still_required_research_boundary"
            ),
            used_as_rule=False,
        )
    )


def _future_data_audit(contract: pd.DataFrame) -> pd.DataFrame:
    if contract.empty:
        return pd.DataFrame(columns=["signal_date", "ticker", "action_variant", "future_data_violation", "reason"])
    out = contract[["signal_date", "next_tradable_date", "ticker", "action_variant", "event_variant"]].copy()
    out["future_data_violation"] = False
    out["reason"] = ""
    return out


def _canonical_from_unknown(ticker: str) -> str:
    text = str(ticker).strip()
    if "." in text:
        return text
    return text


def _load_market_lookup(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    try:
        frame = pd.read_csv(path, usecols=lambda col: col in {"ticker", "market"}).fillna("")
    except (OSError, ValueError):
        return {}
    lookup = {}
    for row in frame.to_dict(orient="records"):
        ticker = str(row.get("ticker") or "").strip().split(".")[0]
        market = str(row.get("market") or "").strip()
        if not ticker or market not in {"TWSE", "TPEx"}:
            continue
        suffix = ".TW" if market == "TWSE" else ".TWO"
        lookup.setdefault(ticker, f"{ticker}{suffix}")
    return lookup


def _canonical_from_market_lookup(lookup: dict[str, str]):
    def convert(ticker: str) -> str:
        text = str(ticker).strip()
        if "." in text:
            return text
        return lookup.get(text, text)

    return convert


def _canonical_from_price_table(price_table: pd.DataFrame):
    lookup = {}
    if "canonical_ticker" in price_table.columns:
        for canonical in price_table["canonical_ticker"].dropna().astype(str).unique():
            lookup.setdefault(canonical.split(".")[0], canonical)
            lookup.setdefault(canonical, canonical)

    def convert(ticker: str) -> str:
        text = str(ticker).strip()
        return lookup.get(text, lookup.get(text.split(".")[0], text))

    return convert


def _as_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def _numeric_series(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(0.0, index=frame.index)
    return pd.to_numeric(frame[column], errors="coerce").fillna(0.0)


def _summary(manifest: dict, contract: pd.DataFrame, caution: pd.DataFrame) -> str:
    if contract.empty:
        variant_summary = "no rows"
    else:
        variant_summary = (
            contract.groupby("action_variant", as_index=False)
            .agg(action_allowed_rows=("action_allowed", "sum"), total_rows=("ticker", "count"))
            .to_csv(index=False)
            .strip()
        )
    if caution.empty:
        caution_summary = "no rows"
    else:
        caution_summary = caution.to_csv(index=False).strip()
    return "\n".join(
        [
            "# Strong stock trend-extension event-to-action contract",
            "",
            "本包只定義 trend-extension bounded event-to-action diagnostic contract；不改正式模型、日報或交易決策，也不執行 portfolio replay。",
            "",
            f"- action contract rows：{manifest['action_contract_rows']}",
            f"- action allowed rows：{manifest['action_allowed_rows']}",
            f"- conflict blocked rows：{manifest['conflict_blocked_rows']}",
            f"- future data violation count：{manifest['future_data_violation_count']}",
            f"- formal direct stock target override count：{manifest['formal_direct_stock_target_override_count']}",
            f"- proxy rows in action contract：{manifest['proxy_rows_in_action_contract']}",
            f"- case trace rows in action allowed：{manifest['case_trace_rows_in_action_allowed']}",
            "",
            "## Action variant summary",
            variant_summary,
            "",
            "## 00631L caution audit",
            caution_summary,
        ]
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--outcome-panel", default=str(DEFAULT_OUTCOME_PANEL))
    parser.add_argument("--candidate-context", default=str(DEFAULT_CANDIDATE_CONTEXT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args(argv)
    manifest = run_strong_stock_trend_extension_event_to_action_contract(
        repo_root=args.repo_root,
        outcome_panel=args.outcome_panel,
        candidate_context=args.candidate_context,
        output_dir=args.output_dir,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
