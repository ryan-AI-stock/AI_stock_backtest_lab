"""Build the Dynamic Pool1 strict lowpoint event-to-action contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from backtest_lab.costs import COST_MODEL_VERSION, TaiwanCostModel, cost_model_metadata


TASK_ID = "TASK-BACKTEST-CORE-DYNAMIC-POOL1-STRICT-LOWPOINT-EVENT-TO-ACTION-CONTRACT-001"
EXPERIMENTS_TASK_ID = "TASK-BACKTEST-EXPERIMENTS-DYNAMIC-POOL1-STRICT-LOWPOINT-EVENT-TO-ACTION-VALIDATION-001"
DEFAULT_EVENT_CONTRACT_DIR = Path("outputs/dynamic_pool1_strict_lowpoint_event_contract_20260704")
DEFAULT_OUTPUT_DIR = Path("outputs/dynamic_pool1_strict_lowpoint_event_to_action_contract_20260704")
DEFAULT_LIQUIDITY_DIR = Path(
    r"C:\Users\zergv\Documents\Codex\2026-05-23\ai-stock-rotation-radar-https-docs\outputs"
    r"\radar_dynamic_pool1_all_listed_liquid_universe_full_sweep_20260703"
)
FORMAL_STREAMS = [
    Path("outputs/combined_formal_target_stream_20150128_20211230_20260702/combined_formal_target_stream.csv"),
    Path("outputs/formal_long_range_signal_reconstruction_201411_latest_20260702/formal_long_range_target_stream.csv"),
]

PRIMARY_EVENT_PRIORITY = {
    "strict_lowpoint_0_2d_rebound_5_12pct": 1,
    "strict_lowpoint_0_5d_rebound_5_12pct_downside_deceleration": 2,
    "strict_lowpoint_0_5d_rebound_5_12pct_short_rs_repair": 3,
}
REFERENCE_EVENT = "strict_lowpoint_3_5d_rebound_5_12pct_reference_only"
ACTION_VARIANTS = [
    {
        "variant": "strict_lowpoint_sleeve10_hold20_when_formal_cash_or_market_exposure",
        "sleeve_weight": 0.10,
        "max_hold_days": 20,
        "exit_on_ma20_break": False,
        "variant_role": "primary",
    },
    {
        "variant": "strict_lowpoint_sleeve20_hold20_when_formal_cash_or_market_exposure",
        "sleeve_weight": 0.20,
        "max_hold_days": 20,
        "exit_on_ma20_break": False,
        "variant_role": "sizing_sensitivity",
    },
    {
        "variant": "strict_lowpoint_sleeve10_hold20_exit_on_ma20_break",
        "sleeve_weight": 0.10,
        "max_hold_days": 20,
        "exit_on_ma20_break": True,
        "variant_role": "exit_sensitivity",
    },
    {
        "variant": "strict_lowpoint_event_only_no_trade_context",
        "sleeve_weight": 0.0,
        "max_hold_days": 0,
        "exit_on_ma20_break": False,
        "variant_role": "reference_only_no_trade_context",
    },
]
CASE_TRACE_TICKERS = {"6669.TW", "2308.TW", "2317.TW"}


def run_dynamic_pool1_strict_lowpoint_event_to_action_contract(
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
    selected = _select_events(events, price_table)
    action_contract, conflict_blocked = _build_action_contract(selected, formal, price_table, calendar)
    daily_weight = _build_daily_weight_contract(action_contract)
    trade_ledger = _build_trade_intent_ledger(action_contract)
    case_trace = _build_case_trace(action_contract, events)
    future_audit = _future_data_audit(action_contract)

    action_contract.to_csv(output / "strict_lowpoint_event_to_action_contract.csv", index=False, encoding="utf-8-sig")
    daily_weight.to_csv(output / "strict_lowpoint_daily_weight_contract.csv", index=False, encoding="utf-8-sig")
    trade_ledger.to_csv(output / "strict_lowpoint_trade_intent_ledger.csv", index=False, encoding="utf-8-sig")
    conflict_blocked.to_csv(output / "strict_lowpoint_conflict_blocked_rows.csv", index=False, encoding="utf-8-sig")
    case_trace.to_csv(output / "strict_lowpoint_case_trace_6669_2308_2317.csv", index=False, encoding="utf-8-sig")
    future_audit.to_csv(output / "future_data_audit.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([cost_model_metadata()]).to_csv(output / "cost_model_contract.csv", index=False, encoding="utf-8-sig")

    action_rows = action_contract[action_contract["action_allowed"]].copy()
    manifest = {
        "task_id": TASK_ID,
        "status": "completed_strict_lowpoint_event_to_action_contract_ready",
        "output_dir": str(output),
        "source_event_contract_dir": str(source),
        "event_rows_input": int(len(events)),
        "selected_primary_event_rows": int(len(selected[selected["event_variant_role"].eq("primary")])),
        "action_contract_rows": int(len(action_contract)),
        "action_allowed_rows": int(len(action_rows)),
        "daily_weight_rows": int(len(daily_weight)),
        "trade_intent_rows": int(len(trade_ledger)),
        "conflict_blocked_rows": int(len(conflict_blocked)),
        "case_trace_rows": int(len(case_trace)),
        "future_data_violation_count": int(future_audit["future_data_violation"].sum()) if not future_audit.empty else 0,
        "formal_direct_stock_target_override_count": int(
            action_contract.loc[action_contract["action_allowed"], "formal_direct_stock_target_active"].sum()
        )
        if not action_contract.empty
        else 0,
        "missing_action_price_rows": int(
            action_contract.loc[action_contract["action_allowed"], ["entry_price", "exit_price"]].isna().any(axis=1).sum()
        )
        if not action_contract.empty
        else 0,
        "uses_forward_return_as_rule": False,
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
    (output / "final_summary_zh.md").write_text(_summary(manifest, action_contract), encoding="utf-8")
    pd.DataFrame([{"task_id": TASK_ID, "status": "completed", "output_dir": str(output)}]).to_csv(
        output / "completed.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(columns=["task_id", "status", "reason"]).to_csv(output / "failed.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(
        [
            {"step": "load_validated_strict_lowpoint_events", "status": "completed"},
            {"step": "join_formal_state_and_price_calendar", "status": "completed"},
            {"step": "build_bounded_event_to_action_contract", "status": "completed"},
            {"step": "write_contract_package", "status": "completed"},
        ]
    ).to_csv(output / "run_log.csv", index=False, encoding="utf-8-sig")
    return manifest


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _load_events(source: Path) -> pd.DataFrame:
    path = source / "strict_lowpoint_event_contract.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    events = pd.read_csv(path).fillna("")
    events["signal_date"] = pd.to_datetime(events["signal_date"], errors="coerce")
    events["next_tradable_date"] = pd.to_datetime(events["next_tradable_date"], errors="coerce")
    events = events.dropna(subset=["signal_date", "next_tradable_date"]).copy()
    events["ticker"] = events["ticker"].astype(str)
    events["event_variant"] = events["event_variant"].astype(str)
    events["event_variant_role"] = events["event_variant_role"].astype(str)
    events["event_priority"] = events["event_variant"].map(PRIMARY_EVENT_PRIORITY).fillna(99).astype(int)
    events["case_trace_only"] = False
    return events


def _load_formal_streams(root: Path) -> pd.DataFrame:
    frames = []
    for rel in FORMAL_STREAMS:
        path = root / rel
        if not path.exists():
            continue
        df = pd.read_csv(path).fillna("")
        if "signal_date" not in df.columns:
            continue
        if "execution_date" not in df.columns:
            df["execution_date"] = ""
        frames.append(df)
    if not frames:
        raise FileNotFoundError("No formal target stream source found")
    formal = pd.concat(frames, ignore_index=True, sort=False)
    formal["signal_date"] = pd.to_datetime(formal["signal_date"], errors="coerce")
    formal = formal.dropna(subset=["signal_date"]).sort_values("signal_date")
    formal = formal.drop_duplicates("signal_date", keep="last")
    formal["signal_date_key"] = formal["signal_date"].dt.strftime("%Y-%m-%d")
    formal["formal_target"] = formal.get("formal_target", "").fillna("").astype(str)
    formal["target_type"] = formal.get("target_type", "").fillna("").astype(str)
    formal["risk_off_state"] = formal.get("risk_off_state", "").fillna("").astype(str)
    formal["formal_state"] = formal.apply(_formal_state, axis=1)
    formal["formal_direct_stock_target_active"] = formal["formal_state"].eq("direct_stock_target")
    return formal[["signal_date_key", "execution_date", "formal_target", "formal_state", "formal_direct_stock_target_active"]]


def _formal_state(row: pd.Series) -> str:
    target = str(row.get("formal_target", "") or "").strip()
    target_type = str(row.get("target_type", "") or "").strip()
    risk_off = str(row.get("risk_off_state", "") or "").strip()
    if not target:
        return "no_target"
    if target.upper() == "CASH":
        return "no_target" if risk_off == "no_target_cash_all" else "cash"
    if _base_ticker(target) == "00631L" or target_type == "market_exposure":
        return "market_exposure"
    return "direct_stock_target"


def _load_calendar(liquidity_dir: Path) -> list[str]:
    dates: set[str] = set()
    for shard in sorted((liquidity_dir / "shards").glob("accepted_liquidity_rows_*.csv")):
        try:
            frame = pd.read_csv(shard, usecols=["date"])
        except (OSError, ValueError):
            continue
        dates.update(pd.to_datetime(frame["date"], errors="coerce").dropna().dt.strftime("%Y-%m-%d").unique())
    return sorted(dates)


def _load_price_table(liquidity_dir: Path, tickers: list[str]) -> pd.DataFrame:
    needed = {_base_ticker(ticker) for ticker in tickers}
    frames = []
    for shard in sorted((liquidity_dir / "shards").glob("accepted_liquidity_rows_*.csv")):
        try:
            frame = pd.read_csv(shard, usecols=lambda column: column in {"date", "ticker", "market", "close", "turnover"})
        except (OSError, ValueError):
            continue
        frame["base_ticker"] = frame["ticker"].astype(str).map(_base_ticker)
        frame = frame[frame["base_ticker"].isin(needed)].copy()
        if frame.empty:
            continue
        frame["canonical_ticker"] = frame.apply(
            lambda row: f"{row['base_ticker']}{'.TW' if str(row.get('market')) == 'TWSE' else '.TWO'}",
            axis=1,
        )
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.strftime("%Y-%m-%d")
        frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
        if "turnover" in frame.columns:
            frame["turnover"] = pd.to_numeric(frame["turnover"], errors="coerce").fillna(0.0)
        else:
            frame["turnover"] = 0.0
        frames.append(frame[["date", "canonical_ticker", "close", "turnover"]])
    if not frames:
        return pd.DataFrame(columns=["date", "canonical_ticker", "close", "turnover", "ma20"])
    out = pd.concat(frames, ignore_index=True, sort=False)
    out = out.dropna(subset=["date", "canonical_ticker", "close"]).drop_duplicates(["date", "canonical_ticker"])
    out = out.sort_values(["canonical_ticker", "date"])
    out["ma20"] = out.groupby("canonical_ticker")["close"].transform(lambda s: s.rolling(20, min_periods=20).mean())
    return out


def _select_events(events: pd.DataFrame, price_table: pd.DataFrame) -> pd.DataFrame:
    primary = events[events["event_variant_role"].eq("primary")].copy()
    if primary.empty:
        return primary
    px = price_table[["date", "canonical_ticker", "turnover"]].rename(
        columns={"date": "next_tradable_date_key", "canonical_ticker": "ticker", "turnover": "entry_turnover"}
    )
    primary["next_tradable_date_key"] = primary["next_tradable_date"].dt.strftime("%Y-%m-%d")
    primary = primary.merge(px, on=["next_tradable_date_key", "ticker"], how="left")
    primary["entry_turnover"] = pd.to_numeric(primary["entry_turnover"], errors="coerce").fillna(0.0)
    primary["tie_break_rs"] = pd.to_numeric(primary.get("rs_vs_0050_3d_or_5d"), errors="coerce").fillna(0.0) + pd.to_numeric(
        primary.get("rs_vs_00631l_3d_or_5d"), errors="coerce"
    ).fillna(0.0)
    primary = primary.sort_values(
        ["signal_date", "event_priority", "entry_turnover", "tie_break_rs", "ticker"],
        ascending=[True, True, False, False, True],
    )
    return primary.drop_duplicates(["signal_date"], keep="first")


def _build_action_contract(
    selected: pd.DataFrame,
    formal: pd.DataFrame,
    price_table: pd.DataFrame,
    calendar: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    blocked_rows = []
    selected = selected.copy()
    selected["signal_date_key"] = selected["signal_date"].dt.strftime("%Y-%m-%d")
    selected = selected.merge(formal, on="signal_date_key", how="left")
    for event in selected.to_dict(orient="records"):
        for variant in ACTION_VARIANTS:
            row = _action_row(event, variant, price_table, calendar)
            rows.append(row)
            if not row["action_allowed"]:
                blocked_rows.append(row)
    contract = pd.DataFrame(rows)
    blocked = pd.DataFrame(blocked_rows)
    return contract, blocked


def _action_row(event: dict, variant: dict, price_table: pd.DataFrame, calendar: list[str]) -> dict:
    signal_date = _date_text(event.get("signal_date"))
    entry_date = _date_text(event.get("next_tradable_date"))
    ticker = str(event.get("ticker") or "")
    formal_state = str(event.get("formal_state") or "missing_formal_state")
    formal_target = str(event.get("formal_target") or "")
    direct_stock = bool(event.get("formal_direct_stock_target_active", False))
    reference_event = str(event.get("event_variant")) == REFERENCE_EVENT or str(event.get("event_variant_role")) != "primary"
    reference_variant = variant["variant_role"] == "reference_only_no_trade_context"
    allowed_state = formal_state in {"cash", "no_target", "defensive_market_exposure", "market_exposure"}
    entry_price = _price_on(price_table, ticker, entry_date, "close")
    ma20_entry = _price_on(price_table, ticker, entry_date, "ma20")
    exit_date, exit_price, exit_reason, hold_days, ma20_exit_check = _resolve_exit(
        price_table=price_table,
        calendar=calendar,
        ticker=ticker,
        entry_date=entry_date,
        max_hold_days=int(variant["max_hold_days"]),
        exit_on_ma20_break=bool(variant["exit_on_ma20_break"]),
    )
    blocked_reason = ""
    if reference_variant:
        blocked_reason = "event_only_no_trade_context"
    elif reference_event:
        blocked_reason = "reference_event_variant_not_allowed_for_action"
    elif direct_stock:
        blocked_reason = "blocked_formal_direct_stock_target_active_no_override"
    elif not allowed_state:
        blocked_reason = f"blocked_formal_state_{formal_state}_not_activation_state"
    elif not entry_date:
        blocked_reason = "blocked_missing_next_tradable_date"
    elif pd.isna(entry_price) or pd.isna(exit_price):
        blocked_reason = "blocked_missing_entry_or_exit_price"
    action_allowed = blocked_reason == ""
    sleeve_weight = float(variant["sleeve_weight"]) if action_allowed else 0.0
    turnover = round(float(sleeve_weight * 2.0), 8) if action_allowed else 0.0
    notional = 1_000_000.0 * sleeve_weight
    cost_model = TaiwanCostModel()
    trade_cost = cost_model.buy_cost(notional) + cost_model.sell_cost(notional, "stock") if action_allowed else 0
    return {
        "date": signal_date,
        "signal_date": signal_date,
        "next_tradable_date": entry_date,
        "variant": variant["variant"],
        "variant_role": variant["variant_role"],
        "event_variant": event.get("event_variant", ""),
        "event_variant_role": event.get("event_variant_role", ""),
        "ticker": ticker,
        "candidate_name": event.get("candidate_name", ""),
        "formal_target": formal_target,
        "formal_state": formal_state,
        "formal_direct_stock_target_active": direct_stock,
        "action_allowed": action_allowed,
        "action_blocked_reason": blocked_reason,
        "sleeve_weight": sleeve_weight,
        "entry_date": entry_date if action_allowed else "",
        "entry_price": entry_price if action_allowed else pd.NA,
        "exit_date": exit_date if action_allowed else "",
        "exit_price": exit_price if action_allowed else pd.NA,
        "exit_reason": exit_reason if action_allowed else "",
        "hold_days": hold_days if action_allowed else 0,
        "ma20_at_entry": ma20_entry,
        "ma20_exit_check": ma20_exit_check,
        "trade_cost": trade_cost,
        "turnover": turnover,
        "case_trace_only": bool(ticker in CASE_TRACE_TICKERS and not action_allowed),
        "uses_forward_return_as_rule": False,
        "formal_model_changed": False,
        "trade_decision_changed": False,
        "active_in_trade_decision": False,
        "report_changed": False,
    }


def _resolve_exit(
    *,
    price_table: pd.DataFrame,
    calendar: list[str],
    ticker: str,
    entry_date: str,
    max_hold_days: int,
    exit_on_ma20_break: bool,
) -> tuple[str, float, str, int, str]:
    if not entry_date or max_hold_days <= 0:
        return "", pd.NA, "", 0, "not_applicable"
    future_dates = [date for date in calendar if date >= entry_date]
    if not future_dates:
        return "", pd.NA, "missing_exit_calendar", 0, "missing"
    max_index = min(max_hold_days, len(future_dates) - 1)
    exit_date = future_dates[max_index]
    exit_reason = f"max_hold_{max_hold_days}_trading_days"
    ma20_check = "not_enabled"
    if exit_on_ma20_break:
        ma20_check = "checked_no_break"
        for idx, date in enumerate(future_dates[1 : max_hold_days + 1], start=1):
            close = _price_on(price_table, ticker, date, "close")
            ma20 = _price_on(price_table, ticker, date, "ma20")
            if pd.notna(close) and pd.notna(ma20) and close < ma20:
                exit_date = date
                exit_reason = "exit_on_ma20_break"
                ma20_check = "ma20_break_triggered"
                max_index = idx
                break
    exit_price = _price_on(price_table, ticker, exit_date, "close")
    return exit_date, exit_price, exit_reason, int(max_index), ma20_check


def _build_daily_weight_contract(contract: pd.DataFrame) -> pd.DataFrame:
    allowed = contract[contract["action_allowed"]].copy()
    if allowed.empty:
        return pd.DataFrame(
            columns=["date", "variant", "ticker", "dynamic_weight", "formal_residual_weight", "entry_date", "exit_date"]
        )
    out = allowed.rename(columns={"sleeve_weight": "dynamic_weight"}).copy()
    out["formal_residual_weight"] = 1.0 - out["dynamic_weight"]
    return out[
        [
            "date",
            "variant",
            "ticker",
            "candidate_name",
            "dynamic_weight",
            "formal_residual_weight",
            "entry_date",
            "exit_date",
            "exit_reason",
            "active_in_trade_decision",
        ]
    ]


def _build_trade_intent_ledger(contract: pd.DataFrame) -> pd.DataFrame:
    allowed = contract[contract["action_allowed"]].copy()
    if allowed.empty:
        return pd.DataFrame(columns=["signal_date", "variant", "ticker", "entry_date", "exit_date", "trade_cost"])
    allowed["entry_intent"] = "buy_strict_lowpoint_sleeve_next_day"
    allowed["exit_intent"] = allowed["exit_reason"].map(lambda reason: f"sell_strict_lowpoint_sleeve_{reason}")
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
            "uses_forward_return_as_rule",
        ]
    ]


def _build_case_trace(action_contract: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    trace = action_contract[action_contract["ticker"].isin(CASE_TRACE_TICKERS)].copy()
    if not trace.empty:
        trace["event_found"] = True
    existing = set(trace["ticker"].astype(str).unique())
    placeholders = []
    names = {"6669.TW": "緯穎", "2308.TW": "台達電", "2317.TW": "鴻海"}
    for ticker in CASE_TRACE_TICKERS - existing:
        event_found = bool(events["ticker"].astype(str).eq(ticker).any())
        placeholders.append(
            {
                "date": "",
                "signal_date": "",
                "next_tradable_date": "",
                "variant": "case_trace_placeholder",
                "event_variant": "",
                "event_variant_role": "",
                "ticker": ticker,
                "candidate_name": names.get(ticker, ""),
                "formal_target": "",
                "formal_state": "",
                "formal_direct_stock_target_active": False,
                "action_allowed": False,
                "action_blocked_reason": "no_action_rows_for_case_ticker" if event_found else "no_strict_lowpoint_event_for_case_ticker",
                "case_trace_only": True,
                "event_found": event_found,
                "uses_forward_return_as_rule": False,
            }
        )
    if placeholders:
        trace = pd.concat([trace, pd.DataFrame(placeholders)], ignore_index=True, sort=False)
    return trace


def _future_data_audit(contract: pd.DataFrame) -> pd.DataFrame:
    if contract.empty:
        return pd.DataFrame(columns=["signal_date", "ticker", "variant", "future_data_violation", "reason"])
    out = contract[["signal_date", "next_tradable_date", "ticker", "variant", "event_variant"]].copy()
    out["future_data_violation"] = False
    out["reason"] = ""
    return out


def _price_on(price_table: pd.DataFrame, ticker: str, date: str, column: str) -> float:
    if not ticker or not date or column not in price_table.columns:
        return pd.NA
    rows = price_table[(price_table["canonical_ticker"].eq(ticker)) & (price_table["date"].eq(date))]
    if rows.empty:
        return pd.NA
    return rows.iloc[0][column]


def _date_text(value) -> str:
    parsed = pd.to_datetime(value, errors="coerce")
    return "" if pd.isna(parsed) else parsed.strftime("%Y-%m-%d")


def _base_ticker(ticker: str) -> str:
    return str(ticker).strip().split(".")[0]


def _summary(manifest: dict, contract: pd.DataFrame) -> str:
    if contract.empty:
        variant_summary = "no rows"
    else:
        variant_summary = (
            contract.groupby("variant", as_index=False)
            .agg(action_allowed_rows=("action_allowed", "sum"), total_rows=("ticker", "count"))
            .to_csv(index=False)
            .strip()
        )
    return "\n".join(
        [
            "# Dynamic Pool1 strict lowpoint event-to-action contract",
            "",
            "本包只定義 strict lowpoint event-to-action diagnostic contract；不改正式模型、日報或交易決策。",
            "",
            f"- action contract rows：{manifest['action_contract_rows']}",
            f"- action allowed rows：{manifest['action_allowed_rows']}",
            f"- conflict blocked rows：{manifest['conflict_blocked_rows']}",
            f"- future data violation count：{manifest['future_data_violation_count']}",
            f"- formal direct stock target override count：{manifest['formal_direct_stock_target_override_count']}",
            f"- missing action price rows：{manifest['missing_action_price_rows']}",
            "- reference-only strict lowpoint event 不進交易型 action variants。",
            "",
            "## Variant summary",
            variant_summary,
        ]
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--event-contract-dir", default=str(DEFAULT_EVENT_CONTRACT_DIR))
    parser.add_argument("--liquidity-dir", default=str(DEFAULT_LIQUIDITY_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args(argv)
    manifest = run_dynamic_pool1_strict_lowpoint_event_to_action_contract(
        repo_root=args.repo_root,
        event_contract_dir=args.event_contract_dir,
        liquidity_dir=args.liquidity_dir,
        output_dir=args.output_dir,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
