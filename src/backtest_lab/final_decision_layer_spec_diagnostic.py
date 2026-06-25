from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_EVENT_PANEL = "outputs/pool3_event_level_decision_diff_independent_base_pit_pool2_20260625/event_decision_diff_panel.csv"
DEFAULT_OUTPUT_DIR = "outputs/final_decision_layer_spec_diagnostic_20260625"
REPORT_ONLY_BOUNDARY = "report_only_diagnostic"
MARKET_EXPOSURE_TICKERS = {"0050", "0050.TW", "00631L", "00631L.TW"}
FORMAL_POOL_IDS = ("pool1", "pool2")
ALL_POOL_IDS = ("pool1", "pool2", "pool3")


def run_final_decision_layer_spec_diagnostic(
    *,
    event_panel_path: str | Path = DEFAULT_EVENT_PANEL,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> Path:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    run_log: list[dict[str, str]] = []

    def log(step: str, status: str, detail: str = "") -> None:
        run_log.append(
            {
                "timestamp": pd.Timestamp.now(tz="Asia/Taipei").strftime("%Y-%m-%d %H:%M:%S%z"),
                "step": step,
                "status": status,
                "detail": detail,
            }
        )
        pd.DataFrame(run_log).to_csv(output / "run_log.csv", index=False, encoding="utf-8-sig")
        (output / "current_step.txt").write_text(step, encoding="utf-8")

    try:
        log("load_inputs", "started", str(event_panel_path))
        events = pd.read_csv(event_panel_path).fillna("")
        _validate_event_panel(events)

        log("build_panels", "started", "")
        pool_panel = build_pool_signal_normalized_panel(events)
        state_panel = build_final_decision_state_panel(events, pool_panel)
        priority_panel = build_target_priority_panel(state_panel, pool_panel)
        exposure_panel = build_market_exposure_tool_panel(state_panel, pool_panel)
        health = build_consensus_health_by_period(state_panel)
        protocol = build_protocol_usage_by_period(state_panel)
        fake = build_fake_health_flags(state_panel)
        transitions = build_state_transition_summary(state_panel)
        distribution = build_state_distribution_summary(state_panel)

        log("write_outputs", "started", "")
        pool_panel.to_csv(output / "pool_signal_normalized_panel.csv", index=False, encoding="utf-8-sig")
        state_panel.to_csv(output / "final_decision_state_panel.csv", index=False, encoding="utf-8-sig")
        priority_panel.to_csv(output / "target_priority_panel.csv", index=False, encoding="utf-8-sig")
        exposure_panel.to_csv(output / "market_exposure_tool_panel.csv", index=False, encoding="utf-8-sig")
        health.to_csv(output / "consensus_health_by_period.csv", index=False, encoding="utf-8-sig")
        protocol.to_csv(output / "protocol_usage_by_period.csv", index=False, encoding="utf-8-sig")
        fake.to_csv(output / "fake_health_flags.csv", index=False, encoding="utf-8-sig")
        transitions.to_csv(output / "state_transition_summary.csv", index=False, encoding="utf-8-sig")
        distribution.to_csv(output / "state_distribution_summary.csv", index=False, encoding="utf-8-sig")
        (output / "final_decision_layer_spec_summary_zh.md").write_text(
            _summary_markdown(health, protocol, fake),
            encoding="utf-8",
        )
        manifest = {
            "schema_version": 1,
            "task_id": "TASK-BACKTEST-CORE-FINAL-DECISION-LAYER-SPEC-DIAGNOSTIC-001",
            "model": "final_decision_layer_spec_diagnostic",
            "status": "completed",
            "event_panel_path": str(event_panel_path),
            "formal_model_changed": False,
            "trade_decision_changed": False,
            "active_in_trade_decision": False,
            "final_decision_layer_boundary": REPORT_ONLY_BOUNDARY,
            "pool3_shadow_used_as_formal": False,
            "etf_counted_as_stock_vote": False,
            "rr_partial_switch_used": False,
            "valuation_used": False,
            "h3_used": False,
            "outputs": {
                "pool_signal_normalized_panel": "pool_signal_normalized_panel.csv",
                "final_decision_state_panel": "final_decision_state_panel.csv",
                "target_priority_panel": "target_priority_panel.csv",
                "market_exposure_tool_panel": "market_exposure_tool_panel.csv",
                "consensus_health_by_period": "consensus_health_by_period.csv",
                "protocol_usage_by_period": "protocol_usage_by_period.csv",
                "fake_health_flags": "fake_health_flags.csv",
                "state_transition_summary": "state_transition_summary.csv",
                "state_distribution_summary": "state_distribution_summary.csv",
                "summary": "final_decision_layer_spec_summary_zh.md",
            },
        }
        (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        pd.DataFrame([{"status": "completed", "output_dir": str(output.resolve())}]).to_csv(
            output / "completed.csv",
            index=False,
            encoding="utf-8-sig",
        )
        pd.DataFrame(columns=["step", "error"]).to_csv(output / "failed.csv", index=False, encoding="utf-8-sig")
        log("completed", "completed", str(output.resolve()))
        (output / "current_step.txt").write_text("completed\n", encoding="utf-8")
        return output
    except Exception as exc:
        pd.DataFrame([{"step": "run_final_decision_layer_spec_diagnostic", "error": str(exc)}]).to_csv(
            output / "failed.csv",
            index=False,
            encoding="utf-8-sig",
        )
        log("failed", "failed", str(exc))
        raise


def build_pool_signal_normalized_panel(events: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, event in events.iterrows():
        for pool_id in ALL_POOL_IDS:
            ticker = _text(event.get(f"{pool_id}_ticker"))
            selection_layer = _text(event.get(f"{pool_id}_selection_layer")) or _infer_selection_layer(ticker)
            vote_state = _text(event.get(f"{pool_id}_vote_state"))
            direction = _text(event.get(f"{pool_id}_direction_state")) or _infer_direction(ticker, selection_layer)
            asset_type = _asset_type(ticker)
            asset_role = _asset_role(ticker, selection_layer)
            is_formal_pool = pool_id in FORMAL_POOL_IDS
            is_pool3_shadow = pool_id == "pool3"
            eligible_base = vote_state == "eligible_vote" and bool(ticker)
            eligible_stock = bool(is_formal_pool and eligible_base and asset_role == "stock_candidate")
            eligible_market = bool(eligible_base and asset_role == "market_exposure_tool")
            rows.append(
                {
                    "period": event.get("period", ""),
                    "signal_date": event.get("signal_date", event.get("date", "")),
                    "pool_id": pool_id,
                    "pool_role": _pool_role(pool_id),
                    "pool_universe_id": pool_id,
                    "pool_target_ticker": ticker,
                    "pool_target_asset_type": asset_type,
                    "pool_target_score": event.get(f"{pool_id}_score", ""),
                    "score_gap_to_second": event.get(f"{pool_id}_score_gap_to_second", ""),
                    "selection_layer": selection_layer,
                    "eligible_stock_vote": eligible_stock,
                    "eligible_market_exposure": eligible_market,
                    "vote_target": ticker if eligible_stock else "",
                    "direction_state": direction,
                    "attack_state": _attack_state(direction, asset_role),
                    "direction_confidence": event.get(f"{pool_id}_direction_confidence", ""),
                    "data_readiness_state": _data_readiness_state(vote_state, event.get(f"{pool_id}_blocked_reason", "")),
                    "blocked_reason": event.get(f"{pool_id}_blocked_reason", ""),
                    "shadow_or_diagnostic_flags": "pool3_shadow_not_formal" if is_pool3_shadow else "",
                    "is_formal_pool": is_formal_pool,
                    "pool3_shadow_not_formal_flag": is_pool3_shadow,
                    "active_in_trade_decision": False,
                }
            )
    return pd.DataFrame(rows)


def build_final_decision_state_panel(events: pd.DataFrame, pool_panel: pd.DataFrame) -> pd.DataFrame:
    pool_by_date = {date: group.copy() for date, group in pool_panel.groupby("signal_date", dropna=False)}
    rows: list[dict[str, Any]] = []
    for _, event in events.iterrows():
        date = _text(event.get("signal_date", event.get("date", "")))
        pools = pool_by_date.get(date, pd.DataFrame())
        formal = pools[pools["is_formal_pool"].astype(bool)] if not pools.empty else pd.DataFrame()
        stock_votes = formal[formal["eligible_stock_vote"].astype(bool)] if not formal.empty else pd.DataFrame()
        market_votes = formal[formal["eligible_market_exposure"].astype(bool)] if not formal.empty else pd.DataFrame()
        exact_group, exact_count = _top_count(stock_votes["vote_target"].tolist() if not stock_votes.empty else [])
        direction_group, direction_count = _top_count(_valid_values(formal.get("direction_state", pd.Series(dtype=str)).tolist()))
        attack_group, attack_count = _top_count(_valid_values(formal.get("attack_state", pd.Series(dtype=str)).tolist()))
        exact_consensus = bool(exact_group and exact_count >= 2)
        direction_consensus = bool(direction_group and direction_count >= 2)
        attack_consensus = bool(attack_group and attack_count >= 2)
        final_target = _text(event.get("formal_final_target", event.get("winner_ticker", "")))
        final_target_role = _asset_role(final_target, "")
        exposure_target = final_target if final_target_role == "market_exposure_tool" else _top_market_exposure(market_votes)
        forced_stop, forced_reason = _forced_stop(event, formal)
        data_insufficient = _data_insufficient(formal)
        fake_direction = bool(direction_consensus and not final_target)
        fake_actionable = bool(final_target and not exact_consensus and final_target_role != "market_exposure_tool")
        protocol_used = False
        if forced_stop:
            state = "forced_stop"
            reason = forced_reason
        elif data_insufficient:
            state = "data_insufficient"
            reason = "Formal pool data readiness is insufficient."
        elif exact_consensus:
            state = "strong_consensus"
            reason = "Formal pools formed exact stock ticker consensus."
        elif exposure_target:
            state = "defensive_market_exposure"
            reason = "Market exposure tool is separated from stock vote consensus."
        elif direction_consensus and not fake_direction:
            state = "weak_consensus"
            reason = "Formal pools align by direction or attack state but not exact ticker."
        elif final_target:
            state = "actionable_divergence"
            reason = "Formal selector has target without healthy exact consensus; diagnostic only."
            protocol_used = True
        else:
            state = "diagnostic_divergence"
            reason = "Formal pools diverge or lack actionable target."
            protocol_used = True
        if fake_direction and state not in {"forced_stop", "data_insufficient"}:
            state = "diagnostic_divergence"
            reason = "Direction consensus lacks actionable target; fail-closed diagnostic."
            protocol_used = True
        target_type = "market_exposure_tool" if final_target_role == "market_exposure_tool" else "stock_attack" if final_target else "none"
        rows.append(
            {
                "period": event.get("period", ""),
                "signal_date": date,
                "exact_ticker_consensus_state": "exact_consensus" if exact_consensus else "no_exact_consensus",
                "exact_ticker_consensus_group": exact_group,
                "exact_ticker_consensus_count": exact_count,
                "direction_consensus_state": "direction_consensus" if direction_consensus else "no_direction_consensus",
                "direction_consensus_group": direction_group,
                "direction_consensus_count": direction_count,
                "attack_or_exposure_state_consensus": "attack_exposure_consensus" if attack_consensus else "no_attack_exposure_consensus",
                "attack_or_exposure_state_consensus_group": attack_group,
                "attack_or_exposure_state_consensus_count": attack_count,
                "actionable_decision_state": "actionable_target_formed" if final_target else "not_actionable",
                "decision_protocol_used": protocol_used,
                "decision_protocol_reason": "tail_divergence_diagnostic" if protocol_used else "",
                "decision_protocol_usage_bucket": _protocol_bucket(protocol_used),
                "final_decision_state": state,
                "final_decision_reason": reason,
                "final_target_ticker": final_target,
                "final_target_type": target_type,
                "target_priority_rank": 1 if final_target else "",
                "target_priority_reason": _target_priority_reason(state, final_target, exposure_target),
                "exposure_target": exposure_target,
                "market_exposure_tool_used": bool(exposure_target),
                "not_eligible_for_formal_selector": bool(state in {"forced_stop", "data_insufficient"} or fake_direction or fake_actionable),
                "fake_direction_consensus_flag": fake_direction,
                "fake_actionable_decision_flag": fake_actionable,
                "protocol_overuse_flag": False,
                "data_insufficient_flag": data_insufficient,
                "forced_stop_reason": forced_reason,
                "formal_pool_count": int(len(formal)),
                "eligible_formal_stock_vote_count": int(len(stock_votes)),
                "eligible_market_exposure_count": int(len(market_votes)),
                "pool3_shadow_used_as_formal": False,
                "etf_counted_as_stock_vote": False,
                "active_in_trade_decision": False,
                "final_decision_layer_boundary": REPORT_ONLY_BOUNDARY,
            }
        )
    panel = pd.DataFrame(rows)
    return _apply_protocol_overuse(panel)


def build_target_priority_panel(state_panel: pd.DataFrame, pool_panel: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for row in state_panel.to_dict(orient="records"):
        if row.get("final_target_ticker"):
            rows.append(
                {
                    "period": row.get("period", ""),
                    "signal_date": row.get("signal_date", ""),
                    "target_priority_rank": 1,
                    "target_ticker": row.get("final_target_ticker", ""),
                    "target_type": row.get("final_target_type", ""),
                    "target_priority_reason": row.get("target_priority_reason", ""),
                    "source": "formal_target_stream",
                    "active_in_trade_decision": False,
                }
            )
        if row.get("exposure_target") and row.get("exposure_target") != row.get("final_target_ticker"):
            rows.append(
                {
                    "period": row.get("period", ""),
                    "signal_date": row.get("signal_date", ""),
                    "target_priority_rank": 2,
                    "target_ticker": row.get("exposure_target", ""),
                    "target_type": "market_exposure_tool",
                    "target_priority_reason": "market exposure tool kept outside stock vote consensus",
                    "source": "market_exposure_layer",
                    "active_in_trade_decision": False,
                }
            )
    if not rows:
        return pd.DataFrame(columns=["period", "signal_date", "target_priority_rank", "target_ticker", "target_type", "target_priority_reason", "source", "active_in_trade_decision"])
    return pd.DataFrame(rows)


def build_market_exposure_tool_panel(state_panel: pd.DataFrame, pool_panel: pd.DataFrame) -> pd.DataFrame:
    pool_rows = pool_panel[pool_panel["eligible_market_exposure"].astype(bool)].copy()
    rows = []
    for row in pool_rows.to_dict(orient="records"):
        rows.append(
            {
                "period": row.get("period", ""),
                "signal_date": row.get("signal_date", ""),
                "source": row.get("pool_id", ""),
                "exposure_target": row.get("pool_target_ticker", ""),
                "exposure_target_type": "market_exposure_tool",
                "eligible_stock_vote": False,
                "eligible_market_exposure": True,
                "active_in_trade_decision": False,
            }
        )
    for row in state_panel[state_panel["market_exposure_tool_used"].astype(bool)].to_dict(orient="records"):
        rows.append(
            {
                "period": row.get("period", ""),
                "signal_date": row.get("signal_date", ""),
                "source": "final_decision_layer",
                "exposure_target": row.get("exposure_target", ""),
                "exposure_target_type": "market_exposure_tool",
                "eligible_stock_vote": False,
                "eligible_market_exposure": True,
                "active_in_trade_decision": False,
            }
        )
    return pd.DataFrame(rows)


def build_consensus_health_by_period(state_panel: pd.DataFrame) -> pd.DataFrame:
    return _period_rates(
        state_panel,
        {
            "exact_ticker_consensus_rate": lambda g: g["exact_ticker_consensus_state"].eq("exact_consensus"),
            "direction_consensus_rate": lambda g: g["direction_consensus_state"].eq("direction_consensus"),
            "attack_or_exposure_state_consensus_rate": lambda g: g["attack_or_exposure_state_consensus"].eq("attack_exposure_consensus"),
            "actionable_decision_rate": lambda g: g["actionable_decision_state"].eq("actionable_target_formed"),
            "final_target_formed_rate": lambda g: g["final_target_ticker"].astype(str).str.strip().ne(""),
            "market_exposure_tool_usage_rate": lambda g: g["market_exposure_tool_used"].astype(bool),
            "data_insufficient_rate": lambda g: g["data_insufficient_flag"].astype(bool),
            "forced_stop_rate": lambda g: g["final_decision_state"].eq("forced_stop"),
        },
    )


def build_protocol_usage_by_period(state_panel: pd.DataFrame) -> pd.DataFrame:
    return _period_rates(
        state_panel,
        {
            "decision_protocol_usage_rate": lambda g: g["decision_protocol_used"].astype(bool),
            "protocol_overuse_rate": lambda g: g["protocol_overuse_flag"].astype(bool),
            "diagnostic_divergence_rate": lambda g: g["final_decision_state"].eq("diagnostic_divergence"),
            "actionable_divergence_rate": lambda g: g["final_decision_state"].eq("actionable_divergence"),
        },
    )


def build_fake_health_flags(state_panel: pd.DataFrame) -> pd.DataFrame:
    return state_panel[
        state_panel["fake_direction_consensus_flag"].astype(bool) | state_panel["fake_actionable_decision_flag"].astype(bool) | state_panel["protocol_overuse_flag"].astype(bool)
    ][
        [
            "period",
            "signal_date",
            "final_decision_state",
            "fake_direction_consensus_flag",
            "fake_actionable_decision_flag",
            "protocol_overuse_flag",
            "not_eligible_for_formal_selector",
            "active_in_trade_decision",
        ]
    ].copy()


def build_state_transition_summary(state_panel: pd.DataFrame) -> pd.DataFrame:
    frame = state_panel.copy()
    frame["previous_state"] = frame.groupby("period")["final_decision_state"].shift(1).fillna("start")
    transitions = frame.groupby(["period", "previous_state", "final_decision_state"], dropna=False).size().reset_index(name="transition_count")
    transitions["active_in_trade_decision"] = False
    return transitions


def build_state_distribution_summary(state_panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for period, group in state_panel.groupby("period", dropna=False):
        total = len(group)
        for state, count in group["final_decision_state"].value_counts(dropna=False).items():
            rows.append(
                {
                    "period": period,
                    "final_decision_state": state,
                    "count": int(count),
                    "rate": _rate(count, total),
                    "active_in_trade_decision": False,
                }
            )
    return pd.DataFrame(rows)


def _validate_event_panel(events: pd.DataFrame) -> None:
    required = {"period", "pool1_ticker", "pool2_ticker"}
    missing = required - set(events.columns)
    if missing:
        raise ValueError("missing final decision spec event columns: " + ",".join(sorted(missing)))
    if "signal_date" not in events.columns and "date" not in events.columns:
        raise ValueError("missing final decision spec event columns: signal_date/date")


def _apply_protocol_overuse(panel: pd.DataFrame) -> pd.DataFrame:
    output = panel.copy()
    output["protocol_overuse_flag"] = False
    for period, group in output.groupby("period", dropna=False):
        usage = group["decision_protocol_used"].astype(bool).mean() if len(group) else 0.0
        if usage > 0.35:
            output.loc[group.index, "protocol_overuse_flag"] = output.loc[group.index, "decision_protocol_used"].astype(bool)
            mask = output.index.isin(group.index) & output["decision_protocol_used"].astype(bool)
            output.loc[mask, "not_eligible_for_formal_selector"] = True
    return output


def _period_rates(frame: pd.DataFrame, specs: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for period, group in frame.groupby("period", dropna=False):
        row: dict[str, Any] = {"period": period, "signal_count": int(len(group))}
        for key, fn in specs.items():
            values = fn(group)
            row[key] = _rate(values.sum(), len(group))
        row["active_in_trade_decision"] = False
        rows.append(row)
    return pd.DataFrame(rows)


def _top_count(values: list[str]) -> tuple[str, int]:
    clean = [str(value).strip() for value in values if str(value).strip()]
    if not clean:
        return "", 0
    counts = pd.Series(clean).value_counts()
    return str(counts.index[0]), int(counts.iloc[0])


def _valid_values(values: list[Any]) -> list[str]:
    return [str(value).strip() for value in values if str(value).strip() and str(value).strip().lower() not in {"none", "no_selection"}]


def _top_market_exposure(market_votes: pd.DataFrame) -> str:
    if market_votes.empty:
        return ""
    target, count = _top_count(market_votes["pool_target_ticker"].astype(str).tolist())
    return target if count else ""


def _forced_stop(event: pd.Series, formal: pd.DataFrame) -> tuple[bool, str]:
    text = " ".join(
        [
            _text(event.get("trade_blocked_reason")),
            _text(event.get("raw_consensus_state")),
            " ".join(formal.get("blocked_reason", pd.Series(dtype=str)).astype(str).tolist()) if not formal.empty else "",
        ]
    ).lower()
    keywords = ("forced_stop", "hard_stop", "future_data", "future violation", "risk_gate_forced_no_trade")
    for keyword in keywords:
        if keyword in text:
            return True, keyword
    return False, ""


def _data_insufficient(formal: pd.DataFrame) -> bool:
    if formal.empty:
        return True
    if int(formal["eligible_stock_vote"].astype(bool).sum() + formal["eligible_market_exposure"].astype(bool).sum()) == 0:
        return True
    readiness = set(formal["data_readiness_state"].astype(str))
    return "blocked" in readiness


def _target_priority_reason(state: str, final_target: str, exposure_target: str) -> str:
    if state == "strong_consensus":
        return "exact formal stock consensus has first priority"
    if exposure_target:
        return "market exposure tool separated from stock vote consensus"
    if final_target:
        return "formal target stream target retained for diagnostic trace"
    return "no final target formed"


def _protocol_bucket(used: bool) -> str:
    return "tail_divergence_candidate" if used else "consensus_or_blocked_bypass"


def _data_readiness_state(vote_state: str, blocked_reason: object) -> str:
    reason = _text(blocked_reason).lower()
    if "blocked" in str(vote_state).lower() or "blocked" in reason or "no_resolved_symbols" in reason:
        return "blocked"
    if "partial" in reason:
        return "partial"
    return "ready"


def _infer_selection_layer(ticker: str) -> str:
    if _asset_role(ticker, "") == "market_exposure_tool":
        return "market_exposure_tool"
    return "formal_candidate" if ticker else "no_selection"


def _infer_direction(ticker: str, selection_layer: str) -> str:
    role = _asset_role(ticker, selection_layer)
    if role == "market_exposure_tool":
        return "market_exposure"
    if role == "stock_candidate":
        return "stock_attack"
    return ""


def _attack_state(direction: str, asset_role: str) -> str:
    if asset_role == "market_exposure_tool" or direction == "market_exposure":
        return "market_exposure"
    if direction in {"stock_attack", "attack", "risk_on"}:
        return "stock_attack"
    if direction in {"risk_off", "defensive"}:
        return direction
    return "no_edge" if not direction else direction


def _asset_type(ticker: str) -> str:
    return "etf" if _asset_role(ticker, "") == "market_exposure_tool" else "stock" if ticker else "none"


def _asset_role(ticker: object, selection_layer: object) -> str:
    value = _text(ticker).upper()
    layer = _text(selection_layer)
    if not value:
        return "none"
    if value in MARKET_EXPOSURE_TICKERS or layer == "market_exposure_tool":
        return "market_exposure_tool"
    return "stock_candidate"


def _pool_role(pool_id: str) -> str:
    return {
        "pool1": "ai_theme_large_cap_formal",
        "pool2": "tw50_breadth_formal",
        "pool3": "shadow_diagnostic_context",
    }.get(pool_id, pool_id)


def _text(value: object) -> str:
    return "" if pd.isna(value) else str(value).strip()


def _rate(numerator: float | int, denominator: float | int) -> float:
    return round(float(numerator) / float(denominator), 6) if denominator else 0.0


def _summary_markdown(health: pd.DataFrame, protocol: pd.DataFrame, fake: pd.DataFrame) -> str:
    lines = [
        "# Final Decision Layer Spec Diagnostic",
        "",
        "本輸出是 report-only diagnostic scaffold，不改正式 selector、vote、target 或 trade action。",
        "",
        "## 邊界",
        "",
        "- formal_model_changed=false",
        "- trade_decision_changed=false",
        "- active_in_trade_decision=false",
        "- Pool3 shadow 不計入 formal exact consensus",
        "- ETF / 0050 / 00631L 不計入 eligible stock vote",
        "",
        "## Health by period",
        "",
    ]
    for row in health.to_dict(orient="records"):
        lines.append(
            f"- {row['period']}：exact={row.get('exact_ticker_consensus_rate')}, "
            f"direction={row.get('direction_consensus_rate')}, actionable={row.get('actionable_decision_rate')}"
        )
    lines.extend(["", "## Protocol usage", ""])
    for row in protocol.to_dict(orient="records"):
        lines.append(f"- {row['period']}：protocol_usage={row.get('decision_protocol_usage_rate')}")
    lines.extend(["", f"Fake health rows：{len(fake)}", ""])
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build final decision layer spec diagnostic report-only panels.")
    parser.add_argument("--event-panel", default=DEFAULT_EVENT_PANEL)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args(argv)
    output = run_final_decision_layer_spec_diagnostic(event_panel_path=args.event_panel, output_dir=args.output_dir)
    print(f"OUTPUT_DIR={output.resolve()}")


if __name__ == "__main__":
    main()
