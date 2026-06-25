from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


REPORT_ONLY_BOUNDARY = "report_only_diagnostic"


def run_final_decision_layer_diagnostic(
    *,
    event_panel_path: str | Path,
    pool3_selector_panel_path: str | Path | None,
    output_dir: str | Path,
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

    log("load_inputs", "started")
    events = pd.read_csv(event_panel_path).fillna("")
    _validate_event_panel(events)
    selector = _load_selector_panel(pool3_selector_panel_path)

    log("build_diagnostic_panel", "started")
    panel = _build_diagnostic_panel(events, selector)
    panel.to_csv(output / "final_decision_layer_diagnostic_panel.csv", index=False, encoding="utf-8-sig")

    summary = _summary_by_period(panel)
    summary.to_csv(output / "final_decision_layer_summary_by_period.csv", index=False, encoding="utf-8-sig")

    pd.DataFrame(_field_contract_rows()).to_csv(output / "final_decision_layer_field_contract.csv", index=False, encoding="utf-8-sig")
    (output / "final_decision_layer_final_summary_zh.md").write_text(_markdown_summary(summary), encoding="utf-8")

    metadata = {
        "schema_version": 1,
        "task_id": "TASK-BACKTEST-CORE-FINAL-DECISION-LAYER-DIAGNOSTIC-SCAFFOLD-001",
        "status": "completed",
        "model": "final_decision_layer_diagnostic_scaffold",
        "formal_model_changed": False,
        "trade_decision_changed": False,
        "active_in_trade_decision": False,
        "final_decision_layer_boundary": REPORT_ONLY_BOUNDARY,
        "pool3_shadow_used_in_trade_decision": False,
        "pool3_shadow_not_formal": True,
        "event_panel_path": str(event_panel_path),
        "pool3_selector_panel_path": str(pool3_selector_panel_path or ""),
        "outputs": {
            "diagnostic_panel": "final_decision_layer_diagnostic_panel.csv",
            "summary_by_period": "final_decision_layer_summary_by_period.csv",
            "field_contract": "final_decision_layer_field_contract.csv",
            "summary": "final_decision_layer_final_summary_zh.md",
        },
    }
    (output / "manifest.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame([{"status": "completed", "output_dir": str(output.resolve())}]).to_csv(
        output / "completed.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(columns=["step", "error"]).to_csv(output / "failed.csv", index=False, encoding="utf-8-sig")
    log("completed", "completed", str(output.resolve()))
    (output / "current_step.txt").write_text("completed\n", encoding="utf-8")
    return output


def _validate_event_panel(events: pd.DataFrame) -> None:
    required = {
        "period",
        "signal_date",
        "pool1_ticker",
        "pool1_vote_state",
        "pool1_direction_state",
        "pool2_ticker",
        "pool2_vote_state",
        "pool2_direction_state",
        "pool3_ticker",
        "pool3_vote_state",
        "pool3_direction_state",
        "formal_final_target",
        "final_target_source",
        "trade_action",
    }
    missing = required - set(events.columns)
    if missing:
        raise ValueError("missing event panel columns: " + ",".join(sorted(missing)))


def _load_selector_panel(path: str | Path | None) -> pd.DataFrame:
    if not path:
        return pd.DataFrame()
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"pool3 selector panel not found: {source}")
    return pd.read_csv(source).fillna("")


def _build_diagnostic_panel(events: pd.DataFrame, selector: pd.DataFrame) -> pd.DataFrame:
    selector_by_date = {}
    if not selector.empty and "signal_date" in selector.columns:
        selector_by_date = {str(row["signal_date"]): row.to_dict() for _, row in selector.iterrows()}
    rows: list[dict[str, Any]] = []
    for _, row in events.iterrows():
        p1_eligible = _eligible(row.get("pool1_vote_state"))
        p2_eligible = _eligible(row.get("pool2_vote_state"))
        p1_ticker = _text(row.get("pool1_ticker"))
        p2_ticker = _text(row.get("pool2_ticker"))
        p1_direction = _text(row.get("pool1_direction_state"))
        p2_direction = _text(row.get("pool2_direction_state"))
        exact_consensus = bool(p1_eligible and p2_eligible and p1_ticker and p1_ticker == p2_ticker)
        direction_consensus = bool(p1_eligible and p2_eligible and p1_direction and p1_direction == p2_direction)
        final_target = _text(row.get("formal_final_target"))
        selector_row = selector_by_date.get(_text(row.get("signal_date")), {})
        state, reason = _decision_state(
            p1_eligible=p1_eligible,
            p2_eligible=p2_eligible,
            p1_ticker=p1_ticker,
            p2_ticker=p2_ticker,
            p1_direction=p1_direction,
            p2_direction=p2_direction,
            exact_consensus=exact_consensus,
            direction_consensus=direction_consensus,
            final_target=final_target,
            pool3_diag_state=_text(selector_row.get("pool3_selector_diagnostic_state")),
        )
        protocol_candidate = state in {
            "direction_consensus_observation",
            "pool1_lead_observation",
            "pool2_lead_observation",
            "pool3_shadow_diagnostic_only",
            "diagnostic_conflict",
        }
        fake_direction = bool(direction_consensus and not final_target)
        fake_actionable = bool(final_target and not exact_consensus)
        exact_state = "exact_consensus" if exact_consensus else "no_exact_consensus"
        direction_state = "direction_consensus" if direction_consensus else "no_direction_consensus"
        attack_exposure_consensus = _attack_exposure_consensus(p1_direction, p2_direction)
        actionable_state = "actionable_target_formed" if final_target else "not_actionable"
        pool3_context_state = _text(selector_row.get("pool3_selector_diagnostic_state")) or "none"
        rows.append(
            {
                "period": row.get("period", ""),
                "signal_date": row.get("signal_date", ""),
                "final_decision_layer_state": state,
                "final_decision_layer_reason": reason,
                "final_decision_diagnostic_state": state,
                "final_decision_diagnostic_reason": reason,
                "final_decision_layer_boundary": REPORT_ONLY_BOUNDARY,
                "final_decision_layer_active_in_trade_decision": False,
                "exact_ticker_consensus_state": exact_state,
                "direction_consensus_state": direction_state,
                "attack_exposure_state_consensus": attack_exposure_consensus,
                "actionable_decision_consensus": actionable_state,
                "pool1_formal_vote": p1_ticker if p1_eligible else "",
                "pool1_direction": p1_direction if p1_eligible else "",
                "pool2_formal_vote": p2_ticker if p2_eligible else "",
                "pool2_direction": p2_direction if p2_eligible else "",
                "pool3_shadow_ticker": row.get("pool3_ticker", ""),
                "pool3_shadow_context_state": pool3_context_state,
                "pool3_shadow_diagnostic_state": pool3_context_state,
                "pool3_shadow_boundary": selector_row.get("pool3_selector_diagnostic_boundary", ""),
                "pool3_shadow_used_in_trade_decision": False,
                "pool3_shadow_not_formal_flag": True,
                "exact_ticker_consensus_rate": 1.0 if exact_consensus else 0.0,
                "direction_consensus_rate": 1.0 if direction_consensus else 0.0,
                "attack_exposure_state_consensus_rate": 1.0 if attack_exposure_consensus else 0.0,
                "actionable_decision_rate": 1.0 if final_target else 0.0,
                "final_target_formed_rate": 1.0 if final_target else 0.0,
                "final_target_source": row.get("final_target_source", ""),
                "formal_final_target": final_target,
                "consensus_type": "exact_ticker" if exact_consensus else "direction" if direction_consensus else "none",
                "formal_pool_count": int(p1_eligible) + int(p2_eligible),
                "shadow_pool_count": 1 if selector_row else 0,
                "fake_direction_consensus_flag": fake_direction,
                "fake_actionable_decision_flag": fake_actionable,
                "decision_protocol_candidate": protocol_candidate,
                "decision_protocol_usage_category": _protocol_usage_category(protocol_candidate, state),
                "decision_protocol_overuse_flag": False,
                "trade_action": row.get("trade_action", ""),
            }
        )
    panel = pd.DataFrame(rows)
    return _apply_overuse_flags(panel)


def _decision_state(
    *,
    p1_eligible: bool,
    p2_eligible: bool,
    p1_ticker: str,
    p2_ticker: str,
    p1_direction: str,
    p2_direction: str,
    exact_consensus: bool,
    direction_consensus: bool,
    final_target: str,
    pool3_diag_state: str,
) -> tuple[str, str]:
    if exact_consensus and final_target:
        return "consensus_passthrough", "Pool1/Pool2 exact ticker consensus formed a formal target."
    if not p1_eligible and not p2_eligible:
        return "data_blocked", "Pool1 and Pool2 have no formal vote; diagnostic layer remains blocked."
    if direction_consensus and not exact_consensus:
        return "direction_consensus_observation", "Pool1/Pool2 directions align but tickers differ; report-only observation."
    if p1_eligible and not p2_eligible:
        return "pool1_lead_observation", "Only Pool1 has a formal vote; report-only lead observation."
    if p2_eligible and not p1_eligible:
        return "pool2_lead_observation", "Only Pool2 has a formal vote; report-only lead observation."
    if p1_ticker and p2_ticker and p1_ticker != p2_ticker:
        return "pool1_pool2_formal_conflict", "Pool1/Pool2 formal tickers conflict."
    if pool3_diag_state in {"veto_explanation", "opportunity_watch_only", "concentration_blocked"}:
        return "pool3_shadow_diagnostic_only", "Pool3 exists only as shadow diagnostic context."
    if p1_direction or p2_direction:
        return "diagnostic_conflict", "Signals are mixed or insufficient for formal action."
    return "risk_off_or_forced_stop", "No actionable formal direction; keep report-only risk-off diagnostic."


def _attack_exposure_consensus(p1_direction: str, p2_direction: str) -> bool:
    return bool(_attack_exposure_state(p1_direction) and _attack_exposure_state(p1_direction) == _attack_exposure_state(p2_direction))


def _attack_exposure_state(direction: str) -> str:
    if direction == "stock_attack":
        return "attack"
    if direction == "market_exposure":
        return "market_exposure"
    return ""


def _protocol_usage_category(protocol_candidate: bool, state: str) -> str:
    if not protocol_candidate:
        return "none"
    if state == "direction_consensus_observation":
        return "direction_observation_only"
    if state in {"pool1_lead_observation", "pool2_lead_observation"}:
        return "single_pool_lead_observation"
    if state == "pool3_shadow_diagnostic_only":
        return "pool3_shadow_context"
    return "diagnostic_only"


def _apply_overuse_flags(panel: pd.DataFrame) -> pd.DataFrame:
    if panel.empty:
        return panel
    output = panel.copy()
    for period, frame in output.groupby("period", dropna=False):
        rate = _rate(frame["decision_protocol_candidate"].map(_truthy).sum(), len(frame))
        if rate > 0.35:
            output.loc[frame.index, "decision_protocol_overuse_flag"] = True
            output.loc[frame.index, "final_decision_layer_state"] = output.loc[frame.index, "final_decision_layer_state"].mask(
                output.loc[frame.index, "decision_protocol_candidate"].map(_truthy),
                "protocol_overuse_warning",
            )
    return output


def _summary_by_period(panel: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for period, frame in panel.groupby("period", dropna=False):
        rows.append(
            {
                "period": period,
                "rows": int(len(frame)),
                "exact_ticker_consensus_rate": round(float(frame["exact_ticker_consensus_rate"].mean()), 6),
                "direction_consensus_rate": round(float(frame["direction_consensus_rate"].mean()), 6),
                "attack_exposure_state_consensus_rate": round(float(frame["attack_exposure_state_consensus_rate"].mean()), 6),
                "actionable_decision_rate": round(float(frame["actionable_decision_rate"].mean()), 6),
                "final_target_formed_rate": round(float(frame["final_target_formed_rate"].mean()), 6),
                "decision_protocol_candidate_rate": _rate(frame["decision_protocol_candidate"].map(_truthy).sum(), len(frame)),
                "decision_protocol_overuse_rate": _rate(frame["decision_protocol_overuse_flag"].map(_truthy).sum(), len(frame)),
                "fake_direction_consensus_rate": _rate(frame["fake_direction_consensus_flag"].map(_truthy).sum(), len(frame)),
                "fake_actionable_decision_rate": _rate(frame["fake_actionable_decision_flag"].map(_truthy).sum(), len(frame)),
            }
        )
    return pd.DataFrame(rows)


def _field_contract_rows() -> list[dict[str, Any]]:
    return [
        {"field": "final_decision_layer_boundary", "value": REPORT_ONLY_BOUNDARY, "active_in_trade_decision": False},
        {"field": "pool3_shadow_not_formal_flag", "value": True, "active_in_trade_decision": False},
        {"field": "decision_protocol_candidate", "value": "diagnostic only", "active_in_trade_decision": False},
    ]


def _markdown_summary(summary: pd.DataFrame) -> str:
    lines = [
        "# Final decision layer diagnostic scaffold",
        "",
        "- 狀態：report-only diagnostic；正式模型未變更。",
        "- Pool3 僅作 shadow context，不計入 formal exact consensus。",
        "",
        "## Period summary",
        "",
    ]
    for _, row in summary.iterrows():
        lines.append(
            f"- {row['period']}: exact={row['exact_ticker_consensus_rate']}, "
            f"direction={row['direction_consensus_rate']}, actionable={row['actionable_decision_rate']}, "
            f"protocol_candidate={row['decision_protocol_candidate_rate']}"
        )
    return "\n".join(lines)


def _eligible(value: object) -> bool:
    return str(value).strip() == "eligible_vote"


def _truthy(value: object) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def _text(value: object) -> str:
    return "" if pd.isna(value) else str(value).strip()


def _rate(numerator: float | int, denominator: float | int) -> float:
    return round(float(numerator) / float(denominator), 6) if denominator else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Build report-only final decision layer diagnostic scaffold.")
    parser.add_argument("--event-panel", required=True)
    parser.add_argument("--pool3-selector-panel", default="")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    output = run_final_decision_layer_diagnostic(
        event_panel_path=args.event_panel,
        pool3_selector_panel_path=args.pool3_selector_panel or None,
        output_dir=args.output_dir,
    )
    print(f"OUTPUT_DIR={output.resolve()}")


if __name__ == "__main__":
    main()
