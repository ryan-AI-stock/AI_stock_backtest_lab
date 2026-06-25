from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


REPORT_ONLY_BOUNDARY = "report_only_diagnostic"
RULE_BOUNDARY_MODE = "report_only_preplan"
MARKET_EXPOSURE_TICKERS = {"0050", "0050.TW", "00631L", "00631L.TW"}


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
    panel.to_csv(output / "final_decision_rule_boundary_panel.csv", index=False, encoding="utf-8-sig")

    summary = _summary_by_period(panel)
    summary.to_csv(output / "final_decision_layer_summary_by_period.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(output / "rule_boundary_summary_by_period.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(_rule_boundary_gate_rows(summary)).to_csv(output / "rule_boundary_gate_report.csv", index=False, encoding="utf-8-sig")

    pd.DataFrame(_field_contract_rows()).to_csv(output / "final_decision_layer_field_contract.csv", index=False, encoding="utf-8-sig")
    (output / "final_decision_layer_final_summary_zh.md").write_text(_markdown_summary(summary), encoding="utf-8")
    (output / "rule_boundary_final_summary_zh.md").write_text(_rule_boundary_markdown_summary(summary), encoding="utf-8")

    metadata = {
        "schema_version": 1,
        "task_id": "TASK-BACKTEST-CORE-FINAL-DECISION-LAYER-RULE-BOUNDARY-001",
        "status": "completed",
        "model": "final_decision_layer_rule_boundary_preplan",
        "formal_model_changed": False,
        "trade_decision_changed": False,
        "active_in_trade_decision": False,
        "final_decision_layer_boundary": REPORT_ONLY_BOUNDARY,
        "final_decision_rule_boundary_mode": RULE_BOUNDARY_MODE,
        "pool3_shadow_used_in_trade_decision": False,
        "pool3_shadow_not_formal": True,
        "event_panel_path": str(event_panel_path),
        "pool3_selector_panel_path": str(pool3_selector_panel_path or ""),
        "outputs": {
            "diagnostic_panel": "final_decision_layer_diagnostic_panel.csv",
            "rule_boundary_panel": "final_decision_rule_boundary_panel.csv",
            "summary_by_period": "final_decision_layer_summary_by_period.csv",
            "rule_boundary_summary_by_period": "rule_boundary_summary_by_period.csv",
            "rule_boundary_gate_report": "rule_boundary_gate_report.csv",
            "field_contract": "final_decision_layer_field_contract.csv",
            "summary": "final_decision_layer_final_summary_zh.md",
            "rule_boundary_summary": "rule_boundary_final_summary_zh.md",
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
        p1_asset_role = _asset_role(p1_ticker)
        p2_asset_role = _asset_role(p2_ticker)
        final_target_asset_role = _asset_role(final_target) if (final_target := _text(row.get("formal_final_target"))) else "none"
        p1_stock_eligible = bool(p1_eligible and p1_asset_role == "stock_candidate")
        p2_stock_eligible = bool(p2_eligible and p2_asset_role == "stock_candidate")
        p1_market_exposure = bool(p1_eligible and p1_asset_role == "market_exposure_tool")
        p2_market_exposure = bool(p2_eligible and p2_asset_role == "market_exposure_tool")
        exact_consensus = bool(p1_stock_eligible and p2_stock_eligible and p1_ticker and p1_ticker == p2_ticker)
        market_exposure_consensus = bool(p1_market_exposure and p2_market_exposure and p1_ticker and p1_ticker == p2_ticker)
        direction_consensus = bool(p1_eligible and p2_eligible and p1_direction and p1_direction == p2_direction)
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
                "pool1_asset_role": p1_asset_role if p1_eligible else "none",
                "pool1_direction": p1_direction if p1_eligible else "",
                "pool2_formal_vote": p2_ticker if p2_eligible else "",
                "pool2_asset_role": p2_asset_role if p2_eligible else "none",
                "pool2_direction": p2_direction if p2_eligible else "",
                "pool3_shadow_ticker": row.get("pool3_ticker", ""),
                "pool3_asset_role": _asset_role(row.get("pool3_ticker", "")),
                "pool3_shadow_context_state": pool3_context_state,
                "pool3_shadow_diagnostic_state": pool3_context_state,
                "pool3_shadow_boundary": selector_row.get("pool3_selector_diagnostic_boundary", ""),
                "pool3_shadow_used_in_trade_decision": False,
                "pool3_shadow_not_formal_flag": True,
                "exact_ticker_consensus_rate": 1.0 if exact_consensus else 0.0,
                "market_exposure_consensus_rate": 1.0 if market_exposure_consensus else 0.0,
                "direction_consensus_rate": 1.0 if direction_consensus else 0.0,
                "attack_exposure_state_consensus_rate": 1.0 if attack_exposure_consensus else 0.0,
                "actionable_decision_rate": 1.0 if final_target else 0.0,
                "final_target_formed_rate": 1.0 if final_target else 0.0,
                "final_target_source": row.get("final_target_source", ""),
                "formal_final_target": final_target,
                "final_target_asset_role": final_target_asset_role,
                "consensus_type": "exact_ticker" if exact_consensus else "direction" if direction_consensus else "none",
                "formal_pool_count": int(p1_eligible) + int(p2_eligible),
                "formal_stock_pool_count": int(p1_stock_eligible) + int(p2_stock_eligible),
                "market_exposure_pool_count": int(p1_market_exposure) + int(p2_market_exposure),
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
    return _apply_rule_boundaries(_apply_overuse_flags(panel))


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


def _apply_rule_boundaries(panel: pd.DataFrame) -> pd.DataFrame:
    if panel.empty:
        return panel
    output = panel.copy()
    for period, frame in output.groupby("period", dropna=False):
        metrics = _period_rule_metrics(frame)
        for idx, row in frame.iterrows():
            labels = _rule_boundary_labels(row, metrics)
            for key, value in labels.items():
                output.loc[idx, key] = value
    return output


def _period_rule_metrics(frame: pd.DataFrame) -> dict[str, float | bool | str]:
    rows = len(frame)
    protocol_candidate_rate = _rate(frame["decision_protocol_candidate"].map(_truthy).sum(), rows)
    protocol_overuse_rate = _rate(frame["decision_protocol_overuse_flag"].map(_truthy).sum(), rows)
    fake_direction_rate = _rate(frame["fake_direction_consensus_flag"].map(_truthy).sum(), rows)
    fake_actionable_rate = _rate(frame["fake_actionable_decision_flag"].map(_truthy).sum(), rows)
    actionable_rate = round(float(frame["actionable_decision_rate"].mean()), 6) if rows else 0.0
    final_target_rate = round(float(frame["final_target_formed_rate"].mean()), 6) if rows else 0.0
    direction_rate = round(float(frame["direction_consensus_rate"].mean()), 6) if rows else 0.0
    data_blocked_rate = _rate((frame["final_decision_diagnostic_state"].astype(str) == "data_blocked").sum(), rows)
    formal_stock_pool_low_rate = _rate((pd.to_numeric(frame["formal_stock_pool_count"], errors="coerce").fillna(0) < 2).sum(), rows)
    return {
        "period": str(frame["period"].iloc[0]) if rows and "period" in frame.columns else "",
        "protocol_candidate_rate": protocol_candidate_rate,
        "protocol_overuse_rate": protocol_overuse_rate,
        "fake_direction_rate": fake_direction_rate,
        "fake_actionable_rate": fake_actionable_rate,
        "actionable_rate": actionable_rate,
        "final_target_rate": final_target_rate,
        "direction_rate": direction_rate,
        "data_blocked_rate": data_blocked_rate,
        "formal_stock_pool_low_rate": formal_stock_pool_low_rate,
        "protocol_tail_allowed": bool(
            protocol_candidate_rate <= 0.25
            and protocol_overuse_rate == 0
            and fake_direction_rate <= 0.15
            and actionable_rate >= 0.50
            and final_target_rate >= 0.50
        ),
        "protocol_observation_only": bool(
            0.25 < protocol_candidate_rate <= 0.35
            and protocol_overuse_rate == 0
            and fake_direction_rate <= 0.15
        ),
        "protocol_overuse_blocked": bool(protocol_candidate_rate > 0.50 or protocol_overuse_rate > 0),
        "fake_direction_fail_closed": bool(fake_direction_rate > 0.15),
        "fake_actionable_watch_only": bool(fake_actionable_rate > 0.20),
        "coverage_or_formal_vote_insufficient": bool(
            data_blocked_rate > 0.50
            or direction_rate == 0
            or final_target_rate < 0.20
            or formal_stock_pool_low_rate > 0.50
        ),
        "direction_actionable_allowed": bool(
            direction_rate >= 0.55
            and actionable_rate >= 0.50
            and final_target_rate >= 0.50
            and fake_direction_rate <= 0.15
            and protocol_candidate_rate <= 0.35
            and protocol_overuse_rate == 0
        ),
        "candidate_for_future_event_study": bool(
            str(frame["period"].iloc[0]) == "2024_now"
            and protocol_candidate_rate <= 0.35
            and protocol_overuse_rate == 0
            and fake_direction_rate <= 0.15
            and actionable_rate >= 0.60
        ),
    }


def _rule_boundary_labels(row: pd.Series, metrics: dict[str, float | bool | str]) -> dict[str, object]:
    single_pool_lead = str(row.get("final_decision_diagnostic_state")) in {"pool1_lead_observation", "pool2_lead_observation"}
    direction_blocked_reason = _direction_blocked_reason(metrics)
    protocol_overuse_blocked = bool(metrics["protocol_overuse_blocked"])
    fake_direction_fail_closed = bool(metrics["fake_direction_fail_closed"])
    fake_actionable_watch_only = bool(metrics["fake_actionable_watch_only"])
    coverage_insufficient = bool(metrics["coverage_or_formal_vote_insufficient"])
    candidate_event_study = bool(metrics["candidate_for_future_event_study"]) and not (
        protocol_overuse_blocked or fake_direction_fail_closed or coverage_insufficient
    )
    not_eligible = bool(
        protocol_overuse_blocked
        or fake_direction_fail_closed
        or coverage_insufficient
        or single_pool_lead
        or fake_actionable_watch_only
        or not bool(metrics["direction_actionable_allowed"])
    )
    state, reason = _rule_boundary_state_reason(
        protocol_overuse_blocked=protocol_overuse_blocked,
        fake_direction_fail_closed=fake_direction_fail_closed,
        coverage_insufficient=coverage_insufficient,
        single_pool_lead=single_pool_lead,
        fake_actionable_watch_only=fake_actionable_watch_only,
        candidate_event_study=candidate_event_study,
        direction_blocked_reason=direction_blocked_reason,
    )
    return {
        "final_decision_rule_boundary_state": state,
        "final_decision_rule_boundary_reason": reason,
        "protocol_tail_allowed": bool(metrics["protocol_tail_allowed"]),
        "protocol_observation_only": bool(metrics["protocol_observation_only"]),
        "protocol_overuse_blocked": protocol_overuse_blocked,
        "direction_consensus_actionable_allowed": bool(metrics["direction_actionable_allowed"]),
        "direction_consensus_blocked_reason": direction_blocked_reason,
        "fake_direction_fail_closed": fake_direction_fail_closed,
        "fake_actionable_watch_only": fake_actionable_watch_only,
        "coverage_or_formal_vote_insufficient": coverage_insufficient,
        "single_pool_lead_observation_only": single_pool_lead,
        "candidate_for_future_event_study": candidate_event_study,
        "not_eligible_for_formal_selector": not_eligible,
        "final_decision_rule_boundary_active_in_trade_decision": False,
        "final_decision_rule_boundary_mode": RULE_BOUNDARY_MODE,
    }


def _direction_blocked_reason(metrics: dict[str, float | bool | str]) -> str:
    reasons: list[str] = []
    if float(metrics["direction_rate"]) < 0.55:
        reasons.append("direction_consensus_below_0.55")
    if float(metrics["actionable_rate"]) < 0.50:
        reasons.append("actionable_decision_below_0.50")
    if float(metrics["final_target_rate"]) < 0.50:
        reasons.append("final_target_formed_below_0.50")
    if float(metrics["fake_direction_rate"]) > 0.15:
        reasons.append("fake_direction_above_0.15")
    if float(metrics["protocol_candidate_rate"]) > 0.35:
        reasons.append("protocol_candidate_above_0.35")
    if float(metrics["protocol_overuse_rate"]) > 0:
        reasons.append("protocol_overuse_present")
    return ";".join(reasons) or "direction_consensus_actionable_allowed"


def _rule_boundary_state_reason(
    *,
    protocol_overuse_blocked: bool,
    fake_direction_fail_closed: bool,
    coverage_insufficient: bool,
    single_pool_lead: bool,
    fake_actionable_watch_only: bool,
    candidate_event_study: bool,
    direction_blocked_reason: str,
) -> tuple[str, str]:
    if protocol_overuse_blocked:
        return "protocol_overuse_blocked", "Decision protocol usage is too broad for tail divergence handling; fail-closed."
    if fake_direction_fail_closed:
        return "fake_direction_fail_closed", "Direction consensus is not actionable enough; fail-closed."
    if coverage_insufficient:
        return "coverage_or_formal_vote_insufficient", "Formal pool votes or data coverage are insufficient; blocked."
    if single_pool_lead:
        return "single_pool_lead_observation_only", "Only one formal pool leads; report-only observation."
    if fake_actionable_watch_only:
        return "fake_actionable_watch_only", "Actionable target is not supported by exact consensus; watch-only."
    if candidate_event_study:
        return "candidate_for_future_event_study", "Low overuse and low fake-health period; event-study candidate only."
    if direction_blocked_reason != "direction_consensus_actionable_allowed":
        return "direction_consensus_observation_blocked", direction_blocked_reason
    return "protocol_tail_allowed", "Rule boundary permits only report-only tail-event study, not formal selector."


def _summary_by_period(panel: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for period, frame in panel.groupby("period", dropna=False):
        rows.append(
            {
                "period": period,
                "rows": int(len(frame)),
                "exact_ticker_consensus_rate": round(float(frame["exact_ticker_consensus_rate"].mean()), 6),
                "market_exposure_consensus_rate": round(float(frame["market_exposure_consensus_rate"].mean()), 6),
                "direction_consensus_rate": round(float(frame["direction_consensus_rate"].mean()), 6),
                "attack_exposure_state_consensus_rate": round(float(frame["attack_exposure_state_consensus_rate"].mean()), 6),
                "actionable_decision_rate": round(float(frame["actionable_decision_rate"].mean()), 6),
                "final_target_formed_rate": round(float(frame["final_target_formed_rate"].mean()), 6),
                "decision_protocol_candidate_rate": _rate(frame["decision_protocol_candidate"].map(_truthy).sum(), len(frame)),
                "decision_protocol_overuse_rate": _rate(frame["decision_protocol_overuse_flag"].map(_truthy).sum(), len(frame)),
                "fake_direction_consensus_rate": _rate(frame["fake_direction_consensus_flag"].map(_truthy).sum(), len(frame)),
                "fake_actionable_decision_rate": _rate(frame["fake_actionable_decision_flag"].map(_truthy).sum(), len(frame)),
                "protocol_overuse_blocked_rate": _rate(frame["protocol_overuse_blocked"].map(_truthy).sum(), len(frame)),
                "fake_direction_fail_closed_rate": _rate(frame["fake_direction_fail_closed"].map(_truthy).sum(), len(frame)),
                "coverage_or_formal_vote_insufficient_rate": _rate(frame["coverage_or_formal_vote_insufficient"].map(_truthy).sum(), len(frame)),
                "candidate_for_future_event_study_rate": _rate(frame["candidate_for_future_event_study"].map(_truthy).sum(), len(frame)),
                "not_eligible_for_formal_selector_rate": _rate(frame["not_eligible_for_formal_selector"].map(_truthy).sum(), len(frame)),
            }
        )
    return pd.DataFrame(rows)


def _field_contract_rows() -> list[dict[str, Any]]:
    return [
        {"field": "final_decision_layer_boundary", "value": REPORT_ONLY_BOUNDARY, "active_in_trade_decision": False},
        {"field": "final_decision_rule_boundary_mode", "value": RULE_BOUNDARY_MODE, "active_in_trade_decision": False},
        {"field": "market_exposure_tool", "value": "0050/00631L excluded from formal stock exact consensus", "active_in_trade_decision": False},
        {"field": "pool3_shadow_not_formal_flag", "value": True, "active_in_trade_decision": False},
        {"field": "decision_protocol_candidate", "value": "diagnostic only", "active_in_trade_decision": False},
        {"field": "protocol_overuse_blocked", "value": "fail-closed label only", "active_in_trade_decision": False},
        {"field": "candidate_for_future_event_study", "value": "research candidate only", "active_in_trade_decision": False},
    ]


def _rule_boundary_gate_rows(summary: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for _, row in summary.iterrows():
        period = str(row["period"])
        rows.extend(
            [
                {
                    "period": period,
                    "gate": "protocol_overuse_blocked",
                    "status": "fail_closed" if float(row["protocol_overuse_blocked_rate"]) > 0 else "pass",
                    "value": row["protocol_overuse_blocked_rate"],
                    "threshold": "must be 0 for formal selector",
                },
                {
                    "period": period,
                    "gate": "fake_direction_fail_closed",
                    "status": "fail_closed" if float(row["fake_direction_fail_closed_rate"]) > 0 else "pass",
                    "value": row["fake_direction_fail_closed_rate"],
                    "threshold": "must be 0 for formal selector",
                },
                {
                    "period": period,
                    "gate": "coverage_or_formal_vote_insufficient",
                    "status": "blocked" if float(row["coverage_or_formal_vote_insufficient_rate"]) > 0 else "pass",
                    "value": row["coverage_or_formal_vote_insufficient_rate"],
                    "threshold": "must be 0 for formal selector",
                },
                {
                    "period": period,
                    "gate": "candidate_for_future_event_study",
                    "status": "candidate_only" if float(row["candidate_for_future_event_study_rate"]) > 0 else "not_candidate",
                    "value": row["candidate_for_future_event_study_rate"],
                    "threshold": "report-only; never formal",
                },
                {
                    "period": period,
                    "gate": "not_eligible_for_formal_selector",
                    "status": "blocked" if float(row["not_eligible_for_formal_selector_rate"]) > 0 else "pass",
                    "value": row["not_eligible_for_formal_selector_rate"],
                    "threshold": "must be 0 for formal selector",
                },
            ]
        )
    return rows


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
            f"market_exposure={row['market_exposure_consensus_rate']}, "
            f"direction={row['direction_consensus_rate']}, actionable={row['actionable_decision_rate']}, "
            f"protocol_candidate={row['decision_protocol_candidate_rate']}"
        )
    return "\n".join(lines)


def _rule_boundary_markdown_summary(summary: pd.DataFrame) -> str:
    lines = [
        "# Final decision layer rule-boundary preplan",
        "",
        "- 狀態：report-only preplan；正式模型、selector、trade action 均未變更。",
        "- 目的：標示 protocol overuse、fake health、coverage/formal vote 不足，不形成正式交易訊號。",
        "- Pool3 shadow 仍不得進 formal decision。",
        "",
        "## Boundary summary",
        "",
    ]
    for _, row in summary.iterrows():
        lines.append(
            f"- {row['period']}: protocol_blocked={row['protocol_overuse_blocked_rate']}, "
            f"fake_direction_closed={row['fake_direction_fail_closed_rate']}, "
            f"insufficient={row['coverage_or_formal_vote_insufficient_rate']}, "
            f"future_candidate={row['candidate_for_future_event_study_rate']}, "
            f"not_formal={row['not_eligible_for_formal_selector_rate']}"
        )
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            "- `candidate_for_future_event_study` 只代表後續研究候選，不代表 formal selector。",
            "- `protocol_overuse_blocked` 與 `fake_direction_fail_closed` 必須 fail-closed。",
            "- `single_pool_lead_observation_only` 僅是 observation，不得產生 formal target。",
        ]
    )
    return "\n".join(lines)


def _eligible(value: object) -> bool:
    return str(value).strip() == "eligible_vote"


def _asset_role(value: object) -> str:
    ticker = _text(value).upper()
    if not ticker:
        return "none"
    if ticker in MARKET_EXPOSURE_TICKERS:
        return "market_exposure_tool"
    return "stock_candidate"


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
