from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from backtest_lab.costs import COST_MODEL_VERSION, cost_model_metadata


TASK_ID = "TASK-BACKTEST-CORE-SHORT-CYCLE-PULLBACK-PORTFOLIO-CHALLENGER-SPEC-001"
DEFAULT_EVENT_PANEL_DIR = "outputs/short_cycle_pullback_reversal_event_panel_20260704"
DEFAULT_FORMAL_STREAM = "outputs/formal_long_range_signal_reconstruction_201411_latest_20260702/formal_long_range_target_stream.csv"
DEFAULT_OUTPUT_DIR = "outputs/short_cycle_pullback_portfolio_challenger_spec_20260704"

PRIMARY_VARIANTS = {
    "strong_stock_ma20_pullback_reclaim",
    "pullback_candidate_wait_for_peer_breadth",
}


def run_short_cycle_pullback_portfolio_challenger_spec(
    *,
    event_panel_dir: str | Path = DEFAULT_EVENT_PANEL_DIR,
    formal_target_stream: str | Path = DEFAULT_FORMAL_STREAM,
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
        (output / "current_step.txt").write_text(f"{step}:{status}\n{detail}", encoding="utf-8")

    try:
        event_root = Path(event_panel_dir)
        log("load_inputs", "started", str(event_root))
        event_manifest = _load_json(event_root / "manifest.json")
        event_panel = _read_csv_required(event_root / "short_cycle_pullback_reversal_event_panel.csv")
        formal_stream_path = Path(formal_target_stream)
        formal_stream = _read_csv_required(formal_stream_path)

        log("build_contract", "started", "")
        eligible_events = _eligible_events(event_panel)
        candidate_schema = _candidate_event_input_schema()
        baseline_contract = _baseline_contract(formal_stream_path, formal_stream)
        variants = _execution_rule_variants()
        cost_contract = _cost_model_contract()
        readiness = _readiness(
            event_manifest=event_manifest,
            event_panel=event_panel,
            eligible_events=eligible_events,
            variants=variants,
            formal_stream=formal_stream,
        )
        manifest = _manifest(output, event_root, formal_stream_path, readiness)
        contract_json = _contract_json(readiness, variants, baseline_contract, cost_contract)
        contract_md = _contract_markdown(readiness, variants, cost_contract)

        log("write_outputs", "started", str(output))
        (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        (output / "portfolio_challenger_contract.json").write_text(
            json.dumps(contract_json, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (output / "portfolio_challenger_contract.md").write_text(contract_md, encoding="utf-8")
        candidate_schema.to_csv(output / "candidate_event_input_schema.csv", index=False, encoding="utf-8-sig")
        variants.to_csv(output / "execution_rule_variants.csv", index=False, encoding="utf-8-sig")
        baseline_contract.to_csv(output / "baseline_contract.csv", index=False, encoding="utf-8-sig")
        cost_contract.to_csv(output / "cost_model_contract.csv", index=False, encoding="utf-8-sig")
        (output / "readiness_for_experiments.json").write_text(
            json.dumps(readiness, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (output / "final_summary_zh.md").write_text(_summary_zh(readiness), encoding="utf-8")
        pd.DataFrame([{"step": TASK_ID, "status": "completed_contract_ready", "output_dir": str(output)}]).to_csv(
            output / "completed.csv",
            index=False,
            encoding="utf-8-sig",
        )
        pd.DataFrame(columns=["step", "status", "reason"]).to_csv(output / "failed.csv", index=False, encoding="utf-8-sig")
        log("completed", "completed", str(output.resolve()))
        return output
    except Exception as exc:
        pd.DataFrame([{"step": TASK_ID, "status": "failed", "reason": str(exc)}]).to_csv(
            output / "failed.csv",
            index=False,
            encoding="utf-8-sig",
        )
        log("failed", "failed", str(exc))
        raise


def _eligible_events(event_panel: pd.DataFrame) -> pd.DataFrame:
    frame = event_panel.copy().fillna("")
    mask = (
        frame["variant_id"].astype(str).isin(PRIMARY_VARIANTS)
        & frame["price_data_ready"].map(_truthy)
        & frame["diagnostic_only"].map(_truthy)
        & ~frame["is_trade_rule"].map(_truthy)
        & ~frame["uses_forward_return_as_live_rule"].map(_truthy)
        & frame["next_tradable_date"].astype(str).str.strip().ne("")
    )
    return frame[mask].copy()


def _candidate_event_input_schema() -> pd.DataFrame:
    rows = [
        ("signal_date", "date", "required", "Event signal date; no same-day execution."),
        ("next_tradable_date", "date", "required", "Next-day fill date for opportunity sleeve."),
        ("ticker", "string", "required", "Event candidate ticker."),
        ("variant_id", "string", "required", "Only primary MA20 reclaim or peer-breadth filter variants are tradable diagnostics."),
        ("candidate_source", "string", "required", "old_ai or pool1b; Pool1B/material remains diagnostic context."),
        ("price_data_ready", "boolean", "required_true", "Must be true; no fake price filling."),
        ("diagnostic_only", "boolean", "required_true", "Must be true; event is not a formal target."),
        ("is_trade_rule", "boolean", "required_false", "Must remain false in source panel."),
        ("uses_forward_return_as_live_rule", "boolean", "required_false", "Forward outcome is evaluation metadata only."),
        ("rs_vs_0050_60d_pct", "number", "live_safe_score", "Primary live-safe ranking input."),
        ("rs_vs_0050_20d_pct", "number", "live_safe_score", "Secondary live-safe ranking input."),
        ("peer_recovery_count", "number", "live_safe_score", "Peer-breadth confidence input."),
        ("drawdown_from_60d_high_pct", "number", "live_safe_score", "Risk tie-break; lower absolute drawdown preferred."),
        ("formal_target", "string", "context_only", "Used for overlap/conflict handling only."),
    ]
    return pd.DataFrame(rows, columns=["field", "type", "requirement", "description"])


def _baseline_contract(formal_stream_path: Path, formal_stream: pd.DataFrame) -> pd.DataFrame:
    start = str(formal_stream["signal_date"].min()) if "signal_date" in formal_stream else ""
    end = str(formal_stream["signal_date"].max()) if "signal_date" in formal_stream else ""
    return pd.DataFrame(
        [
            {
                "baseline_id": "baseline_formal_next_day",
                "source": str(formal_stream_path),
                "execution_basis": "next_day",
                "target_policy": "current formal target stream, no sleeve",
                "signal_start": start,
                "signal_end": end,
                "active_in_trade_decision": False,
                "diagnostic_only": True,
            },
            {
                "baseline_id": "report_only_context_no_trade_change",
                "source": str(formal_stream_path),
                "execution_basis": "next_day",
                "target_policy": "same as formal baseline; event panel recorded as opportunity context only",
                "signal_start": start,
                "signal_end": end,
                "active_in_trade_decision": False,
                "diagnostic_only": True,
            },
            {
                "baseline_id": "event_only_reference",
                "source": "short_cycle_pullback_reversal_event_panel",
                "execution_basis": "next_day",
                "target_policy": "reference-only event sleeve behavior; not a formal selector replacement",
                "signal_start": "",
                "signal_end": "",
                "active_in_trade_decision": False,
                "diagnostic_only": True,
            },
        ]
    )


def _execution_rule_variants() -> pd.DataFrame:
    rows = [
        _variant(
            "baseline_formal_next_day",
            "baseline",
            "",
            0.0,
            "none",
            "none",
            "Formal target stream only.",
        ),
        _variant(
            "report_only_context_no_trade_change",
            "baseline_context",
            "",
            0.0,
            "none",
            "none",
            "Record events only; no trade change.",
        ),
        _variant(
            "ma20_reclaim_overlay_20_when_formal_cash_or_market_exposure",
            "diagnostic_overlay",
            "strong_stock_ma20_pullback_reclaim",
            0.20,
            "cash_or_market_exposure",
            "hold_20d_or_signal_break",
            "Small opportunity sleeve only when formal target is cash/risk-off or market exposure.",
        ),
        _variant(
            "ma20_reclaim_overlay_20_all_formal_states_except_same_ticker",
            "diagnostic_overlay",
            "strong_stock_ma20_pullback_reclaim",
            0.20,
            "all_except_same_ticker",
            "hold_20d_or_signal_break",
            "Small sleeve in all formal states, keeping formal core and avoiding duplicate ticker.",
        ),
        _variant(
            "peer_breadth_overlay_20_when_formal_cash_or_market_exposure",
            "diagnostic_overlay",
            "pullback_candidate_wait_for_peer_breadth",
            0.20,
            "cash_or_market_exposure",
            "hold_20d_or_signal_break",
            "Peer breadth confidence filter; cash/market-exposure states only.",
        ),
        _variant(
            "peer_breadth_overlay_20_all_formal_states_except_same_ticker",
            "diagnostic_overlay",
            "pullback_candidate_wait_for_peer_breadth",
            0.20,
            "all_except_same_ticker",
            "hold_20d_or_signal_break",
            "Peer breadth confidence filter; all formal states except same ticker.",
        ),
        _variant(
            "ma20_reclaim_overlay_20_when_formal_cash_or_market_exposure_hold60",
            "diagnostic_overlay",
            "strong_stock_ma20_pullback_reclaim",
            0.20,
            "cash_or_market_exposure",
            "hold_60d_or_signal_break",
            "Hold-window sensitivity; still small sleeve only.",
        ),
        _variant(
            "ma20_reclaim_overlay_10_when_formal_cash_or_market_exposure",
            "sensitivity",
            "strong_stock_ma20_pullback_reclaim",
            0.10,
            "cash_or_market_exposure",
            "hold_20d_or_signal_break",
            "Weight sensitivity requested by strategy center; not a grid.",
        ),
    ]
    return pd.DataFrame(rows)


def _variant(
    variant_id: str,
    role: str,
    event_variant_id: str,
    sleeve_weight: float,
    formal_state_scope: str,
    exit_rule: str,
    note: str,
) -> dict[str, Any]:
    return {
        "variant_id": variant_id,
        "role": role,
        "event_variant_id": event_variant_id,
        "sleeve_weight": sleeve_weight,
        "entry_basis": "next_tradable_date",
        "candidate_selection": (
            "top1_by_live_safe_score: rs60 desc, rs20 desc, peer_recovery_count desc, abs(drawdown_from_60d_high_pct) asc"
            if event_variant_id
            else "none"
        ),
        "formal_state_scope": formal_state_scope,
        "conflict_handling": "do_not_double_count_same_ticker; keep formal core holding; sleeve is diagnostic only",
        "exit_rule": exit_rule,
        "signal_break_rule": "exit next day after MA20 below for 2 consecutive trading days or MA60 break beyond threshold",
        "same_day_allowed": False,
        "active_in_trade_decision": False,
        "formal_model_changed": False,
        "trade_decision_changed": False,
        "diagnostic_only": True,
        "note": note,
    }


def _cost_model_contract() -> pd.DataFrame:
    meta = cost_model_metadata()
    return pd.DataFrame(
        [
            {
                "cost_model_version": COST_MODEL_VERSION,
                "scope": "buy",
                "broker_fee_rate": meta["broker_fee_rate"],
                "minimum_fee_twd": meta["minimum_fee_twd"],
                "securities_transaction_tax_rate": 0.0,
                "asset_type": "stock_or_etf",
                "applies_to": "opportunity sleeve and formal baseline fills",
            },
            {
                "cost_model_version": COST_MODEL_VERSION,
                "scope": "sell_stock",
                "broker_fee_rate": meta["broker_fee_rate"],
                "minimum_fee_twd": meta["minimum_fee_twd"],
                "securities_transaction_tax_rate": meta["stock_sell_tax_rate"],
                "asset_type": "stock",
                "applies_to": "opportunity sleeve stock exits",
            },
            {
                "cost_model_version": COST_MODEL_VERSION,
                "scope": "sell_etf",
                "broker_fee_rate": meta["broker_fee_rate"],
                "minimum_fee_twd": meta["minimum_fee_twd"],
                "securities_transaction_tax_rate": meta["etf_sell_tax_rate"],
                "asset_type": "etf",
                "applies_to": "formal ETF exits when baseline is replayed",
            },
        ]
    )


def _readiness(
    *,
    event_manifest: dict[str, Any],
    event_panel: pd.DataFrame,
    eligible_events: pd.DataFrame,
    variants: pd.DataFrame,
    formal_stream: pd.DataFrame,
) -> dict[str, Any]:
    material_mask = eligible_events["candidate_source"].astype(str).eq("pool1b") | eligible_events[
        "supply_chain_layer"
    ].astype(str).str.contains("material|wafer|substrate", case=False, na=False)
    return {
        "schema_version": 1,
        "task_id": TASK_ID,
        "status": "completed_portfolio_challenger_contract_ready",
        "source_event_panel_status": event_manifest.get("status", ""),
        "event_rows": int(len(event_panel)),
        "eligible_event_rows": int(len(eligible_events)),
        "eligible_unique_tickers": int(eligible_events["ticker"].nunique()) if not eligible_events.empty else 0,
        "eligible_pool1b_rows": int(eligible_events["candidate_source"].astype(str).eq("pool1b").sum())
        if not eligible_events.empty
        else 0,
        "eligible_material_layer_rows": int(material_mask.sum()) if not eligible_events.empty else 0,
        "case_6488_two_rows": int(eligible_events["ticker"].astype(str).eq("6488.TWO").sum())
        if not eligible_events.empty
        else 0,
        "formal_stream_rows": int(len(formal_stream)),
        "execution_basis": "next_day",
        "score_formula": "rs_vs_0050_60d_pct desc; rs_vs_0050_20d_pct desc; peer_recovery_count desc; abs(drawdown_from_60d_high_pct) asc",
        "exit_rules": ["hold_20d_or_signal_break", "hold_60d_or_signal_break"],
        "sleeve_weights": [0.10, 0.20],
        "variant_count": int(len(variants)),
        "ready_for_experiments_portfolio_challenger_validation": True,
        "ready_for_strategy_replay": False,
        "ready_for_formal_absorption": False,
        "diagnostic_only": True,
        "formal_model_changed": False,
        "trade_decision_changed": False,
        "active_in_trade_decision": False,
        "uses_forward_return_as_live_rule": False,
        "same_day_execution_allowed": False,
        "material_layer_case_only": True,
        "case_6488_two_case_only": True,
        "future_data_violation_count": int(event_manifest.get("future_data_violation_count", 0) or 0),
    }


def _manifest(output: Path, event_root: Path, formal_stream: Path, readiness: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "task_id": TASK_ID,
        "status": readiness["status"],
        "generated_at": pd.Timestamp.now(tz="Asia/Taipei").isoformat(),
        "output_dir": str(output),
        "event_panel_dir": str(event_root),
        "formal_target_stream": str(formal_stream),
        **{
            key: readiness[key]
            for key in [
                "eligible_event_rows",
                "eligible_unique_tickers",
                "eligible_pool1b_rows",
                "eligible_material_layer_rows",
                "case_6488_two_rows",
                "execution_basis",
                "ready_for_experiments_portfolio_challenger_validation",
                "ready_for_strategy_replay",
                "ready_for_formal_absorption",
                "diagnostic_only",
                "formal_model_changed",
                "trade_decision_changed",
                "active_in_trade_decision",
                "uses_forward_return_as_live_rule",
                "future_data_violation_count",
            ]
        },
        "outputs": {
            "portfolio_challenger_contract_md": "portfolio_challenger_contract.md",
            "portfolio_challenger_contract_json": "portfolio_challenger_contract.json",
            "candidate_event_input_schema": "candidate_event_input_schema.csv",
            "execution_rule_variants": "execution_rule_variants.csv",
            "baseline_contract": "baseline_contract.csv",
            "cost_model_contract": "cost_model_contract.csv",
            "readiness_for_experiments": "readiness_for_experiments.json",
            "final_summary_zh": "final_summary_zh.md",
        },
    }


def _contract_json(
    readiness: dict[str, Any],
    variants: pd.DataFrame,
    baseline: pd.DataFrame,
    cost: pd.DataFrame,
) -> dict[str, Any]:
    return {
        "task_id": TASK_ID,
        "status": readiness["status"],
        "entry_contract": {
            "execution_basis": "next_day",
            "eligible_variants": sorted(PRIMARY_VARIANTS),
            "filters": [
                "price_data_ready=true",
                "diagnostic_only=true",
                "is_trade_rule=false",
                "uses_forward_return_as_live_rule=false",
                "next_tradable_date present",
            ],
            "top1_score_formula": readiness["score_formula"],
            "forward_return_allowed_as_live_rule": False,
        },
        "exit_contract": {
            "variants": readiness["exit_rules"],
            "signal_break_rule": "MA20 below for 2 consecutive trading days or MA60 break beyond threshold; fill next day",
        },
        "baseline_contract": baseline.to_dict(orient="records"),
        "execution_rule_variants": variants.to_dict(orient="records"),
        "cost_model_contract": cost.to_dict(orient="records"),
        "formal_target_relationship": {
            "formal_stock_target": "keep formal core holding; sleeve only in variants that allow all states",
            "formal_cash_or_risk_off": "opportunity sleeve allowed in bounded variants",
            "formal_market_exposure": "opportunity sleeve allowed in bounded variants",
            "same_ticker": "do not double count; keep formal position only",
        },
        "boundary": {
            "diagnostic_only": True,
            "formal_model_changed": False,
            "trade_decision_changed": False,
            "active_in_trade_decision": False,
            "material_layer_case_only": True,
            "case_6488_two_case_only": True,
        },
    }


def _contract_markdown(readiness: dict[str, Any], variants: pd.DataFrame, cost: pd.DataFrame) -> str:
    variant_lines = [
        f"- `{row['variant_id']}`：{row['role']}，sleeve={row['sleeve_weight']:.0%}，exit={row['exit_rule']}。"
        for row in variants.to_dict(orient="records")
    ]
    return "\n".join(
        [
            "# Short-cycle pullback portfolio challenger contract",
            "",
            "## Boundary",
            "",
            "- Diagnostic-only. This contract does not change the formal selector, target, report, or trade action.",
            "- Formal target remains the core holding. Event candidates are only bounded opportunity sleeves.",
            "- 6488.TWO and material-layer evidence remain case-only / shadow context.",
            "- Forward returns may be used only by Experiments for evaluation, never as live rules.",
            "",
            "## Entry",
            "",
            "- Use next-day execution from `next_tradable_date`; same-day fills are not allowed.",
            "- Eligible rows must have price ready, diagnostic-only, non-trade-rule source flags, and no forward return live-rule usage.",
            f"- Top-1 event score: {readiness['score_formula']}.",
            "",
            "## Exit",
            "",
            "- `hold_20d_or_signal_break` and `hold_60d_or_signal_break` are the only first-pass exit contracts.",
            "- Signal break is MA20 below for 2 consecutive trading days or MA60 break beyond threshold, filled next day.",
            "",
            "## Variants",
            "",
            *variant_lines,
            "",
            "## Cost",
            "",
            f"- Cost model: `{COST_MODEL_VERSION}`.",
            "- Buy and sell both include broker fee; sell additionally includes securities transaction tax.",
            "- Stock sell tax 0.3%; ETF sell tax 0.1%; no Yuanta discount assumed unless separately provided.",
            "",
            "## Readiness",
            "",
            f"- Eligible event rows: {readiness['eligible_event_rows']}",
            f"- Eligible Pool1B rows: {readiness['eligible_pool1b_rows']}",
            f"- Eligible material-layer rows: {readiness['eligible_material_layer_rows']}",
            f"- 6488.TWO rows: {readiness['case_6488_two_rows']}",
            "- Ready for Experiments portfolio challenger validation: true",
        ]
    )


def _summary_zh(readiness: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Short-cycle pullback portfolio challenger spec",
            "",
            f"- 狀態：{readiness['status']}",
            f"- eligible event rows：{readiness['eligible_event_rows']}",
            f"- eligible unique tickers：{readiness['eligible_unique_tickers']}",
            f"- Pool1B eligible rows：{readiness['eligible_pool1b_rows']}",
            f"- material-layer eligible rows：{readiness['eligible_material_layer_rows']}",
            f"- 6488.TWO case rows：{readiness['case_6488_two_rows']}",
            "- 這是 diagnostic portfolio challenger contract，不改正式模型、不改每日報告、不改交易指令。",
            "- 下一棒可交 Experiments 依此 contract 跑 next-day portfolio challenger validation。",
        ]
    )


def _read_csv_required(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path).fillna("")


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build short-cycle pullback portfolio challenger contract package.")
    parser.add_argument("--event-panel-dir", default=DEFAULT_EVENT_PANEL_DIR)
    parser.add_argument("--formal-target-stream", default=DEFAULT_FORMAL_STREAM)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args(argv)
    run_short_cycle_pullback_portfolio_challenger_spec(
        event_panel_dir=args.event_panel_dir,
        formal_target_stream=args.formal_target_stream,
        output_dir=args.output_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
