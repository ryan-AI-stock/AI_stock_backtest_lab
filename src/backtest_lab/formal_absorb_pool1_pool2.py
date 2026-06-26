from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from backtest_lab.formal_model_contract import FORMAL_MODEL_TARGET, get_formal_model_contract


TASK_ID = "TASK-BACKTEST-CORE-FORMAL-ABSORB-POOL1-POOL2-COMBINED-CAP40-CONFIRMATION1-001"
DEFAULT_CANDIDATE_DIR = "outputs/pool1_pool2_veto_cap_downweight_panels_20260626"
DEFAULT_THREE_POOL_DIR = "outputs/stock_pool_formal_daily_replay_pit_pool2_daily_final_combined_20260624"
DEFAULT_LABEL_DIR = "outputs/pool1_pool2_0050x2_opportunity_label_20260626"
DEFAULT_OUTPUT_DIR = "outputs/formal_absorb_pool1_pool2_combined_cap40_confirmation1_20260626"


def run_formal_absorb_pool1_pool2(
    *,
    candidate_dir: str | Path = DEFAULT_CANDIDATE_DIR,
    three_pool_dir: str | Path = DEFAULT_THREE_POOL_DIR,
    opportunity_label_dir: str | Path = DEFAULT_LABEL_DIR,
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
        candidate = Path(candidate_dir)
        three_pool = Path(three_pool_dir)
        label = Path(opportunity_label_dir)
        log("load_inputs", "started", f"candidate={candidate}")
        period_perf = pd.read_csv(candidate / "period_performance_by_variant.csv").fillna("")
        daily = pd.read_csv(candidate / "daily_equity_by_variant.csv").fillna("")
        trades = pd.read_csv(candidate / "trade_ledger_by_variant.csv").fillna("")
        events = pd.read_csv(candidate / "pool2_disagreement_variant_events.csv").fillna("")
        variant_matrix = pd.read_csv(candidate / "variant_parameter_matrix.csv").fillna("")
        three_pool_summary = pd.read_csv(three_pool / "formal_three_pool_summary.csv").fillna("")
        label_manifest = json.loads((label / "manifest.json").read_text(encoding="utf-8"))

        log("validate_contract", "started", "")
        _validate_candidate_inputs(period_perf, daily, trades, events, variant_matrix, label_manifest)
        contract = get_formal_model_contract()
        candidate_full = _period_row(period_perf, "combined_cap40_confirmation1", "full")
        candidate_hard_gate = _period_row(period_perf, "combined_cap40_confirmation1", "2024_hard_gate")
        old_full = three_pool_summary.iloc[0].to_dict() if not three_pool_summary.empty else {}
        latest_complete_common_date = str(daily["date"].iloc[-1]) if not daily.empty else ""

        log("build_outputs", "started", "")
        before_after = _before_after(contract, old_full, candidate_full)
        blocker_matrix = _blocker_matrix()
        decision_sample = _formal_trade_decision_sample(daily, events)
        report_diff = _report_output_before_after(candidate_full, candidate_hard_gate)
        contract_md = _formal_model_contract_markdown(contract)
        summary = _summary_markdown(candidate_full, old_full, blocker_matrix)

        before_after.to_csv(output / "formal_selector_before_after.csv", index=False, encoding="utf-8-sig")
        decision_sample.to_csv(output / "formal_trade_decision_sample.csv", index=False, encoding="utf-8-sig")
        blocker_matrix.to_csv(output / "blocker_matrix.csv", index=False, encoding="utf-8-sig")
        (output / "formal_model_contract.md").write_text(contract_md, encoding="utf-8")
        (output / "report_output_before_after.md").write_text(report_diff, encoding="utf-8")
        (output / "pool1_pool2_formal_absorption_summary_zh.md").write_text(summary, encoding="utf-8")

        manifest = {
            "schema_version": 1,
            "task_id": TASK_ID,
            "status": "completed_formal_absorption",
            "formal_model_target": FORMAL_MODEL_TARGET,
            "formal_model_route": contract["formal_model_route"],
            "formal_model_changed": True,
            "trade_decision_changed": True,
            "formal_absorption_ready": True,
            "three_pool_formal_route_abandoned": True,
            "pool3_shadow_used_as_formal": False,
            "0050x2_opportunity_label_active_in_trade_decision": False,
            "market_exposure_override_absorbed": False,
            "rr_partial_switch_used_in_performance": False,
            "valuation_used": False,
            "h3_used": False,
            "uses_forward_return_as_rule": False,
            "candidate_dir": str(candidate),
            "previous_three_pool_dir": str(three_pool),
            "opportunity_label_dir": str(label),
            "latest_complete_common_date": latest_complete_common_date,
            "candidate_full_return_pct": _float(candidate_full.get("return_pct")),
            "candidate_full_max_drawdown_pct": _float(candidate_full.get("max_drawdown_pct")),
            "previous_three_pool_return_pct": _float(old_full.get("total_return_pct")),
            "previous_three_pool_max_drawdown_pct": _float(old_full.get("max_drawdown_pct")),
            "residual_caveat_count": int(len(blocker_matrix)),
            "blocking_caveat_count": int(blocker_matrix["blocks_formal_absorption"].map(_truthy).sum()),
            "output_files": {
                "manifest": "manifest.json",
                "before_after": "formal_selector_before_after.csv",
                "contract": "formal_model_contract.md",
                "sample": "formal_trade_decision_sample.csv",
                "report_diff": "report_output_before_after.md",
                "summary": "pool1_pool2_formal_absorption_summary_zh.md",
                "blocker_matrix": "blocker_matrix.csv",
            },
        }
        (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        pd.DataFrame([{"status": "completed_formal_absorption", "output_dir": str(output.resolve())}]).to_csv(
            output / "completed.csv",
            index=False,
            encoding="utf-8-sig",
        )
        pd.DataFrame(columns=["step", "error"]).to_csv(output / "failed.csv", index=False, encoding="utf-8-sig")
        log("completed", "completed", str(output.resolve()))
        (output / "current_step.txt").write_text("completed\n", encoding="utf-8")
        return output
    except Exception as exc:
        pd.DataFrame([{"step": "run_formal_absorb_pool1_pool2", "error": str(exc)}]).to_csv(
            output / "failed.csv",
            index=False,
            encoding="utf-8-sig",
        )
        log("failed", "failed", str(exc))
        raise


def _validate_candidate_inputs(
    period_perf: pd.DataFrame,
    daily: pd.DataFrame,
    trades: pd.DataFrame,
    events: pd.DataFrame,
    variant_matrix: pd.DataFrame,
    label_manifest: dict[str, Any],
) -> None:
    required_variant = "combined_cap40_confirmation1"
    if required_variant not in set(period_perf.get("variant", pd.Series(dtype=str)).astype(str)):
        raise ValueError("candidate period performance missing combined_cap40_confirmation1")
    if required_variant not in set(daily.get("variant", pd.Series(dtype=str)).astype(str)):
        raise ValueError("candidate daily equity missing combined_cap40_confirmation1")
    if required_variant not in set(variant_matrix.get("variant", pd.Series(dtype=str)).astype(str)):
        raise ValueError("candidate matrix missing combined_cap40_confirmation1")
    for name, frame, required in [
        ("daily", daily, {"variant", "date", "target_weights", "position_ticker", "equity", "drawdown", "action"}),
        ("trades", trades, {"variant", "date", "ticker", "action", "gross_amount", "costs"}),
        ("events", events, {"variant", "date", "pool1_vote", "pool2_vote", "target_weights", "event_reason"}),
    ]:
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"{name} missing columns: {missing}")
    if _truthy(label_manifest.get("opportunity_cost_label_active_in_trade_decision")):
        raise ValueError("0050x2 opportunity label is active in trade decision")
    if _truthy(label_manifest.get("market_exposure_override_absorbed")):
        raise ValueError("market exposure override was absorbed")
    if label_manifest.get("forbidden_word_positive_hits"):
        raise ValueError("0050x2 opportunity label wording has forbidden positive hits")


def _period_row(perf: pd.DataFrame, variant: str, period_label: str) -> dict[str, Any]:
    subset = perf[perf["variant"].astype(str).eq(variant) & perf["period_label"].astype(str).eq(period_label)]
    if subset.empty:
        return {}
    return subset.iloc[0].to_dict()


def _before_after(contract: dict[str, Any], old_full: dict[str, Any], candidate_full: dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "dimension": "formal_route",
                "before": "current_formal_three_pool_baseline",
                "after": contract["formal_model_target"],
                "changed": True,
                "evidence": "three_pool_formal_route_abandoned=true",
            },
            {
                "dimension": "selector_logic",
                "before": "2_of_3_three_pool_consensus",
                "after": "pool1_primary_plus_pool2_confirmation1_risk_layer",
                "changed": True,
                "evidence": "Pool1 primary; Pool2 disagreement requires one prior same Pool1 signal day.",
            },
            {
                "dimension": "00631L_exposure",
                "before": "uncapped_by_absorption_contract",
                "after": "cap40_residual_cash",
                "changed": True,
                "evidence": "00631L max target weight 40%.",
            },
            {
                "dimension": "pool3",
                "before": "formal_vote_in_three_pool_baseline",
                "after": "shadow_or_diagnostic_only",
                "changed": True,
                "evidence": "pool3_shadow_used_as_formal=false",
            },
            {
                "dimension": "0050x2_label",
                "before": "report_only_caveat",
                "after": "report_only_caveat",
                "changed": False,
                "evidence": "opportunity_cost_label_active_in_trade_decision=false",
            },
            {
                "dimension": "full_period_return_pct",
                "before": old_full.get("total_return_pct", ""),
                "after": candidate_full.get("return_pct", ""),
                "changed": True,
                "evidence": "same historical panel; candidate absorbed as formal route.",
            },
            {
                "dimension": "full_period_mdd_pct",
                "before": old_full.get("max_drawdown_pct", ""),
                "after": candidate_full.get("max_drawdown_pct", ""),
                "changed": True,
                "evidence": "same historical panel; candidate absorbed as formal route.",
            },
        ]
    )


def _formal_trade_decision_sample(daily: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    candidate_daily = daily[daily["variant"].astype(str).eq("combined_cap40_confirmation1")].copy()
    candidate_events = events[events["variant"].astype(str).eq("combined_cap40_confirmation1")].copy()
    merged = candidate_daily.merge(
        candidate_events[["date", "pool1_vote", "pool2_vote", "event_reason", "pool2_disagreement"]],
        on="date",
        how="left",
    )
    sample = merged.tail(30).copy()
    sample["formal_model_target"] = FORMAL_MODEL_TARGET
    sample["formal_selector_route"] = "pool1_primary_pool2_confirmation_cap40"
    sample["active_in_trade_decision"] = True
    sample["pool3_shadow_used_as_formal"] = False
    sample["0050x2_opportunity_label_active_in_trade_decision"] = False
    sample["market_exposure_override_absorbed"] = False
    sample["rr_partial_switch_used_in_performance"] = False
    sample["uses_forward_return_as_rule"] = False
    cols = [
        "date",
        "period",
        "formal_model_target",
        "formal_selector_route",
        "pool1_vote",
        "pool2_vote",
        "pool2_disagreement",
        "target_weights",
        "position_ticker",
        "action",
        "event_reason",
        "active_in_trade_decision",
        "pool3_shadow_used_as_formal",
        "0050x2_opportunity_label_active_in_trade_decision",
        "market_exposure_override_absorbed",
        "rr_partial_switch_used_in_performance",
        "uses_forward_return_as_rule",
    ]
    return sample[[col for col in cols if col in sample.columns]]


def _blocker_matrix() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "blocker_id": "2024_hard_gate_0050x2_caveat",
                "description": "2024 hard gate period still lagged 0050x2; this is a residual opportunity-cost caveat.",
                "severity": "medium",
                "blocks_formal_absorption": False,
                "owner": "Core/Research report boundary",
                "minimal_handling": "Keep 0050x2 opportunity-cost label report-only; do not add override trade rule.",
                "status": "accepted_caveat",
            },
            {
                "blocker_id": "market_exposure_override_rejected",
                "description": "Market exposure override improved hard gate partially but hurt full period/MDD/execution stability.",
                "severity": "medium",
                "blocks_formal_absorption": False,
                "owner": "Core",
                "minimal_handling": "Keep market_exposure_override_absorbed=false.",
                "status": "closed_as_not_absorbed",
            },
            {
                "blocker_id": "pool3_not_formal",
                "description": "Pool3 Radar/style/pure-stock routes remain diagnostic and are not part of this formal selector.",
                "severity": "low",
                "blocks_formal_absorption": False,
                "owner": "Research/Data future candidate source",
                "minimal_handling": "Preserve pool3_shadow_used_as_formal=false.",
                "status": "non_blocking_boundary",
            },
            {
                "blocker_id": "execution_layer_not_absorbed",
                "description": "RR partial switch and execution/exit layer remain paper-trade or diagnostic only.",
                "severity": "low",
                "blocks_formal_absorption": False,
                "owner": "Core/Experiments future execution validation",
                "minimal_handling": "Keep rr_partial_switch_used_in_performance=false.",
                "status": "non_blocking_boundary",
            },
        ]
    )


def _formal_model_contract_markdown(contract: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Formal Model Contract",
            "",
            f"- formal_model_target: `{contract['formal_model_target']}`",
            f"- formal_model_route: `{contract['formal_model_route']}`",
            "- selector: Pool1 primary attack selector.",
            "- Pool2 layer: confirmation/risk layer; when Pool2 disagrees with Pool1, Pool1 target must persist for one prior signal day.",
            "- 00631L cap: max target weight 40%; residual stays as cash in the absorbed contract.",
            "- three_pool_formal_route_abandoned=true.",
            "- Pool3 remains shadow/diagnostic only and is not a formal vote source.",
            "- 0050x2 opportunity-cost label remains report-only and is not active in trade decision.",
            "- market exposure override is not absorbed.",
            "- RR partial switch execution layer is not absorbed.",
            "- valuation/H3/forward return are not used as rules.",
            "",
            "This contract is an AI-assisted market observation and replay contract, not a complete execution/exit system.",
            "",
        ]
    )


def _report_output_before_after(candidate_full: dict[str, Any], candidate_hard_gate: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Report Output Boundary Before / After",
            "",
            "## Before",
            "",
            "- Formal wording referenced current three-pool baseline and three-pool consensus as the main formal route.",
            "",
            "## After",
            "",
            "- Formal wording must identify `combined_cap40_confirmation1_base` as the active formal baseline.",
            "- Three-pool consensus may remain as diagnostic/history context, but not as the formal performance selector.",
            "- Pool3 Radar / Pool3 shadow remains report-only.",
            "- 0050x2 opportunity-cost label remains report-only and cannot change target or trade decision.",
            "",
            "## Residual Caveat",
            "",
            f"- Absorbed candidate full return: {candidate_full.get('return_pct', '')}%, MDD: {candidate_full.get('max_drawdown_pct', '')}%.",
            f"- 2024 hard gate return: {candidate_hard_gate.get('return_pct', '')}%; prior validation still requires 0050x2 opportunity-cost caveat wording.",
            "",
        ]
    )


def _summary_markdown(candidate_full: dict[str, Any], old_full: dict[str, Any], blocker_matrix: pd.DataFrame) -> str:
    return "\n".join(
        [
            "# Pool1+Pool2 Formal Absorption Summary",
            "",
            "狀態：已正式吸收 `combined_cap40_confirmation1_base` 作為目前 formal model target。",
            "",
            "## 正式切換",
            "",
            "- 放棄三池 as-is performance selector formal route。",
            "- 新正式路線：Pool1 主攻 selector + Pool2 confirmation/risk layer。",
            "- 00631L 目標權重上限 40%，剩餘部位保留現金處理。",
            "- Pool3 不進正式票；0050正二 opportunity-cost label 保持 report-only。",
            "",
            "## Before / After",
            "",
            f"- 舊三池 baseline full return: {old_full.get('total_return_pct', '')}%，MDD: {old_full.get('max_drawdown_pct', '')}%。",
            f"- 新正式候選 full return: {candidate_full.get('return_pct', '')}%，MDD: {candidate_full.get('max_drawdown_pct', '')}%。",
            "",
            "## Residual Caveats",
            "",
            f"- blocker_matrix rows: {len(blocker_matrix)}。",
            "- 剩餘 caveat 不阻止本次 formal absorption；主要是 2024 hard gate 對 0050正二的機會成本警示。",
            "",
            "本輸出是模型路線吸收與報告邊界，不是買賣建議。",
            "",
        ]
    )


def _float(value: Any) -> float | None:
    try:
        if value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _truthy(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Absorb Pool1+Pool2 combined cap40 confirmation1 as formal model contract.")
    parser.add_argument("--candidate-dir", default=DEFAULT_CANDIDATE_DIR)
    parser.add_argument("--three-pool-dir", default=DEFAULT_THREE_POOL_DIR)
    parser.add_argument("--opportunity-label-dir", default=DEFAULT_LABEL_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    output = run_formal_absorb_pool1_pool2(
        candidate_dir=args.candidate_dir,
        three_pool_dir=args.three_pool_dir,
        opportunity_label_dir=args.opportunity_label_dir,
        output_dir=args.output_dir,
    )
    print(f"OUTPUT_DIR={output.resolve()}")


if __name__ == "__main__":
    main()
