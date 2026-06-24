from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd


POOL3_ID = "large_core_bluechip_v0"
GATE_RULE_ID_V2 = "core_style_complement_opportunity_gate_v2"
MAX_DRAWDOWN20_V2 = -0.08
VARIANTS = (
    "style_complement_v1_e68d31f",
    "style_complement_v2_mdd_cap",
    "style_complement_v2_consensus_aware",
    "style_complement_v2_mdd_cap_plus_consensus_aware",
    "style_complement_v2_mdd_cap_plus_consensus_aware_plus_fallback",
)


def run_pool3_style_complement_v2_challenger(
    *,
    replay_panel_path: str | Path,
    top_candidates_path: str | Path,
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

    log("load_inputs", "started", "")
    replay = pd.read_csv(replay_panel_path).fillna("")
    top_candidates = pd.read_csv(top_candidates_path).fillna("")
    _validate_inputs(replay, top_candidates)

    log("build_variant_panels", "started", "")
    diff_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    outputs: dict[str, str] = {}
    for variant in VARIANTS:
        panel, variant_diff = _build_variant_panel(replay, top_candidates, variant=variant)
        panel_path = output / f"{variant}_replay_panel.csv"
        panel.to_csv(panel_path, index=False, encoding="utf-8-sig")
        outputs[variant] = str(panel_path)
        diff_rows.extend(variant_diff)
        summary_rows.append(_variant_summary(panel, variant_diff, variant=variant))

    diff_path = output / "pool3_style_complement_v2_decision_diff.csv"
    summary_path = output / "pool3_style_complement_v2_variant_summary.csv"
    pd.DataFrame(diff_rows).to_csv(diff_path, index=False, encoding="utf-8-sig")
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False, encoding="utf-8-sig")
    metadata = {
        "schema_version": 1,
        "task_id": "TASK-BACKTEST-CORE-POOL3-STYLE-COMPLEMENT-V2-GATES-001",
        "status": "completed",
        "model": "pool3_style_complement_v2_challenger",
        "formal_model_changed": False,
        "trade_decision_changed": False,
        "active_in_trade_decision": False,
        "replay_panel_path": str(replay_panel_path),
        "top_candidates_path": str(top_candidates_path),
        "gate_rule_id": GATE_RULE_ID_V2,
        "max_drawdown20_v2": MAX_DRAWDOWN20_V2,
        "variants": list(VARIANTS),
        "outputs": {
            "variant_replay_panels": outputs,
            "decision_diff": str(diff_path),
            "variant_summary": str(summary_path),
            "run_log": str(output / "run_log.csv"),
        },
        "boundaries": [
            "challenger_replay_panel_only",
            "formal_model_unchanged",
            "pool3_radar_not_used",
            "valuation_not_used",
            "h3_day_trading_margin_overheat_not_used",
        ],
    }
    (output / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame([{"status": "completed", "output_dir": str(output.resolve())}]).to_csv(
        output / "completed.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(columns=["step", "error"]).to_csv(output / "failed.csv", index=False, encoding="utf-8-sig")
    log("completed", "completed", str(output.resolve()))
    (output / "current_step.txt").write_text("completed\n", encoding="utf-8")
    return output


def _validate_inputs(replay: pd.DataFrame, top_candidates: pd.DataFrame) -> None:
    required_replay = {"period", "requested_signal_date", "pool_id", "top_ticker", "selection_layer", "eligible_for_pool_selection"}
    required_candidates = {"period", "requested_signal_date", "pool_id", "ticker", "selection_layer", "eligible_for_pool_selection"}
    missing_replay = required_replay - set(replay.columns)
    missing_candidates = required_candidates - set(top_candidates.columns)
    if missing_replay:
        raise ValueError("missing replay panel columns: " + ",".join(sorted(missing_replay)))
    if missing_candidates:
        raise ValueError("missing top candidate columns: " + ",".join(sorted(missing_candidates)))


def _build_variant_panel(replay: pd.DataFrame, top_candidates: pd.DataFrame, *, variant: str) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    panel = replay.copy().astype(object)
    for column in (
        "pool3_style_gate_v2_pass",
        "pool3_style_gate_v2_reason",
        "pool3_mdd_risk_gate_state",
        "pool3_consensus_alignment_state",
        "pool3_consensus_alignment_pool",
        "pool3_lead_observation",
        "pool3_fallback_state",
        "pool3_protocol_state",
    ):
        if column not in panel.columns:
            panel[column] = ""
    diffs: list[dict[str, Any]] = []
    if variant == "style_complement_v1_e68d31f":
        panel["pool3_style_gate_v2_pass"] = ""
        return panel, diffs

    for index, row in panel[panel["pool_id"].astype(str) == POOL3_ID].iterrows():
        original = row.to_dict()
        evaluation = _evaluate_pool3_row(panel, top_candidates, row)
        adjusted = _apply_variant(row.to_dict(), top_candidates, evaluation, variant=variant)
        for key, value in adjusted.items():
            panel.at[index, key] = value
        if _changed(original, adjusted):
            diffs.append(
                {
                    "variant": variant,
                    "period": row.get("period", ""),
                    "requested_signal_date": row.get("requested_signal_date", row.get("signal_date", "")),
                    "original_ticker": original.get("top_ticker", ""),
                    "challenger_ticker": adjusted.get("top_ticker", ""),
                    "original_selection_layer": original.get("selection_layer", ""),
                    "challenger_selection_layer": adjusted.get("selection_layer", ""),
                    "original_eligible": original.get("eligible_for_pool_selection", ""),
                    "challenger_eligible": adjusted.get("eligible_for_pool_selection", ""),
                    "mdd_state": adjusted.get("pool3_mdd_risk_gate_state", ""),
                    "alignment_state": adjusted.get("pool3_consensus_alignment_state", ""),
                    "fallback_state": adjusted.get("pool3_fallback_state", ""),
                    "reason": adjusted.get("pool3_style_gate_v2_reason", ""),
                }
            )
    return panel, diffs


def _evaluate_pool3_row(panel: pd.DataFrame, top_candidates: pd.DataFrame, row: pd.Series) -> dict[str, Any]:
    asset_type = str(row.get("top_asset_type") or "").strip()
    selection_layer = str(row.get("selection_layer") or "").strip()
    ticker = str(row.get("top_ticker") or "").strip()
    is_stock_formal = asset_type == "stock" and selection_layer == "formal_candidate" and _truthy(row.get("eligible_for_pool_selection"))
    drawdown20 = _parse_drawdown20(str(row.get("gate_reason") or row.get("selection_reason") or ""))
    mdd_pass = drawdown20 is None or drawdown20 >= MAX_DRAWDOWN20_V2
    alignment_pool = _alignment_pool(panel, row, ticker)
    alignment_pass = bool(alignment_pool)
    return {
        "is_stock_formal": is_stock_formal,
        "drawdown20": drawdown20,
        "mdd_pass": mdd_pass,
        "alignment_pass": alignment_pass,
        "alignment_pool": alignment_pool,
        "fallback": _market_exposure_fallback(top_candidates, row),
    }


def _apply_variant(row: dict[str, Any], top_candidates: pd.DataFrame, evaluation: dict[str, Any], *, variant: str) -> dict[str, Any]:
    adjusted = dict(row)
    if str(row.get("pool_id") or "") != POOL3_ID:
        return adjusted
    use_mdd = "mdd_cap" in variant
    use_alignment = "consensus_aware" in variant
    use_fallback = variant.endswith("_plus_fallback")
    mdd_pass = (not use_mdd) or bool(evaluation["mdd_pass"])
    alignment_pass = (not use_alignment) or bool(evaluation["alignment_pass"])
    gate_pass = bool(evaluation["is_stock_formal"] and mdd_pass and alignment_pass)
    adjusted.update(
        {
            "gate_rule_id": GATE_RULE_ID_V2,
            "pool3_style_gate_v2_pass": _bool_text(gate_pass),
            "pool3_mdd_risk_gate_state": _mdd_state(evaluation, use_mdd=use_mdd),
            "pool3_consensus_alignment_state": _alignment_state(evaluation, use_alignment=use_alignment),
            "pool3_consensus_alignment_pool": evaluation.get("alignment_pool", ""),
            "pool3_lead_observation": _bool_text(bool(evaluation["is_stock_formal"] and not alignment_pass)),
            "pool3_protocol_state": "pool3_lead_observation" if evaluation["is_stock_formal"] and not alignment_pass else "",
        }
    )
    if gate_pass or not evaluation["is_stock_formal"]:
        adjusted["pool3_fallback_state"] = "not_needed"
        adjusted["pool3_style_gate_v2_reason"] = "Pool3 v2 pass or non-stock/market-exposure row unchanged."
        return adjusted

    reason = _blocked_reason(mdd_pass=mdd_pass, alignment_pass=alignment_pass)
    if use_fallback and evaluation.get("fallback"):
        fallback = evaluation["fallback"]
        adjusted.update(
            {
                "top_ticker": fallback["ticker"],
                "top_display": fallback.get("display", fallback["ticker"]),
                "top_asset_type": "etf",
                "selection_layer": "market_exposure_tool",
                "eligible_for_pool_selection": "true",
                "attack_gate_open": "",
                "gate_reason": f"Pool3 v2 fallback：{reason}；改用池內市場曝險工具。",
                "selection_reason": f"Pool3 v2 fallback：{reason}；改用池內市場曝險工具。",
                "pool3_fallback_state": "market_exposure_tool",
                "pool3_style_gate_v2_reason": reason,
            }
        )
        return adjusted

    adjusted.update(
        {
            "selection_layer": "observation_only",
            "eligible_for_pool_selection": "false",
            "attack_gate_open": "false",
            "gate_reason": f"Pool3 v2 observation：{reason}",
            "selection_reason": f"Pool3 v2 observation：{reason}",
            "pool3_fallback_state": "no_vote",
            "pool3_style_gate_v2_reason": reason,
        }
    )
    return adjusted


def _parse_drawdown20(text: str) -> float | None:
    match = re.search(r"20日回撤控管=([+-]?\d+(?:\.\d+)?)%", text)
    if not match:
        return None
    return float(match.group(1)) / 100.0


def _alignment_pool(panel: pd.DataFrame, row: pd.Series, ticker: str) -> str:
    if not ticker:
        return ""
    date = str(row.get("requested_signal_date") or row.get("signal_date") or "")
    period = str(row.get("period") or "")
    subset = panel[
        (panel["period"].astype(str) == period)
        & (panel["requested_signal_date"].astype(str) == date)
        & (panel["pool_id"].astype(str) != POOL3_ID)
        & panel["eligible_for_pool_selection"].map(_truthy)
    ]
    for _, item in subset.iterrows():
        if str(item.get("top_ticker") or "").strip() == ticker:
            return str(item.get("pool_id") or "")
    return ""


def _market_exposure_fallback(top_candidates: pd.DataFrame, row: pd.Series) -> dict[str, Any] | None:
    date = str(row.get("requested_signal_date") or row.get("signal_date") or "")
    period = str(row.get("period") or "")
    subset = top_candidates[
        (top_candidates["period"].astype(str) == period)
        & (top_candidates["requested_signal_date"].astype(str) == date)
        & (top_candidates["pool_id"].astype(str) == POOL3_ID)
        & (top_candidates["selection_layer"].astype(str) == "market_exposure_tool")
        & top_candidates["eligible_for_pool_selection"].map(_truthy)
    ].copy()
    if subset.empty:
        return None
    if "rank" in subset.columns:
        subset["_rank_number"] = pd.to_numeric(subset["rank"], errors="coerce").fillna(999)
        subset = subset.sort_values("_rank_number")
    item = subset.iloc[0]
    return {
        "ticker": str(item.get("ticker") or "").strip(),
        "display": str(item.get("display") or item.get("ticker") or "").strip(),
    }


def _mdd_state(evaluation: dict[str, Any], *, use_mdd: bool) -> str:
    if not use_mdd:
        return "not_applied"
    drawdown = evaluation.get("drawdown20")
    if drawdown is None:
        return "unknown_pass"
    return "pass" if evaluation["mdd_pass"] else "blocked"


def _alignment_state(evaluation: dict[str, Any], *, use_alignment: bool) -> str:
    if not use_alignment:
        return "not_applied"
    return "aligned" if evaluation["alignment_pass"] else "pool3_lead_unaligned"


def _blocked_reason(*, mdd_pass: bool, alignment_pass: bool) -> str:
    reasons = []
    if not mdd_pass:
        reasons.append("20日回撤風險超過 v2 cap")
    if not alignment_pass:
        reasons.append("未與池1/池2形成同標的一致")
    return "；".join(reasons) or "未通過 Pool3 v2 gate"


def _changed(original: dict[str, Any], adjusted: dict[str, Any]) -> bool:
    keys = ("top_ticker", "selection_layer", "eligible_for_pool_selection")
    return any(str(original.get(key, "")) != str(adjusted.get(key, "")) for key in keys)


def _variant_summary(panel: pd.DataFrame, diff_rows: list[dict[str, Any]], *, variant: str) -> dict[str, Any]:
    pool3 = panel[panel["pool_id"].astype(str) == POOL3_ID]
    return {
        "variant": variant,
        "pool3_rows": int(len(pool3)),
        "pool3_eligible_rows": int(pool3["eligible_for_pool_selection"].map(_truthy).sum()),
        "pool3_formal_candidate_rows": int((pool3["selection_layer"].astype(str) == "formal_candidate").sum()),
        "pool3_market_exposure_rows": int((pool3["selection_layer"].astype(str) == "market_exposure_tool").sum()),
        "pool3_observation_only_rows": int((pool3["selection_layer"].astype(str) == "observation_only").sum()),
        "changed_rows": len(diff_rows),
        "active_in_trade_decision": False,
        "formal_model_changed": False,
    }


def _truthy(value: object) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def _bool_text(value: bool) -> str:
    return "true" if value else "false"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Pool3 style-complement v2 challenger replay panels.")
    parser.add_argument("--replay-panel", required=True)
    parser.add_argument("--top-candidates", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    run_pool3_style_complement_v2_challenger(
        replay_panel_path=args.replay_panel,
        top_candidates_path=args.top_candidates,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
