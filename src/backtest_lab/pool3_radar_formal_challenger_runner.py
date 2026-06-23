from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_OUTPUT_DIR = "outputs/pool3_radar_formal_challenger_runner_20260623"
PRIMARY_CANDIDATE = "ma200_radar20_00631l80_else_top10"
REQUIRED_SCENARIOS = (
    "baseline_existing_pool3",
    "pool3_00631l_with_radar20_satellite",
    "pool3_ma200_00631l_else_radar_top1",
    "pool3_radar_top1_always",
)
ALLOWED_PERFORMANCE_VARIANTS = {
    PRIMARY_CANDIDATE,
    "mix_top10_70_00631l_30",
    "mix_top10_50_00631l_50",
    "top10_base",
}


def run_pool3_radar_formal_challenger_runner(
    *,
    opportunity_overlay_dir: str | Path,
    three_pool_shadow_dir: str | Path,
    attack_pool_dir: str | Path | None,
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
    opportunity_dir = Path(opportunity_overlay_dir)
    shadow_dir = Path(three_pool_shadow_dir)
    attack_dir = Path(attack_pool_dir) if attack_pool_dir else None
    overlay_performance = _read_csv(opportunity_dir / "overlay_performance.csv")
    vote_panel = _read_csv(shadow_dir / "three_pool_shadow_vote_panel.csv")
    attack_yearly = _read_optional_csv(attack_dir / "yearly_performance.csv" if attack_dir else None)
    ticker_concentration = _read_optional_csv(attack_dir / "ticker_concentration.csv" if attack_dir else None)
    theme_concentration = _read_optional_csv(attack_dir / "theme_concentration.csv" if attack_dir else None)
    readiness = _read_optional_json(attack_dir / "readiness_summary.json" if attack_dir else None)
    log("load_inputs", "completed", f"overlay_rows={len(overlay_performance)};vote_rows={len(vote_panel)}")

    log("build_contract_outputs", "started", "")
    performance_summary = _build_performance_summary(overlay_performance, attack_yearly)
    decision_diff = _build_decision_diff_panel(vote_panel)
    consensus_summary = _build_consensus_summary(vote_panel)
    concentration = _build_concentration_summary(ticker_concentration, theme_concentration, vote_panel)
    readiness_frame = _readiness_frame(readiness)
    hard_gate = _build_2024_hard_gate(performance_summary)
    metadata = _build_metadata(
        opportunity_dir=opportunity_dir,
        shadow_dir=shadow_dir,
        attack_dir=attack_dir,
        performance_summary=performance_summary,
        decision_diff=decision_diff,
        readiness=readiness,
        hard_gate=hard_gate,
    )

    performance_summary.to_csv(output / "baseline_vs_challengers.csv", index=False, encoding="utf-8-sig")
    decision_diff.to_csv(output / "decision_diff_panel.csv", index=False, encoding="utf-8-sig")
    consensus_summary.to_csv(output / "consensus_state_summary.csv", index=False, encoding="utf-8-sig")
    concentration.to_csv(output / "concentration_summary.csv", index=False, encoding="utf-8-sig")
    readiness_frame.to_csv(output / "readiness_gap.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(hard_gate).to_csv(output / "hard_gate_2024.csv", index=False, encoding="utf-8-sig")
    (output / "manifest.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "final_summary_zh.md").write_text(_markdown_summary(metadata, performance_summary, hard_gate), encoding="utf-8")
    (output / "completed.txt").write_text("completed\n", encoding="utf-8")
    log("completed", "completed", str(output.resolve()))
    (output / "current_step.txt").write_text("completed\n", encoding="utf-8")
    return output


def _build_performance_summary(overlay_performance: pd.DataFrame, attack_yearly: pd.DataFrame) -> pd.DataFrame:
    frame = overlay_performance.copy()
    frame = frame[frame["variant_id"].astype(str).isin(ALLOWED_PERFORMANCE_VARIANTS)].copy()
    frame["source"] = "opportunity_overlay"
    frame["active_in_trade_decision"] = False
    frame["formal_model_changed"] = False
    frame["valuation_used"] = False
    frame["h3_used"] = False
    if not attack_yearly.empty:
        top10 = attack_yearly[attack_yearly["variant_id"].astype(str).str.contains("top10", na=False)].copy()
        if not top10.empty:
            cols = [
                "period",
                "variant_id",
                "total_return_pct",
                "max_drawdown_pct",
                "trade_count",
                "benchmark_0050_return_pct",
                "benchmark_00631l_return_pct",
                "excess_vs_0050_pct",
                "excess_vs_00631l_pct",
            ]
            available = [col for col in cols if col in top10.columns]
            top10 = top10[available].copy()
            top10["source"] = "pool3_radar_top10_attack_pool"
            top10["active_in_trade_decision"] = False
            top10["formal_model_changed"] = False
            top10["valuation_used"] = False
            top10["h3_used"] = False
            frame = pd.concat([frame, top10], ignore_index=True, sort=False)
    return frame.sort_values(["period", "variant_id"]).reset_index(drop=True)


def _build_decision_diff_panel(vote_panel: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for row in vote_panel.to_dict(orient="records"):
        scenario = str(row.get("scenario") or "")
        rows.append(
            {
                "period": row.get("period", ""),
                "signal_date": row.get("signal_date", ""),
                "scenario": scenario,
                "pool1_vote": row.get("pool1_vote", ""),
                "pool2_vote": row.get("pool2_vote", ""),
                "pool3_vote": row.get("pool3_vote", ""),
                "winner_ticker": row.get("winner_ticker", ""),
                "winner_vote_count": _number(row.get("winner_vote_count")),
                "result_state": row.get("result_state", ""),
                "pool3_shadow_risk_on_0050_ma200": _bool(row.get("pool3_shadow_risk_on_0050_ma200")),
                "radar_top1_ticker": row.get("radar_top1_ticker", ""),
                "radar_top1_theme": row.get("radar_top1_theme", ""),
                "radar_top1_weight": _number(row.get("radar_top1_weight")),
                "changed_pool3_vote_from_baseline": _pool3_changed_from_baseline(vote_panel, row),
                "active_in_trade_decision": False,
                "formal_model_changed": False,
                "valuation_used": False,
                "h3_used": False,
            }
        )
    return pd.DataFrame(rows)


def _build_consensus_summary(vote_panel: pd.DataFrame) -> pd.DataFrame:
    if vote_panel.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for scenario, frame in vote_panel.groupby("scenario", dropna=False):
        rows.append(
            {
                "scenario": scenario,
                "sample_rows": len(frame),
                "consensus_count": int((frame["result_state"].astype(str) == "consensus").sum()),
                "no_vote_count": int((frame["result_state"].astype(str) == "no_vote").sum()),
                "insufficient_votes_count": int((frame["result_state"].astype(str) == "insufficient_votes").sum()),
                "radar_top1_present_count": int(frame["radar_top1_ticker"].astype(str).str.strip().ne("").sum()) if "radar_top1_ticker" in frame.columns else 0,
                "active_in_trade_decision": False,
            }
        )
    return pd.DataFrame(rows)


def _build_concentration_summary(
    ticker_concentration: pd.DataFrame,
    theme_concentration: pd.DataFrame,
    vote_panel: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if not ticker_concentration.empty:
        top = ticker_concentration.head(20)
        for row in top.to_dict(orient="records"):
            rows.append({"source": "ticker_concentration", **row})
    if not theme_concentration.empty:
        top = theme_concentration.head(20)
        for row in top.to_dict(orient="records"):
            rows.append({"source": "theme_concentration", **row})
    if vote_panel.empty:
        return pd.DataFrame(rows)
    radar = vote_panel[vote_panel["radar_top1_ticker"].astype(str).str.strip().ne("")].copy() if "radar_top1_ticker" in vote_panel.columns else pd.DataFrame()
    if not radar.empty:
        counts = radar["radar_top1_ticker"].astype(str).value_counts(normalize=True).head(20)
        for ticker, share in counts.items():
            rows.append(
                {
                    "source": "quick_vote_radar_top1_share",
                    "ticker": ticker,
                    "share": round(float(share), 6),
                    "sample_rows": len(radar),
                }
            )
    return pd.DataFrame(rows)


def _readiness_frame(readiness: dict[str, Any]) -> pd.DataFrame:
    if not readiness:
        return pd.DataFrame([{"status": "unknown", "reason": "readiness_summary_missing"}])
    rows = []
    for key, value in readiness.items():
        if isinstance(value, (dict, list)):
            text = json.dumps(value, ensure_ascii=False)
        else:
            text = str(value)
        rows.append({"field": key, "value": text})
    return pd.DataFrame(rows)


def _build_2024_hard_gate(performance_summary: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    subset = performance_summary[
        (performance_summary["period"].astype(str) == "2024")
        & (performance_summary["variant_id"].astype(str) == PRIMARY_CANDIDATE)
    ]
    if subset.empty:
        return [
            {
                "variant_id": PRIMARY_CANDIDATE,
                "period": "2024",
                "status": "blocked",
                "reason": "primary_candidate_2024_result_missing",
            }
        ]
    row = subset.iloc[0]
    mdd = _number(row.get("max_drawdown_pct"))
    excess = _number(row.get("excess_vs_0050_pct"))
    status = "needs_research_review" if mdd <= -25 or excess < 5 else "pass"
    rows.append(
        {
            "variant_id": PRIMARY_CANDIDATE,
            "period": "2024",
            "status": status,
            "max_drawdown_pct": round(mdd, 4),
            "excess_vs_0050_pct": round(excess, 4),
            "reason": "2024 MDD too deep or excess vs 0050 too small" if status != "pass" else "2024 hard gate passed",
        }
    )
    return rows


def _build_metadata(
    *,
    opportunity_dir: Path,
    shadow_dir: Path,
    attack_dir: Path | None,
    performance_summary: pd.DataFrame,
    decision_diff: pd.DataFrame,
    readiness: dict[str, Any],
    hard_gate: list[dict[str, Any]],
) -> dict[str, Any]:
    required_missing = []
    if performance_summary.empty:
        required_missing.append("overlay_performance.csv usable rows")
    if decision_diff.empty:
        required_missing.append("three_pool_shadow_vote_panel.csv usable rows")
    readiness_blockers = _radar_readiness_blockers(readiness)
    full_replay_blockers = [
        "daily weighted basket holdings/equity are not available in current Experiments handoff",
        "baseline three-pool daily equity curve is not available in current Experiments handoff",
    ]
    full_replay_blockers.extend(readiness_blockers)
    status = "blocked_data_readiness" if readiness_blockers else "partial_contract"
    return {
        "model": "pool3_radar_formal_challenger_runner_v1",
        "status": status,
        "decision_layer": "formal_challenger_contract",
        "active_in_trade_decision": False,
        "formal_model_changed": False,
        "pool3_formal_vote_changed": False,
        "valuation_used": False,
        "h3_used": False,
        "primary_candidate": PRIMARY_CANDIDATE,
        "required_scenarios": list(REQUIRED_SCENARIOS),
        "inputs": {
            "opportunity_overlay_dir": str(opportunity_dir),
            "three_pool_shadow_dir": str(shadow_dir),
            "attack_pool_dir": str(attack_dir) if attack_dir else "",
        },
        "outputs": {
            "baseline_vs_challengers": "baseline_vs_challengers.csv",
            "decision_diff_panel": "decision_diff_panel.csv",
            "consensus_state_summary": "consensus_state_summary.csv",
            "concentration_summary": "concentration_summary.csv",
            "hard_gate_2024": "hard_gate_2024.csv",
            "readiness_gap": "readiness_gap.csv",
        },
        "rows": {
            "performance_summary": int(len(performance_summary)),
            "decision_diff": int(len(decision_diff)),
        },
        "readiness_status": readiness.get("status") or readiness.get("formal_top3_status") or "unknown",
        "radar_readiness_blockers": readiness_blockers,
        "full_replay_ready": False,
        "full_replay_blockers": full_replay_blockers,
        "required_missing": required_missing,
        "hard_gate_2024": hard_gate,
        "experiments_next_step": (
            "Use this contract to add daily weighted basket equity/holdings, then rerun full replay. "
            "Do not promote Pool3 Radar overlay until full replay passes 2024 hard gate and concentration checks."
        ),
    }


def _radar_readiness_blockers(readiness: dict[str, Any]) -> list[str]:
    blockers: list[str] = []
    if not readiness:
        return ["radar readiness summary missing"]
    if readiness.get("formal_top3_ready") is False:
        issues = readiness.get("formal_top3_blocking_issues") or []
        detail = ",".join(str(item) for item in issues) if issues else "formal_top3_ready=false"
        blockers.append(f"radar formal_top3 not ready: {detail}")
    if readiness.get("theme_membership_v2_ready") is False:
        status = readiness.get("theme_membership_v2_formal_top3_status") or "unknown"
        blockers.append(f"radar theme_membership_v2 not ready: {status}")
    coverage_rows = readiness.get("price_cache_coverage_by_year") or []
    low_coverage = []
    for row in coverage_rows:
        try:
            ratio = float(row.get("price_cache_coverage_ratio") or 0.0)
        except (TypeError, ValueError):
            ratio = 0.0
        if ratio < 0.95:
            low_coverage.append(f"{row.get('period', 'unknown')}={ratio:.2%}")
    if low_coverage:
        blockers.append("radar member price cache coverage below 95%: " + "; ".join(low_coverage))
    return blockers


def _markdown_summary(metadata: dict[str, Any], performance_summary: pd.DataFrame, hard_gate: list[dict[str, Any]]) -> str:
    lines = [
        "# Pool3 Radar Formal Challenger Runner",
        "",
        f"- status: `{metadata['status']}`",
        f"- active_in_trade_decision: `{metadata['active_in_trade_decision']}`",
        f"- formal_model_changed: `{metadata['formal_model_changed']}`",
        f"- valuation_used: `{metadata['valuation_used']}`",
        f"- h3_used: `{metadata['h3_used']}`",
        f"- primary_candidate: `{PRIMARY_CANDIDATE}`",
        "",
        "本輸出是 Core formal challenger contract，不是正式模型升級。",
        "",
        "## 2024 hard gate",
        "",
    ]
    for row in hard_gate:
        lines.append(f"- {row.get('variant_id')}: {row.get('status')}；{row.get('reason')}")
    lines.extend(["", "## Yearly performance rows", ""])
    if performance_summary.empty:
        lines.append("- 無可用績效列。")
    else:
        cols = ["period", "variant_id", "total_return_pct", "max_drawdown_pct", "excess_vs_0050_pct", "source"]
        available = [col for col in cols if col in performance_summary.columns]
        lines.append("| " + " | ".join(available) + " |")
        lines.append("| " + " | ".join(["---"] * len(available)) + " |")
        for row in performance_summary[available].to_dict(orient="records")[:80]:
            lines.append("| " + " | ".join(str(row.get(col, "")) for col in available) + " |")
    if metadata["full_replay_blockers"]:
        lines.extend(["", "## Full replay blockers", ""])
        for blocker in metadata["full_replay_blockers"]:
            lines.append(f"- {blocker}")
    return "\n".join(lines) + "\n"


def _pool3_changed_from_baseline(vote_panel: pd.DataFrame, row: dict[str, Any]) -> bool:
    if str(row.get("scenario") or "") == "baseline_existing_pool3":
        return False
    baseline = vote_panel[
        (vote_panel["period"].astype(str) == str(row.get("period") or ""))
        & (vote_panel["signal_date"].astype(str) == str(row.get("signal_date") or ""))
        & (vote_panel["scenario"].astype(str) == "baseline_existing_pool3")
    ]
    if baseline.empty:
        return False
    return str(baseline.iloc[0].get("pool3_vote") or "") != str(row.get("pool3_vote") or "")


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(str(path))
    return pd.read_csv(path).fillna("")


def _read_optional_csv(path: Path | None) -> pd.DataFrame:
    if path is None or not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path).fillna("")


def _read_optional_json(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _number(value: object) -> float:
    try:
        text = str(value).replace(",", "").replace("%", "").strip()
        return float(text) if text else 0.0
    except (TypeError, ValueError):
        return 0.0


def _bool(value: object) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Pool3 Radar formal challenger contract outputs.")
    parser.add_argument("--opportunity-overlay-dir", required=True)
    parser.add_argument("--three-pool-shadow-dir", required=True)
    parser.add_argument("--attack-pool-dir", default="")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    output = run_pool3_radar_formal_challenger_runner(
        opportunity_overlay_dir=args.opportunity_overlay_dir,
        three_pool_shadow_dir=args.three_pool_shadow_dir,
        attack_pool_dir=args.attack_pool_dir or None,
        output_dir=args.output_dir,
    )
    print(f"OUTPUT_DIR={output.resolve()}")


if __name__ == "__main__":
    main()
