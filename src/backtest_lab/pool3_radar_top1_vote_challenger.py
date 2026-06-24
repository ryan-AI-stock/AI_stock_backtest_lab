from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_OUTPUT_DIR = "outputs/pool3_radar_top1_vote_challenger_20260624"
TOP1_VARIANT = "top10_base"


def run_pool3_radar_top1_vote_challenger(
    *,
    weighted_basket_daily: str | Path,
    membership_csv: str | Path,
    readiness_manifest: str | Path | None,
    output_dir: str | Path,
    min_persistence_days: int = 3,
    max_abs_daily_return: float = 0.20,
    max_observed_drawdown_pct: float = -40.0,
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
    source = pd.read_csv(weighted_basket_daily).fillna("")
    membership = pd.read_csv(membership_csv).fillna("")
    readiness = _read_json(Path(readiness_manifest) if readiness_manifest else None)
    accepted = _accepted_tickers(membership)
    log("load_inputs", "completed", f"source_rows={len(source)};accepted={len(accepted)}")

    log("build_top1_panel", "started", "")
    top1_panel = _build_top1_panel(
        source,
        accepted_tickers=accepted,
        readiness=readiness,
        min_persistence_days=min_persistence_days,
        max_abs_daily_return=max_abs_daily_return,
        max_observed_drawdown_pct=max_observed_drawdown_pct,
    )
    summary = _summary(top1_panel)
    manifest = {
        "model": "pool3_radar_top1_vote_shadow_formal_challenger_v1",
        "status": "completed",
        "decision_layer": "shadow_formal_challenger",
        "active_in_trade_decision": False,
        "formal_model_changed": False,
        "pool3_formal_vote_changed": False,
        "pool3_radar_candidate_source": "batch07",
        "pool3_radar_vote_mode": "top1_shadow",
        "top1_variant": TOP1_VARIANT,
        "gate_parameters": {
            "min_persistence_days": min_persistence_days,
            "max_abs_daily_return": max_abs_daily_return,
            "max_observed_drawdown_pct": max_observed_drawdown_pct,
        },
        "readiness": _readiness_status(readiness),
        "rows": {"top1_vote_panel": len(top1_panel), "summary": len(summary)},
        "outputs": {
            "top1_vote_panel": "pool3_radar_top1_vote_panel.csv",
            "summary": "pool3_radar_top1_vote_summary.csv",
        },
        "next_step": "Experiments can use challenger_eligible_for_pool_selection and vote_target for full daily shadow decision diff.",
    }
    top1_panel.to_csv(output / "pool3_radar_top1_vote_panel.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(output / "pool3_radar_top1_vote_summary.csv", index=False, encoding="utf-8-sig")
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "final_summary_zh.md").write_text(_markdown_summary(manifest, summary), encoding="utf-8")
    (output / "completed.csv").write_text("status\ncompleted\n", encoding="utf-8")
    (output / "failed.csv").write_text("status,reason\n", encoding="utf-8")
    log("completed", "completed", str(output.resolve()))
    (output / "current_step.txt").write_text("completed\n", encoding="utf-8")
    return output


def _build_top1_panel(
    source: pd.DataFrame,
    *,
    accepted_tickers: set[str],
    readiness: dict[str, Any],
    min_persistence_days: int,
    max_abs_daily_return: float,
    max_observed_drawdown_pct: float,
) -> pd.DataFrame:
    frame = source.copy()
    frame = frame[frame["variant"].astype(str).eq(TOP1_VARIANT)]
    frame = frame[~frame["ticker"].astype(str).isin({"", "cash", "00631L.TW"})].copy()
    if frame.empty:
        return pd.DataFrame(columns=_columns())
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame = frame[frame["date"].notna()].copy()
    frame["weight"] = pd.to_numeric(frame["weight"], errors="coerce").fillna(0.0)
    frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
    frame["return"] = pd.to_numeric(frame["return"], errors="coerce")
    frame = frame.sort_values(["date", "weight", "ticker"], ascending=[True, False, True])

    consecutive: dict[str, int] = {}
    observed_high: dict[str, float] = {}
    rows: list[dict[str, Any]] = []
    for date, day in frame.groupby("date", dropna=False):
        ranked = day.sort_values(["weight", "ticker"], ascending=[False, True]).reset_index(drop=True)
        top = ranked.iloc[0]
        ticker = str(top.get("ticker") or "").strip()
        score = float(top.get("weight") or 0.0)
        second = float(ranked.iloc[1].get("weight") or 0.0) if len(ranked) > 1 else 0.0
        for key in list(consecutive):
            if key != ticker:
                consecutive[key] = 0
        consecutive[ticker] = consecutive.get(ticker, 0) + 1
        close = _number(top.get("close"))
        if close > 0:
            observed_high[ticker] = max(observed_high.get(ticker, close), close)
        drawdown_pct = ((close / observed_high[ticker] - 1.0) * 100.0) if close > 0 and ticker in observed_high else 0.0
        daily_return = _number(top.get("return"))

        formal_ready = _readiness_passed(readiness)
        accepted_pass = ticker in accepted_tickers
        price_pass = close > 0
        top1_gate_pass = bool(ticker and score > 0)
        persistence_days = int(consecutive.get(ticker, 0))
        persistence_pass = persistence_days >= min_persistence_days
        overheat_flags = []
        if abs(daily_return) > max_abs_daily_return:
            overheat_flags.append(f"abs_daily_return>{max_abs_daily_return:.2f}")
        risk_flags = []
        mdd_gate_pass = drawdown_pct >= max_observed_drawdown_pct
        if not mdd_gate_pass:
            risk_flags.append(f"observed_drawdown_pct<{max_observed_drawdown_pct:.1f}")
        gates = {
            "readiness": formal_ready,
            "accepted_universe": accepted_pass,
            "price_coverage": price_pass,
            "top1": top1_gate_pass,
            "overheat": not overheat_flags,
            "mdd_risk": mdd_gate_pass,
            "persistence": persistence_pass,
        }
        eligible = all(gates.values())
        ineligible = [key for key, passed in gates.items() if not passed]
        rows.append(
            {
                "date": pd.Timestamp(date).strftime("%Y-%m-%d"),
                "period": str(top.get("period") or ""),
                "pool3_radar_candidate_source": "batch07",
                "pool3_radar_formal_ready": formal_ready,
                "pool3_radar_vote_mode": "top1_shadow",
                "pool3_radar_top1_ticker": ticker,
                "pool3_radar_top1_theme": str(top.get("theme") or ""),
                "pool3_radar_top1_score": round(score, 8),
                "pool3_radar_top1_rank": 1,
                "pool3_radar_top1_score_gap_to_second": round(score - second, 8),
                "pool3_radar_top1_persistence_days": persistence_days,
                "pool3_radar_top1_persistence_weeks": round(persistence_days / 5.0, 2),
                "pool3_radar_top1_overheat_flags": ";".join(overheat_flags),
                "pool3_radar_top1_risk_flags": ";".join(risk_flags),
                "pool3_radar_top1_mdd_gate_pass": mdd_gate_pass,
                "pool3_radar_top1_observed_drawdown_pct": round(drawdown_pct, 4),
                "pool3_radar_top1_eligible_for_challenger_vote": eligible,
                "pool3_radar_top1_ineligible_reason": ";".join(ineligible),
                "vote_target": ticker if eligible else "",
                "challenger_eligible_for_pool_selection": eligible,
                "challenger_selection_layer": "formal_candidate" if eligible else "observation_only",
                "eligible_for_pool_selection": False,
                "selection_layer": "observation_only",
                "active_in_trade_decision": False,
                "formal_model_changed": False,
                "pool3_formal_vote_changed": False,
                "valuation_used": False,
                "h3_used": False,
            }
        )
    return pd.DataFrame(rows, columns=_columns())


def _columns() -> list[str]:
    return [
        "date",
        "period",
        "pool3_radar_candidate_source",
        "pool3_radar_formal_ready",
        "pool3_radar_vote_mode",
        "pool3_radar_top1_ticker",
        "pool3_radar_top1_theme",
        "pool3_radar_top1_score",
        "pool3_radar_top1_rank",
        "pool3_radar_top1_score_gap_to_second",
        "pool3_radar_top1_persistence_days",
        "pool3_radar_top1_persistence_weeks",
        "pool3_radar_top1_overheat_flags",
        "pool3_radar_top1_risk_flags",
        "pool3_radar_top1_mdd_gate_pass",
        "pool3_radar_top1_observed_drawdown_pct",
        "pool3_radar_top1_eligible_for_challenger_vote",
        "pool3_radar_top1_ineligible_reason",
        "vote_target",
        "challenger_eligible_for_pool_selection",
        "challenger_selection_layer",
        "eligible_for_pool_selection",
        "selection_layer",
        "active_in_trade_decision",
        "formal_model_changed",
        "pool3_formal_vote_changed",
        "valuation_used",
        "h3_used",
    ]


def _accepted_tickers(membership: pd.DataFrame) -> set[str]:
    if membership.empty or "ticker" not in membership.columns:
        return set()
    if "usable_for_formal_replay" in membership.columns:
        membership = membership[membership["usable_for_formal_replay"].map(_truthy)]
    if "review_status" in membership.columns:
        membership = membership[membership["review_status"].astype(str).str.lower().eq("accepted")]
    return {str(value).strip() for value in membership["ticker"].tolist() if str(value).strip()}


def _readiness_status(readiness: dict[str, Any]) -> dict[str, Any]:
    return {
        "theme_membership_v2_ready": readiness.get("theme_membership_v2_ready"),
        "coverage_ratio_after_batch": readiness.get("coverage_ratio_after_batch"),
        "future_data_violation_count": readiness.get("future_data_violation_count"),
        "remaining_gap_symbol_count": readiness.get("remaining_gap_symbol_count"),
    }


def _readiness_passed(readiness: dict[str, Any]) -> bool:
    if not readiness:
        return False
    if readiness.get("theme_membership_v2_ready") is not True:
        return False
    if int(readiness.get("future_data_violation_count") or 0) != 0:
        return False
    if int(readiness.get("remaining_gap_symbol_count") or 0) != 0:
        return False
    threshold = float(readiness.get("ready_threshold") or 0.95)
    return float(readiness.get("coverage_ratio_after_batch") or 0.0) >= threshold


def _summary(top1_panel: pd.DataFrame) -> pd.DataFrame:
    if top1_panel.empty:
        return pd.DataFrame([{"period": "all", "rows": 0, "eligible_rows": 0, "eligible_rate": 0.0}])
    rows: list[dict[str, Any]] = []
    for period, frame in top1_panel.groupby("period", dropna=False):
        eligible = int(frame["pool3_radar_top1_eligible_for_challenger_vote"].astype(bool).sum())
        rows.append(
            {
                "period": period,
                "rows": len(frame),
                "eligible_rows": eligible,
                "eligible_rate": round(eligible / len(frame), 6) if len(frame) else 0.0,
                "unique_top1_tickers": frame["pool3_radar_top1_ticker"].nunique(),
                "top_ticker_share": round(frame["pool3_radar_top1_ticker"].value_counts(normalize=True).iloc[0], 6)
                if not frame.empty
                else 0.0,
            }
        )
    rows.append(
        {
            "period": "all",
            "rows": len(top1_panel),
            "eligible_rows": int(top1_panel["pool3_radar_top1_eligible_for_challenger_vote"].astype(bool).sum()),
            "eligible_rate": round(float(top1_panel["pool3_radar_top1_eligible_for_challenger_vote"].astype(bool).mean()), 6),
            "unique_top1_tickers": top1_panel["pool3_radar_top1_ticker"].nunique(),
            "top_ticker_share": round(top1_panel["pool3_radar_top1_ticker"].value_counts(normalize=True).iloc[0], 6),
        }
    )
    return pd.DataFrame(rows)


def _markdown_summary(manifest: dict[str, Any], summary: pd.DataFrame) -> str:
    lines = [
        "# Pool3 Radar Top1 Vote Shadow Challenger",
        "",
        f"- status: `{manifest['status']}`",
        f"- active_in_trade_decision: `{manifest['active_in_trade_decision']}`",
        f"- formal_model_changed: `{manifest['formal_model_changed']}`",
        f"- pool3_formal_vote_changed: `{manifest['pool3_formal_vote_changed']}`",
        f"- vote_mode: `{manifest['pool3_radar_vote_mode']}`",
        "",
        "本輸出只供 Experiments 做 shadow formal challenger replay，不會改正式三池投票。",
        "",
        "## Summary",
        "",
        _markdown_table(summary),
    ]
    return "\n".join(lines) + "\n"


def _markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "- 無資料。"
    columns = [str(col) for col in frame.columns]
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in frame.to_dict(orient="records"):
        lines.append("| " + " | ".join(str(row.get(col, "")) for col in columns) + " |")
    return "\n".join(lines)


def _read_json(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _number(value: object) -> float:
    try:
        text = str(value).replace(",", "").replace("%", "").strip()
        return float(text) if text else 0.0
    except (TypeError, ValueError):
        return 0.0


def _truthy(value: object) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Pool3 Radar Top1 vote shadow challenger panel.")
    parser.add_argument("--weighted-basket-daily", required=True)
    parser.add_argument("--membership-csv", required=True)
    parser.add_argument("--readiness-manifest", default="")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--min-persistence-days", type=int, default=3)
    parser.add_argument("--max-abs-daily-return", type=float, default=0.20)
    parser.add_argument("--max-observed-drawdown-pct", type=float, default=-40.0)
    args = parser.parse_args()
    output = run_pool3_radar_top1_vote_challenger(
        weighted_basket_daily=args.weighted_basket_daily,
        membership_csv=args.membership_csv,
        readiness_manifest=args.readiness_manifest or None,
        output_dir=args.output_dir,
        min_persistence_days=args.min_persistence_days,
        max_abs_daily_return=args.max_abs_daily_return,
        max_observed_drawdown_pct=args.max_observed_drawdown_pct,
    )
    print(f"OUTPUT_DIR={output.resolve()}")


if __name__ == "__main__":
    main()
