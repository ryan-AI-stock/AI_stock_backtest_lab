from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from backtest_lab.formal_model_contract import FORMAL_MODEL_ROUTE, FORMAL_MODEL_TARGET
from backtest_lab.pool1_pool2_veto_cap_downweight import VariantSpec as DecisionVariantSpec
from backtest_lab.pool1_pool2_veto_cap_downweight import _build_target_weights


TASK_ID = "TASK-BACKTEST-CORE-FORMAL-LONG-RANGE-SIGNAL-RECONSTRUCTION-201411-20260702"
DEFAULT_2014_PANEL_DIR = "outputs/current_formal_pool1_pool2_signal_panels_201411_202112_20260630"
DEFAULT_FORMAL_REPLAY_DIR = "outputs/stock_pool_formal_daily_replay_pit_pool2_daily_final_combined_20260624"
DEFAULT_PIT_READINESS_DIR = "outputs/core_0050_pit_candidate_backtest_data_readiness_201411_202312_20260629"
DEFAULT_PRICE_ABSORPTION_DIR = "outputs/core_0050_pit_price_coverage_absorption_201411_202312_20260629"
DEFAULT_OUTPUT_DIR = "outputs/formal_long_range_signal_reconstruction_201411_latest_20260702"


def run_formal_long_range_signal_reconstruction(
    *,
    panel_2014_dir: str | Path = DEFAULT_2014_PANEL_DIR,
    formal_replay_dir: str | Path = DEFAULT_FORMAL_REPLAY_DIR,
    pit_readiness_dir: str | Path = DEFAULT_PIT_READINESS_DIR,
    price_absorption_dir: str | Path = DEFAULT_PRICE_ABSORPTION_DIR,
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
        panel_root = Path(panel_2014_dir)
        formal_root = Path(formal_replay_dir)
        log("load_existing_packages", "started", f"{panel_root}; {formal_root}")
        panel_manifest = _load_json(panel_root / "manifest.json")
        panel_readiness = pd.read_csv(panel_root / "formal_policy_input_readiness.csv")
        panel_blockers = pd.read_csv(panel_root / "data_blockers.csv")
        decision = pd.read_csv(formal_root / "formal_three_pool_decision_panel.csv").fillna("")

        log("build_partial_target_stream", "started", "")
        blocked_stream = _blocked_2014_2021_stream(panel_readiness)
        formal_stream = _formal_2022_stream(decision)
        partial_stream = pd.concat([blocked_stream, formal_stream], ignore_index=True)
        formal_ready_stream = formal_stream[formal_stream["readiness_state"].eq("formal_ready")].copy()

        log("build_readiness_ledgers", "started", "")
        readiness = _data_readiness(panel_manifest, decision, pit_readiness_dir, price_absorption_dir)
        coverage = _signal_coverage(panel_manifest, panel_readiness, decision, formal_stream)
        blocked_periods = _blocked_periods(panel_manifest, panel_blockers, decision)
        source_decisions = _source_decisions()
        handoff = _handoff_to_experiments(formal_ready_stream, blocked_periods)

        log("write_outputs", "started", "")
        readiness.to_csv(output / "formal_long_range_data_readiness.csv", index=False, encoding="utf-8-sig")
        coverage.to_csv(output / "formal_long_range_signal_coverage.csv", index=False, encoding="utf-8-sig")
        formal_ready_stream.to_csv(output / "formal_long_range_target_stream.csv", index=False, encoding="utf-8-sig")
        partial_stream.to_csv(output / "partial_target_stream.csv", index=False, encoding="utf-8-sig")
        blocked_periods.to_csv(output / "blocked_periods.csv", index=False, encoding="utf-8-sig")
        source_decisions.to_csv(output / "proxy_or_formal_source_decision.csv", index=False, encoding="utf-8-sig")
        (output / "handoff_to_experiments.md").write_text(handoff, encoding="utf-8")
        (output / "final_summary_zh.md").write_text(_final_summary(coverage, blocked_periods), encoding="utf-8")

        manifest = {
            "schema_version": 1,
            "task_id": TASK_ID,
            "status": "completed_partial_reconstruction_package",
            "formal_model_target": FORMAL_MODEL_TARGET,
            "formal_model_route": FORMAL_MODEL_ROUTE,
            "formal_execution_risk_control": "no_target_cash_all",
            "no_target_risk_off_policy": "cash_all",
            "execution_basis": "next_day",
            "requested_start_date": "2014-11-01",
            "blocked_signal_start": _first_date(blocked_stream),
            "blocked_signal_end": _last_date(blocked_stream),
            "formal_ready_signal_start": _first_date(formal_ready_stream),
            "formal_ready_signal_end": _last_date(formal_ready_stream),
            "formal_ready_signal_rows": int(len(formal_ready_stream)),
            "partial_target_stream_rows": int(len(partial_stream)),
            "2014_2021_formal_target_stream_ready": False,
            "2022_latest_formal_target_stream_ready": bool(not formal_ready_stream.empty),
            "latest_available_formal_stream_date": _last_date(formal_ready_stream),
            "latest_requested_not_fully_ready": True,
            "formal_model_changed": False,
            "trade_decision_changed": False,
            "active_in_trade_decision": False,
            "uses_forward_return_as_rule": False,
            "pool3_shadow_used": False,
            "proxy_used_as_formal": False,
            "outputs": {
                "data_readiness": "formal_long_range_data_readiness.csv",
                "signal_coverage": "formal_long_range_signal_coverage.csv",
                "formal_target_stream": "formal_long_range_target_stream.csv",
                "partial_target_stream": "partial_target_stream.csv",
                "blocked_periods": "blocked_periods.csv",
                "source_decision": "proxy_or_formal_source_decision.csv",
                "handoff": "handoff_to_experiments.md",
                "summary": "final_summary_zh.md",
            },
        }
        (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        pd.DataFrame([{"status": "completed", "output_dir": str(output.resolve())}]).to_csv(output / "completed.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame(columns=["step", "error"]).to_csv(output / "failed.csv", index=False, encoding="utf-8-sig")
        log("completed", "completed", str(output.resolve()))
        (output / "current_step.txt").write_text("completed\n", encoding="utf-8")
        return output
    except Exception as exc:
        pd.DataFrame([{"step": "run_formal_long_range_signal_reconstruction", "error": str(exc)}]).to_csv(output / "failed.csv", index=False, encoding="utf-8-sig")
        log("failed", "failed", str(exc))
        raise


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _blocked_2014_2021_stream(readiness: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for item in readiness.to_dict(orient="records"):
        signal_date = str(item.get("date") or "")
        rows.append(
            {
                "signal_date": signal_date,
                "execution_date": _next_business_day(signal_date),
                "formal_target": "",
                "formal_target_display": "",
                "target_weights": "{}",
                "no_target_reason": "",
                "risk_off_state": "not_evaluated_blocked",
                "pool1_top_candidate": _clean_text(item.get("pool1_top_candidate")),
                "pool2_confirmation_state": _clean_text(item.get("pool2_vote")),
                "execution_action_basis": "blocked_before_execution",
                "next_day_tradable_flag": False,
                "source_decision": "blocked_2014_2021_signal_panel",
                "readiness_state": _clean_text(item.get("readiness_state")) or "blocked_for_formal_target_stream",
                "blocked_reason": _clean_text(item.get("blocker_reason")),
            }
        )
    return pd.DataFrame(rows)


def _formal_2022_stream(decision: pd.DataFrame) -> pd.DataFrame:
    spec = DecisionVariantSpec(
        "current_formal_no_target_cash_all_long_range",
        "confirmation",
        confirmation_days=1,
        no_formal_target_policy="exit_to_cash",
    )
    target_panel = _build_target_weights(decision, spec)
    rows: list[dict[str, Any]] = []
    for item in target_panel.to_dict(orient="records"):
        signal_date = str(item.get("date") or "")
        weights = _parse_weights(item.get("target_weights"))
        if weights:
            formal_target = next(iter(weights))
            risk_state = "formal_target_active"
            no_target_reason = ""
            display = _display_for_target(formal_target)
        else:
            formal_target = "CASH"
            risk_state = "no_target_cash_all"
            no_target_reason = str(item.get("event_reason") or "no_formal_target_risk_control_cash")
            display = "風險控管空手 / 現金"
        rows.append(
            {
                "signal_date": signal_date,
                "execution_date": _next_business_day(signal_date),
                "formal_target": formal_target,
                "formal_target_display": display,
                "target_weights": json.dumps(weights, ensure_ascii=False),
                "no_target_reason": no_target_reason,
                "risk_off_state": risk_state,
                "pool1_top_candidate": str(item.get("pool1_vote") or ""),
                "pool2_confirmation_state": _pool2_state(item),
                "execution_action_basis": "next_day",
                "next_day_tradable_flag": True,
                "source_decision": "formal_decision_panel_2022_latest_with_no_target_cash_all",
                "readiness_state": "formal_ready",
                "blocked_reason": "",
            }
        )
    return pd.DataFrame(rows)


def _data_readiness(
    panel_manifest: dict[str, Any],
    decision: pd.DataFrame,
    pit_readiness_dir: str | Path,
    price_absorption_dir: str | Path,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "layer": "0050_pcf_daily_monthly_anchor_pit_candidate",
                "status": "ready_as_source_backed_candidate_not_exact",
                "coverage": "2014-11 to 2023-12",
                "source_path": str(pit_readiness_dir),
                "formal_ready": False,
                "proxy_or_candidate": "source_backed_manual_candidate",
                "notes": "可作 PIT candidate monthly anchor；formal_exact=false，不可包裝成 exact index constituent archive。",
            },
            {
                "layer": "pit_universe_price_coverage",
                "status": "price_only_ready_with_caveats",
                "coverage": "2014-11 to 2023-12",
                "source_path": str(price_absorption_dir),
                "formal_ready": False,
                "proxy_or_candidate": "price_only",
                "notes": "76/76 price-only readiness 已收斂；4 檔 unadjusted-only 仍是 total-return replay caveat。",
            },
            {
                "layer": "pool1_pool2_signal_panels_2014_2021",
                "status": "panels_generated_target_stream_blocked",
                "coverage": f"{panel_manifest.get('date_start')} to {panel_manifest.get('date_end')}",
                "source_path": DEFAULT_2014_PANEL_DIR,
                "formal_ready": False,
                "proxy_or_candidate": "partial_signal_panel",
                "notes": "Pool1/Pool2 panel 已產，但正式 attack gate / lifecycle / Pool2 persistence contract 尚未完整。",
            },
            {
                "layer": "formal_decision_panel_2022_latest",
                "status": "formal_ready",
                "coverage": f"{decision['date'].iloc[0]} to {decision['date'].iloc[-1]}" if not decision.empty else "",
                "source_path": DEFAULT_FORMAL_REPLAY_DIR,
                "formal_ready": True,
                "proxy_or_candidate": "formal",
                "notes": "可套 current formal + no-target cash-all 產 next-day target stream。",
            },
            {
                "layer": "no_target_cash_all_execution_rule",
                "status": "formal_active",
                "coverage": "formal-ready signal dates only",
                "source_path": "src/backtest_lab/formal_model_contract.py",
                "formal_ready": True,
                "proxy_or_candidate": "formal",
                "notes": "沒有正式 target 時 next-day 100% 現金；blocked data days 不得被當成 no-target。"
            },
        ]
    )


def _signal_coverage(
    panel_manifest: dict[str, Any],
    panel_readiness: pd.DataFrame,
    decision: pd.DataFrame,
    formal_stream: pd.DataFrame,
) -> pd.DataFrame:
    no_target_days = int(formal_stream["risk_off_state"].eq("no_target_cash_all").sum()) if not formal_stream.empty else 0
    return pd.DataFrame(
        [
            {
                "period": "2014-11-03_to_2021-12-31",
                "start_date": panel_manifest.get("date_start", ""),
                "end_date": panel_manifest.get("date_end", ""),
                "signal_days": int(len(panel_readiness)),
                "pool1_panel_generated": bool(panel_manifest.get("pool1_daily_candidate_ranking_panel_generated")),
                "pool2_panel_generated": bool(panel_manifest.get("pool2_daily_confirmation_panel_generated")),
                "formal_target_stream_ready": False,
                "target_stream_rows": 0,
                "no_target_cash_all_days": 0,
                "coverage_state": "blocked_partial_signal_panels_only",
            },
            {
                "period": "2022-01-03_to_latest_formal_decision_panel",
                "start_date": str(decision["date"].iloc[0]) if not decision.empty else "",
                "end_date": str(decision["date"].iloc[-1]) if not decision.empty else "",
                "signal_days": int(len(decision)),
                "pool1_panel_generated": True,
                "pool2_panel_generated": True,
                "formal_target_stream_ready": bool(not formal_stream.empty),
                "target_stream_rows": int(len(formal_stream)),
                "no_target_cash_all_days": no_target_days,
                "coverage_state": "formal_ready_target_stream",
            },
            {
                "period": "after_latest_formal_decision_panel",
                "start_date": _next_business_day(str(decision["date"].iloc[-1])) if not decision.empty else "",
                "end_date": "latest_available_not_integrated",
                "signal_days": 0,
                "pool1_panel_generated": False,
                "pool2_panel_generated": False,
                "formal_target_stream_ready": False,
                "target_stream_rows": 0,
                "no_target_cash_all_days": 0,
                "coverage_state": "blocked_until_daily_formal_target_stream_ingested",
            },
        ]
    )


def _blocked_periods(panel_manifest: dict[str, Any], blockers: pd.DataFrame, decision: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for blocker in blockers.to_dict(orient="records"):
        rows.append(
            {
                "period_start": panel_manifest.get("date_start", "2014-11-03"),
                "period_end": panel_manifest.get("date_end", "2021-12-31"),
                "blocker": blocker.get("blocker", ""),
                "severity": blocker.get("status", ""),
                "blocks_formal_target_stream": blocker.get("blocks_formal_target_stream", ""),
                "detail": blocker.get("detail", ""),
                "next_owner": blocker.get("next_owner", ""),
            }
        )
    if not decision.empty:
        rows.append(
            {
                "period_start": _next_business_day(str(decision["date"].iloc[-1])),
                "period_end": "latest_available_not_integrated",
                "blocker": "post_latest_formal_decision_panel_ingestion",
                "severity": "missing",
                "blocks_formal_target_stream": True,
                "detail": "Current combined formal decision panel ends before latest report date; latest daily formal report target stream needs to be ingested before long-range latest replay.",
                "next_owner": "Core",
            }
        )
    return pd.DataFrame(rows)


def _source_decisions() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "source_layer": "0050_pcf_daily_monthly_anchor",
                "decision": "usable_as_pit_candidate_not_formal_exact",
                "formal": False,
                "proxy_or_sensitivity": "candidate",
                "notes": "source_backed_manual_candidate; no current snapshot backfill.",
            },
            {
                "source_layer": "00631L_twse_stock_day_backfill",
                "decision": "usable_as_true_price_source_price_only",
                "formal": False,
                "proxy_or_sensitivity": "price_only",
                "notes": "true TWSE rows for 2014/11-2015; adjusted-close policy remains a caveat.",
            },
            {
                "source_layer": "2022_formal_decision_panel",
                "decision": "usable_as_formal_target_stream_input",
                "formal": True,
                "proxy_or_sensitivity": "formal",
                "notes": "Current formal target stream can be reconstructed from existing decision panel with no-target cash-all.",
            },
            {
                "source_layer": "2014_2021_pool1_pool2_signal_panels",
                "decision": "partial_not_formal_target_stream",
                "formal": False,
                "proxy_or_sensitivity": "partial",
                "notes": "Panels exist but formal policy input readiness is zero days.",
            },
        ]
    )


def _handoff_to_experiments(formal_stream: pd.DataFrame, blocked: pd.DataFrame) -> str:
    return "\n".join(
        [
            "【Core 交 Experiments｜Long-range formal target stream partial validation】",
            "",
            "Core 已產出 2014/11～latest long-range signal reconstruction package。",
            "",
            "可直接驗收/回測的部分：",
            f"- 2022+ formal-ready target stream rows: {len(formal_stream)}",
            "- execution_basis=next_day",
            "- no-target rule=no_target_cash_all",
            "",
            "不可包裝成正式長區間績效的部分：",
            "- 2014/11～2021/12 只有 Pool1/Pool2 signal panels，formal target stream 尚 blocked。",
            "- blocked_periods.csv 已列 pool1 attack gate、ticker lifecycle、Pool2 persistence、post-latest ingestion 等 blocker。",
            "",
            "請 Experiments 僅對 formal_ready rows 做 apples-to-apples smoke；不得把 partial/blocked rows 當正式績效。",
        ]
    )


def _final_summary(coverage: pd.DataFrame, blocked: pd.DataFrame) -> str:
    ready = coverage[coverage["coverage_state"].eq("formal_ready_target_stream")]
    blocked_main = coverage[coverage["coverage_state"].eq("blocked_partial_signal_panels_only")]
    ready_row = ready.iloc[0].to_dict() if not ready.empty else {}
    blocked_row = blocked_main.iloc[0].to_dict() if not blocked_main.empty else {}
    return "\n".join(
        [
            "# Formal long-range signal reconstruction summary",
            "",
            "結論：本棒產出 partial reconstruction package；2014/11～2021/12 尚不能正式回測，2022+ 可產 formal-ready target stream。",
            "",
            "## 2014/11～2021/12",
            "",
            f"- signal days: {blocked_row.get('signal_days', 0)}",
            "- Pool1 / Pool2 signal panels 已存在。",
            "- formal target stream: blocked。",
            "- 主要 blocker：Pool1 date-aware formal attack gate、上市前 ticker lifecycle、Pool2 10 日 persistence gate。",
            "",
            "## 2022+",
            "",
            f"- formal-ready rows: {ready_row.get('target_stream_rows', 0)}",
            f"- signal range: {ready_row.get('start_date', '')} 到 {ready_row.get('end_date', '')}",
            f"- no-target cash-all days: {ready_row.get('no_target_cash_all_days', 0)}",
            "",
            "## Boundary",
            "",
            "- no-target cash-all 只套用在 formal-ready signal dates；資料 blocked day 不得被當成 no-target。",
            "- 0050 PCF/Daily monthly anchor 是 source-backed PIT candidate，formal_exact=false。",
            "- 4 檔 unadjusted-only 價格 source 仍是 total-return replay caveat。",
            f"- blocker rows: {len(blocked)}",
            "",
        ]
    )


def _parse_weights(value: object) -> dict[str, float]:
    if not value:
        return {}
    try:
        payload = json.loads(str(value))
    except json.JSONDecodeError:
        return {}
    return {str(key): float(weight) for key, weight in payload.items()}


def _clean_text(value: object) -> str:
    if value is None:
        return ""
    if pd.isna(value):
        return ""
    return str(value)


def _pool2_state(item: dict[str, Any]) -> str:
    reason = str(item.get("event_reason") or "")
    if "confirmation" in reason and "not_met" in reason:
        return "pool2_disagreement_confirmation_not_met"
    if bool(item.get("pool2_disagreement")):
        return "pool2_disagreement_but_confirmed_or_nonblocking"
    return "pool2_aligned_or_not_required"


def _display_for_target(ticker: str) -> str:
    labels = {
        "00631L.TW": "0050正二(00631L)",
        "0050.TW": "元大台灣50(0050)",
        "2454.TW": "聯發科(2454)",
        "2330.TW": "台積電(2330)",
        "2308.TW": "台達電(2308)",
        "2317.TW": "鴻海(2317)",
        "2382.TW": "廣達(2382)",
        "3231.TW": "緯創(3231)",
        "6669.TW": "緯穎(6669)",
    }
    return labels.get(ticker, ticker)


def _next_business_day(value: str) -> str:
    if not value:
        return ""
    try:
        return (pd.Timestamp(value) + pd.offsets.BDay(1)).strftime("%Y-%m-%d")
    except Exception:
        return ""


def _first_date(frame: pd.DataFrame) -> str:
    return str(frame["signal_date"].iloc[0]) if not frame.empty and "signal_date" in frame.columns else ""


def _last_date(frame: pd.DataFrame) -> str:
    return str(frame["signal_date"].iloc[-1]) if not frame.empty and "signal_date" in frame.columns else ""


def main() -> None:
    parser = argparse.ArgumentParser(description="Build formal long-range signal reconstruction readiness package.")
    parser.add_argument("--panel-2014-dir", default=DEFAULT_2014_PANEL_DIR)
    parser.add_argument("--formal-replay-dir", default=DEFAULT_FORMAL_REPLAY_DIR)
    parser.add_argument("--pit-readiness-dir", default=DEFAULT_PIT_READINESS_DIR)
    parser.add_argument("--price-absorption-dir", default=DEFAULT_PRICE_ABSORPTION_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    run_formal_long_range_signal_reconstruction(
        panel_2014_dir=args.panel_2014_dir,
        formal_replay_dir=args.formal_replay_dir,
        pit_readiness_dir=args.pit_readiness_dir,
        price_absorption_dir=args.price_absorption_dir,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
