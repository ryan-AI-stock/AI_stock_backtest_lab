from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from backtest_lab.formal_model_contract import FORMAL_MODEL_ROUTE, FORMAL_MODEL_TARGET


TASK_ID = "TASK-BACKTEST-CORE-POOL1-DATE-AWARE-FORMAL-ATTACK-GATE-CONTRACT-20260702"
DEFAULT_PANEL_DIR = "outputs/current_formal_pool1_pool2_signal_panels_201411_202112_20260630"
DEFAULT_OUTPUT_DIR = "outputs/pool1_date_aware_formal_attack_gate_contract_201411_202112_20260702"


def run_pool1_date_aware_attack_gate_contract(
    *,
    panel_dir: str | Path = DEFAULT_PANEL_DIR,
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
        panel_root = Path(panel_dir)
        log("load_2014_2021_pool1_panel", "started", str(panel_root))
        manifest = _load_json(panel_root / "manifest.json")
        pool1_panel = pd.read_csv(panel_root / "pool1_daily_candidate_ranking_panel.csv").fillna("")
        readiness = pd.read_csv(panel_root / "formal_policy_input_readiness.csv").fillna("")

        log("build_contract_ledgers", "started", "")
        contract = _pool1_attack_gate_contract()
        candidate_availability = _candidate_availability(pool1_panel)
        coverage = _attack_gate_coverage(readiness, pool1_panel)
        blocked_rows = _blocked_signal_rows(coverage)
        blockers = _blocker_by_field()
        source_decision = _source_decisions()

        log("write_outputs", "started", "")
        contract.to_csv(output / "pool1_attack_gate_contract.csv", index=False, encoding="utf-8-sig")
        candidate_availability.to_csv(
            output / "pool1_date_aware_candidate_availability.csv",
            index=False,
            encoding="utf-8-sig",
        )
        coverage.to_csv(output / "pool1_attack_gate_coverage_201411_202112.csv", index=False, encoding="utf-8-sig")
        blocked_rows.to_csv(output / "blocked_signal_rows.csv", index=False, encoding="utf-8-sig")
        blockers.to_csv(output / "blocker_by_field.csv", index=False, encoding="utf-8-sig")
        source_decision.to_csv(output / "proxy_or_formal_source_decision.csv", index=False, encoding="utf-8-sig")
        (output / "next_step_handoff.md").write_text(_next_step_handoff(), encoding="utf-8")
        (output / "final_summary_zh.md").write_text(
            _final_summary(manifest, coverage, candidate_availability, blockers),
            encoding="utf-8",
        )

        output_manifest = {
            "schema_version": 1,
            "task_id": TASK_ID,
            "status": "completed_blocked_contract_package",
            "formal_model_target": FORMAL_MODEL_TARGET,
            "formal_model_route": FORMAL_MODEL_ROUTE,
            "source_panel_dir": str(panel_root),
            "date_start": _manifest_value(manifest, "date_start", _first_date(coverage)),
            "date_end": _manifest_value(manifest, "date_end", _last_date(coverage)),
            "pool1_ranking_panel_rows": int(len(pool1_panel)),
            "pool1_attack_gate_formal_ready": False,
            "formal_vote_ready_days": 0,
            "blocked_signal_rows": int(len(blocked_rows)),
            "candidate_count": int(candidate_availability["candidate_ticker"].nunique()) if not candidate_availability.empty else 0,
            "ranking_reconstructed": True,
            "formal_attack_gate_reconstructed": False,
            "no_target_cash_all_applied": False,
            "no_target_cash_all_reason": "not_applied_until_pool1_attack_gate_formal_ready",
            "proxy_used_as_formal": False,
            "uses_forward_return_as_rule": False,
            "formal_model_changed": False,
            "trade_decision_changed": False,
            "active_in_trade_decision": False,
            "blocking_fields": sorted(blockers.loc[blockers["formal_ready"].eq(False), "field_name"].astype(str).tolist()),
            "outputs": {
                "contract": "pool1_attack_gate_contract.csv",
                "candidate_availability": "pool1_date_aware_candidate_availability.csv",
                "coverage": "pool1_attack_gate_coverage_201411_202112.csv",
                "blocked_signal_rows": "blocked_signal_rows.csv",
                "blockers": "blocker_by_field.csv",
                "source_decision": "proxy_or_formal_source_decision.csv",
                "handoff": "next_step_handoff.md",
                "summary": "final_summary_zh.md",
            },
        }
        (output / "manifest.json").write_text(json.dumps(output_manifest, ensure_ascii=False, indent=2), encoding="utf-8")
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
        pd.DataFrame([{"step": "run_pool1_date_aware_attack_gate_contract", "error": str(exc)}]).to_csv(
            output / "failed.csv",
            index=False,
            encoding="utf-8-sig",
        )
        log("failed", "failed", str(exc))
        raise


def _pool1_attack_gate_contract() -> pd.DataFrame:
    rows = [
        {
            "field_name": "candidate_ticker",
            "formal_2022_source": "pool1 ranking row / FrozenStrategySignal.ranking",
            "formal_2022_condition": "date-aware tradable candidate identifier",
            "2014_2021_status": "available_in_partial_panel",
            "formal_ready": True,
            "blocker": "",
        },
        {
            "field_name": "candidate_name",
            "formal_2022_source": "labels / pool display mapping",
            "formal_2022_condition": "user-facing display only",
            "2014_2021_status": "available_in_partial_panel",
            "formal_ready": True,
            "blocker": "",
        },
        {
            "field_name": "candidate_rank_score",
            "formal_2022_source": "relative_strength_scores(prices_by_ticker, signal_date)",
            "formal_2022_condition": "used to rank attack candidates against fallback benchmark",
            "2014_2021_status": "reconstructed_price_only",
            "formal_ready": True,
            "blocker": "",
        },
        {
            "field_name": "base_pool_passed",
            "formal_2022_source": "StockPoolObservationCandidate.passed",
            "formal_2022_condition": "candidate must pass base Pool1 ranking/gate layer",
            "2014_2021_status": "available_as_partial_panel_passed",
            "formal_ready": True,
            "blocker": "",
        },
        {
            "field_name": "candidate_listed_and_tradable_on_signal_date",
            "formal_2022_source": "date-aware price data plus lifecycle policy",
            "formal_2022_condition": "not-yet-listed or untradable tickers cannot enter daily candidate set",
            "2014_2021_status": "partial_from_price_panel_only",
            "formal_ready": False,
            "blocker": "missing_pool1_not_yet_listed_ticker_lifecycle_contract",
        },
        {
            "field_name": "fallback_benchmark_score",
            "formal_2022_source": "relative_strength_scores fallback ticker 0050.TW",
            "formal_2022_condition": "top attack candidate must beat fallback by configured margin",
            "2014_2021_status": "available_price_only_but_not_bound_to_formal_state_machine",
            "formal_ready": False,
            "blocker": "missing_formal_attack_gate_state_replay",
        },
        {
            "field_name": "attack_gate_active",
            "formal_2022_source": "simulate_regime_mode_switch equity_curve.attack_gate_active",
            "formal_2022_condition": "formal stock target only opens after gate state turns active",
            "2014_2021_status": "missing",
            "formal_ready": False,
            "blocker": "missing_date_aware_pool1_attack_gate_state",
        },
        {
            "field_name": "attack_gate_ever_activated",
            "formal_2022_source": "simulate_regime_mode_switch equity_curve.attack_gate_ever_activated",
            "formal_2022_condition": "affects re-entry and market risk-off interaction",
            "2014_2021_status": "missing",
            "formal_ready": False,
            "blocker": "missing_stateful_gate_history_replay",
        },
        {
            "field_name": "risk_off_active",
            "formal_2022_source": "simulate_regime_mode_switch equity_curve.risk_off_active",
            "formal_2022_condition": "market risk-off state changes formal mode after gate interaction",
            "2014_2021_status": "missing",
            "formal_ready": False,
            "blocker": "missing_regime_mode_switch_state_replay",
        },
        {
            "field_name": "target_is_actionable",
            "formal_2022_source": "FrozenStrategySignal.target_is_actionable",
            "formal_2022_condition": "only actionable target can become formal Pool1 target",
            "2014_2021_status": "missing",
            "formal_ready": False,
            "blocker": "missing_frozen_strategy_signal_replay",
        },
        {
            "field_name": "model_target_status",
            "formal_2022_source": "FrozenStrategySignal.model_target_status",
            "formal_2022_condition": "distinguishes formal target from risk-control/no target state",
            "2014_2021_status": "missing",
            "formal_ready": False,
            "blocker": "missing_frozen_strategy_signal_replay",
        },
        {
            "field_name": "formal_selection_layer",
            "formal_2022_source": "stock_pool_observation._candidate_gate_evaluation",
            "formal_2022_condition": "stock formal candidate requires passed candidate and active attack gate",
            "2014_2021_status": "cannot_determine_without_attack_gate_active",
            "formal_ready": False,
            "blocker": "missing_formal_selection_layer_inputs",
        },
    ]
    return pd.DataFrame(rows)


def _candidate_availability(panel: pd.DataFrame) -> pd.DataFrame:
    if panel.empty:
        return pd.DataFrame(
            columns=[
                "candidate_ticker",
                "candidate_name",
                "first_panel_date",
                "last_panel_date",
                "panel_rows",
                "rank1_days",
                "passed_rows",
                "price_only_used_rows",
                "adjusted_close_available_rows",
                "availability_source",
                "formal_lifecycle_contract_ready",
                "date_aware_availability_status",
            ]
        )
    frame = panel.copy()
    frame["rank_num"] = pd.to_numeric(frame.get("rank", ""), errors="coerce")
    frame["passed_bool"] = frame.get("passed", "").map(_bool_like)
    frame["price_only_bool"] = frame.get("price_only_used", "").map(_bool_like)
    frame["adjusted_bool"] = frame.get("adjusted_close_available", "").map(_bool_like)
    rows: list[dict[str, Any]] = []
    for ticker, group in frame.groupby("candidate_ticker", dropna=False):
        if not str(ticker):
            continue
        rows.append(
            {
                "candidate_ticker": str(ticker),
                "candidate_name": _first_non_empty(group.get("candidate_name", pd.Series(dtype=str))),
                "first_panel_date": str(group["date"].min()),
                "last_panel_date": str(group["date"].max()),
                "panel_rows": int(len(group)),
                "rank1_days": int(group["rank_num"].eq(1).sum()),
                "passed_rows": int(group["passed_bool"].sum()),
                "price_only_used_rows": int(group["price_only_bool"].sum()),
                "adjusted_close_available_rows": int(group["adjusted_bool"].sum()),
                "availability_source": "partial_pool1_panel_price_presence",
                "formal_lifecycle_contract_ready": False,
                "date_aware_availability_status": "partial_price_seen_not_formal_lifecycle",
            }
        )
    return pd.DataFrame(rows).sort_values(["candidate_ticker"]).reset_index(drop=True)


def _attack_gate_coverage(readiness: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    top_by_date: dict[str, dict[str, Any]] = {}
    if not panel.empty:
        frame = panel.copy()
        frame["rank_num"] = pd.to_numeric(frame.get("rank", ""), errors="coerce")
        top = frame.sort_values(["date", "rank_num", "candidate_ticker"]).groupby("date", as_index=False).first()
        for row in top.to_dict(orient="records"):
            date = str(row.get("date") or "")
            top_by_date[date] = row
    candidate_counts = panel.groupby("date").size().to_dict() if not panel.empty else {}

    rows: list[dict[str, Any]] = []
    for item in readiness.to_dict(orient="records"):
        date = str(item.get("date") or "")
        top = top_by_date.get(date, {})
        rows.append(
            {
                "date": date,
                "pool1_candidate_rows": int(candidate_counts.get(date, 0)),
                "pool1_top_candidate_ticker": str(top.get("candidate_ticker") or ""),
                "pool1_top_candidate_name": str(top.get("candidate_name") or ""),
                "pool1_top_candidate_score": _clean_text(top.get("score")),
                "ranking_available": bool(candidate_counts.get(date, 0)),
                "base_pool_passed_available": "passed" in panel.columns,
                "date_aware_candidate_availability_ready": False,
                "formal_attack_gate_ready": False,
                "formal_vote_ready": False,
                "formal_target_stream_ready": False,
                "blocked_reason": _clean_text(item.get("blocker_reason"))
                or "missing_date_aware_pool1_attack_gate_contract",
                "no_target_cash_all_applied": False,
            }
        )
    return pd.DataFrame(rows)


def _blocked_signal_rows(coverage: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for item in coverage.to_dict(orient="records"):
        rows.append(
            {
                "signal_date": item["date"],
                "pool1_top_candidate": item["pool1_top_candidate_ticker"],
                "pool1_top_candidate_name": item["pool1_top_candidate_name"],
                "pool1_top_candidate_score": item["pool1_top_candidate_score"],
                "formal_attack_gate_ready": False,
                "formal_vote_ready": False,
                "formal_target": "",
                "target_weights": "{}",
                "risk_off_state": "not_evaluated_blocked",
                "no_target_cash_all_applied": False,
                "blocked_reason": item["blocked_reason"],
            }
        )
    return pd.DataFrame(rows)


def _blocker_by_field() -> pd.DataFrame:
    contract = _pool1_attack_gate_contract()
    rows = []
    for item in contract.to_dict(orient="records"):
        if bool(item["formal_ready"]):
            continue
        rows.append(
            {
                "field_name": item["field_name"],
                "blocker": item["blocker"],
                "severity": "blocking_formal_target_stream",
                "required_for": "Pool1 formal attack gate / formal target stream",
                "can_core_reconstruct_now": False,
                "next_action": _next_action_for_blocker(str(item["blocker"])),
                "formal_ready": False,
            }
        )
    return pd.DataFrame(rows)


def _source_decisions() -> pd.DataFrame:
    rows = [
        {
            "source_layer": "pool1_daily_candidate_ranking_panel",
            "source_path": DEFAULT_PANEL_DIR + "/pool1_daily_candidate_ranking_panel.csv",
            "status": "usable_partial_input",
            "formal_or_proxy": "partial_formal_input_not_target_stream",
            "decision": "可用來看每日候選排序，但不能單獨決定 formal target。",
        },
        {
            "source_layer": "pool1_attack_gate_state",
            "source_path": "simulate_regime_mode_switch / FrozenStrategySignal",
            "status": "missing_for_2014_2021",
            "formal_or_proxy": "missing",
            "decision": "缺 attack_gate_active / target_is_actionable 狀態重放；不可用排名 proxy 替代。",
        },
        {
            "source_layer": "candidate_lifecycle",
            "source_path": "price panel first/last date only",
            "status": "partial",
            "formal_or_proxy": "partial_proxy",
            "decision": "價格出現日可排除尚無價格資料標的，但仍需正式 lifecycle contract 才能 formal-ready。",
        },
        {
            "source_layer": "no_target_cash_all",
            "source_path": "formal execution risk-control rule",
            "status": "not_applied_in_this_step",
            "formal_or_proxy": "formal_rule_waiting_for_formal_signal",
            "decision": "只有 Pool1/Pool2 formal target stream 完成後才可套用；2014-2021 blocked rows 不得硬套空手。",
        },
    ]
    return pd.DataFrame(rows)


def _next_step_handoff() -> str:
    return "\n".join(
        [
            "# Pool1 date-aware formal attack gate handoff",
            "",
            "## 結論",
            "2014/11～2021 目前只能重建 Pool1 候選排序，不能 formal-ready 重建 Pool1 攻擊閘門。",
            "",
            "## 下一步",
            "1. Core 需補 `pool1_not_yet_listed_ticker_lifecycle_contract`，把候選股上市日、下市/併購/代碼停用與可交易狀態做成正式表。",
            "2. Core 需補 `formal_attack_gate_state_replay`，使用現行 `simulate_regime_mode_switch` 所需欄位重放 `attack_gate_active`、`attack_gate_ever_activated`、`risk_off_active`。",
            "3. 完成上述後，才能合成 2014～2021 Pool1 formal vote，並交下一棒處理 Pool2 persistence full reconstruction。",
            "",
            "## 邊界",
            "- 不得把 `pool1_daily_candidate_ranking_panel.csv` 的第一名直接視為 formal target。",
            "- 不得在 blocked 2014～2021 rows 上套用 no-target cash-all。",
            "- `formal_model_changed=false`、`trade_decision_changed=false`。",
        ]
    ) + "\n"


def _final_summary(
    manifest: dict[str, Any],
    coverage: pd.DataFrame,
    candidate_availability: pd.DataFrame,
    blockers: pd.DataFrame,
) -> str:
    blocked_days = int(len(coverage))
    candidate_count = int(candidate_availability["candidate_ticker"].nunique()) if not candidate_availability.empty else 0
    return "\n".join(
        [
            "# Pool1 date-aware formal attack gate contract",
            "",
            "## 判定",
            "2014/11～2021 的 Pool1 排名 panel 已存在，但 formal attack gate 尚未 formal-ready。",
            "原因是現行 2022+ 正式 Pool1 target 不是單純候選排名，而是 `simulate_regime_mode_switch` 的狀態機輸出；需要 `attack_gate_active`、`attack_gate_ever_activated`、`risk_off_active`、`target_is_actionable`、`model_target_status` 等欄位。",
            "",
            "## 本批輸出",
            f"- 來源區間：{manifest.get('date_start')}～{manifest.get('date_end')}",
            f"- blocked signal rows：{blocked_days}",
            f"- Pool1 候選 ticker 數：{candidate_count}",
            f"- blocking fields：{len(blockers)}",
            "- no-target cash-all：本步未套用，因為 2014～2021 formal signal 尚未完成。",
            "",
            "## 結論",
            "不能把 2014～2021 partial ranking 包裝成正式 target stream。下一個最小任務是補候選 lifecycle contract 與 formal attack gate state replay。",
            "",
            "## 邊界",
            "- formal_model_changed=false",
            "- trade_decision_changed=false",
            "- proxy_used_as_formal=false",
        ]
    ) + "\n"


def _next_action_for_blocker(blocker: str) -> str:
    if "lifecycle" in blocker:
        return "建立候選 ticker 上市/下市/可交易日期 contract，確認當日不可交易標的不能進 formal candidate set。"
    if "state" in blocker or "replay" in blocker:
        return "用現行 frozen strategy / regime mode switch 欄位重放 2014～2021 attack gate state。"
    if "selection" in blocker:
        return "在 attack gate state 與 lifecycle 完成後重建 formal selection layer。"
    return "補齊欄位來源與正式驗收測試。"


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _manifest_value(manifest: dict[str, Any], key: str, fallback: str) -> str:
    value = manifest.get(key)
    return str(value) if value else fallback


def _bool_like(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _first_non_empty(series: pd.Series) -> str:
    for value in series.astype(str).tolist():
        if value:
            return value
    return ""


def _clean_text(value: Any) -> str:
    if pd.isna(value):
        return ""
    return str(value)


def _first_date(frame: pd.DataFrame) -> str:
    if frame.empty:
        return ""
    column = "date" if "date" in frame.columns else "signal_date"
    return str(frame[column].iloc[0])


def _last_date(frame: pd.DataFrame) -> str:
    if frame.empty:
        return ""
    column = "date" if "date" in frame.columns else "signal_date"
    return str(frame[column].iloc[-1])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build Pool1 date-aware formal attack gate contract package.")
    parser.add_argument("--panel-dir", default=DEFAULT_PANEL_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args(argv)
    output = run_pool1_date_aware_attack_gate_contract(panel_dir=args.panel_dir, output_dir=args.output_dir)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
