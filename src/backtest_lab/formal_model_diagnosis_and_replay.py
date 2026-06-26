from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from backtest_lab.data import load_price_csv
from backtest_lab.formal_model_contract import FORMAL_MODEL_ROUTE, FORMAL_MODEL_TARGET, get_formal_model_contract


TASK_ID = "TASK-BACKTEST-CORE-FORMAL-MODEL-DIAGNOSIS-AND-REPLAY-001"
DEFAULT_ABSORPTION_DIR = "outputs/formal_absorb_pool1_pool2_combined_cap40_confirmation1_20260626"
DEFAULT_FORMAL_CANDIDATE_DIR = "outputs/pool1_pool2_veto_cap_downweight_panels_20260626"
DEFAULT_COMPARISON_DIR = "outputs/three_pool_vs_pool1_comparison_panels_20260626"
DEFAULT_0050X2_LABEL_DIR = "outputs/pool1_pool2_0050x2_opportunity_label_20260626"
DEFAULT_COOLDOWN_LABEL_DIR = "outputs/execution_cooldown3_report_label_20260626"
DEFAULT_PRICE_CACHE_DIR = "backtest_cache/stock_pool_triad_v1_corrected"
DEFAULT_OUTPUT_DIR = "outputs/formal_model_diagnosis_and_replay_20260626"
FORMAL_SOURCE_VARIANT = "combined_cap40_confirmation1"
FORMAL_OUTPUT_VARIANT = "combined_cap40_confirmation1_base"
BENCHMARKS = {
    "0050": {"ticker": "0050.TW", "kind": "actual_etf"},
    "0050正二_00631L": {"ticker": "00631L.TW", "kind": "actual_leveraged_etf"},
    "0050x2_synthetic": {"ticker": "0050.TW", "kind": "synthetic_2x_daily_0050"},
}


def run_formal_model_diagnosis_and_replay(
    *,
    absorption_dir: str | Path = DEFAULT_ABSORPTION_DIR,
    formal_candidate_dir: str | Path = DEFAULT_FORMAL_CANDIDATE_DIR,
    comparison_dir: str | Path = DEFAULT_COMPARISON_DIR,
    opportunity_label_dir: str | Path = DEFAULT_0050X2_LABEL_DIR,
    cooldown_label_dir: str | Path = DEFAULT_COOLDOWN_LABEL_DIR,
    price_cache_dir: str | Path = DEFAULT_PRICE_CACHE_DIR,
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
        absorption = Path(absorption_dir)
        candidate = Path(formal_candidate_dir)
        comparison = Path(comparison_dir)
        opportunity = Path(opportunity_label_dir)
        cooldown = Path(cooldown_label_dir)
        cache = Path(price_cache_dir)

        log("load_inputs", "started", f"candidate={candidate}")
        absorption_manifest = json.loads((absorption / "manifest.json").read_text(encoding="utf-8"))
        candidate_perf = pd.read_csv(candidate / "period_performance_by_variant.csv").fillna("")
        candidate_daily = pd.read_csv(candidate / "daily_equity_by_variant.csv").fillna("")
        candidate_trades = pd.read_csv(candidate / "trade_ledger_by_variant.csv").fillna("")
        prior_perf = pd.read_csv(comparison / "period_performance_by_variant.csv").fillna("")
        opportunity_manifest = json.loads((opportunity / "manifest.json").read_text(encoding="utf-8"))
        cooldown_manifest = json.loads((cooldown / "manifest.json").read_text(encoding="utf-8"))

        log("validate_boundaries", "started", "")
        _validate_inputs(absorption_manifest, candidate_perf, candidate_daily, opportunity_manifest, cooldown_manifest)

        log("build_core_tables", "started", "")
        formal_period_perf = _formal_period_performance(candidate_perf)
        formal_daily = _formal_daily(candidate_daily)
        formal_trades = _formal_trades(candidate_trades)
        identity = _formal_identity(absorption_manifest)
        cost_turnover = _trade_cost_turnover_summary(formal_daily)
        prior_models = _formal_vs_prior_models(formal_period_perf, prior_perf, cost_turnover)
        prices = _load_benchmark_prices(cache)
        benchmarks = _formal_vs_benchmarks(formal_period_perf, prices)
        drawdown_risk = _drawdown_and_risk_summary(formal_daily)
        holding = _holding_transition_summary(formal_daily)
        diagnostics = _report_only_diagnostics_inventory(opportunity, cooldown)
        caveats = _residual_caveat_matrix()
        manifest = _manifest(absorption_manifest, formal_period_perf, diagnostics, caveats, output)

        log("write_outputs", "started", "")
        identity.to_csv(output / "formal_model_identity.csv", index=False, encoding="utf-8-sig")
        benchmarks.to_csv(output / "formal_vs_benchmarks_performance.csv", index=False, encoding="utf-8-sig")
        prior_models.to_csv(output / "formal_vs_prior_models_performance.csv", index=False, encoding="utf-8-sig")
        formal_period_perf.to_csv(output / "period_performance.csv", index=False, encoding="utf-8-sig")
        drawdown_risk.to_csv(output / "drawdown_and_risk_summary.csv", index=False, encoding="utf-8-sig")
        cost_turnover.to_csv(output / "trade_cost_turnover_summary.csv", index=False, encoding="utf-8-sig")
        holding.to_csv(output / "holding_transition_summary.csv", index=False, encoding="utf-8-sig")
        diagnostics.to_csv(output / "report_only_diagnostics_inventory.csv", index=False, encoding="utf-8-sig")
        caveats.to_csv(output / "residual_caveat_matrix.csv", index=False, encoding="utf-8-sig")
        (output / "formal_model_diagnosis_summary_zh.md").write_text(
            _diagnosis_summary_markdown(formal_period_perf, prior_models, benchmarks, caveats),
            encoding="utf-8",
        )
        (output / "model_boundary_for_report_zh.md").write_text(_model_boundary_markdown(), encoding="utf-8")
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
        pd.DataFrame([{"step": "run_formal_model_diagnosis_and_replay", "error": str(exc)}]).to_csv(
            output / "failed.csv",
            index=False,
            encoding="utf-8-sig",
        )
        log("failed", "failed", str(exc))
        raise


def _validate_inputs(
    absorption_manifest: dict[str, Any],
    candidate_perf: pd.DataFrame,
    candidate_daily: pd.DataFrame,
    opportunity_manifest: dict[str, Any],
    cooldown_manifest: dict[str, Any],
) -> None:
    if absorption_manifest.get("formal_model_target") != FORMAL_MODEL_TARGET:
        raise ValueError("absorption manifest formal_model_target mismatch")
    if not _truthy(absorption_manifest.get("formal_absorption_ready")):
        raise ValueError("absorption manifest is not formal-ready")
    for frame_name, frame, required in [
        ("candidate_perf", candidate_perf, {"variant", "period_label", "return_pct", "max_drawdown_pct"}),
        ("candidate_daily", candidate_daily, {"variant", "date", "position_ticker", "equity", "drawdown", "turnover", "transaction_cost", "action"}),
    ]:
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"{frame_name} missing columns: {missing}")
    if FORMAL_SOURCE_VARIANT not in set(candidate_perf["variant"].astype(str)):
        raise ValueError(f"missing {FORMAL_SOURCE_VARIANT} period performance")
    if FORMAL_SOURCE_VARIANT not in set(candidate_daily["variant"].astype(str)):
        raise ValueError(f"missing {FORMAL_SOURCE_VARIANT} daily equity")
    for name, manifest, flag in [
        ("0050x2 opportunity label", opportunity_manifest, "opportunity_cost_label_active_in_trade_decision"),
        ("cooldown3 execution label", cooldown_manifest, "execution_label_active_in_trade_decision"),
    ]:
        if _truthy(manifest.get(flag)):
            raise ValueError(f"{name} is active in trade decision")


def _formal_period_performance(candidate_perf: pd.DataFrame) -> pd.DataFrame:
    frame = candidate_perf[candidate_perf["variant"].astype(str).eq(FORMAL_SOURCE_VARIANT)].copy()
    frame["variant"] = FORMAL_OUTPUT_VARIANT
    frame["formal_model_route"] = FORMAL_MODEL_ROUTE
    frame["active_in_trade_decision"] = True
    frame["performance_scope"] = "formal_same_day_replay"
    return frame.reset_index(drop=True)


def _formal_daily(candidate_daily: pd.DataFrame) -> pd.DataFrame:
    frame = candidate_daily[candidate_daily["variant"].astype(str).eq(FORMAL_SOURCE_VARIANT)].copy()
    frame["variant"] = FORMAL_OUTPUT_VARIANT
    frame["date_ts"] = pd.to_datetime(frame["date"])
    return frame.sort_values("date_ts").reset_index(drop=True)


def _formal_trades(candidate_trades: pd.DataFrame) -> pd.DataFrame:
    if candidate_trades.empty or "variant" not in candidate_trades.columns:
        return candidate_trades.copy()
    frame = candidate_trades[candidate_trades["variant"].astype(str).eq(FORMAL_SOURCE_VARIANT)].copy()
    frame["variant"] = FORMAL_OUTPUT_VARIANT
    return frame.reset_index(drop=True)


def _formal_identity(absorption_manifest: dict[str, Any]) -> pd.DataFrame:
    contract = get_formal_model_contract()
    return pd.DataFrame(
        [
            {
                "formal_model_target": FORMAL_MODEL_TARGET,
                "formal_model_route": FORMAL_MODEL_ROUTE,
                "formal_model_effective_date": contract.get("formal_model_effective_date", ""),
                "active_components": "Pool1 primary attack selector; PIT-ready Pool2 confirmation/risk layer; 00631L cap40 cash residual",
                "abandoned_formal_route": "current_formal_three_pool_baseline",
                "pool3_role": "shadow_or_diagnostic_only",
                "formal_absorption_commit_evidence": absorption_manifest.get("task_id", ""),
                "formal_absorption_ready": True,
                "active_in_trade_decision": True,
            }
        ]
    )


def _formal_vs_prior_models(formal_perf: pd.DataFrame, prior_perf: pd.DataFrame, cost_turnover: pd.DataFrame) -> pd.DataFrame:
    formal = formal_perf.copy()
    if not cost_turnover.empty and "period_label" in cost_turnover.columns:
        turnover_map = cost_turnover.set_index("period_label")["total_turnover"].to_dict()
        formal["total_turnover"] = formal["period_label"].map(turnover_map)
    formal["comparison_group"] = "new_formal_model"
    prior = prior_perf.copy()
    prior["comparison_group"] = prior["variant"].map(
        {
            "current_formal_three_pool_baseline": "prior_formal_model",
            "pool1_only_formal_replay": "pool1_only_reference",
            "three_pool_with_report_only_labels": "report_only_same_as_prior",
            "three_pool_with_execution_shadow_diagnostics": "execution_shadow_same_as_prior",
        }
    ).fillna("prior_reference")
    keep = [
        "comparison_group",
        "variant",
        "period_label",
        "status",
        "start_date",
        "end_date",
        "start_equity",
        "final_equity",
        "return_pct",
        "max_drawdown_pct",
        "trade_days",
        "total_transaction_cost",
        "total_turnover",
    ]
    for col in keep:
        if col not in formal.columns:
            formal[col] = ""
        if col not in prior.columns:
            prior[col] = ""
    return pd.concat([formal[keep], prior[keep]], ignore_index=True)


def _load_benchmark_prices(cache_dir: Path) -> dict[str, pd.Series]:
    prices: dict[str, pd.Series] = {}
    for config in BENCHMARKS.values():
        ticker = config["ticker"]
        if ticker in prices:
            continue
        path = cache_dir / f"{ticker.replace('.', '_')}.csv"
        if path.exists():
            prices[ticker] = pd.to_numeric(load_price_csv(path)["adj_close"], errors="coerce").dropna()
    return prices


def _formal_vs_benchmarks(formal_perf: pd.DataFrame, prices: dict[str, pd.Series]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for item in formal_perf.to_dict(orient="records"):
        start = _text(item.get("start_date"))
        end = _text(item.get("end_date"))
        period = _text(item.get("period_label"))
        formal_ret = _float(item.get("return_pct"))
        for label, config in BENCHMARKS.items():
            ticker = str(config["ticker"])
            kind = str(config["kind"])
            bench = _benchmark_perf(prices.get(ticker), start, end, kind)
            ret = bench.get("return_pct")
            rows.append(
                {
                    "period_label": period,
                    "formal_model": FORMAL_OUTPUT_VARIANT,
                    "benchmark": label,
                    "benchmark_ticker": ticker,
                    "benchmark_kind": kind,
                    "start_date": bench.get("start_date", start),
                    "end_date": bench.get("end_date", end),
                    "formal_return_pct": formal_ret,
                    "benchmark_return_pct": ret,
                    "formal_excess_return_pp": _round(None if ret is None or formal_ret is None else formal_ret - ret),
                    "benchmark_max_drawdown_pct": bench.get("max_drawdown_pct", ""),
                    "formal_max_drawdown_pct": item.get("max_drawdown_pct", ""),
                    "data_complete": bench.get("data_complete", False),
                }
            )
    return pd.DataFrame(rows)


def _benchmark_perf(series: pd.Series | None, start: str, end: str, kind: str) -> dict[str, Any]:
    if series is None or series.empty or not start or not end:
        return {"data_complete": False}
    subset = series[(series.index >= pd.Timestamp(start)) & (series.index <= pd.Timestamp(end))].dropna()
    if len(subset) < 2:
        return {"data_complete": False}
    if kind == "synthetic_2x_daily_0050":
        daily = subset.pct_change().fillna(0.0) * 2
        equity = (1.0 + daily).cumprod()
    else:
        equity = subset / float(subset.iloc[0])
    running_max = equity.cummax()
    drawdown = equity / running_max - 1
    return {
        "start_date": subset.index[0].strftime("%Y-%m-%d"),
        "end_date": subset.index[-1].strftime("%Y-%m-%d"),
        "return_pct": _round((float(equity.iloc[-1]) / float(equity.iloc[0]) - 1) * 100),
        "max_drawdown_pct": _round(float(drawdown.min()) * 100),
        "data_complete": True,
    }


def _drawdown_and_risk_summary(daily: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for label, subset in _period_slices(daily).items():
        if subset.empty:
            continue
        position = subset["position_ticker"].astype(str)
        active = subset[~position.eq("cash")]
        top_position = position.value_counts().index[0] if len(position) else ""
        rows.append(
            {
                "period_label": label,
                "start_date": subset["date"].iloc[0],
                "end_date": subset["date"].iloc[-1],
                "max_drawdown_pct": _round(float(pd.to_numeric(subset["drawdown"], errors="coerce").min()) * 100),
                "worst_drawdown_date": subset.loc[pd.to_numeric(subset["drawdown"], errors="coerce").idxmin(), "date"],
                "active_days": int(len(active)),
                "cash_days": int(position.eq("cash").sum()),
                "00631L_position_days": int(position.str.contains("00631L.TW", regex=False).sum()),
                "00631L_position_day_share": _round(position.str.contains("00631L.TW", regex=False).sum() / len(subset)),
                "top_position": top_position,
                "top_position_day_share": _round(position.eq(top_position).sum() / len(subset)) if len(subset) else "",
            }
        )
    return pd.DataFrame(rows)


def _trade_cost_turnover_summary(daily: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for label, subset in _period_slices(daily).items():
        if subset.empty:
            continue
        cost = pd.to_numeric(subset["transaction_cost"], errors="coerce").fillna(0)
        turnover = pd.to_numeric(subset["turnover"], errors="coerce").fillna(0)
        rows.append(
            {
                "period_label": label,
                "start_date": subset["date"].iloc[0],
                "end_date": subset["date"].iloc[-1],
                "trade_days": int(subset["action"].astype(str).ne("hold").sum()),
                "total_transaction_cost": _round(float(cost.sum())),
                "total_turnover": _round(float(turnover.sum())),
                "cost_to_turnover_ratio": _round(float(cost.sum() / turnover.sum())) if float(turnover.sum()) else "",
                "average_trade_day_cost": _round(float(cost[cost > 0].mean())) if len(cost[cost > 0]) else "",
            }
        )
    return pd.DataFrame(rows)


def _holding_transition_summary(daily: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for label, subset in _period_slices(daily).items():
        if subset.empty:
            continue
        target = subset["position_ticker"].astype(str).tolist()
        changes = [idx for idx in range(1, len(target)) if target[idx] != target[idx - 1]]
        holding_spans = _holding_spans(subset)
        rows.append(
            {
                "period_label": label,
                "start_date": subset["date"].iloc[0],
                "end_date": subset["date"].iloc[-1],
                "target_change_count": int(len(changes)),
                "formal_target_changed_within_1d_count": int(_changed_within(changes, 1)),
                "formal_target_changed_within_3d_count": int(_changed_within(changes, 3)),
                "rapid_flip_same_target_window_1_3d_count": int(_rapid_flip(target, window=3)),
                "average_holding_days": _round(sum(holding_spans) / len(holding_spans)) if holding_spans else "",
                "minimum_holding_days": min(holding_spans) if holding_spans else "",
                "possible_execution_layer_issue": bool(_changed_within(changes, 3) > 0),
                "execution_layer_status": "not_formal_report_only_diagnostic",
            }
        )
    return pd.DataFrame(rows)


def _report_only_diagnostics_inventory(opportunity_dir: Path, cooldown_dir: Path) -> pd.DataFrame:
    rows = [
        {
            "component": "0050x2_opportunity_cost_label",
            "source_dir": str(opportunity_dir),
            "boundary": "report_only",
            "active_in_trade_decision": False,
            "used_in_performance": False,
            "status": "smoke_pass",
            "description_zh": "0050正二機會成本警示，只提醒特定權值槓桿行情可能落後，不改正式 target。",
        },
        {
            "component": "execution_cooldown3_label",
            "source_dir": str(cooldown_dir),
            "boundary": "report_only",
            "active_in_trade_decision": False,
            "used_in_performance": False,
            "status": "smoke_pass",
            "description_zh": "Cooldown3 只保留執行層診斷標籤，未啟用正式換倉規則。",
        },
        {
            "component": "pool3_radar_and_pool3_shadow",
            "source_dir": "",
            "boundary": "shadow_diagnostic",
            "active_in_trade_decision": False,
            "used_in_performance": False,
            "status": "not_formal",
            "description_zh": "Pool3 / Radar 不進正式投票，也不阻塞目前 Pool1+Pool2 主線。",
        },
        {
            "component": "final_decision_layer_labels",
            "source_dir": "",
            "boundary": "report_only",
            "active_in_trade_decision": False,
            "used_in_performance": False,
            "status": "not_formal_selector",
            "description_zh": "Final decision layer 只保留 report boundary，不作 formal selector。",
        },
        {
            "component": "rr_partial_switch_paper_trade",
            "source_dir": "",
            "boundary": "paper_trade_shadow",
            "active_in_trade_decision": False,
            "used_in_performance": False,
            "status": "sample_limited",
            "description_zh": "RR partial switch 仍是 paper-trade shadow，未啟用 execution layer。",
        },
        {
            "component": "valuation_h3_chip_extra_rules",
            "source_dir": "",
            "boundary": "excluded_or_blocked",
            "active_in_trade_decision": False,
            "used_in_performance": False,
            "status": "not_used",
            "description_zh": "本正式模型未加入 valuation、H3 或額外籌碼規則。",
        },
    ]
    return pd.DataFrame(rows)


def _residual_caveat_matrix() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "caveat_id": "2024_hard_gate_0050x2_opportunity_cost",
                "severity": "high",
                "blocks_current_formal_baseline": False,
                "active_trade_rule": False,
                "summary_zh": "2024 權值槓桿行情中，正式候選仍落後 0050正二；此 caveat 只作 report-only 機會成本警示。",
                "minimum_handling": "報告揭露，不開 market exposure override。",
            },
            {
                "caveat_id": "execution_layer_not_activated",
                "severity": "medium",
                "blocks_current_formal_baseline": False,
                "active_trade_rule": False,
                "summary_zh": "正式 replay 仍是 selector baseline，不代表完整 next-day execution / exit layer 已完成。",
                "minimum_handling": "Cooldown3 與 next-day 診斷保留 report-only。",
            },
            {
                "caveat_id": "pool3_not_formal",
                "severity": "medium",
                "blocks_current_formal_baseline": False,
                "active_trade_rule": False,
                "summary_zh": "Pool3/Radar 目前不具正式第三票資格，不再卡住 Pool1+Pool2 正式主線。",
                "minimum_handling": "保留為 shadow/diagnostic 或後續資料線。",
            },
            {
                "caveat_id": "same_day_replay_vs_real_execution",
                "severity": "medium",
                "blocks_current_formal_baseline": False,
                "active_trade_rule": False,
                "summary_zh": "主要正式績效數字來自 same-day replay；next-day 口徑另有診斷結果，不應混成正式績效。",
                "minimum_handling": "報告中清楚標示 replay 口徑。",
            },
        ]
    )


def _manifest(
    absorption_manifest: dict[str, Any],
    formal_perf: pd.DataFrame,
    diagnostics: pd.DataFrame,
    caveats: pd.DataFrame,
    output: Path,
) -> dict[str, Any]:
    full = _row(formal_perf, "period_label", "full")
    return {
        "schema_version": 1,
        "task_id": TASK_ID,
        "status": "completed",
        "formal_model_target": FORMAL_MODEL_TARGET,
        "formal_model_route": FORMAL_MODEL_ROUTE,
        "three_pool_formal_route_abandoned": True,
        "formal_model_changed_in_this_task": False,
        "trade_decision_changed_in_this_task": False,
        "pool3_shadow_used_as_formal": False,
        "0050x2_opportunity_label_active_in_trade_decision": False,
        "execution_label_active_in_trade_decision": False,
        "market_exposure_override_absorbed": False,
        "rr_partial_switch_used_in_performance": False,
        "valuation_used": False,
        "h3_used": False,
        "uses_forward_return_as_rule": False,
        "report_only_diagnostics_active_count": int(diagnostics["active_in_trade_decision"].map(_truthy).sum()),
        "residual_caveat_count": int(len(caveats)),
        "blocking_caveat_count": int(caveats["blocks_current_formal_baseline"].map(_truthy).sum()),
        "latest_complete_common_date": str(full.get("end_date", absorption_manifest.get("latest_complete_common_date", ""))),
        "formal_same_day_full_return_pct": _float(full.get("return_pct")),
        "formal_same_day_full_max_drawdown_pct": _float(full.get("max_drawdown_pct")),
        "formal_report_baseline_usable": True,
        "output_dir": str(output.resolve()),
    }


def _period_slices(daily: pd.DataFrame) -> dict[str, pd.DataFrame]:
    frame = daily.copy()
    frame["date_ts"] = pd.to_datetime(frame["date"])
    periods = {
        "2022": ("2022-01-01", "2022-12-31"),
        "2023": ("2023-01-01", "2023-12-31"),
        "2024_now": ("2024-01-01", None),
        "2024_hard_gate": ("2024-01-01", "2024-12-31"),
        "full": (None, None),
    }
    result: dict[str, pd.DataFrame] = {}
    for label, (start, end) in periods.items():
        subset = frame
        if start:
            subset = subset[subset["date_ts"] >= pd.Timestamp(start)]
        if end:
            subset = subset[subset["date_ts"] <= pd.Timestamp(end)]
        result[label] = subset.copy()
    return result


def _holding_spans(subset: pd.DataFrame) -> list[int]:
    targets = subset["position_ticker"].astype(str).tolist()
    if not targets:
        return []
    spans: list[int] = []
    current = targets[0]
    length = 1
    for target in targets[1:]:
        if target == current:
            length += 1
        else:
            spans.append(length)
            current = target
            length = 1
    spans.append(length)
    return spans


def _changed_within(change_indexes: list[int], window: int) -> int:
    return sum(1 for prev, cur in zip(change_indexes, change_indexes[1:]) if cur - prev <= window)


def _rapid_flip(targets: list[str], window: int = 3) -> int:
    count = 0
    for idx, target in enumerate(targets):
        lookback = targets[max(0, idx - window) : idx]
        if target and target != "cash" and target in lookback and targets[idx - 1] != target:
            count += 1
    return count


def _row(frame: pd.DataFrame, column: str, value: str) -> dict[str, Any]:
    if column not in frame.columns:
        return {}
    subset = frame[frame[column].astype(str).eq(value)]
    return subset.iloc[0].to_dict() if not subset.empty else {}


def _text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value)
    return "" if text.lower() == "nan" else text.strip()


def _float(value: Any) -> float | None:
    try:
        if value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _round(value: Any) -> Any:
    try:
        return round(float(value), 4)
    except (TypeError, ValueError):
        return ""


def _truthy(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def _diagnosis_summary_markdown(
    formal_perf: pd.DataFrame,
    prior_models: pd.DataFrame,
    benchmarks: pd.DataFrame,
    caveats: pd.DataFrame,
) -> str:
    full = _row(formal_perf, "period_label", "full")
    hard = _row(formal_perf, "period_label", "2024_hard_gate")
    old = prior_models[
        prior_models["variant"].astype(str).eq("current_formal_three_pool_baseline")
        & prior_models["period_label"].astype(str).eq("full")
    ]
    pool1 = prior_models[
        prior_models["variant"].astype(str).eq("pool1_only_formal_replay")
        & prior_models["period_label"].astype(str).eq("full")
    ]
    hard_0050x2 = benchmarks[
        benchmarks["period_label"].astype(str).eq("2024_hard_gate")
        & benchmarks["benchmark"].astype(str).eq("0050正二_00631L")
    ]
    old_row = old.iloc[0].to_dict() if not old.empty else {}
    pool1_row = pool1.iloc[0].to_dict() if not pool1.empty else {}
    hard_bench = hard_0050x2.iloc[0].to_dict() if not hard_0050x2.empty else {}
    lines = [
        "# 新正式模型整體診斷與正式回測包",
        "",
        "## 正式模型是什麼",
        "",
        f"- 正式模型：`{FORMAL_MODEL_TARGET}`。",
        f"- 正式路線：`{FORMAL_MODEL_ROUTE}`。",
        "- 模型語意：Pool1 作為主攻 selector；Pool2 作為 confirmation / risk layer；00631L 目標權重上限 40%，超出部分留現金。",
        "- 三池表決已停止作為正式 performance selector；Pool3 / Radar 僅保留 shadow 或 diagnostic。",
        "",
        "## 核心績效",
        "",
        f"- 新正式模型 full same-day replay：報酬 `{full.get('return_pct')}%`，MDD `{full.get('max_drawdown_pct')}%`。",
        f"- 舊三池 full：報酬 `{old_row.get('return_pct', '')}%`，MDD `{old_row.get('max_drawdown_pct', '')}%`。",
        f"- Pool1-only full：報酬 `{pool1_row.get('return_pct', '')}%`，MDD `{pool1_row.get('max_drawdown_pct', '')}%`。",
        f"- 2024 hard gate：新正式模型 `{hard.get('return_pct')}%`；0050正二/00631L `{hard_bench.get('benchmark_return_pct', '')}%`，這是需保留的機會成本 caveat。",
        "",
        "## 邊界",
        "",
        "- 0050正二 opportunity-cost label 是 report-only，不改正式 target。",
        "- Cooldown3 / next-day execution 是 report-only diagnostic，不是正式換倉規則。",
        "- 未使用 Pool3 shadow、market exposure override、RR partial switch、valuation、H3 或 forward return 規則。",
        "",
        "## 結論",
        "",
        "這一版可以作為後續正式股票池觀察報告的模型基準；但報告必須揭露 2024 權值槓桿行情落後 0050正二，以及正式 replay 不等於完整 execution layer。",
        "",
        "## Residual Caveats",
        "",
    ]
    for row in caveats.to_dict(orient="records"):
        lines.append(f"- `{row.get('caveat_id')}`: {row.get('summary_zh')}")
    return "\n".join(lines) + "\n"


def _model_boundary_markdown() -> str:
    return "\n".join(
        [
            "# 正式模型與報告邊界",
            "",
            "正式 active model：",
            "- Pool1 primary selector。",
            "- PIT-ready Pool2 confirmation / risk layer。",
            "- 00631L cap40，超出權重留現金。",
            "",
            "非正式 / report-only：",
            "- Pool3 / Radar：shadow 或 diagnostic，不進正式投票。",
            "- 0050正二 opportunity-cost label：只作機會成本說明，不改 target。",
            "- Cooldown3 / next-day execution：只作執行層診斷，不是正式換倉規則。",
            "- Final decision layer labels：只作閱讀邊界，不是 formal selector。",
            "- RR partial switch、valuation、H3、forward return：未用於正式模型。",
            "",
            "公開語氣：",
            "- 這是 AI 輔助市場觀察與策略回測基準，不是買賣指令。",
            "- 報告可呈現模型 target 與風險 caveat，但不可寫成保證績效或替使用者決定交易。",
            "",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build formal model diagnosis and replay package.")
    parser.add_argument("--absorption-dir", default=DEFAULT_ABSORPTION_DIR)
    parser.add_argument("--formal-candidate-dir", default=DEFAULT_FORMAL_CANDIDATE_DIR)
    parser.add_argument("--comparison-dir", default=DEFAULT_COMPARISON_DIR)
    parser.add_argument("--opportunity-label-dir", default=DEFAULT_0050X2_LABEL_DIR)
    parser.add_argument("--cooldown-label-dir", default=DEFAULT_COOLDOWN_LABEL_DIR)
    parser.add_argument("--price-cache-dir", default=DEFAULT_PRICE_CACHE_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    output = run_formal_model_diagnosis_and_replay(
        absorption_dir=args.absorption_dir,
        formal_candidate_dir=args.formal_candidate_dir,
        comparison_dir=args.comparison_dir,
        opportunity_label_dir=args.opportunity_label_dir,
        cooldown_label_dir=args.cooldown_label_dir,
        price_cache_dir=args.price_cache_dir,
        output_dir=args.output_dir,
    )
    print(f"OUTPUT_DIR={output.resolve()}")


if __name__ == "__main__":
    main()
