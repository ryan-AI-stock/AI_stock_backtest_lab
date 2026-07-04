"""Build a bounded Dynamic Pool1 portfolio challenger contract package."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from backtest_lab.costs import COST_MODEL_VERSION, cost_model_metadata


TASK_ID = "TASK-BACKTEST-CORE-DYNAMIC-POOL1-PORTFOLIO-CHALLENGER-CONTRACT-001"
DEFAULT_CANDIDATE_DIR = Path("outputs/dynamic_pool1_candidate_panel_v0_20260704")
DEFAULT_OUTPUT_DIR = Path("outputs/dynamic_pool1_portfolio_challenger_contract_20260704")
FORMAL_STREAMS = [
    Path("outputs/combined_formal_target_stream_20150128_20211230_20260702/combined_formal_target_stream.csv"),
    Path("outputs/formal_long_range_signal_reconstruction_201411_latest_20260702/formal_long_range_target_stream.csv"),
]
BENCHMARK_PRICE_PATHS = {
    "0050.TW": Path("backtest_cache/0050_TW.csv"),
    "00631L.TW": Path("backtest_cache/00631L_TW.csv"),
}


def run_dynamic_pool1_portfolio_contract(
    *,
    repo_root: str | Path = ".",
    candidate_dir: str | Path = DEFAULT_CANDIDATE_DIR,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> dict:
    root = Path(repo_root).resolve()
    candidate_path = _resolve(root, candidate_dir)
    output = _resolve(root, output_dir)
    output.mkdir(parents=True, exist_ok=True)

    candidate_pool = pd.read_csv(candidate_path / "candidate_pool_by_month.csv")
    candidate_panel = pd.read_csv(candidate_path / "candidate_panel_monthly.csv")
    monthly = _monthly_candidate_summary(candidate_pool)
    formal = _load_formal_streams(root)
    daily = _build_daily_contract(formal, monthly, candidate_panel)
    benchmark_availability = _benchmark_availability(root, daily)
    daily = daily.merge(benchmark_availability, on="trade_date", how="left")
    daily["cost_model_id"] = COST_MODEL_VERSION

    variant_matrix = _variant_matrix()
    execution_rules = _execution_rules()
    baseline_contract = _baseline_contract()
    cost_contract = pd.DataFrame([cost_model_metadata()])
    schema = _input_schema()

    daily.to_csv(output / "daily_portfolio_contract_panel.csv", index=False, encoding="utf-8-sig")
    variant_matrix.to_csv(output / "portfolio_variant_matrix.csv", index=False, encoding="utf-8-sig")
    execution_rules.to_csv(output / "execution_rule_variants.csv", index=False, encoding="utf-8-sig")
    baseline_contract.to_csv(output / "baseline_contract.csv", index=False, encoding="utf-8-sig")
    cost_contract.to_csv(output / "cost_model_contract.csv", index=False, encoding="utf-8-sig")
    schema.to_csv(output / "candidate_event_input_schema.csv", index=False, encoding="utf-8-sig")
    _contract_md(output).write_text(_contract_text(), encoding="utf-8")
    (output / "portfolio_challenger_contract.json").write_text(
        json.dumps(_contract_json(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    readiness = {
        "task_id": TASK_ID,
        "status": "completed_contract_ready_for_experiments_validation",
        "ready_for_experiments": True,
        "experiments_task_id": "TASK-BACKTEST-EXPERIMENTS-DYNAMIC-POOL1-PORTFOLIO-CHALLENGER-VALIDATION-001",
        "ready_for_formal_absorption": False,
        "diagnostic_only": True,
        "strategy_replay_executed_by_core": False,
        "formal_model_changed": False,
        "trade_decision_changed": False,
        "active_in_trade_decision": False,
        "daily_contract_rows": int(len(daily)),
        "benchmark_0050_available_rows": int(daily["benchmark_0050_return_available"].fillna(False).sum()),
        "benchmark_00631l_available_rows": int(daily["benchmark_00631l_return_available"].fillna(False).sum()),
        "benchmark_join_boundary": "explicit local cache availability only; no cross-section median substitute",
    }
    (output / "readiness_for_experiments.json").write_text(
        json.dumps(readiness, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    manifest = {
        "task_id": TASK_ID,
        "status": "completed_portfolio_challenger_contract_only",
        "output_dir": str(output),
        "candidate_source": str(candidate_path),
        "daily_contract_rows": int(len(daily)),
        "variant_count": int(len(variant_matrix)),
        "formal_model_changed": False,
        "trade_decision_changed": False,
        "active_in_trade_decision": False,
        "report_changed": False,
        "strategy_replay_executed_by_core": False,
        "forward_return_used_as_live_rule": False,
        "ready_for_experiments_validation": True,
        "handoff_to_experiments_task": readiness["experiments_task_id"],
    }
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "final_summary_zh.md").write_text(_summary(manifest, readiness), encoding="utf-8")
    pd.DataFrame([{"task_id": TASK_ID, "status": "completed", "output_dir": str(output)}]).to_csv(
        output / "completed.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(columns=["task_id", "status", "reason"]).to_csv(output / "failed.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(
        [
            {"step": "load_candidate_panel", "status": "completed"},
            {"step": "load_formal_streams", "status": "completed"},
            {"step": "build_daily_contract_panel", "status": "completed"},
            {"step": "write_contract_package", "status": "completed"},
        ]
    ).to_csv(output / "run_log.csv", index=False, encoding="utf-8-sig")
    return manifest


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _monthly_candidate_summary(pool: pd.DataFrame) -> pd.DataFrame:
    pool = pool.copy()
    pool["year_month"] = pool["year_month"].astype(str)
    pool["candidate_rank_v0"] = pd.to_numeric(pool["candidate_rank_v0"], errors="coerce")
    pool["dynamic_pool1_score_v0"] = pd.to_numeric(pool["dynamic_pool1_score_v0"], errors="coerce")
    pool = pool.sort_values(["year_month", "candidate_rank_v0"])
    rows = []
    for month, group in pool.groupby("year_month", sort=True):
        top = group.head(5)
        scores = top["dynamic_pool1_score_v0"].tolist()
        rows.append(
            {
                "candidate_refresh_month": month,
                "candidate_pool_tickers": ";".join(group["ticker"].astype(str).tolist()),
                "selected_top1_ticker": str(top.iloc[0]["ticker"]) if not top.empty else "",
                "selected_top3_tickers": ";".join(top.head(3)["ticker"].astype(str).tolist()),
                "selected_top5_tickers": ";".join(top.head(5)["ticker"].astype(str).tolist()),
                "top1_candidate_score": scores[0] if len(scores) >= 1 else None,
                "score_margin_top1_top2": scores[0] - scores[1] if len(scores) >= 2 else None,
                "score_margin_top1_top5": scores[0] - scores[4] if len(scores) >= 5 else None,
                "candidate_source": "dynamic_pool1_candidate_panel_v0_monthly",
                "candidate_readiness_state": str(top.iloc[0].get("feature_readiness_state", "")) if not top.empty else "data_blocked",
                "candidate_data_blocked_reason": "" if not top.empty else "no_candidate_for_refresh_month",
            }
        )
    return pd.DataFrame(rows)


def _load_formal_streams(root: Path) -> pd.DataFrame:
    frames = []
    for rel in FORMAL_STREAMS:
        path = root / rel
        if not path.exists():
            continue
        df = pd.read_csv(path)
        if "signal_date" not in df.columns:
            continue
        if "execution_date" not in df.columns:
            df["execution_date"] = ""
        frames.append(df)
    if not frames:
        raise FileNotFoundError("No formal target stream inputs found")
    formal = pd.concat(frames, ignore_index=True, sort=False)
    formal["signal_date"] = pd.to_datetime(formal["signal_date"], errors="coerce")
    formal = formal.dropna(subset=["signal_date"]).sort_values("signal_date")
    formal = formal.drop_duplicates("signal_date", keep="last")
    formal["trade_date"] = formal["signal_date"].dt.strftime("%Y-%m-%d")
    formal["candidate_refresh_month"] = formal["signal_date"].dt.to_period("M").astype(str)
    return formal


def _build_daily_contract(formal: pd.DataFrame, monthly: pd.DataFrame, candidate_panel: pd.DataFrame) -> pd.DataFrame:
    daily = formal.merge(monthly, on="candidate_refresh_month", how="left")
    daily["next_tradable_date"] = daily["execution_date"].astype(str)
    daily["dynamic_pool_version"] = "dynamic_pool1_candidate_panel_v0"
    daily["candidate_rank"] = 1
    daily["candidate_score"] = daily["top1_candidate_score"]
    daily["formal_target_ticker"] = daily.get("formal_target", "").fillna("").astype(str)
    daily["formal_target_state"] = daily["formal_target_ticker"].map(_formal_target_state)
    daily["cash_state"] = daily["formal_target_state"].eq("cash")
    daily["market_exposure_state"] = daily["formal_target_state"].eq("market_exposure")
    daily["formal_conflict_state"] = daily.apply(_conflict_state, axis=1)
    daily["blocked_fill_state"] = daily.apply(_blocked_fill_state, axis=1)
    daily["uses_forward_return_as_live_rule"] = False
    daily["diagnostic_only"] = True
    daily["active_in_trade_decision"] = False
    columns = [
        "trade_date",
        "next_tradable_date",
        "dynamic_pool_version",
        "candidate_refresh_month",
        "candidate_pool_tickers",
        "selected_top1_ticker",
        "selected_top3_tickers",
        "selected_top5_tickers",
        "candidate_rank",
        "candidate_score",
        "score_margin_top1_top2",
        "score_margin_top1_top5",
        "candidate_source",
        "candidate_readiness_state",
        "candidate_data_blocked_reason",
        "formal_target_ticker",
        "formal_target_state",
        "formal_conflict_state",
        "cash_state",
        "market_exposure_state",
        "blocked_fill_state",
        "cost_model_id",
        "diagnostic_only",
        "active_in_trade_decision",
        "uses_forward_return_as_live_rule",
    ]
    return daily[[col for col in columns if col in daily.columns]].copy()


def _formal_target_state(target: str) -> str:
    target = (target or "").strip()
    if not target or target.upper() == "CASH":
        return "cash"
    if target.startswith("00631L"):
        return "market_exposure"
    return "stock_target"


def _conflict_state(row: pd.Series) -> str:
    if not row.get("selected_top1_ticker"):
        return "data_blocked"
    state = row.get("formal_target_state", "")
    target = str(row.get("formal_target_ticker", ""))
    candidate = str(row.get("selected_top1_ticker", ""))
    if state == "cash":
        return "no_conflict_formal_cash"
    if state == "market_exposure":
        return "no_conflict_market_exposure"
    if target.startswith(candidate):
        return "same_ticker_no_double_count"
    if state == "stock_target":
        return "conflict_with_formal_stock_target"
    return "data_blocked"


def _blocked_fill_state(row: pd.Series) -> str:
    if not row.get("next_tradable_date") or str(row.get("next_tradable_date")).lower() == "nan":
        return "blocked_missing_next_tradable_date"
    if not row.get("selected_top1_ticker"):
        return "blocked_missing_dynamic_candidate"
    if str(row.get("candidate_readiness_state", "")).startswith("blocked"):
        return "blocked_candidate_readiness"
    return "fill_contract_ready"


def _benchmark_availability(root: Path, daily: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame({"trade_date": daily["trade_date"].unique()})
    for ticker, rel in BENCHMARK_PRICE_PATHS.items():
        path = root / rel
        key = ticker.replace(".", "_").lower()
        if path.exists():
            prices = pd.read_csv(path, usecols=lambda col: col in {"date", "close", "adj_close"})
            dates = set(pd.to_datetime(prices["date"], errors="coerce").dropna().dt.strftime("%Y-%m-%d"))
            out[f"benchmark_{key}_return_available"] = out["trade_date"].isin(dates)
            out[f"benchmark_{key}_price_source"] = str(rel)
        else:
            out[f"benchmark_{key}_return_available"] = False
            out[f"benchmark_{key}_price_source"] = "missing_local_cache"
    out = out.rename(
        columns={
            "benchmark_0050_tw_return_available": "benchmark_0050_return_available",
            "benchmark_0050_tw_price_source": "benchmark_0050_price_source",
            "benchmark_00631l_tw_return_available": "benchmark_00631l_return_available",
            "benchmark_00631l_tw_price_source": "benchmark_00631l_price_source",
        }
    )
    return out


def _variant_matrix() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "variant_id": "baseline_formal_next_day",
                "sleeve_weight": 0.0,
                "entry_contract": "current formal target only",
                "exit_contract": "current formal target stream",
                "allowed_formal_states": "all",
                "diagnostic_stress_only": False,
                "formal_route_candidate": False,
            },
            {
                "variant_id": "dynamic_top1_when_formal_cash_or_market_exposure",
                "sleeve_weight": 0.20,
                "entry_contract": "top1 next-day only when formal state is cash or market exposure",
                "exit_contract": "fixed_hold_20_or_fixed_hold_60_by_experiments_variant",
                "allowed_formal_states": "cash;market_exposure",
                "diagnostic_stress_only": False,
                "formal_route_candidate": False,
            },
            {
                "variant_id": "dynamic_top1_when_formal_no_target_only",
                "sleeve_weight": 0.20,
                "entry_contract": "top1 next-day only when formal no-target/risk-control cash",
                "exit_contract": "fixed_hold_20_or_fixed_hold_60_by_experiments_variant",
                "allowed_formal_states": "cash",
                "diagnostic_stress_only": False,
                "formal_route_candidate": False,
            },
            {
                "variant_id": "dynamic_top3_equal_weight_when_formal_cash_or_market_exposure",
                "sleeve_weight": 0.20,
                "entry_contract": "top3 equal-weight sleeve next-day when formal state is cash or market exposure",
                "exit_contract": "fixed_hold_20_or_fixed_hold_60_by_experiments_variant",
                "allowed_formal_states": "cash;market_exposure",
                "diagnostic_stress_only": False,
                "formal_route_candidate": False,
            },
            {
                "variant_id": "dynamic_top1_all_formal_states_diagnostic_only",
                "sleeve_weight": 0.20,
                "entry_contract": "top1 next-day in all formal states; diagnostic stress only",
                "exit_contract": "fixed_hold_20_or_fixed_hold_60_by_experiments_variant",
                "allowed_formal_states": "all",
                "diagnostic_stress_only": True,
                "formal_route_candidate": False,
            },
        ]
    )


def _execution_rules() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"rule_id": "execution_basis", "value": "next_day_only"},
            {"rule_id": "primary_sleeve_weight", "value": "20_percent"},
            {"rule_id": "sensitivity_sleeve_weight", "value": "10_percent_optional_experiments_only"},
            {"rule_id": "exit_fixed_hold_20", "value": "exit after 20 tradable days unless fill/data blocked"},
            {"rule_id": "exit_fixed_hold_60", "value": "exit after 60 tradable days unless fill/data blocked"},
            {"rule_id": "blocked_fill", "value": "do not fill if next_tradable_date, price, or candidate readiness is missing"},
            {"rule_id": "same_ticker_overlap", "value": "do not double count if formal target equals dynamic candidate"},
            {"rule_id": "forward_return_live_rule", "value": "forbidden"},
        ]
    )


def _baseline_contract() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"baseline_id": "baseline_formal_next_day", "description": "Current formal next-day target stream with no dynamic sleeve"},
            {"baseline_id": "report_only_context_no_trade_change", "description": "Dynamic candidates shown only as diagnostic context, no trade change"},
            {"baseline_id": "0050_buy_and_hold", "description": "Benchmark only where explicit 0050 local price cache is available"},
            {"baseline_id": "00631L_buy_and_hold", "description": "Benchmark only where explicit 00631L local price cache is available"},
        ]
    )


def _input_schema() -> pd.DataFrame:
    fields = [
        "trade_date",
        "next_tradable_date",
        "dynamic_pool_version",
        "candidate_refresh_month",
        "candidate_pool_tickers",
        "selected_top1_ticker",
        "selected_top3_tickers",
        "selected_top5_tickers",
        "candidate_rank",
        "candidate_score",
        "score_margin_top1_top2",
        "score_margin_top1_top5",
        "candidate_source",
        "candidate_readiness_state",
        "candidate_data_blocked_reason",
        "formal_target_ticker",
        "formal_target_state",
        "formal_conflict_state",
        "cash_state",
        "market_exposure_state",
        "benchmark_0050_return_available",
        "benchmark_00631l_return_available",
        "cost_model_id",
    ]
    return pd.DataFrame({"field": fields, "required_for_experiments": True, "live_safe": True})


def _contract_json() -> dict:
    return {
        "task_id": TASK_ID,
        "scope": "diagnostic next-day portfolio challenger contract",
        "entry_variants": _variant_matrix().to_dict(orient="records"),
        "execution_basis": "next_day_only",
        "cost_model_id": COST_MODEL_VERSION,
        "formal_model_changed": False,
        "trade_decision_changed": False,
        "active_in_trade_decision": False,
    }


def _contract_text() -> str:
    return "\n".join(
        [
            "# Dynamic Pool1 portfolio challenger contract",
            "",
            "本包只定義 Experiments 可執行的 bounded next-day diagnostic contract。",
            "",
            "- 不改 formal selector / formal target / daily report / trade action。",
            "- Entry 只使用 signal date 可得的 candidate panel 欄位。",
            "- Forward return 不得作 live rule。",
            "- Primary variants 只允許 formal cash 或 market exposure 狀態開 20% opportunity sleeve。",
            "- all-formal-states 只作 diagnostic stress test，不得當 formal route。",
            "- 成本使用 current formal Taiwan fee/tax model。",
            "- 0050/00631L benchmark 只用 explicit local cache availability，不使用 cross-section median 替代。",
        ]
    )


def _contract_md(output: Path) -> Path:
    return output / "portfolio_challenger_contract.md"


def _summary(manifest: dict, readiness: dict) -> str:
    return "\n".join(
        [
            "# Dynamic Pool1 portfolio challenger contract",
            "",
            "已建立 bounded next-day portfolio challenger contract，可交 Experiments 做 shadow/diagnostic validation。",
            "",
            f"- daily contract rows：{manifest['daily_contract_rows']}",
            f"- variants：{manifest['variant_count']}",
            f"- 0050 benchmark explicit availability rows：{readiness['benchmark_0050_available_rows']}",
            f"- 00631L benchmark explicit availability rows：{readiness['benchmark_00631l_available_rows']}",
            "- Core 未跑 strategy replay，未改正式模型、交易或報告。",
        ]
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--candidate-dir", default=str(DEFAULT_CANDIDATE_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args(argv)
    manifest = run_dynamic_pool1_portfolio_contract(
        repo_root=args.repo_root,
        candidate_dir=args.candidate_dir,
        output_dir=args.output_dir,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
