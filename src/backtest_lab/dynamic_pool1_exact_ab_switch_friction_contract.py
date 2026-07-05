"""Build the Dynamic Pool1 exact A/B switch friction contract.

This package converts switch-friction attribution rows into a live-safe
candidate contract. Forward returns are retained only as evaluation metadata;
they are never used as rule inputs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


TASK_ID = "TASK-BACKTEST-CORE-DYNAMIC-POOL1-EXACT-AB-SWITCH-FRICTION-CONTRACT-001"
EXPERIMENTS_TASK_ID = "TASK-BACKTEST-EXPERIMENTS-DYNAMIC-POOL1-EXACT-AB-SWITCH-FRICTION-CONTRACT-VALIDATION-001"
DEFAULT_ATTRIBUTION_DIR = Path(
    r"C:\Users\zergv\Documents\Codex\2026-06-17\repo-ai-stock-backtest-lab-repo\outputs"
    r"\experiments_dynamic_pool1_switch_friction_entry_exit_attribution_20260704"
)
DEFAULT_CANDIDATE_CONTRACT = Path(
    "outputs/dynamic_pool1_benchmark_aware_candidate_contract_20260704"
    "/dynamic_pool1_benchmark_aware_candidate_contract.csv"
)
DEFAULT_CANDIDATE_CONTEXT = Path("outputs/dynamic_pool1_candidate_panel_v0_20260704/candidate_pool_by_month.csv")
DEFAULT_V2_SIGNAL_PANEL = Path("outputs/dynamic_pool1_v2_bounded_portfolio_contract_20260704/daily_signal_panel.csv")
DEFAULT_OUTPUT_DIR = Path("outputs/dynamic_pool1_exact_ab_switch_friction_contract_20260705")
DEFAULT_BACKTEST_PERIOD_CONTRACT = [
    {
        "period_label": "default_backtest_period_1",
        "requested_start": "2015-01-02",
        "requested_end": "2022-12-29",
    },
    {
        "period_label": "default_backtest_period_2",
        "requested_start": "2023-01-02",
        "requested_end": "2026-06-30",
    },
]


def run_dynamic_pool1_exact_ab_switch_friction_contract(
    *,
    repo_root: str | Path = ".",
    attribution_dir: str | Path = DEFAULT_ATTRIBUTION_DIR,
    candidate_contract: str | Path = DEFAULT_CANDIDATE_CONTRACT,
    candidate_context: str | Path = DEFAULT_CANDIDATE_CONTEXT,
    v2_signal_panel: str | Path = DEFAULT_V2_SIGNAL_PANEL,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    attribution_root = _resolve(root, attribution_dir)
    attribution_path = attribution_root / "ab_switch_comparison_panel.csv"
    candidate_path = _resolve(root, candidate_contract)
    context_path = _resolve(root, candidate_context)
    signal_path = _resolve(root, v2_signal_panel)
    output = _resolve(root, output_dir)
    output.mkdir(parents=True, exist_ok=True)

    attribution = pd.read_csv(attribution_path)
    candidate = pd.read_csv(candidate_path)
    context = pd.read_csv(context_path)
    signal_panel = pd.read_csv(signal_path) if signal_path.exists() else pd.DataFrame()

    contract = _build_contract(attribution, candidate, context, signal_panel)
    missing_audit = _missing_feature_audit(contract)
    benchmark_audit = _benchmark_rs_readiness_audit(contract)
    quality_audit = _quality_proxy_readiness_audit(contract)
    future_audit = _future_data_audit(contract)
    rule_distribution = _rule_candidate_distribution(contract)
    evaluation_trace = _evaluation_metadata_trace(contract)
    readiness = _contract_readiness_summary(contract, missing_audit, benchmark_audit, quality_audit, future_audit)

    contract.to_csv(output / "exact_ab_switch_friction_contract.csv", index=False, encoding="utf-8-sig")
    readiness.to_csv(output / "contract_readiness_summary.csv", index=False, encoding="utf-8-sig")
    missing_audit.to_csv(output / "missing_feature_audit.csv", index=False, encoding="utf-8-sig")
    benchmark_audit.to_csv(output / "benchmark_rs_readiness_audit.csv", index=False, encoding="utf-8-sig")
    quality_audit.to_csv(output / "quality_proxy_readiness_audit.csv", index=False, encoding="utf-8-sig")
    future_audit.to_csv(output / "future_data_audit.csv", index=False, encoding="utf-8-sig")
    rule_distribution.to_csv(output / "rule_candidate_distribution.csv", index=False, encoding="utf-8-sig")
    evaluation_trace.to_csv(output / "evaluation_metadata_trace.csv", index=False, encoding="utf-8-sig")

    future_count = int(future_audit["future_data_violation"].sum()) if len(future_audit) else 0
    manifest: dict[str, Any] = {
        "task_id": TASK_ID,
        "status": "completed_exact_ab_switch_friction_contract",
        "output_dir": str(output),
        "source_attribution_panel": str(attribution_path),
        "source_candidate_contract": str(candidate_path),
        "source_candidate_context": str(context_path),
        "source_v2_signal_panel": str(signal_path) if signal_path.exists() else "",
        "contract_rows": int(len(contract)),
        "unique_switch_events": int(contract["switch_event_id"].nunique()),
        "variant_count": int(contract["variant_id"].nunique()),
        "benchmark_rs_complete_rows": int(contract["benchmark_rs_ready"].sum()),
        "quality_proxy_complete_rows": int(contract["quality_proxy_ready"].sum()),
        "combined_ab_switch_friction_strict_rows": int(contract["combined_ab_switch_friction_strict"].sum()),
        "missing_feature_rows": int((missing_audit["missing_count"] > 0).sum()),
        "future_data_violation_count": future_count,
        "default_backtest_period_contract": DEFAULT_BACKTEST_PERIOD_CONTRACT,
        "actual_contract_start": _date_text(contract["date"].min()),
        "actual_contract_end": _date_text(contract["date"].max()),
        "forward_return_used_as_evaluation_metadata": True,
        "uses_forward_return_as_rule": False,
        "formal_model_changed": False,
        "trade_decision_changed": False,
        "active_in_trade_decision": False,
        "report_changed": False,
        "portfolio_replay_executed": False,
        "ready_for_strategy_replay": False,
        "ready_for_formal_absorption": False,
        "handoff_to_experiments_task": EXPERIMENTS_TASK_ID,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "final_summary_zh.md").write_text(_summary_text(manifest), encoding="utf-8")
    pd.DataFrame([{"task_id": TASK_ID, "status": "completed", "output_dir": str(output)}]).to_csv(
        output / "completed.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(columns=["task_id", "status", "reason"]).to_csv(output / "failed.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(
        [
            {"step": "load_attribution_panel", "status": "completed"},
            {"step": "asof_join_candidate_contract", "status": "completed"},
            {"step": "asof_join_candidate_context", "status": "completed"},
            {"step": "build_live_safe_rule_flags", "status": "completed"},
            {"step": "write_contract_package", "status": "completed"},
        ]
    ).to_csv(output / "run_log.csv", index=False, encoding="utf-8-sig")
    return manifest


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _build_contract(
    attribution: pd.DataFrame,
    candidate: pd.DataFrame,
    context: pd.DataFrame,
    signal_panel: pd.DataFrame,
) -> pd.DataFrame:
    rows = attribution.copy()
    rows["date"] = pd.to_datetime(rows["date"], errors="coerce")
    rows["switch_event_id"] = [
        f"ABSW-{d:%Y%m%d}-{i:05d}" if pd.notna(d) else f"ABSW-unknown-{i:05d}"
        for i, d in enumerate(rows["date"], start=1)
    ]
    rows["incumbent_ticker_A"] = rows["incumbent_ticker"].map(_ticker_text)
    rows["challenger_ticker_B"] = rows["challenger_ticker"].map(_ticker_text)
    rows["incumbent_ticker_A_base"] = rows["incumbent_ticker_A"].map(_base_ticker)
    rows["challenger_ticker_B_base"] = rows["challenger_ticker_B"].map(_base_ticker)
    rows["variant_id"] = rows["variant"].astype(str)
    rows["top1_or_top3_source"] = rows["variant_id"].str.contains("top3", case=False, na=False).map(
        {True: "top3", False: "top1"}
    )
    rows = _attach_next_tradable_date(rows, signal_panel)

    candidate_lookup = _AsOfLookup(
        candidate,
        ticker_col="ticker",
        date_col="candidate_as_of_date",
        value_cols=[
            "candidate_month",
            "candidate_as_of_date",
            "candidate_rank",
            "candidate_score",
            "candidate_layer",
            "price_ready_flag",
            "benchmark_0050_ready_flag",
            "benchmark_00631l_ready_flag",
            "ret_20d_vs_0050_trailing",
            "ret_60d_vs_0050_trailing",
            "ret_20d_vs_00631L_trailing",
            "ret_60d_vs_00631L_trailing",
            "benchmark_blocked_reason",
        ],
    )
    context_lookup = _AsOfLookup(
        _prepare_context(context),
        ticker_col="ticker",
        date_col="context_as_of_date",
        value_cols=[
            "context_as_of_date",
            "name",
            "market",
            "fundamental_quality_raw",
            "fundamentals_score",
            "revenue_yoy_3m_avg_pct",
            "liquidity_score",
            "avg_turnover_value",
            "candidate_scope",
            "candidate_source_scope",
        ],
    )

    enriched = []
    for _, row in rows.iterrows():
        record = row.to_dict()
        for side, ticker_col, rank_col, score_col in [
            ("A", "incumbent_ticker_A_base", "incumbent_rank", "score_A"),
            ("B", "challenger_ticker_B_base", "challenger_rank", "score_B"),
        ]:
            c = candidate_lookup.latest(record[ticker_col], record["date"])
            x = context_lookup.latest(record[ticker_col], record["date"])
            _assign_side_features(record, side, c, x, row, rank_col, score_col)
        enriched.append(record)
    out = pd.DataFrame(enriched)

    _coerce_numeric(
        out,
        [
            "incumbent_holding_age_days",
            "rank_A",
            "rank_B",
            "score_A",
            "score_B",
            "rs20_A_vs_0050",
            "rs20_B_vs_0050",
            "rs60_A_vs_0050",
            "rs60_B_vs_0050",
            "rs20_A_vs_00631l",
            "rs20_B_vs_00631l",
            "rs60_A_vs_00631l",
            "rs60_B_vs_00631l",
            "close_vs_ma20_A",
            "close_vs_ma20_B",
            "close_vs_ma60_A",
            "close_vs_ma60_B",
            "quality_proxy_A",
            "quality_proxy_B",
            "fundamental_proxy_A",
            "fundamental_proxy_B",
            "liquidity_A",
            "liquidity_B",
            "turnover_A",
            "turnover_B",
            "A_forward_return_5d",
            "B_forward_return_5d",
            "A_forward_return_10d",
            "B_forward_return_10d",
            "A_forward_return_20d",
            "B_forward_return_20d",
            "A_forward_return_40d",
            "B_forward_return_40d",
        ],
    )
    out["rank_margin"] = out["rank_A"] - out["rank_B"]
    out["score_margin"] = out["score_B"] - out["score_A"]
    out["rs20_B_minus_A_vs_0050"] = out["rs20_B_vs_0050"] - out["rs20_A_vs_0050"]
    out["rs60_B_minus_A_vs_0050"] = out["rs60_B_vs_0050"] - out["rs60_A_vs_0050"]
    out["rs20_B_minus_A_vs_00631l"] = out["rs20_B_vs_00631l"] - out["rs20_A_vs_00631l"]
    out["rs60_B_minus_A_vs_00631l"] = out["rs60_B_vs_00631l"] - out["rs60_A_vs_00631l"]
    out["deviation_gap_B_minus_A_ma20"] = out["close_vs_ma20_B"] - out["close_vs_ma20_A"]
    out["deviation_gap_B_minus_A_ma60"] = out["close_vs_ma60_B"] - out["close_vs_ma60_A"]
    out["quality_margin"] = out["quality_proxy_B"] - out["quality_proxy_A"]
    out["fundamental_margin"] = out["fundamental_proxy_B"] - out["fundamental_proxy_A"]
    out["liquidity_margin"] = out["liquidity_B"] - out["liquidity_A"]
    out["B_more_overheated_ma20"] = out["deviation_gap_B_minus_A_ma20"] > 5
    out["B_more_overheated_ma60"] = out["deviation_gap_B_minus_A_ma60"] > 8
    out["short_heat_only"] = _bool_series(out.get("short_heat_only_flag", False))
    out["medium_quality_confirmed"] = _bool_series(out.get("medium_quality_confirmed_flag", False)) | (
        out["quality_proxy_B"].notna() & (out["quality_proxy_B"] >= out["quality_proxy_A"].fillna(out["quality_proxy_B"]))
    )
    out["rank_score_superiority"] = (out["rank_margin"] >= 2) & (out["score_margin"] >= 0.05)
    out["rs_superiority"] = (
        (out["rs60_B_minus_A_vs_0050"] >= 0)
        & (out["rs60_B_minus_A_vs_00631l"] >= 0)
        & out["rs60_B_minus_A_vs_0050"].notna()
        & out["rs60_B_minus_A_vs_00631l"].notna()
    )
    out["quality_not_lower"] = out["quality_margin"].notna() & (out["quality_margin"] >= 0)
    out["not_more_overheated"] = (
        out["deviation_gap_B_minus_A_ma20"].notna()
        & out["deviation_gap_B_minus_A_ma60"].notna()
        & ~out["B_more_overheated_ma20"]
        & ~out["B_more_overheated_ma60"]
    )
    out["switch_margin_rank2_score5"] = out["rank_score_superiority"]
    out["switch_margin_rank3_score10"] = (out["rank_margin"] >= 3) & (out["score_margin"] >= 0.10)
    out["switch_not_more_overheated_ma20_5pp"] = out["deviation_gap_B_minus_A_ma20"].notna() & (
        out["deviation_gap_B_minus_A_ma20"] <= 5
    )
    out["switch_not_more_overheated_ma60_8pp"] = out["deviation_gap_B_minus_A_ma60"].notna() & (
        out["deviation_gap_B_minus_A_ma60"] <= 8
    )
    out["switch_quality_not_lower"] = out["quality_not_lower"]
    out["switch_no_short_heat_only"] = ~out["short_heat_only"]
    out["switch_after_min_hold5"] = out["incumbent_holding_age_days"] >= 5
    out["min_hold5_protection"] = ~out["switch_after_min_hold5"]
    out["combined_ab_switch_friction_candidate"] = (
        out["switch_margin_rank2_score5"]
        & out["switch_not_more_overheated_ma20_5pp"]
        & out["switch_not_more_overheated_ma60_8pp"]
        & out["switch_quality_not_lower"]
        & out["switch_no_short_heat_only"]
        & out["switch_after_min_hold5"]
        & out["rs_superiority"]
    )
    out["combined_ab_switch_friction_strict"] = out["combined_ab_switch_friction_candidate"]
    out["benchmark_rs_ready"] = out[
        [
            "rs20_A_vs_0050",
            "rs20_B_vs_0050",
            "rs60_A_vs_0050",
            "rs60_B_vs_0050",
            "rs20_A_vs_00631l",
            "rs20_B_vs_00631l",
            "rs60_A_vs_00631l",
            "rs60_B_vs_00631l",
        ]
    ].notna().all(axis=1)
    out["quality_proxy_ready"] = out[["quality_proxy_A", "quality_proxy_B"]].notna().all(axis=1)
    out["forward_return_used_as_evaluation_metadata"] = True
    out["uses_forward_return_as_rule"] = False
    out["formal_model_changed"] = False
    out["trade_decision_changed"] = False
    out["active_in_trade_decision"] = False
    out["report_changed"] = False
    out["portfolio_replay_executed"] = False
    out["ready_for_strategy_replay"] = False
    out["ready_for_formal_absorption"] = False
    if "formal_state" not in out.columns:
        out["formal_state"] = "not_provided_by_upstream_attribution"
    out["B_minus_A_forward_delta_5d"] = out["B_forward_return_5d"] - out["A_forward_return_5d"]
    out["B_minus_A_forward_delta_10d"] = out["B_forward_return_10d"] - out["A_forward_return_10d"]
    out["B_minus_A_forward_delta_20d"] = out["B_forward_return_20d"] - out["A_forward_return_20d"]
    out["B_minus_A_forward_delta_40d"] = out["B_forward_return_40d"] - out["A_forward_return_40d"]
    out["future_data_violation"] = _future_violation(out)
    for col in _contract_columns():
        if col not in out.columns:
            out[col] = pd.NA
    return out[_contract_columns()]


def _attach_next_tradable_date(rows: pd.DataFrame, signal_panel: pd.DataFrame) -> pd.DataFrame:
    out = rows.copy()
    if signal_panel.empty or "date" not in signal_panel.columns or "next_tradable_date" not in signal_panel.columns:
        out["next_tradable_date"] = pd.NA
        out["next_tradable_date_source"] = "missing_v2_signal_panel"
        return out
    map_df = signal_panel[["date", "next_tradable_date"]].drop_duplicates("date").copy()
    map_df["date"] = pd.to_datetime(map_df["date"], errors="coerce")
    out = out.merge(map_df, on="date", how="left")
    out["next_tradable_date_source"] = "v2_daily_signal_panel"
    out.loc[out["next_tradable_date"].isna(), "next_tradable_date_source"] = "missing_for_switch_date"
    return out


def _prepare_context(context: pd.DataFrame) -> pd.DataFrame:
    out = context.copy()
    if "available_date" in out.columns:
        out["context_as_of_date"] = out["available_date"]
    else:
        out["context_as_of_date"] = pd.to_datetime(out["year_month"].astype(str) + "-01", errors="coerce") + pd.offsets.MonthEnd(0)
    return out


class _AsOfLookup:
    def __init__(self, frame: pd.DataFrame, *, ticker_col: str, date_col: str, value_cols: list[str]) -> None:
        self.value_cols = [c for c in value_cols if c in frame.columns]
        data = frame.copy()
        data["_ticker_key"] = data[ticker_col].map(_base_ticker)
        data["_asof_date"] = pd.to_datetime(data[date_col], errors="coerce")
        data = data[data["_ticker_key"].ne("") & data["_asof_date"].notna()].sort_values(["_ticker_key", "_asof_date"])
        self.by_ticker = {ticker: group for ticker, group in data.groupby("_ticker_key", sort=False)}

    def latest(self, ticker: object, date: object) -> dict[str, Any]:
        key = _base_ticker(ticker)
        when = pd.to_datetime(date, errors="coerce")
        if not key or pd.isna(when) or key not in self.by_ticker:
            return {}
        group = self.by_ticker[key]
        eligible = group[group["_asof_date"] <= when]
        if eligible.empty:
            return {}
        row = eligible.iloc[-1]
        return {col: row.get(col) for col in self.value_cols}


def _assign_side_features(
    record: dict[str, Any],
    side: str,
    candidate: dict[str, Any],
    context: dict[str, Any],
    source_row: pd.Series,
    attribution_rank_col: str,
    attribution_score_col: str,
) -> None:
    record[f"candidate_as_of_date_{side}"] = candidate.get("candidate_as_of_date")
    record[f"candidate_month_{side}"] = candidate.get("candidate_month")
    record[f"candidate_layer_{side}"] = candidate.get("candidate_layer")
    record[f"candidate_name_{side}"] = context.get("name")
    record[f"candidate_market_{side}"] = context.get("market")
    record[f"rank_{side}"] = _first_value(candidate.get("candidate_rank"), source_row.get(attribution_rank_col))
    record[f"score_{side}"] = _first_value(candidate.get("candidate_score"), source_row.get(attribution_score_col))
    record[f"rs20_{side}_vs_0050"] = _first_value(candidate.get("ret_20d_vs_0050_trailing"), source_row.get(f"rs20_{side}_vs_0050"))
    record[f"rs60_{side}_vs_0050"] = _first_value(candidate.get("ret_60d_vs_0050_trailing"), source_row.get(f"rs60_{side}_vs_0050"))
    record[f"rs20_{side}_vs_00631l"] = _first_value(
        candidate.get("ret_20d_vs_00631L_trailing"), source_row.get(f"rs20_{side}_vs_00631l")
    )
    record[f"rs60_{side}_vs_00631l"] = _first_value(
        candidate.get("ret_60d_vs_00631L_trailing"), source_row.get(f"rs60_{side}_vs_00631l")
    )
    record[f"quality_proxy_{side}"] = _first_value(
        context.get("fundamental_quality_raw"),
        context.get("fundamentals_score"),
        source_row.get(f"quality_score_{side}"),
    )
    record[f"fundamental_proxy_{side}"] = _first_value(
        context.get("revenue_yoy_3m_avg_pct"),
        context.get("fundamentals_score"),
        context.get("fundamental_quality_raw"),
    )
    record[f"liquidity_{side}"] = _first_value(context.get("liquidity_score"), source_row.get(f"liquidity_{side}"))
    record[f"turnover_{side}"] = _first_value(context.get("avg_turnover_value"), source_row.get(f"turnover_{side}"))
    record[f"benchmark_blocked_reason_{side}"] = candidate.get("benchmark_blocked_reason", "")


def _first_value(*values: object) -> object:
    for value in values:
        if value is None:
            continue
        try:
            if pd.isna(value):
                continue
        except TypeError:
            pass
        return value
    return pd.NA


def _ticker_text(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip()
    if text.endswith(".0") and text[:-2].isdigit():
        text = text[:-2]
    return text


def _base_ticker(value: object) -> str:
    text = _ticker_text(value).upper()
    if "." in text:
        text = text.split(".", 1)[0]
    return text


def _coerce_numeric(frame: pd.DataFrame, cols: list[str]) -> None:
    for col in cols:
        if col in frame.columns:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
        else:
            frame[col] = pd.NA


def _bool_series(value: object) -> pd.Series:
    if isinstance(value, pd.Series):
        if value.dtype == bool:
            return value.fillna(False)
        return value.astype(str).str.lower().isin(["true", "1", "yes"])
    return pd.Series([bool(value)])


def _future_violation(out: pd.DataFrame) -> pd.Series:
    dates = pd.to_datetime(out["date"], errors="coerce")
    violations = pd.Series(False, index=out.index)
    for col in ["candidate_as_of_date_A", "candidate_as_of_date_B"]:
        if col in out.columns:
            violations = violations | (pd.to_datetime(out[col], errors="coerce") > dates)
    return violations.fillna(False)


def _missing_feature_audit(contract: pd.DataFrame) -> pd.DataFrame:
    rows = []
    fields = [
        "next_tradable_date",
        "rank_A",
        "rank_B",
        "score_A",
        "score_B",
        "rs20_A_vs_0050",
        "rs20_B_vs_0050",
        "rs60_A_vs_0050",
        "rs60_B_vs_0050",
        "rs20_A_vs_00631l",
        "rs20_B_vs_00631l",
        "rs60_A_vs_00631l",
        "rs60_B_vs_00631l",
        "quality_proxy_A",
        "quality_proxy_B",
        "fundamental_proxy_A",
        "fundamental_proxy_B",
        "liquidity_A",
        "liquidity_B",
        "turnover_A",
        "turnover_B",
        "close_vs_ma20_A",
        "close_vs_ma20_B",
        "close_vs_ma60_A",
        "close_vs_ma60_B",
    ]
    for field in fields:
        missing = int(contract[field].isna().sum()) if field in contract.columns else len(contract)
        rows.append(
            {
                "field": field,
                "missing_count": missing,
                "total_rows": int(len(contract)),
                "missing_rate": round(missing / len(contract), 6) if len(contract) else 0.0,
                "status": "ready" if missing == 0 else "partial",
            }
        )
    return pd.DataFrame(rows)


def _benchmark_rs_readiness_audit(contract: pd.DataFrame) -> pd.DataFrame:
    fields = [
        "rs20_A_vs_0050",
        "rs20_B_vs_0050",
        "rs60_A_vs_0050",
        "rs60_B_vs_0050",
        "rs20_A_vs_00631l",
        "rs20_B_vs_00631l",
        "rs60_A_vs_00631l",
        "rs60_B_vs_00631l",
    ]
    return pd.DataFrame(
        [
            {
                "field": field,
                "ready_rows": int(contract[field].notna().sum()),
                "missing_rows": int(contract[field].isna().sum()),
                "total_rows": int(len(contract)),
                "ready_rate": round(float(contract[field].notna().mean()), 6) if len(contract) else 0.0,
            }
            for field in fields
        ]
    )


def _quality_proxy_readiness_audit(contract: pd.DataFrame) -> pd.DataFrame:
    fields = ["quality_proxy_A", "quality_proxy_B", "fundamental_proxy_A", "fundamental_proxy_B", "liquidity_A", "liquidity_B"]
    return pd.DataFrame(
        [
            {
                "field": field,
                "ready_rows": int(contract[field].notna().sum()),
                "missing_rows": int(contract[field].isna().sum()),
                "total_rows": int(len(contract)),
                "ready_rate": round(float(contract[field].notna().mean()), 6) if len(contract) else 0.0,
                "source_note": "asof candidate_pool_v0 context plus attribution fallback",
            }
            for field in fields
        ]
    )


def _future_data_audit(contract: pd.DataFrame) -> pd.DataFrame:
    return contract[
        [
            "switch_event_id",
            "date",
            "incumbent_ticker_A",
            "challenger_ticker_B",
            "candidate_as_of_date_A",
            "candidate_as_of_date_B",
            "future_data_violation",
            "uses_forward_return_as_rule",
        ]
    ].copy()


def _rule_candidate_distribution(contract: pd.DataFrame) -> pd.DataFrame:
    rules = [
        "switch_margin_rank2_score5",
        "switch_margin_rank3_score10",
        "switch_not_more_overheated_ma20_5pp",
        "switch_not_more_overheated_ma60_8pp",
        "switch_quality_not_lower",
        "switch_no_short_heat_only",
        "switch_after_min_hold5",
        "combined_ab_switch_friction_strict",
    ]
    return pd.DataFrame(
        [
            {
                "rule_candidate": rule,
                "passed_rows": int(contract[rule].sum()),
                "total_rows": int(len(contract)),
                "passed_rate": round(float(contract[rule].mean()), 6) if len(contract) else 0.0,
                "uses_forward_return_as_rule": False,
            }
            for rule in rules
        ]
    )


def _evaluation_metadata_trace(contract: pd.DataFrame) -> pd.DataFrame:
    fields = [
        "switch_event_id",
        "date",
        "variant_id",
        "incumbent_ticker_A",
        "challenger_ticker_B",
        "A_forward_return_5d",
        "B_forward_return_5d",
        "B_minus_A_forward_delta_5d",
        "A_forward_return_10d",
        "B_forward_return_10d",
        "B_minus_A_forward_delta_10d",
        "A_forward_return_20d",
        "B_forward_return_20d",
        "B_minus_A_forward_delta_20d",
        "A_forward_return_40d",
        "B_forward_return_40d",
        "B_minus_A_forward_delta_40d",
        "forward_return_used_as_evaluation_metadata",
        "uses_forward_return_as_rule",
    ]
    return contract[fields].copy()


def _contract_readiness_summary(
    contract: pd.DataFrame,
    missing: pd.DataFrame,
    benchmark: pd.DataFrame,
    quality: pd.DataFrame,
    future: pd.DataFrame,
) -> pd.DataFrame:
    rows = [
            {
                "metric": "contract_rows",
                "value": int(len(contract)),
                "status": "completed",
                "note": "one row per potential A/B switch from upstream attribution panel",
            },
            {
                "metric": "benchmark_rs_complete_rows",
                "value": int(contract["benchmark_rs_ready"].sum()),
                "status": "partial" if not contract["benchmark_rs_ready"].all() else "ready",
                "note": "numeric 0050/00631L RS joined as-of where available",
            },
            {
                "metric": "quality_proxy_complete_rows",
                "value": int(contract["quality_proxy_ready"].sum()),
                "status": "partial" if not contract["quality_proxy_ready"].all() else "ready",
                "note": "quality proxy remains diagnostic and source-backed",
            },
            {
                "metric": "combined_ab_switch_friction_strict_rows",
                "value": int(contract["combined_ab_switch_friction_strict"].sum()),
                "status": "diagnostic_only",
                "note": "rule flag only; no strategy replay executed",
            },
            {
                "metric": "future_data_violation_count",
                "value": int(future["future_data_violation"].sum()),
                "status": "ready" if int(future["future_data_violation"].sum()) == 0 else "blocked",
                "note": "candidate features use as-of date not later than switch date",
            },
            {
                "metric": "missing_feature_fields",
                "value": int((missing["missing_count"] > 0).sum()),
                "status": "partial" if int((missing["missing_count"] > 0).sum()) else "ready",
                "note": f"benchmark audit fields={len(benchmark)}, quality audit fields={len(quality)}",
            },
        ]
    for period in DEFAULT_BACKTEST_PERIOD_CONTRACT:
        requested_start = pd.to_datetime(period["requested_start"])
        requested_end = pd.to_datetime(period["requested_end"])
        in_period = contract[(pd.to_datetime(contract["date"]) >= requested_start) & (pd.to_datetime(contract["date"]) <= requested_end)]
        rows.append(
            {
                "metric": f"{period['period_label']}_date_contract",
                "value": int(len(in_period)),
                "status": "governance_recorded",
                "note": (
                    f"requested_start={period['requested_start']};requested_end={period['requested_end']};"
                    f"actual_start={_date_text(in_period['date'].min())};actual_end={_date_text(in_period['date'].max())}"
                ),
            }
        )
    return pd.DataFrame(rows)


def _contract_columns() -> list[str]:
    return [
        "date",
        "next_tradable_date",
        "variant_id",
        "period_label",
        "formal_state",
        "incumbent_ticker_A",
        "challenger_ticker_B",
        "incumbent_holding_age_days",
        "switch_event_id",
        "top1_or_top3_source",
        "candidate_name_A",
        "candidate_name_B",
        "candidate_market_A",
        "candidate_market_B",
        "candidate_as_of_date_A",
        "candidate_as_of_date_B",
        "candidate_month_A",
        "candidate_month_B",
        "candidate_layer_A",
        "candidate_layer_B",
        "rank_A",
        "rank_B",
        "rank_margin",
        "score_A",
        "score_B",
        "score_margin",
        "rs20_A_vs_0050",
        "rs20_B_vs_0050",
        "rs20_B_minus_A_vs_0050",
        "rs60_A_vs_0050",
        "rs60_B_vs_0050",
        "rs60_B_minus_A_vs_0050",
        "rs20_A_vs_00631l",
        "rs20_B_vs_00631l",
        "rs20_B_minus_A_vs_00631l",
        "rs60_A_vs_00631l",
        "rs60_B_vs_00631l",
        "rs60_B_minus_A_vs_00631l",
        "close_vs_ma20_A",
        "close_vs_ma20_B",
        "deviation_gap_B_minus_A_ma20",
        "B_more_overheated_ma20",
        "close_vs_ma60_A",
        "close_vs_ma60_B",
        "deviation_gap_B_minus_A_ma60",
        "B_more_overheated_ma60",
        "quality_proxy_A",
        "quality_proxy_B",
        "quality_margin",
        "liquidity_A",
        "liquidity_B",
        "liquidity_margin",
        "turnover_A",
        "turnover_B",
        "fundamental_proxy_A",
        "fundamental_proxy_B",
        "fundamental_margin",
        "short_heat_only",
        "medium_quality_confirmed",
        "rank_score_superiority",
        "rs_superiority",
        "quality_not_lower",
        "not_more_overheated",
        "min_hold5_protection",
        "switch_margin_rank2_score5",
        "switch_margin_rank3_score10",
        "switch_not_more_overheated_ma20_5pp",
        "switch_not_more_overheated_ma60_8pp",
        "switch_quality_not_lower",
        "switch_no_short_heat_only",
        "switch_after_min_hold5",
        "combined_ab_switch_friction_candidate",
        "combined_ab_switch_friction_strict",
        "benchmark_rs_ready",
        "quality_proxy_ready",
        "benchmark_blocked_reason_A",
        "benchmark_blocked_reason_B",
        "A_forward_return_5d",
        "B_forward_return_5d",
        "B_minus_A_forward_delta_5d",
        "A_forward_return_10d",
        "B_forward_return_10d",
        "B_minus_A_forward_delta_10d",
        "A_forward_return_20d",
        "B_forward_return_20d",
        "B_minus_A_forward_delta_20d",
        "A_forward_return_40d",
        "B_forward_return_40d",
        "B_minus_A_forward_delta_40d",
        "forward_return_used_as_evaluation_metadata",
        "uses_forward_return_as_rule",
        "future_data_violation",
        "formal_model_changed",
        "trade_decision_changed",
        "active_in_trade_decision",
        "report_changed",
        "portfolio_replay_executed",
        "ready_for_strategy_replay",
        "ready_for_formal_absorption",
    ]


def _summary_text(manifest: dict[str, Any]) -> str:
    period_lines = [
        f"- {p['period_label']}：requested {p['requested_start']}～{p['requested_end']}"
        for p in manifest["default_backtest_period_contract"]
    ]
    return "\n".join(
        [
            "# Dynamic Pool1 exact A/B switch friction contract",
            "",
            "本包把 Dynamic Pool1 v2 A/B switch attribution 轉成 exact contract；只提供 live-safe rule flags 與 evaluation metadata，不跑策略、不改正式模型。",
            "",
            f"- contract rows：{manifest['contract_rows']}",
            f"- benchmark RS complete rows：{manifest['benchmark_rs_complete_rows']}",
            f"- quality proxy complete rows：{manifest['quality_proxy_complete_rows']}",
            f"- combined strict candidate rows：{manifest['combined_ab_switch_friction_strict_rows']}",
            f"- future_data_violation_count：{manifest['future_data_violation_count']}",
            f"- actual contract range：{manifest['actual_contract_start']}～{manifest['actual_contract_end']}",
            "- default_backtest_period_contract：",
            *period_lines,
            "- forward returns 僅為 evaluation metadata，uses_forward_return_as_rule=false。",
            "- ready_for_strategy_replay=false；ready_for_formal_absorption=false。",
            f"- 下一棒：{manifest['handoff_to_experiments_task']}",
        ]
    )


def _date_text(value: object) -> str:
    date = pd.to_datetime(value, errors="coerce")
    if pd.isna(date):
        return ""
    return str(date.date())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--attribution-dir", default=str(DEFAULT_ATTRIBUTION_DIR))
    parser.add_argument("--candidate-contract", default=str(DEFAULT_CANDIDATE_CONTRACT))
    parser.add_argument("--candidate-context", default=str(DEFAULT_CANDIDATE_CONTEXT))
    parser.add_argument("--v2-signal-panel", default=str(DEFAULT_V2_SIGNAL_PANEL))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args(argv)
    manifest = run_dynamic_pool1_exact_ab_switch_friction_contract(
        repo_root=args.repo_root,
        attribution_dir=args.attribution_dir,
        candidate_contract=args.candidate_contract,
        candidate_context=args.candidate_context,
        v2_signal_panel=args.v2_signal_panel,
        output_dir=args.output_dir,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
