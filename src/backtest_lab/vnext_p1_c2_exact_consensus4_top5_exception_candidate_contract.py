from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS_ROOT = Path("C:/Users/zergv/Documents/Codex/2026-07-06/backtest-lab-experiments-diagnostic-validation-attribution")
LEGACY_P1_TRACE = (
    EXPERIMENTS_ROOT
    / "outputs"
    / "vnext_p1_legacy_regime_unadjusted_ohlc_cost_timing_diagnostic_20260708"
    / "p1_legacy_regime_unadjusted_ohlc_trade_path_trace.csv"
)
PRIOR_POLICY_TRACE = (
    EXPERIMENTS_ROOT
    / "outputs"
    / "vnext_p1_defensive_policy_benchmark_comparison_diagnostic_20260708"
    / "p1_defensive_policy_benchmark_policy_trace.csv"
)
C2_STATE_MACHINE = (
    REPO_ROOT
    / "outputs"
    / "vnext_p1_c2_market_health_consensus4_adjusted_state_machine_contract_20260708"
    / "p1_c2_market_health_consensus4_state_machine_contract.csv"
)
PREV_COST_DESIGN = (
    REPO_ROOT
    / "outputs"
    / "vnext_p1_c2_top5_ohlc_absorption_and_exright_review_20260708"
    / "p1_c2_top5_exception_candidate_transition_cost_design.csv"
)
OUTPUT_DIR = REPO_ROOT / "outputs" / "vnext_p1_c2_exact_consensus4_top5_exception_candidate_contract_20260708"

TASK_ID = "TASK-BACKTEST-CORE-VNEXT-P1-C2-EXACT-CONSENSUS4-TOP5-EXCEPTION-CANDIDATE-CONTRACT-001"
PRIMARY_TIMING = "next_day_close_entry_fixed_5td_exit"
RETURN_COL = "net_return_local_ep05_cost_unit_notional"
SOURCE_VARIANTS = [
    "hybrid_pullback_base_mega_override",
    "conservative_hurdle_route",
    "pool_breadth_route",
    "market_bias_pool_trend_route",
    "dispersion_route",
]
FLAGS = {
    "formal_model_changed": False,
    "trade_decision_changed": False,
    "active_in_trade_decision": False,
    "report_changed": False,
    "portfolio_replay_executed": False,
    "ready_for_strategy_replay": False,
    "ready_for_formal": False,
    "not_live_rule": True,
    "forward_returns_live_rule_usage": False,
}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _as_bool(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s
    return s.astype(str).str.lower().isin(["true", "1", "yes"])


def _ticker(value: Any) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    return text.zfill(4) if text.isdigit() and len(text) < 4 else text


def _load_ordinary_trace() -> pd.DataFrame:
    df = pd.read_csv(LEGACY_P1_TRACE, low_memory=False, dtype={"ticker": str, "executable_ticker": str})
    df["signal_date"] = pd.to_datetime(df["signal_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    for col in ["entry_date", "exit_date"]:
        df[col] = pd.to_datetime(df[col], errors="coerce").dt.strftime("%Y-%m-%d")
    df["ticker"] = df["ticker"].map(_ticker)
    df["path_ready"] = _as_bool(df["path_ready"])
    if "price_path_ready" in df.columns:
        df["price_path_ready"] = _as_bool(df["price_path_ready"])
    else:
        df["price_path_ready"] = df["path_ready"]
    return df[
        df["timing_variant"].eq(PRIMARY_TIMING)
        & df["path_bucket"].eq("ordinary_stock")
        & df["variant"].isin(SOURCE_VARIANTS)
        & df["path_ready"]
        & df[RETURN_COL].notna()
    ].copy()


def _route_flags(values: pd.Series) -> str:
    return "|".join(sorted(set(map(str, values.dropna()))))


def _variant_flags(values: pd.Series) -> str:
    return "|".join(sorted(set(map(str, values.dropna()))))


def _representative_path_rows(trace: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "signal_date",
        "ticker",
        "entry_date",
        "exit_date",
        "entry_open",
        "entry_close",
        "exit_close",
        "entry_price",
        "exit_price",
        "gross_return_unadjusted",
        RETURN_COL,
        "source_quality",
        "entry_source_route",
        "exit_source_route",
        "entry_adjustment_policy",
        "exit_adjustment_policy",
        "total_cost_twd",
        "cost_application_status",
    ]
    existing = [c for c in cols if c in trace.columns]
    rep = trace.sort_values(["signal_date", "ticker", "variant", "route_or_mode"])[existing].drop_duplicates(["signal_date", "ticker"])
    return rep.rename(columns={RETURN_COL: "representative_net_return_local_ep05_cost_unit_notional"})


def _consensus_candidates(trace: pd.DataFrame) -> pd.DataFrame:
    rep = _representative_path_rows(trace)
    grouped = (
        trace.groupby(["signal_date", "ticker"], as_index=False)
        .agg(
            consensus_count=("variant", "nunique"),
            route_count=("route_or_mode", "nunique"),
            route_source_flags=("route_or_mode", _route_flags),
            variant_source_flags=("variant", _variant_flags),
            route_row_count=("variant", "size"),
        )
        .sort_values(["signal_date", "consensus_count", "route_count", "ticker"], ascending=[True, False, False, True])
    )
    grouped = grouped[grouped["consensus_count"] >= 4].copy()
    grouped["candidate_rank"] = grouped.groupby("signal_date").cumcount() + 1
    grouped = grouped[grouped["candidate_rank"] <= 5].copy()
    grouped["candidate_score"] = grouped["consensus_count"] * 100 + grouped["route_count"]
    grouped["rank_source"] = "exact_multi_route_same_ticker_consensus_count_then_route_count_then_ticker"
    grouped["route_source_policy"] = "source variants: " + "|".join(SOURCE_VARIANTS)
    return grouped.merge(rep, on=["signal_date", "ticker"], how="left")


def _prior_single_exception() -> pd.DataFrame:
    prior = pd.read_csv(PRIOR_POLICY_TRACE, low_memory=False, dtype={"ticker": str, "recommendation": str})
    prior["signal_date"] = pd.to_datetime(prior["signal_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    prior["ticker"] = prior["ticker"].map(_ticker)
    prior = prior[
        prior["policy_id"].eq("consensus4_else_00631L")
        & prior["timing_variant"].eq(PRIMARY_TIMING)
    ].copy()
    prior["prior_single_exception_ticker"] = prior["ticker"].where(prior["exposure_type"].eq("stock"), "")
    return prior[["signal_date", "prior_single_exception_ticker", "exposure_type", "selection_reason"]].drop_duplicates("signal_date")


def _c2_gate() -> pd.DataFrame:
    c2 = pd.read_csv(C2_STATE_MACHINE, low_memory=False, dtype={"holding_ticker": str})
    c2["signal_date"] = pd.to_datetime(c2["signal_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    c2["holding_ticker"] = c2["holding_ticker"].map(_ticker)
    c2 = c2[c2["signal_date"].notna()].copy()
    for col in ["c2_market_health_gate", "raw_consensus4_exception_active", "exception_allowed_by_c2"]:
        c2[col] = _as_bool(c2[col])
    c2["prior_c2_allowed_exception_ticker"] = c2["holding_ticker"].where(c2["exception_allowed_by_c2"], "")
    return c2[
        [
            "signal_date",
            "c2_market_health_gate",
            "raw_consensus4_exception_active",
            "exception_allowed_by_c2",
            "prior_c2_allowed_exception_ticker",
            "0050_above_ma60_flag",
            "0050_return_20d",
            "0050_return_40d",
        ]
    ].drop_duplicates("signal_date")


def _build_contract(candidates: pd.DataFrame, prior: pd.DataFrame, c2: pd.DataFrame) -> pd.DataFrame:
    df = candidates.merge(c2, on="signal_date", how="left").merge(prior, on="signal_date", how="left")
    bool_cols = ["c2_market_health_gate", "raw_consensus4_exception_active", "exception_allowed_by_c2"]
    for col in bool_cols:
        df[col] = df[col].fillna(False).astype(bool)
    df["top1_matches_prior_single_exception"] = (
        (df["candidate_rank"] == 1)
        & (df["ticker"].eq(df["prior_single_exception_ticker"].fillna("")))
        & df["prior_single_exception_ticker"].fillna("").ne("")
    )
    df["top1_matches_prior_c2_allowed_exception"] = (
        (df["candidate_rank"] == 1)
        & (df["ticker"].eq(df["prior_c2_allowed_exception_ticker"].fillna("")))
        & df["prior_c2_allowed_exception_ticker"].fillna("").ne("")
    )
    df["official_unadjusted_ohlc_path_ready"] = df[["entry_close", "exit_close"]].notna().all(axis=1)
    df["candidate_available"] = True
    df["adjusted_close_ready"] = False
    df["adjustment_policy"] = "official_unadjusted_ohlc_diagnostic_only_adjusted_close_blocked"
    df["candidate_contract_policy"] = "exact_consensus4_route_candidates_not_layer4_proxy_top5"
    df["future_data_violation_count"] = 0
    df["diagnostic_only"] = True
    for key, value in FLAGS.items():
        df[key] = value
    ordered = [
        "signal_date",
        "candidate_rank",
        "ticker",
        "candidate_available",
        "consensus_count",
        "route_count",
        "candidate_score",
        "route_source_flags",
        "variant_source_flags",
        "route_row_count",
        "rank_source",
        "route_source_policy",
        "c2_market_health_gate",
        "raw_consensus4_exception_active",
        "exception_allowed_by_c2",
        "prior_single_exception_ticker",
        "prior_c2_allowed_exception_ticker",
        "top1_matches_prior_single_exception",
        "top1_matches_prior_c2_allowed_exception",
        "entry_date",
        "exit_date",
        "entry_open",
        "entry_close",
        "exit_close",
        "gross_return_unadjusted",
        "representative_net_return_local_ep05_cost_unit_notional",
        "official_unadjusted_ohlc_path_ready",
        "adjusted_close_ready",
        "source_quality",
        "entry_source_route",
        "exit_source_route",
        "entry_adjustment_policy",
        "exit_adjustment_policy",
        "total_cost_twd",
        "cost_application_status",
        "adjustment_policy",
        "candidate_contract_policy",
        "future_data_violation_count",
        "diagnostic_only",
        *FLAGS.keys(),
    ]
    return df[[c for c in ordered if c in df.columns]]


def _add_shortfall_slots(contract: pd.DataFrame) -> pd.DataFrame:
    signal_dates = sorted(contract["signal_date"].dropna().unique())
    slots = pd.MultiIndex.from_product([signal_dates, range(1, 6)], names=["signal_date", "candidate_rank"]).to_frame(index=False)
    expanded = slots.merge(contract, on=["signal_date", "candidate_rank"], how="left", suffixes=("", "_actual"))
    expanded["candidate_available"] = expanded["candidate_available"].fillna(False).astype(bool)

    per_date = contract.sort_values(["signal_date", "candidate_rank"]).groupby("signal_date").first(numeric_only=False).reset_index()
    fill_cols = [
        "c2_market_health_gate",
        "raw_consensus4_exception_active",
        "exception_allowed_by_c2",
        "prior_single_exception_ticker",
        "prior_c2_allowed_exception_ticker",
    ]
    expanded = expanded.merge(per_date[["signal_date", *[c for c in fill_cols if c in per_date.columns]]], on="signal_date", how="left", suffixes=("", "_datefill"))
    for col in fill_cols:
        fill_col = f"{col}_datefill"
        if fill_col in expanded.columns:
            expanded[col] = expanded[col].where(expanded[col].notna(), expanded[fill_col])
            expanded = expanded.drop(columns=[fill_col])

    expanded["ticker"] = expanded["ticker"].fillna("")
    expanded["official_unadjusted_ohlc_path_ready"] = expanded["official_unadjusted_ohlc_path_ready"].fillna(False).astype(bool)
    expanded.loc[~expanded["candidate_available"], "candidate_contract_policy"] = "shortfall_no_additional_consensus4_candidate"
    expanded.loc[~expanded["candidate_available"], "adjustment_policy"] = "not_applicable_shortfall_slot"
    expanded.loc[~expanded["candidate_available"], "rank_source"] = "shortfall_slot_explicit_no_proxy_fill"
    expanded.loc[~expanded["candidate_available"], "route_source_policy"] = "no candidate: do not backfill with Layer4 generic top5"
    expanded.loc[~expanded["candidate_available"], "adjusted_close_ready"] = False
    expanded.loc[~expanded["candidate_available"], "future_data_violation_count"] = 0
    expanded.loc[~expanded["candidate_available"], "diagnostic_only"] = True
    for key, value in FLAGS.items():
        if key not in expanded.columns:
            expanded[key] = value
        else:
            expanded[key] = expanded[key].where(expanded[key].notna(), value)
    return expanded


def _match_audit(contract: pd.DataFrame, prior: pd.DataFrame, c2: pd.DataFrame) -> pd.DataFrame:
    top1 = contract[contract["candidate_rank"].eq(1)][
        ["signal_date", "ticker", "consensus_count", "route_count", "top1_matches_prior_single_exception", "top1_matches_prior_c2_allowed_exception"]
    ].rename(columns={"ticker": "exact_top1_ticker"})
    audit = prior.merge(c2, on="signal_date", how="outer").merge(top1, on="signal_date", how="left")
    audit["prior_has_single_exception"] = audit["prior_single_exception_ticker"].fillna("").ne("")
    audit["prior_c2_has_allowed_exception"] = audit["prior_c2_allowed_exception_ticker"].fillna("").ne("")
    audit["exact_top1_available"] = audit["exact_top1_ticker"].fillna("").ne("")
    audit["exact_top1_matches_prior_single_exception"] = (
        audit["prior_has_single_exception"] & audit["exact_top1_ticker"].fillna("").eq(audit["prior_single_exception_ticker"].fillna(""))
    )
    audit["exact_top1_matches_prior_c2_allowed_exception"] = (
        audit["prior_c2_has_allowed_exception"] & audit["exact_top1_ticker"].fillna("").eq(audit["prior_c2_allowed_exception_ticker"].fillna(""))
    )
    audit["match_policy"] = "top1 must reproduce prior consensus4 single-exception ticker; C2 subset checks allowed exception ticker"
    return audit.sort_values("signal_date")


def _coverage(contract: pd.DataFrame) -> pd.DataFrame:
    rows = []
    actual = contract[contract["candidate_available"].fillna(False).astype(bool)]
    scopes = {
        "all_exact_consensus4_rank_le5_candidate_rows": actual,
        "all_exact_consensus4_rank_le5_slots_including_shortfall": contract,
        "c2_true_rank_le5_candidate_rows": actual[actual["c2_market_health_gate"]],
        "c2_true_rank_le5_slots_including_shortfall": contract[contract["c2_market_health_gate"]],
        "c2_allowed_exception_rank_le5_candidate_rows": actual[actual["exception_allowed_by_c2"]],
    }
    for scope, df in scopes.items():
        rows.append(
            {
                "scope": scope,
                "rows": int(len(df)),
                "signal_dates": int(df["signal_date"].nunique()) if len(df) else 0,
                "unique_tickers": int(df["ticker"].nunique()) if len(df) else 0,
                "official_unadjusted_ohlc_ready_rows": int(df["official_unadjusted_ohlc_path_ready"].fillna(False).astype(bool).sum()) if len(df) else 0,
                "official_unadjusted_ohlc_ready_share": float(df["official_unadjusted_ohlc_path_ready"].fillna(False).astype(bool).mean()) if len(df) else None,
                "adjusted_close_ready_rows": 0,
                "adjusted_close_ready_share": 0.0,
            }
        )
    return pd.DataFrame(rows)


def _blocked_ledger(contract: pd.DataFrame, match_audit: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    actual = contract[contract["candidate_available"].fillna(False).astype(bool)]
    missing_path = actual[~actual["official_unadjusted_ohlc_path_ready"].fillna(False).astype(bool)]
    for row in missing_path.itertuples(index=False):
        rows.append(
            {
                "signal_date": row.signal_date,
                "ticker": row.ticker,
                "candidate_rank": row.candidate_rank,
                "blocked_item": "official_unadjusted_ohlc_path",
                "blocked_reason": "missing_entry_or_exit_close",
                "policy": "no silent fill",
            }
        )
    shortfall = contract[~contract["candidate_available"].fillna(False).astype(bool)]
    for row in shortfall.itertuples(index=False):
        rows.append(
            {
                "signal_date": row.signal_date,
                "ticker": "",
                "candidate_rank": row.candidate_rank,
                "blocked_item": "rank_shortfall",
                "blocked_reason": "no additional exact consensus4 candidate for this rank",
                "policy": "leave blank; no Layer4 proxy backfill",
            }
        )
    unmatched = match_audit[match_audit["prior_has_single_exception"] & ~match_audit["exact_top1_matches_prior_single_exception"]]
    for row in unmatched.itertuples(index=False):
        rows.append(
            {
                "signal_date": row.signal_date,
                "ticker": getattr(row, "exact_top1_ticker", ""),
                "candidate_rank": 1,
                "blocked_item": "top1_match_prior_single_exception",
                "blocked_reason": f"prior={getattr(row, 'prior_single_exception_ticker', '')}; exact_top1={getattr(row, 'exact_top1_ticker', '')}",
                "policy": "audit mismatch; do not substitute Layer4 proxy",
            }
        )
    rows.append(
        {
            "signal_date": "",
            "ticker": "",
            "candidate_rank": "",
            "blocked_item": "adjusted_close",
            "blocked_reason": "exact historical ex-right date and capital-change adjustment route incomplete",
            "policy": "use official unadjusted OHLC diagnostic-only; not formal",
        }
    )
    return pd.DataFrame(rows)


def _readiness(contract: pd.DataFrame, match_audit: pd.DataFrame, coverage: pd.DataFrame, blocked: pd.DataFrame) -> dict[str, Any]:
    prior_single = match_audit[match_audit["prior_has_single_exception"]]
    prior_c2 = match_audit[match_audit["prior_c2_has_allowed_exception"]]
    actual = contract[contract["candidate_available"].fillna(False).astype(bool)]
    c2_rank = actual[actual["c2_market_health_gate"]]
    ready_share = float(c2_rank["official_unadjusted_ohlc_path_ready"].mean()) if len(c2_rank) else 0.0
    top1_prior_share = (
        float(prior_single["exact_top1_matches_prior_single_exception"].mean()) if len(prior_single) else 0.0
    )
    top1_c2_share = (
        float(prior_c2["exact_top1_matches_prior_c2_allowed_exception"].mean()) if len(prior_c2) else 0.0
    )
    ready = ready_share == 1.0 and top1_prior_share == 1.0 and top1_c2_share == 1.0
    return {
        "task_id": TASK_ID,
        "status": "exact_consensus4_top5_contract_ready_unadjusted_diagnostic_adjusted_blocked" if ready else "exact_consensus4_top5_contract_partial_requires_review",
        "ready_for_p1_c2_exact_consensus4_top5_multi_stock_diagnostic": bool(ready),
        "ready_for_experiments": bool(ready),
        "exact_consensus_source": "P1 legacy/regime trade path trace grouped by signal_date+ticker across five source variants",
        "not_layer4_proxy_top5": True,
        "exact_top1_matches_prior_single_exception_share": top1_prior_share,
        "exact_top1_matches_prior_c2_allowed_exception_share": top1_c2_share,
        "prior_single_exception_signal_dates": int(len(prior_single)),
        "prior_c2_allowed_exception_signal_dates": int(len(prior_c2)),
        "contract_rows": int(len(contract)),
        "actual_candidate_rows": int(len(actual)),
        "shortfall_slot_rows": int((~contract["candidate_available"].fillna(False).astype(bool)).sum()),
        "contract_signal_dates": int(contract["signal_date"].nunique()),
        "c2_true_rank_le5_rows": int(len(c2_rank)),
        "official_unadjusted_ohlc_ready_share": ready_share,
        "official_unadjusted_ohlc_ready_rows": int(c2_rank["official_unadjusted_ohlc_path_ready"].sum()) if len(c2_rank) else 0,
        "official_unadjusted_ohlc_blocked_rows": int((~c2_rank["official_unadjusted_ohlc_path_ready"]).sum()) if len(c2_rank) else 0,
        "adjusted_close_ready": False,
        "transition_cost_fields_ready": True,
        "future_data_violation_count": 0,
        "ready_for_formal": False,
        "ready_for_strategy_replay": False,
        "not_live_rule": True,
        "forward_returns_live_rule_usage": False,
        "formal_model_changed": False,
        "trade_decision_changed": False,
        "active_in_trade_decision": False,
        "report_changed": False,
        "portfolio_replay_executed": False,
    }


def _manifest(files: list[Path], readiness: dict[str, Any]) -> dict[str, Any]:
    return {
        "task_id": TASK_ID,
        "output_dir": str(OUTPUT_DIR),
        "inputs": {
            "legacy_p1_trace": str(LEGACY_P1_TRACE),
            "prior_policy_trace": str(PRIOR_POLICY_TRACE),
            "c2_state_machine": str(C2_STATE_MACHINE),
            "previous_cost_design": str(PREV_COST_DESIGN),
        },
        "artifacts": [
            {"path": str(path), "sha256": _sha256(path), "bytes": path.stat().st_size}
            for path in files
        ],
        "readiness": readiness,
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    trace = _load_ordinary_trace()
    candidates = _consensus_candidates(trace)
    prior = _prior_single_exception()
    c2 = _c2_gate()
    contract = _add_shortfall_slots(_build_contract(candidates, prior, c2))
    match_audit = _match_audit(contract, prior, c2)
    coverage = _coverage(contract)
    blocked = _blocked_ledger(contract, match_audit)
    cost = pd.read_csv(PREV_COST_DESIGN, low_memory=False)
    future = pd.DataFrame(
        [
            {
                "audit_item": "candidate_ranking",
                "future_return_used_as_rule": False,
                "rule_source": "same-date route consensus_count only",
                "future_data_violation_count": 0,
            },
            {
                "audit_item": "ohlc_path",
                "future_return_used_as_rule": False,
                "rule_source": "evaluation/path metadata only",
                "future_data_violation_count": 0,
            },
        ]
    )
    readiness = _readiness(contract, match_audit, coverage, blocked)

    paths = {
        "contract": OUTPUT_DIR / "p1_c2_exact_consensus4_top5_exception_candidate_contract.csv",
        "rank_audit": OUTPUT_DIR / "p1_c2_exact_consensus4_top5_rank_score_audit.csv",
        "match_audit": OUTPUT_DIR / "p1_c2_exact_consensus4_top1_match_audit.csv",
        "coverage": OUTPUT_DIR / "p1_c2_exact_consensus4_top5_ohlc_path_coverage.csv",
        "cost": OUTPUT_DIR / "p1_c2_exact_consensus4_top5_transition_cost_fields.csv",
        "blocked": OUTPUT_DIR / "p1_c2_exact_consensus4_top5_blocked_ledger.csv",
        "future": OUTPUT_DIR / "p1_c2_exact_consensus4_top5_future_data_audit.csv",
        "readiness": OUTPUT_DIR / "readiness_for_p1_c2_exact_consensus4_top5_exception_diagnostic.json",
        "summary": OUTPUT_DIR / "final_summary_zh.md",
        "manifest": OUTPUT_DIR / "manifest.json",
    }
    contract.to_csv(paths["contract"], index=False, encoding="utf-8-sig")
    contract[
        [
            "signal_date",
            "candidate_rank",
            "ticker",
            "consensus_count",
            "route_count",
            "candidate_score",
            "route_source_flags",
            "variant_source_flags",
            "rank_source",
        ]
    ].to_csv(paths["rank_audit"], index=False, encoding="utf-8-sig")
    match_audit.to_csv(paths["match_audit"], index=False, encoding="utf-8-sig")
    coverage.to_csv(paths["coverage"], index=False, encoding="utf-8-sig")
    cost.to_csv(paths["cost"], index=False, encoding="utf-8-sig")
    blocked.to_csv(paths["blocked"], index=False, encoding="utf-8-sig")
    future.to_csv(paths["future"], index=False, encoding="utf-8-sig")
    paths["readiness"].write_text(json.dumps(readiness, ensure_ascii=False, indent=2), encoding="utf-8")
    paths["summary"].write_text(
        "\n".join(
            [
                "# P1 C2 exact consensus4 top5 exception candidate contract",
                "",
                "- 本輪已修正前一輪偏差：top1~top5 直接由原 consensus4 多 route 同股確認邏輯重建，不使用 Layer4 generic top5 proxy。",
                f"- exact top1 對齊 prior single consensus4 exception share = {readiness['exact_top1_matches_prior_single_exception_share']:.4f}。",
                f"- exact top1 對齊 C2 allowed prior exception share = {readiness['exact_top1_matches_prior_c2_allowed_exception_share']:.4f}。",
                f"- C2=true rank<=5 official unadjusted OHLC ready share = {readiness['official_unadjusted_ohlc_ready_share']:.4f}。",
                "- adjusted_close_ready=false；official unadjusted OHLC 只能作 diagnostic path，不可 formal。",
                "- 後續 Experiments 主結論必須 net after transaction cost；gross/no-cost 只能 secondary。",
                "- ready_for_formal=false；ready_for_strategy_replay=false。",
                "",
                "下一棒：交 Experiments rerun exact consensus4 top1~top5 multi-stock exception diagnostic。",
                "",
                "Flags: formal_model_changed=false; trade_decision_changed=false; active_in_trade_decision=false; report_changed=false; portfolio_replay_executed=false; ready_for_strategy_replay=false; ready_for_formal=false; not_live_rule=true; forward_returns_live_rule_usage=false.",
                "",
                "完成後如果下一棒明確，請直接指派下一個 thread；如果下一棒不明確，請回報 Strategy Center 判斷。不要完成後停住不回報。",
            ]
        ),
        encoding="utf-8",
    )
    manifest = _manifest([p for key, p in paths.items() if key != "manifest"], readiness)
    paths["manifest"].write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(readiness, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
