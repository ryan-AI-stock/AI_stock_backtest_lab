from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS_ROOT = Path("C:/Users/zergv/Documents/Codex/2026-07-06/backtest-lab-experiments-diagnostic-validation-attribution")
P1_LEGACY_TRACE = (
    EXPERIMENTS_ROOT
    / "outputs"
    / "vnext_p1_legacy_regime_unadjusted_ohlc_cost_timing_diagnostic_20260708"
    / "p1_legacy_regime_unadjusted_ohlc_trade_path_trace.csv"
)
P1_EXACT = (
    REPO_ROOT
    / "outputs"
    / "vnext_p1_c2_exact_consensus4_top5_exception_candidate_contract_20260708"
    / "p1_c2_exact_consensus4_top5_exception_candidate_contract.csv"
)
FULL_ROUTE_TRACE = (
    REPO_ROOT
    / "outputs"
    / "vnext_full_period_regime_switch_benchmark_exception_path_20260708"
    / "full_period_regime_switch_route_signal_trace.csv"
)
OUTPUT_DIR = REPO_ROOT / "outputs" / "vnext_full_period_exact_consensus_trigger_contract_20260708"

TASK_ID = "TASK-BACKTEST-CORE-VNEXT-FULL-PERIOD-EXACT-CONSENSUS-TRIGGER-CONTRACT-001"
PRIMARY_TIMING = "next_day_close_entry_fixed_5td_exit"
SOURCE_VARIANTS = [
    "hybrid_pullback_base_mega_override",
    "conservative_hurdle_route",
    "pool_breadth_route",
    "market_bias_pool_trend_route",
    "dispersion_route",
]
PERIODS = {
    "P1": ("2015-01-02", "2022-12-29"),
    "P2": ("2023-01-02", "2026-06-30"),
    "2024_latest": ("2024-01-02", "2026-06-30"),
    "2026YTD": ("2026-01-02", "2026-06-30"),
}
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


def _ticker(value: Any) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    return text.zfill(4) if text.isdigit() and len(text) < 4 else text


def _as_bool(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s
    return s.astype(str).str.lower().isin(["true", "1", "yes"])


def _period_flags(date_text: str) -> dict[str, bool]:
    date = pd.Timestamp(date_text)
    return {f"in_{period}": pd.Timestamp(start) <= date <= pd.Timestamp(end) for period, (start, end) in PERIODS.items()}


def _period_label(date_text: str) -> str:
    flags = _period_flags(date_text)
    labels = [key[3:] for key, active in flags.items() if active]
    return "|".join(labels) if labels else "outside_requested_periods"


def _p1_source_matrix() -> pd.DataFrame:
    df = pd.read_csv(P1_LEGACY_TRACE, low_memory=False, dtype={"ticker": str})
    df["signal_date"] = pd.to_datetime(df["signal_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    df["ticker"] = df["ticker"].map(_ticker)
    df["path_ready"] = _as_bool(df["path_ready"])
    df = df[
        df["timing_variant"].eq(PRIMARY_TIMING)
        & df["path_bucket"].eq("ordinary_stock")
        & df["variant"].isin(SOURCE_VARIANTS)
        & df["path_ready"]
    ].copy()
    out = df.rename(columns={"variant": "source_variant", "route_or_mode": "route_mode"})[
        ["signal_date", "source_variant", "route_mode", "ticker", "path_ready"]
    ].drop_duplicates(["signal_date", "source_variant", "ticker"])
    out["period_label"] = out["signal_date"].map(_period_label)
    out["candidate_ticker"] = out["ticker"]
    out["source_available"] = True
    out["candidate_is_stock"] = out["candidate_ticker"].ne("")
    out["source_family"] = "p1_legacy_route_trace"
    out["exact_trigger_source_quality"] = "exact_p1_legacy_source_variant_trace"
    return out.drop(columns=["ticker"])


def _full_period_source_matrix() -> pd.DataFrame:
    df = pd.read_csv(FULL_ROUTE_TRACE, low_memory=False, dtype={"ticker": str})
    df["signal_date"] = pd.to_datetime(df["route_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    df["ticker"] = df["ticker"].map(_ticker)
    df = df[df["route_variant"].isin(SOURCE_VARIANTS)].copy()
    df = df[~df["period_label"].astype(str).eq("P1")].copy()
    out = df.rename(columns={"route_variant": "source_variant", "selection_reason": "route_mode"})[
        ["signal_date", "period_label", "source_variant", "route_mode", "ticker", "exposure_flag", "path_source_status"]
    ].drop_duplicates(["signal_date", "source_variant"])
    out["candidate_ticker"] = out["ticker"].where(out["exposure_flag"].astype(str).eq("stock"), "")
    out["source_available"] = True
    out["candidate_is_stock"] = out["candidate_ticker"].ne("")
    out["source_family"] = "full_period_regime_switch_route_signal_trace"
    out["exact_trigger_source_quality"] = "exact_full_period_source_variant_trace"
    return out.drop(columns=["ticker"])


def _complete_matrix(matrix: pd.DataFrame) -> pd.DataFrame:
    dates = sorted(matrix["signal_date"].dropna().unique())
    slots = pd.MultiIndex.from_product([dates, SOURCE_VARIANTS], names=["signal_date", "source_variant"]).to_frame(index=False)
    out = slots.merge(matrix, on=["signal_date", "source_variant"], how="left")
    out["period_label"] = out["period_label"].fillna(out["signal_date"].map(_period_label))
    out["candidate_ticker"] = out["candidate_ticker"].fillna("")
    out["candidate_is_stock"] = out["candidate_ticker"].ne("")
    out["source_available"] = out["source_available"].fillna(False).astype(bool)
    out["source_family"] = out["source_family"].fillna("missing_source_variant")
    out["exact_trigger_source_quality"] = out["exact_trigger_source_quality"].fillna("missing_source_variant")
    out["route_mode"] = out["route_mode"].fillna("")
    for period in PERIODS:
        out[f"in_{period}"] = out["signal_date"].map(lambda x, p=period: _period_flags(x)[f"in_{p}"])
    out["future_return_used"] = False
    out["diagnostic_only"] = True
    for key, value in FLAGS.items():
        out[key] = value
    return out.sort_values(["signal_date", "source_variant"])


def _contract(matrix: pd.DataFrame) -> pd.DataFrame:
    stock = matrix[matrix["source_available"] & matrix["candidate_is_stock"]].copy()
    grouped = (
        stock.groupby(["signal_date", "candidate_ticker"], as_index=False)
        .agg(
            consensus_count=("source_variant", "nunique"),
            route_count=("route_mode", "nunique"),
            trigger_source_variants=("source_variant", lambda x: "|".join(sorted(set(map(str, x.dropna()))))),
            trigger_route_modes=("route_mode", lambda x: "|".join(sorted(set(map(str, x.dropna()))))),
        )
    )
    grouped["period_label"] = grouped["signal_date"].map(_period_label)
    for period in PERIODS:
        grouped[f"in_{period}"] = grouped["signal_date"].map(lambda x, p=period: _period_flags(x)[f"in_{p}"])
    source_counts = matrix.groupby("signal_date", as_index=False).agg(
        source_variant_available_count=("source_available", "sum"),
        source_variant_expected_count=("source_variant", "size"),
    )
    grouped = grouped.merge(source_counts, on="signal_date", how="left")
    grouped["exact_trigger_pass"] = grouped["consensus_count"] >= 4
    grouped["candidate_score"] = grouped["consensus_count"] * 100 + grouped["route_count"]
    grouped = grouped.sort_values(["signal_date", "candidate_score", "candidate_ticker"], ascending=[True, False, True])
    grouped["candidate_rank"] = grouped.groupby("signal_date").cumcount() + 1
    grouped["rank_source"] = "exact_same_ticker_consensus_count_then_route_count_then_ticker"
    grouped["trigger_definition"] = "same ticker selected by >=4 of 5 PIT source variants"
    grouped["proxy_trigger_used"] = False
    grouped["future_return_used"] = False
    grouped["diagnostic_only"] = True
    for key, value in FLAGS.items():
        grouped[key] = value
    cols = [
        "signal_date", "period_label", "candidate_rank", "candidate_ticker", "exact_trigger_pass",
        "consensus_count", "route_count", "candidate_score", "trigger_source_variants",
        "trigger_route_modes", "source_variant_available_count", "source_variant_expected_count",
        "rank_source", "trigger_definition", "proxy_trigger_used", "future_return_used",
        *[f"in_{p}" for p in PERIODS], "diagnostic_only", *FLAGS.keys(),
    ]
    return grouped[[c for c in cols if c in grouped.columns]]


def _p1_match_audit(contract: pd.DataFrame) -> pd.DataFrame:
    p1_exact = pd.read_csv(P1_EXACT, low_memory=False, dtype={"ticker": str})
    p1_exact = p1_exact[p1_exact["candidate_rank"].eq(1) & p1_exact["candidate_available"].astype(str).str.lower().eq("true")].copy()
    p1_exact["signal_date"] = pd.to_datetime(p1_exact["signal_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    p1_exact["prior_exact_top1_ticker"] = p1_exact["ticker"].map(_ticker)
    ours = contract[contract["candidate_rank"].eq(1) & contract["in_P1"] & contract["exact_trigger_pass"]].copy()
    audit = p1_exact[["signal_date", "prior_exact_top1_ticker", "consensus_count", "route_count"]].merge(
        ours[["signal_date", "candidate_ticker", "consensus_count", "route_count", "exact_trigger_pass"]],
        on="signal_date",
        how="outer",
        suffixes=("_prior", "_rebuilt"),
    )
    audit["rebuilt_matches_prior_exact_top1"] = audit["prior_exact_top1_ticker"].fillna("").eq(audit["candidate_ticker"].fillna(""))
    audit["match_policy"] = "rebuilt full-period exact trigger must reproduce P1 exact consensus4 top1"
    return audit.sort_values("signal_date")


def _missing_source_ledger(matrix: pd.DataFrame) -> pd.DataFrame:
    missing = matrix[~matrix["source_available"]].copy()
    rows = []
    for row in missing.itertuples(index=False):
        rows.append({
            "signal_date": row.signal_date,
            "period_label": row.period_label,
            "source_variant": row.source_variant,
            "blocked_item": "source_variant_missing",
            "blocked_reason": "required exact consensus source variant row is not materialized for this date",
            "policy": "do not substitute route_support score threshold",
        })
    if not rows:
        rows.append({
            "signal_date": "",
            "period_label": "all",
            "source_variant": "",
            "blocked_item": "none",
            "blocked_reason": "all required exact consensus source variant rows present for materialized signal dates",
            "policy": "exact trigger contract can be rebuilt without proxy threshold",
        })
    return pd.DataFrame(rows)


def _proxy_options() -> pd.DataFrame:
    return pd.DataFrame([
        {
            "option_id": "route_support_ge4_proxy",
            "description": "Use route_support_variant_count>=4 as date/candidate proxy.",
            "accepted_as_primary": False,
            "reason": "Strategy Center rejected this as primary full-period same-basis conclusion.",
        },
        {
            "option_id": "exact_same_ticker_consensus_ge4",
            "description": "Use same ticker selected by >=4 of the five PIT source variants.",
            "accepted_as_primary": True,
            "reason": "This matches P1 exact consensus semantics and is rebuilt from source variant matrix.",
        },
    ])


def _readiness(contract: pd.DataFrame, matrix: pd.DataFrame, p1_audit: pd.DataFrame) -> dict[str, Any]:
    p1_match = float(p1_audit["rebuilt_matches_prior_exact_top1"].mean()) if len(p1_audit) else 0.0
    p2_matrix = matrix[matrix["in_P2"]]
    recent_matrix = matrix[matrix["in_2024_latest"] | matrix["in_2026YTD"]]
    p2_ready = bool(p2_matrix.groupby("signal_date")["source_available"].sum().eq(len(SOURCE_VARIANTS)).all()) if len(p2_matrix) else False
    recent_ready = bool(recent_matrix.groupby("signal_date")["source_available"].sum().eq(len(SOURCE_VARIANTS)).all()) if len(recent_matrix) else False
    ready = p1_match == 1.0 and p2_ready and recent_ready
    return {
        "task_id": TASK_ID,
        "status": "full_period_exact_consensus_trigger_ready" if ready else "full_period_exact_consensus_trigger_partial_or_blocked",
        "ready_for_route_support_max1_full_period_exact_same_basis_contract_refresh": bool(ready),
        "p1_exact_trigger_match_share": p1_match,
        "p2_exact_trigger_ready": p2_ready,
        "recent_exact_trigger_ready": recent_ready,
        "proxy_only": False,
        "ready_for_experiments": False,
        "ready_for_formal": False,
        "ready_for_strategy_replay": False,
        "contract_rows": int(len(contract)),
        "exact_trigger_pass_rows": int(contract["exact_trigger_pass"].sum()) if len(contract) else 0,
        "source_matrix_rows": int(len(matrix)),
        "missing_source_variant_rows": int((~matrix["source_available"]).sum()),
        "future_data_violation_count": 0,
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
            "p1_legacy_trace": str(P1_LEGACY_TRACE),
            "p1_exact_reference": str(P1_EXACT),
            "full_route_trace": str(FULL_ROUTE_TRACE),
        },
        "artifacts": [{"path": str(path), "sha256": _sha256(path), "bytes": path.stat().st_size} for path in files],
        "readiness": readiness,
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    source_matrix = _complete_matrix(pd.concat([_p1_source_matrix(), _full_period_source_matrix()], ignore_index=True, sort=False))
    contract = _contract(source_matrix)
    p1_audit = _p1_match_audit(contract)
    missing = _missing_source_ledger(source_matrix)
    proxy = _proxy_options()
    readiness = _readiness(contract, source_matrix, p1_audit)
    paths = {
        "contract": OUTPUT_DIR / "full_period_exact_consensus_trigger_contract.csv",
        "matrix": OUTPUT_DIR / "full_period_exact_consensus_trigger_source_variant_matrix.csv",
        "p1_audit": OUTPUT_DIR / "full_period_exact_consensus_trigger_p1_match_audit.csv",
        "missing": OUTPUT_DIR / "full_period_exact_consensus_trigger_missing_source_ledger.csv",
        "proxy": OUTPUT_DIR / "full_period_exact_consensus_trigger_proxy_alternative_options.csv",
        "readiness": OUTPUT_DIR / "readiness_for_full_period_exact_consensus_trigger_contract.json",
        "summary": OUTPUT_DIR / "final_summary_zh.md",
        "manifest": OUTPUT_DIR / "manifest.json",
    }
    contract.to_csv(paths["contract"], index=False, encoding="utf-8-sig")
    source_matrix.to_csv(paths["matrix"], index=False, encoding="utf-8-sig")
    p1_audit.to_csv(paths["p1_audit"], index=False, encoding="utf-8-sig")
    missing.to_csv(paths["missing"], index=False, encoding="utf-8-sig")
    proxy.to_csv(paths["proxy"], index=False, encoding="utf-8-sig")
    paths["readiness"].write_text(json.dumps(readiness, ensure_ascii=False, indent=2), encoding="utf-8")
    paths["summary"].write_text(
        "\n".join([
            "# full-period exact consensus trigger contract",
            "",
            "- P1 exact consensus trigger 已明文化：同一 ticker 被五個 PIT source variants 中至少四個指向，即 exact_same_ticker_consensus_ge4。",
            "- 排名依序為 consensus_count、route_count、ticker；不使用 future return，也不使用 route_support score threshold。",
            f"- p1_exact_trigger_match_share = {readiness['p1_exact_trigger_match_share']:.4f}。",
            f"- p2_exact_trigger_ready = {str(readiness['p2_exact_trigger_ready']).lower()}；recent_exact_trigger_ready = {str(readiness['recent_exact_trigger_ready']).lower()}。",
            "- route_support_ge4_proxy 保留為 rejected secondary option，不作 primary。",
            "- 若 readiness true，下一步應刷新 route_support max1 full-period same-basis state-machine contract，再交 Experiments。",
            "",
            "Flags: formal_model_changed=false; trade_decision_changed=false; active_in_trade_decision=false; report_changed=false; portfolio_replay_executed=false; ready_for_strategy_replay=false; ready_for_formal=false; not_live_rule=true; forward_returns_live_rule_usage=false.",
            "",
            "完成後如果下一棒明確，請直接指派下一個 thread；如果下一棒不明確，請回報 Strategy Center 判斷。不要完成後停住不回報。",
        ]),
        encoding="utf-8",
    )
    manifest = _manifest([p for k, p in paths.items() if k != "manifest"], readiness)
    paths["manifest"].write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(readiness, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
