"""Build the Dynamic Pool1 v2 bounded portfolio challenger contract package."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from backtest_lab.costs import COST_MODEL_VERSION, cost_model_metadata


TASK_ID = "TASK-BACKTEST-CORE-DYNAMIC-POOL1-V2-BOUNDED-PORTFOLIO-CONTRACT-001"
EXPERIMENTS_TASK_ID = "TASK-BACKTEST-EXPERIMENTS-DYNAMIC-POOL1-V2-BOUNDED-PORTFOLIO-VALIDATION-001"
DEFAULT_V2_MEMBER_PANEL = Path(
    r"C:\Users\zergv\Documents\Codex\2026-06-17\repo-ai-stock-backtest-lab-repo\outputs"
    r"\experiments_dynamic_pool1_benchmark_aware_candidate_pool_v2_quality_diagnostic_20260704"
    r"\candidate_pool_v2_member_panel.csv"
)
DEFAULT_OUTPUT_DIR = Path("outputs/dynamic_pool1_v2_bounded_portfolio_contract_20260704")
DEFAULT_CANDIDATE_V0_POOL = Path("outputs/dynamic_pool1_candidate_panel_v0_20260704/candidate_pool_by_month.csv")
DEFAULT_RADAR_LIQUIDITY_DIR = Path(
    r"C:\Users\zergv\Documents\Codex\2026-05-23\ai-stock-rotation-radar-https-docs\outputs"
    r"\radar_dynamic_pool1_all_listed_liquid_universe_full_sweep_20260703"
)
FORMAL_STREAMS = [
    Path("outputs/combined_formal_target_stream_20150128_20211230_20260702/combined_formal_target_stream.csv"),
    Path("outputs/formal_long_range_signal_reconstruction_201411_latest_20260702/formal_long_range_target_stream.csv"),
]
BENCHMARK_PRICE_PATHS = {
    "0050": Path("backtest_cache/stock_pool_observations/0050_TW.csv"),
    "00631L": Path("backtest_cache/stock_pool_observations/00631L_TW.csv"),
}


VARIANTS = [
    {
        "dynamic_pool_variant": "v2_top15_top1_when_formal_cash_or_market_exposure_hold20",
        "source_variant_id": "v2_primary_rs60_top15_monthly",
        "variant_role": "primary",
        "top_n": 1,
        "dynamic_sleeve_weight": 0.20,
        "hold_days": 20,
        "report_only_reference": False,
    },
    {
        "dynamic_pool_variant": "v2_top15_top3_equal_weight_when_formal_cash_or_market_exposure_hold20",
        "source_variant_id": "v2_primary_rs60_top15_monthly",
        "variant_role": "diversification_sensitivity",
        "top_n": 3,
        "dynamic_sleeve_weight": 0.20,
        "hold_days": 20,
        "report_only_reference": False,
    },
    {
        "dynamic_pool_variant": "v2_top10_top1_when_formal_cash_or_market_exposure_hold20",
        "source_variant_id": "v2_primary_rs60_top10_monthly",
        "variant_role": "pool_size_sensitivity",
        "top_n": 1,
        "dynamic_sleeve_weight": 0.20,
        "hold_days": 20,
        "report_only_reference": False,
    },
    {
        "dynamic_pool_variant": "v2_broad_watchlist_no_trade_context",
        "source_variant_id": "v2_broad_watchlist_rs60_all",
        "variant_role": "report_only_watchlist_context",
        "top_n": 0,
        "dynamic_sleeve_weight": 0.0,
        "hold_days": 0,
        "report_only_reference": True,
    },
]


def run_dynamic_pool1_v2_bounded_portfolio_contract(
    *,
    repo_root: str | Path = ".",
    v2_member_panel: str | Path = DEFAULT_V2_MEMBER_PANEL,
    candidate_v0_pool: str | Path = DEFAULT_CANDIDATE_V0_POOL,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    liquidity_calendar_dir: str | Path = DEFAULT_RADAR_LIQUIDITY_DIR,
) -> dict:
    root = Path(repo_root).resolve()
    panel_path = _resolve(root, v2_member_panel)
    v0_pool_path = _resolve(root, candidate_v0_pool)
    liquidity_dir = _resolve(root, liquidity_calendar_dir)
    output = _resolve(root, output_dir)
    output.mkdir(parents=True, exist_ok=True)

    market_lookup = _load_market_lookup(v0_pool_path, liquidity_dir)
    member_panel = _load_member_panel(panel_path, market_lookup)
    monthly_candidates = _build_monthly_candidates(member_panel)
    formal = _load_formal_streams(root)
    trading_calendar = _load_local_trading_calendar(root, liquidity_dir)
    signal_panel = _build_daily_signal_panel(formal, monthly_candidates, trading_calendar, liquidity_dir)

    weight_ledger = _build_weight_ledger(signal_panel, monthly_candidates)
    trade_ledger = _build_trade_ledger(weight_ledger)
    cash_ledger = _build_cash_ledger(signal_panel)
    next_day_audit = _build_next_day_execution_audit(signal_panel)
    blocked_fill_audit = signal_panel[signal_panel["dynamic_blocked_reason"].astype(str) != ""].copy()
    cost_ledger = _build_cost_ledger(weight_ledger)
    benchmark_panel = _build_benchmark_comparison_panel(root, signal_panel)
    traceability = _build_candidate_traceability(weight_ledger, monthly_candidates)
    future_data_audit = _build_future_data_audit(signal_panel, monthly_candidates)
    variant_matrix = _build_variant_matrix()
    price_mapping_audit = _build_price_mapping_audit(member_panel)

    signal_panel.to_csv(output / "daily_signal_panel.csv", index=False, encoding="utf-8-sig")
    weight_ledger.to_csv(output / "portfolio_weight_ledger.csv", index=False, encoding="utf-8-sig")
    trade_ledger.to_csv(output / "trade_ledger.csv", index=False, encoding="utf-8-sig")
    cash_ledger.to_csv(output / "cash_ledger.csv", index=False, encoding="utf-8-sig")
    next_day_audit.to_csv(output / "next_day_execution_audit.csv", index=False, encoding="utf-8-sig")
    blocked_fill_audit.to_csv(output / "blocked_fill_audit.csv", index=False, encoding="utf-8-sig")
    cost_ledger.to_csv(output / "cost_ledger.csv", index=False, encoding="utf-8-sig")
    benchmark_panel.to_csv(output / "benchmark_comparison_panel.csv", index=False, encoding="utf-8-sig")
    traceability.to_csv(output / "candidate_source_traceability.csv", index=False, encoding="utf-8-sig")
    future_data_audit.to_csv(output / "future_data_audit.csv", index=False, encoding="utf-8-sig")
    price_mapping_audit.to_csv(output / "price_mapping_audit.csv", index=False, encoding="utf-8-sig")
    variant_matrix.to_csv(output / "portfolio_variant_matrix.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([cost_model_metadata()]).to_csv(output / "cost_model_contract.csv", index=False, encoding="utf-8-sig")

    active_signal = signal_panel[~signal_panel["report_only_reference"]]
    fillable = active_signal["dynamic_blocked_reason"].eq("")
    candidate_available = active_signal["dynamic_selected_ticker"].fillna("").astype(str).ne("")
    price_ready = active_signal.loc[candidate_available, "candidate_price_ready_all"].fillna(False)
    benchmark_ready = active_signal.loc[candidate_available, "benchmark_ready_all"].fillna(False)
    manifest = {
        "task_id": TASK_ID,
        "status": "completed_v2_bounded_portfolio_contract_ready",
        "output_dir": str(output),
        "v2_member_panel_source": str(panel_path),
        "candidate_v0_pool_source": str(v0_pool_path),
        "daily_signal_rows": int(len(signal_panel)),
        "portfolio_weight_rows": int(len(weight_ledger)),
        "trade_contract_rows": int(len(trade_ledger)),
        "blocked_fill_rows": int(len(blocked_fill_audit)),
        "market_mapping_blocked_rows": int(member_panel["market_mapping_blocked_reason"].astype(str).ne("").sum()),
        "price_mapping_audit_rows": int(len(price_mapping_audit)),
        "next_tradable_date_blocked_rows": int(
            next_day_audit["calendar_status"].astype(str).str.startswith("blocked").sum()
        ),
        "execution_calendar_adjusted_rows": int(next_day_audit["calendar_adjusted"].fillna(False).sum()),
        "candidate_price_availability_rate": round(float(price_ready.mean()) if len(price_ready) else 0.0, 6),
        "benchmark_availability_rate": round(float(benchmark_ready.mean()) if len(benchmark_ready) else 0.0, 6),
        "candidate_available_active_rows": int(candidate_available.sum()),
        "fill_contract_ready_rows": int(fillable.sum()),
        "variant_count": int(len(variant_matrix)),
        "cost_model_version": COST_MODEL_VERSION,
        "uses_forward_return_as_rule": False,
        "uses_cross_section_median_as_primary_benchmark": False,
        "same_day_execution_mixed": False,
        "formal_direct_stock_target_override_allowed": False,
        "portfolio_replay_executed": False,
        "diagnostic_challenger_only": True,
        "formal_model_changed": False,
        "trade_decision_changed": False,
        "active_in_trade_decision": False,
        "report_changed": False,
        "future_data_violation_count": int(future_data_audit["future_data_violation"].sum()),
        "ready_for_experiments_validation": True,
        "handoff_to_experiments_task": EXPERIMENTS_TASK_ID,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "readiness_for_experiments.json").write_text(
        json.dumps(_readiness(manifest), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output / "final_summary_zh.md").write_text(_summary(manifest), encoding="utf-8")
    pd.DataFrame([{"task_id": TASK_ID, "status": "completed", "output_dir": str(output)}]).to_csv(
        output / "completed.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(columns=["task_id", "status", "reason"]).to_csv(output / "failed.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(
        [
            {"step": "load_v2_member_panel", "status": "completed"},
            {"step": "load_formal_streams", "status": "completed"},
            {"step": "asof_join_monthly_candidates", "status": "completed"},
            {"step": "write_contract_ledgers", "status": "completed"},
        ]
    ).to_csv(output / "run_log.csv", index=False, encoding="utf-8-sig")
    return manifest


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _load_member_panel(path: Path, market_lookup: pd.DataFrame) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    rename = {
        "ret_60d_vs_00631L_trailing": "ret_60d_vs_00631l_trailing",
        "ret_20d_vs_00631L_trailing": "ret_20d_vs_00631l_trailing",
    }
    df = df.rename(columns=rename).copy()
    required = {
        "candidate_month",
        "candidate_as_of_date",
        "ticker",
        "candidate_rank",
        "candidate_score",
        "candidate_layer",
        "price_ready_flag",
        "benchmark_0050_ready_flag",
        "benchmark_00631l_ready_flag",
        "ret_60d_vs_0050_trailing",
        "ret_60d_vs_00631l_trailing",
        "ret_20d_vs_0050_trailing",
        "ret_20d_vs_00631l_trailing",
        "rs60_positive_vs_both",
        "rs20_and_rs60_positive_vs_both",
        "top10_and_rs60_positive_vs_both",
        "benchmark_blocked_reason",
        "uses_cross_section_median_as_primary_benchmark",
        "forward_return_used_as_contract_rule",
        "variant_id",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing v2 member panel fields: {missing}")
    df["candidate_as_of_date"] = pd.to_datetime(df["candidate_as_of_date"], errors="coerce")
    df["candidate_rank"] = pd.to_numeric(df["candidate_rank"], errors="coerce")
    df["candidate_score"] = pd.to_numeric(df["candidate_score"], errors="coerce")
    df["ticker"] = df["ticker"].astype(str).map(_ticker_norm)
    for col in [
        "price_ready_flag",
        "benchmark_0050_ready_flag",
        "benchmark_00631l_ready_flag",
        "rs60_positive_vs_both",
        "rs20_and_rs60_positive_vs_both",
        "top10_and_rs60_positive_vs_both",
        "uses_cross_section_median_as_primary_benchmark",
        "forward_return_used_as_contract_rule",
    ]:
        df[col] = df[col].map(_as_bool)
    df = df.dropna(subset=["candidate_as_of_date", "candidate_rank"]).copy()
    df["candidate_month"] = df["candidate_month"].astype(str)
    df = df.merge(market_lookup, on=["candidate_month", "ticker"], how="left")
    df["ticker_market"] = df["market"].fillna("").astype(str)
    df["canonical_ticker"] = df.apply(lambda row: _canonical_ticker(row["ticker"], row["ticker_market"]), axis=1)
    df["price_source_cache_key"] = df["canonical_ticker"]
    df["market_mapping_blocked_reason"] = df["ticker_market"].map(
        lambda value: "" if value in {"TWSE", "TPEx"} else "blocked_missing_market_for_canonical_ticker"
    )
    return df


def _load_market_lookup(candidate_v0_pool: Path, liquidity_dir: Path) -> pd.DataFrame:
    frames = []
    if candidate_v0_pool.exists():
        v0 = pd.read_csv(candidate_v0_pool, usecols=lambda col: col in {"year_month", "ticker", "market"})
        v0 = v0.rename(columns={"year_month": "candidate_month"})
        v0["ticker"] = v0["ticker"].astype(str).map(_ticker_norm)
        v0["candidate_month"] = v0["candidate_month"].astype(str)
        v0["market"] = v0["market"].astype(str)
        frames.append(v0[["candidate_month", "ticker", "market"]])
    shard_dir = liquidity_dir / "shards"
    if shard_dir.exists():
        shard_frames = []
        for shard in sorted(shard_dir.glob("accepted_liquidity_rows_*.csv")):
            try:
                liquidity = pd.read_csv(shard, usecols=["date", "ticker", "market"])
            except (OSError, ValueError):
                continue
            liquidity["candidate_month"] = pd.to_datetime(liquidity["date"], errors="coerce").dt.to_period("M").astype(str)
            liquidity["ticker"] = liquidity["ticker"].astype(str).map(_ticker_norm)
            liquidity["market"] = liquidity["market"].astype(str)
            shard_frames.append(liquidity[["candidate_month", "ticker", "market"]].drop_duplicates())
        if shard_frames:
            frames.append(pd.concat(shard_frames, ignore_index=True, sort=False))
    if not frames:
        return pd.DataFrame(columns=["candidate_month", "ticker", "market"])
    lookup = pd.concat(frames, ignore_index=True, sort=False)
    lookup = lookup[lookup["market"].isin(["TWSE", "TPEx"])]
    return lookup.drop_duplicates(["candidate_month", "ticker"], keep="last")


def _build_monthly_candidates(member_panel: pd.DataFrame) -> dict[str, pd.DataFrame]:
    out = {}
    for variant in VARIANTS:
        source = variant["source_variant_id"]
        selected = member_panel[member_panel["variant_id"].eq(source)].copy()
        selected = selected.sort_values(["candidate_as_of_date", "candidate_rank", "ticker"])
        out[source] = selected
    return out


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
        raise FileNotFoundError("No formal stream source found")
    formal = pd.concat(frames, ignore_index=True, sort=False)
    formal["signal_date"] = pd.to_datetime(formal["signal_date"], errors="coerce")
    formal = formal.dropna(subset=["signal_date"]).sort_values("signal_date")
    formal = formal.drop_duplicates("signal_date", keep="last")
    formal["date"] = formal["signal_date"].dt.strftime("%Y-%m-%d")
    formal["candidate_lookup_date"] = formal["signal_date"]
    formal["formal_target"] = formal.get("formal_target", "").fillna("").astype(str)
    formal["target_type"] = formal.get("target_type", "").fillna("").astype(str)
    formal["risk_off_state"] = formal.get("risk_off_state", "").fillna("").astype(str)
    return formal


def _load_local_trading_calendar(root: Path, liquidity_calendar_dir: Path) -> list[str]:
    dates: set[str] = set()
    shard_dir = liquidity_calendar_dir / "shards"
    if shard_dir.exists():
        for shard in sorted(shard_dir.glob("accepted_liquidity_rows_*.csv")):
            try:
                df = pd.read_csv(shard, usecols=["date"])
            except (OSError, ValueError):
                continue
            dates.update(pd.to_datetime(df["date"], errors="coerce").dropna().dt.strftime("%Y-%m-%d").unique())
    if dates:
        return sorted(dates)
    for rel in BENCHMARK_PRICE_PATHS.values():
        path = root / rel
        if not path.exists():
            continue
        try:
            df = pd.read_csv(path, usecols=["date"])
        except (OSError, ValueError):
            continue
        dates.update(pd.to_datetime(df["date"], errors="coerce").dropna().dt.strftime("%Y-%m-%d").unique())
    return sorted(dates)


def _build_daily_signal_panel(
    formal: pd.DataFrame,
    monthly_candidates: dict[str, pd.DataFrame],
    trading_calendar: list[str],
    liquidity_dir: Path,
) -> pd.DataFrame:
    rows = []
    calendar = sorted(set(trading_calendar))
    for formal_row in formal.to_dict(orient="records"):
        signal_date = pd.to_datetime(formal_row["signal_date"])
        raw_next = str(formal_row.get("execution_date", "") or "")
        next_date, calendar_status, calendar_adjusted = _resolve_next_tradable_date(raw_next, calendar)
        formal_state = _formal_state(formal_row)
        for variant in VARIANTS:
            source_panel = monthly_candidates[variant["source_variant_id"]]
            candidate_rows = _latest_candidates_asof(source_panel, signal_date, variant["top_n"])
            selected_tickers = candidate_rows["ticker"].astype(str).tolist()
            selected_canonical_tickers = candidate_rows["canonical_ticker"].fillna("").astype(str).tolist()
            selected_markets = candidate_rows["ticker_market"].fillna("").astype(str).tolist()
            selected_cache_keys = candidate_rows["price_source_cache_key"].fillna("").astype(str).tolist()
            selected_weights = _selected_weights(selected_tickers, variant["dynamic_sleeve_weight"])
            candidate_price_ready_all = bool(candidate_rows["price_ready_flag"].all()) if len(candidate_rows) else False
            benchmark_ready_all = (
                bool(candidate_rows["benchmark_0050_ready_flag"].all() and candidate_rows["benchmark_00631l_ready_flag"].all())
                if len(candidate_rows)
                else False
            )
            market_mapping_ready_all = bool(candidate_rows["market_mapping_blocked_reason"].fillna("").astype(str).eq("").all()) if len(candidate_rows) else False
            active_state = formal_state in {"cash", "no_target_cash", "market_exposure"}
            report_only = bool(variant["report_only_reference"])
            blocked = _dynamic_blocked_reason(
                report_only=report_only,
                active_state=active_state,
                formal_state=formal_state,
                selected_tickers=selected_tickers,
                next_tradable_date=next_date,
                candidate_price_ready_all=candidate_price_ready_all,
                benchmark_ready_all=benchmark_ready_all,
                market_mapping_ready_all=market_mapping_ready_all,
                candidate_rows=candidate_rows,
            )
            rows.append(
                {
                    "date": signal_date.strftime("%Y-%m-%d"),
                    "next_tradable_date": next_date,
                    "raw_next_tradable_date": raw_next,
                    "calendar_status": calendar_status,
                    "calendar_adjusted": calendar_adjusted,
                    "formal_target": formal_row.get("formal_target", ""),
                    "formal_state": formal_state,
                    "dynamic_pool_variant": variant["dynamic_pool_variant"],
                    "dynamic_candidate_pool_month": _pool_month(candidate_rows),
                    "dynamic_candidate_as_of_date": _pool_asof(candidate_rows),
                    "dynamic_candidate_pool_source": variant["source_variant_id"],
                    "dynamic_candidate_pool_count": int(len(candidate_rows)),
                    "dynamic_selected_ticker": ";".join(selected_tickers),
                    "dynamic_selected_canonical_ticker": ";".join(selected_canonical_tickers),
                    "dynamic_selected_ticker_market": ";".join(selected_markets),
                    "dynamic_selected_price_source_cache_key": ";".join(selected_cache_keys),
                    "dynamic_candidate_price_source_path": _price_source_path(liquidity_dir, _pool_asof(candidate_rows)),
                    "dynamic_execution_price_source_path": _price_source_path(liquidity_dir, next_date),
                    "dynamic_selected_weight": ";".join(str(round(w, 8)) for w in selected_weights),
                    "dynamic_selection_reason": _selection_reason(variant, candidate_rows),
                    "dynamic_blocked_reason": blocked,
                    "formal_conflict_state": _formal_conflict_state(formal_state, selected_tickers, formal_row.get("formal_target", "")),
                    "active_only_when_formal_cash_or_market_exposure": True,
                    "candidate_price_ready_all": candidate_price_ready_all,
                    "benchmark_ready_all": benchmark_ready_all,
                    "market_mapping_ready_all": market_mapping_ready_all,
                    "uses_forward_return_as_rule": False,
                    "uses_cross_section_median_as_primary_benchmark": False,
                    "diagnostic_challenger_only": True,
                    "report_only_reference": report_only,
                    "portfolio_replay_executed": False,
                    "formal_model_changed": False,
                    "trade_decision_changed": False,
                    "active_in_trade_decision": False,
                    "report_changed": False,
                    "cost_model_version": COST_MODEL_VERSION,
                }
            )
    return pd.DataFrame(rows)


def _latest_candidates_asof(source_panel: pd.DataFrame, signal_date: pd.Timestamp, top_n: int) -> pd.DataFrame:
    if top_n <= 0:
        return pd.DataFrame(columns=source_panel.columns)
    available = source_panel[source_panel["candidate_as_of_date"].le(signal_date)].copy()
    if available.empty:
        return available
    latest_asof = available["candidate_as_of_date"].max()
    latest = available[available["candidate_as_of_date"].eq(latest_asof)].copy()
    return latest.sort_values(["candidate_rank", "ticker"]).head(top_n)


def _resolve_next_tradable_date(raw: str, calendar: list[str]) -> tuple[str, str, bool]:
    raw_date = pd.to_datetime(raw, errors="coerce")
    if pd.isna(raw_date):
        return "", "blocked_missing_next_tradable_date", False
    raw_text = raw_date.strftime("%Y-%m-%d")
    calendar_set = set(calendar)
    if raw_text in calendar_set:
        return raw_text, "valid_local_trading_date", False
    later = [date for date in calendar if date > raw_text]
    if later:
        return later[0], "adjusted_to_next_available_local_trading_date", True
    return "", "blocked_no_later_local_trading_date", False


def _formal_state(row: dict) -> str:
    target = str(row.get("formal_target", "") or "").strip()
    target_type = str(row.get("target_type", "") or "").strip()
    risk_off = str(row.get("risk_off_state", "") or "").strip()
    if not target:
        return "no_target_cash" if risk_off == "no_target_cash_all" else "cash"
    if target.upper() == "CASH":
        return "no_target_cash" if risk_off == "no_target_cash_all" or target_type == "risk_control_cash" else "cash"
    if _ticker_norm(target) == "00631L" or target_type == "market_exposure":
        return "market_exposure"
    return "direct_stock_target"


def _dynamic_blocked_reason(
    *,
    report_only: bool,
    active_state: bool,
    formal_state: str,
    selected_tickers: list[str],
    next_tradable_date: str,
    candidate_price_ready_all: bool,
    benchmark_ready_all: bool,
    market_mapping_ready_all: bool,
    candidate_rows: pd.DataFrame,
) -> str:
    if report_only:
        return "report_only_watchlist_context_no_trade"
    if not active_state:
        return f"blocked_formal_state_{formal_state}_no_override"
    if not next_tradable_date:
        return "blocked_missing_next_tradable_date"
    if not selected_tickers:
        return "blocked_no_asof_dynamic_candidate_pool"
    if not market_mapping_ready_all:
        reasons = sorted(set(candidate_rows["market_mapping_blocked_reason"].fillna("").astype(str)) - {""})
        return "blocked_price_mapping_not_ready" + (":" + ";".join(reasons) if reasons else "")
    if not candidate_price_ready_all:
        return "blocked_candidate_price_not_ready"
    if not benchmark_ready_all:
        reasons = sorted(set(candidate_rows["benchmark_blocked_reason"].fillna("").astype(str)) - {""})
        return "blocked_benchmark_not_ready" + (":" + ";".join(reasons) if reasons else "")
    return ""


def _build_weight_ledger(signal_panel: pd.DataFrame, monthly_candidates: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    source_by_variant = {item["dynamic_pool_variant"]: item for item in VARIANTS}
    for row in signal_panel.to_dict(orient="records"):
        if row["dynamic_blocked_reason"] or row["report_only_reference"]:
            continue
        variant = source_by_variant[row["dynamic_pool_variant"]]
        source_panel = monthly_candidates[variant["source_variant_id"]]
        candidates = _latest_candidates_asof(source_panel, pd.to_datetime(row["date"]), variant["top_n"])
        weights = _selected_weights(candidates["ticker"].astype(str).tolist(), variant["dynamic_sleeve_weight"])
        for candidate, weight in zip(candidates.to_dict(orient="records"), weights, strict=False):
            rows.append(
                {
                    "date": row["date"],
                    "next_tradable_date": row["next_tradable_date"],
                    "dynamic_pool_variant": row["dynamic_pool_variant"],
                    "ticker": candidate["ticker"],
                    "ticker_market": candidate["ticker_market"],
                    "canonical_ticker": candidate["canonical_ticker"],
                    "price_source_cache_key": candidate["price_source_cache_key"],
                    "candidate_price_source_path": row["dynamic_candidate_price_source_path"],
                    "execution_price_source_path": row["dynamic_execution_price_source_path"],
                    "dynamic_weight": weight,
                    "formal_residual_weight": round(1.0 - variant["dynamic_sleeve_weight"], 8),
                    "candidate_rank": candidate["candidate_rank"],
                    "candidate_score": candidate["candidate_score"],
                    "candidate_layer": candidate["candidate_layer"],
                    "candidate_month": candidate["candidate_month"],
                    "candidate_as_of_date": candidate["candidate_as_of_date"].strftime("%Y-%m-%d"),
                    "entry_basis": "next_day_contract",
                    "exit_contract": f"hold_{variant['hold_days']}_tradable_days_or_experiments_blocked_fill",
                    "diagnostic_challenger_only": True,
                    "active_in_trade_decision": False,
                }
            )
    return pd.DataFrame(rows)


def _build_trade_ledger(weight_ledger: pd.DataFrame) -> pd.DataFrame:
    if weight_ledger.empty:
        return pd.DataFrame(
            columns=[
                "entry_signal_date",
                "entry_execution_date",
                "dynamic_pool_variant",
                "ticker",
                "ticker_market",
                "canonical_ticker",
                "price_source_cache_key",
                "candidate_price_source_path",
                "execution_price_source_path",
                "target_weight",
                "entry_reason",
                "exit_contract",
                "portfolio_replay_executed",
            ]
        )
    out = weight_ledger.rename(
        columns={"date": "entry_signal_date", "next_tradable_date": "entry_execution_date", "dynamic_weight": "target_weight"}
    ).copy()
    out["entry_reason"] = "v2_benchmark_aware_candidate_selected_when_formal_cash_or_market_exposure"
    out["portfolio_replay_executed"] = False
    return out[
        [
            "entry_signal_date",
            "entry_execution_date",
            "dynamic_pool_variant",
            "ticker",
            "ticker_market",
            "canonical_ticker",
            "price_source_cache_key",
            "candidate_price_source_path",
            "execution_price_source_path",
            "target_weight",
            "candidate_rank",
            "candidate_score",
            "entry_reason",
            "exit_contract",
            "portfolio_replay_executed",
        ]
    ]


def _build_cash_ledger(signal_panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for row in signal_panel.to_dict(orient="records"):
        fillable = not row["dynamic_blocked_reason"] and not row["report_only_reference"]
        sleeve = _variant_by_name(row["dynamic_pool_variant"])["dynamic_sleeve_weight"] if fillable else 0.0
        cash_weight = 1.0 - sleeve if row["formal_state"] in {"cash", "no_target_cash"} else 0.0
        rows.append(
            {
                "date": row["date"],
                "next_tradable_date": row["next_tradable_date"],
                "dynamic_pool_variant": row["dynamic_pool_variant"],
                "formal_state": row["formal_state"],
                "dynamic_sleeve_weight": sleeve,
                "cash_weight_after_dynamic_contract": round(cash_weight, 8),
                "cash_ledger_reason": "cash_reduced_by_dynamic_sleeve" if cash_weight < 1 and row["formal_state"] in {"cash", "no_target_cash"} else "no_cash_change",
                "portfolio_replay_executed": False,
            }
        )
    return pd.DataFrame(rows)


def _build_next_day_execution_audit(signal_panel: pd.DataFrame) -> pd.DataFrame:
    return signal_panel[
        [
            "date",
            "raw_next_tradable_date",
            "next_tradable_date",
            "calendar_status",
            "calendar_adjusted",
            "dynamic_pool_variant",
            "dynamic_blocked_reason",
            "uses_forward_return_as_rule",
            "diagnostic_challenger_only",
        ]
    ].copy()


def _build_cost_ledger(weight_ledger: pd.DataFrame) -> pd.DataFrame:
    if weight_ledger.empty:
        return pd.DataFrame(columns=["date", "dynamic_pool_variant", "ticker", "cost_model_version", "cost_contract_boundary"])
    out = weight_ledger[
        [
            "date",
            "next_tradable_date",
            "dynamic_pool_variant",
            "ticker",
            "ticker_market",
            "canonical_ticker",
            "price_source_cache_key",
            "dynamic_weight",
        ]
    ].copy()
    out["asset_type"] = out["ticker"].map(lambda ticker: "etf" if str(ticker).startswith("00") else "stock")
    out["cost_model_version"] = COST_MODEL_VERSION
    out["cost_contract_boundary"] = "Experiments applies Taiwan buy fee, sell fee, and sell tax using formal cost model"
    return out


def _build_benchmark_comparison_panel(root: Path, signal_panel: pd.DataFrame) -> pd.DataFrame:
    base = signal_panel[["date", "next_tradable_date", "dynamic_pool_variant"]].drop_duplicates().copy()
    for label, rel in BENCHMARK_PRICE_PATHS.items():
        path = root / rel
        dates = set()
        source = "missing_local_cache"
        if path.exists():
            prices = pd.read_csv(path, usecols=lambda col: col in {"date", "close", "adj_close"})
            dates = set(pd.to_datetime(prices["date"], errors="coerce").dropna().dt.strftime("%Y-%m-%d"))
            source = str(rel)
        base[f"benchmark_{label.lower()}_price_ready"] = base["next_tradable_date"].isin(dates)
        base[f"benchmark_{label.lower()}_price_source"] = source
    base["uses_cross_section_median_as_primary_benchmark"] = False
    return base


def _build_candidate_traceability(weight_ledger: pd.DataFrame, monthly_candidates: dict[str, pd.DataFrame]) -> pd.DataFrame:
    if weight_ledger.empty:
        return pd.DataFrame()
    source = pd.concat(monthly_candidates.values(), ignore_index=True, sort=False)
    source["candidate_as_of_date_text"] = source["candidate_as_of_date"].dt.strftime("%Y-%m-%d")
    keys = ["ticker", "candidate_month", "candidate_as_of_date_text", "candidate_rank", "candidate_score"]
    trace = weight_ledger.merge(
        source,
        left_on=["ticker", "candidate_month", "candidate_as_of_date", "candidate_rank", "candidate_score"],
        right_on=keys,
        how="left",
        suffixes=("", "_source"),
    )
    keep = [
        "date",
        "dynamic_pool_variant",
        "ticker",
        "ticker_market",
        "canonical_ticker",
        "price_source_cache_key",
        "candidate_month",
        "candidate_as_of_date",
        "candidate_rank",
        "candidate_score",
        "candidate_layer_source",
        "variant_id",
        "variant_role",
        "rs60_positive_vs_both",
        "rs20_and_rs60_positive_vs_both",
        "top10_and_rs60_positive_vs_both",
        "price_ready_flag",
        "benchmark_0050_ready_flag",
        "benchmark_00631l_ready_flag",
        "market_mapping_blocked_reason",
        "uses_cross_section_median_as_primary_benchmark",
        "forward_return_used_as_contract_rule",
    ]
    return trace[[col for col in keep if col in trace.columns]].copy()


def _build_future_data_audit(signal_panel: pd.DataFrame, monthly_candidates: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for row in signal_panel.to_dict(orient="records"):
        asof = pd.to_datetime(row.get("dynamic_candidate_as_of_date"), errors="coerce")
        signal = pd.to_datetime(row.get("date"), errors="coerce")
        violation = bool(pd.notna(asof) and pd.notna(signal) and asof > signal)
        rows.append(
            {
                "date": row["date"],
                "dynamic_pool_variant": row["dynamic_pool_variant"],
                "candidate_as_of_date": row.get("dynamic_candidate_as_of_date", ""),
                "future_data_violation": violation,
                "reason": "candidate_as_of_date_after_signal_date" if violation else "",
            }
        )
    return pd.DataFrame(rows)


def _build_price_mapping_audit(member_panel: pd.DataFrame) -> pd.DataFrame:
    audit = member_panel[
        [
            "candidate_month",
            "ticker",
            "ticker_market",
            "canonical_ticker",
            "price_source_cache_key",
            "variant_id",
            "market_mapping_blocked_reason",
        ]
    ].drop_duplicates()
    audit["price_mapping_ready"] = audit["market_mapping_blocked_reason"].astype(str).eq("")
    return audit.sort_values(["candidate_month", "ticker", "variant_id"])


def _build_variant_matrix() -> pd.DataFrame:
    rows = []
    for variant in VARIANTS:
        rows.append(
            {
                **variant,
                "execution_basis": "next_day_only",
                "exit_contract": f"hold_{variant['hold_days']}_tradable_days" if variant["hold_days"] else "no_trade_context",
                "allowed_formal_states": "cash;no_target_cash;market_exposure",
                "formal_direct_stock_target_override_allowed": False,
                "uses_forward_return_as_rule": False,
                "uses_cross_section_median_as_primary_benchmark": False,
                "formal_model_changed": False,
                "trade_decision_changed": False,
                "active_in_trade_decision": False,
            }
        )
    return pd.DataFrame(rows)


def _readiness(manifest: dict) -> dict:
    return {
        "task_id": TASK_ID,
        "status": "ready_for_experiments_bounded_portfolio_validation",
        "ready_for_experiments_validation": bool(manifest["ready_for_experiments_validation"]),
        "handoff_to_experiments_task": EXPERIMENTS_TASK_ID,
        "candidate_price_availability_rate": manifest["candidate_price_availability_rate"],
        "benchmark_availability_rate": manifest["benchmark_availability_rate"],
        "next_tradable_date_blocked_rows": manifest["next_tradable_date_blocked_rows"],
        "future_data_violation_count": manifest["future_data_violation_count"],
        "portfolio_replay_executed": False,
        "formal_model_changed": False,
        "trade_decision_changed": False,
        "active_in_trade_decision": False,
        "report_changed": False,
    }


def _summary(manifest: dict) -> str:
    return "\n".join(
        [
            "# Dynamic Pool1 v2 bounded portfolio contract",
            "",
            "本包只建立 v2 benchmark-aware candidate pool 的 bounded next-day portfolio challenger contract。",
            "",
            f"- daily signal rows：{manifest['daily_signal_rows']}",
            f"- portfolio weight rows：{manifest['portfolio_weight_rows']}",
            f"- blocked fill rows：{manifest['blocked_fill_rows']}",
            f"- market mapping blocked rows：{manifest['market_mapping_blocked_rows']}",
            f"- price mapping audit rows：{manifest['price_mapping_audit_rows']}",
            f"- candidate price availability rate：{manifest['candidate_price_availability_rate']}",
            f"- benchmark availability rate：{manifest['benchmark_availability_rate']}",
            f"- next tradable date blocked rows：{manifest['next_tradable_date_blocked_rows']}",
            f"- future data violation count：{manifest['future_data_violation_count']}",
            "- 只允許 formal cash / no-target / 00631L market exposure 狀態啟用 dynamic sleeve。",
            "- 不覆蓋 formal direct stock target，不跑 portfolio replay，不改正式模型、交易或日報。",
        ]
    )


def _selected_weights(tickers: list[str], sleeve_weight: float) -> list[float]:
    if not tickers or sleeve_weight <= 0:
        return []
    return [round(sleeve_weight / len(tickers), 8) for _ in tickers]


def _selection_reason(variant: dict, candidate_rows: pd.DataFrame) -> str:
    if variant["report_only_reference"]:
        return "broad_watchlist_context_only_no_trade"
    if candidate_rows.empty:
        return "no_asof_candidate_pool"
    return f"{variant['source_variant_id']}_rank_top{variant['top_n']}_next_day_hold{variant['hold_days']}"


def _pool_month(candidate_rows: pd.DataFrame) -> str:
    return "" if candidate_rows.empty else str(candidate_rows.iloc[0]["candidate_month"])


def _pool_asof(candidate_rows: pd.DataFrame) -> str:
    return "" if candidate_rows.empty else pd.to_datetime(candidate_rows.iloc[0]["candidate_as_of_date"]).strftime("%Y-%m-%d")


def _formal_conflict_state(formal_state: str, selected_tickers: list[str], formal_target: str) -> str:
    if not selected_tickers:
        return "no_dynamic_candidate"
    if formal_state in {"cash", "no_target_cash"}:
        return "no_conflict_formal_cash_or_no_target"
    if formal_state == "market_exposure":
        return "no_conflict_market_exposure"
    formal_norm = _ticker_norm(formal_target)
    if formal_norm in selected_tickers:
        return "same_ticker_no_double_count"
    return "blocked_formal_direct_stock_target_no_override"


def _variant_by_name(name: str) -> dict:
    return next(item for item in VARIANTS if item["dynamic_pool_variant"] == name)


def _canonical_ticker(ticker: str, market: str) -> str:
    suffix = {"TWSE": ".TW", "TPEx": ".TWO"}.get(str(market), "")
    return f"{_ticker_norm(ticker)}{suffix}" if suffix else ""


def _price_source_path(liquidity_dir: Path, date_text: str) -> str:
    parsed = pd.to_datetime(date_text, errors="coerce")
    if pd.isna(parsed):
        return ""
    shard = liquidity_dir / "shards" / f"accepted_liquidity_rows_{parsed.strftime('%Y_%m')}.csv"
    return str(shard)


def _ticker_norm(value: str) -> str:
    text = str(value).strip()
    return text.replace(".TW", "").replace(".TWO", "")


def _as_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if pd.isna(value):
        return False
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--v2-member-panel", default=str(DEFAULT_V2_MEMBER_PANEL))
    parser.add_argument("--candidate-v0-pool", default=str(DEFAULT_CANDIDATE_V0_POOL))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--liquidity-calendar-dir", default=str(DEFAULT_RADAR_LIQUIDITY_DIR))
    args = parser.parse_args(argv)
    manifest = run_dynamic_pool1_v2_bounded_portfolio_contract(
        repo_root=args.repo_root,
        v2_member_panel=args.v2_member_panel,
        candidate_v0_pool=args.candidate_v0_pool,
        output_dir=args.output_dir,
        liquidity_calendar_dir=args.liquidity_calendar_dir,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
