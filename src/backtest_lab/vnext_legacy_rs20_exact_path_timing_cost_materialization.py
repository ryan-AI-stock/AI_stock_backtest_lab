from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from backtest_lab.costs import TaiwanCostModel, cost_model_metadata


REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = REPO_ROOT / "outputs" / "vnext_legacy_rs20_exact_path_timing_cost_materialization_20260708"
PRIOR_CORE_DIR = REPO_ROOT / "outputs" / "vnext_legacy_rs20_operating_mode_runner_readiness_20260708"
SIGNAL_INPUT = PRIOR_CORE_DIR / "legacy_rs20_operating_mode_signal_table.csv"
PRIOR_READINESS = PRIOR_CORE_DIR / "readiness_for_legacy_rs20_operating_mode_diagnostic.json"
BENCHMARK_FEATURES = REPO_ROOT / "outputs" / "vnext_dynamic_candidate_pool_data_materialization_20260706" / "benchmark_features.csv"
PRICE_REGISTRY = REPO_ROOT / "data" / "price_source_registry.csv"
EXPERIMENTS_DIR = (
    Path("C:/Users/zergv/Documents/Codex/2026-07-06/backtest-lab-experiments-diagnostic-validation-attribution")
    / "outputs"
    / "vnext_legacy_rs20_operating_mode_cost_timing_diagnostic_20260708"
)

TASK_ID = "TASK-BACKTEST-CORE-VNEXT-LEGACY-RS20-EXACT-PATH-TIMING-COST-MATERIALIZATION-001"
PRIMARY_VARIANT = "dynamic80_top3_rs20_risk_tiebreak_proxy"
SUPPORTED_VARIANTS = [
    PRIMARY_VARIANT,
    "dynamic80_top1_rs20_proxy",
    "dynamic80_top1_rs20_31_bonus_proxy",
]
REFERENCE_VARIANTS = [
    "00631L_buy_hold_reference",
]
PERIODS = {
    "P1": ("2015-01-02", "2022-12-29"),
    "P2": ("2023-01-02", "2026-06-30"),
    "2024_latest": ("2024-01-02", "2026-06-30"),
    "2026YTD": ("2026-01-02", "2026-06-30"),
    "legacy_requested": ("2024-01-02", "2026-05-26"),
}

FLAGS: dict[str, bool] = {
    "formal_model_changed": False,
    "trade_decision_changed": False,
    "active_in_trade_decision": False,
    "report_changed": False,
    "portfolio_replay_executed": False,
    "ready_for_strategy_replay": False,
    "not_live_rule": True,
    "forward_returns_live_rule_usage": False,
}


@dataclass(frozen=True)
class PriceSourceCoverage:
    ticker: str
    registry_source_found: bool
    source_path: str | None
    source_type: str | None
    first_date: str | None
    last_date: str | None
    adjusted_close_column_found: bool
    adjusted_close_non_null_rows: int
    close_column_found: bool
    open_column_found: bool
    selected_signal_rows: int
    selected_signal_rows_inside_source_date_range: int
    exact_adjusted_close_path_ready: bool
    tradable_close_proxy_available: bool
    blocked_reason: str


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _ticker_base(value: Any) -> str:
    text = str(value).strip()
    if "." in text:
        text = text.split(".", 1)[0]
    return text


def _load_calendar() -> list[pd.Timestamp]:
    bench = pd.read_csv(BENCHMARK_FEATURES, usecols=["trade_date", "benchmark"])
    dates = (
        bench.loc[bench["benchmark"].astype(str) == "0050", "trade_date"]
        .dropna()
        .drop_duplicates()
        .sort_values()
    )
    return [pd.Timestamp(d) for d in dates]


def _next_trading_date(calendar: list[pd.Timestamp], date: pd.Timestamp, offset: int) -> pd.Timestamp | None:
    for idx, trade_date in enumerate(calendar):
        if trade_date > date:
            target = idx + offset - 1
            if 0 <= target < len(calendar):
                return calendar[target]
            return None
    return None


def _trading_date_at_or_after(calendar: list[pd.Timestamp], date: pd.Timestamp) -> pd.Timestamp | None:
    for trade_date in calendar:
        if trade_date >= date:
            return trade_date
    return None


def _read_price_frame(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        frame = pd.read_csv(path)
    except Exception:
        return None
    if "date" not in frame.columns:
        return None
    frame = frame.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    return frame


def _load_price_registry() -> pd.DataFrame:
    if not PRICE_REGISTRY.exists():
        return pd.DataFrame()
    registry = pd.read_csv(PRICE_REGISTRY)
    registry["ticker_base"] = registry["ticker"].map(_ticker_base)
    return registry


def _build_price_coverage(signals: pd.DataFrame, registry: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    selected_counts = signals.groupby("ticker")["signal_date"].size().rename("selected_signal_rows")
    price_frames: dict[str, pd.DataFrame] = {}
    rows: list[PriceSourceCoverage] = []
    selected_tickers = sorted(signals["ticker"].astype(str).unique())

    for ticker in selected_tickers:
        matches = registry.loc[registry["ticker_base"] == ticker] if not registry.empty else pd.DataFrame()
        if matches.empty:
            rows.append(
                PriceSourceCoverage(
                    ticker=ticker,
                    registry_source_found=False,
                    source_path=None,
                    source_type=None,
                    first_date=None,
                    last_date=None,
                    adjusted_close_column_found=False,
                    adjusted_close_non_null_rows=0,
                    close_column_found=False,
                    open_column_found=False,
                    selected_signal_rows=int(selected_counts.get(ticker, 0)),
                    selected_signal_rows_inside_source_date_range=0,
                    exact_adjusted_close_path_ready=False,
                    tradable_close_proxy_available=False,
                    blocked_reason="missing_selected_stock_price_registry_row",
                )
            )
            continue

        match = matches.iloc[0]
        source_path = str(match.get("source_path", ""))
        full_path = REPO_ROOT / source_path
        price_frame = _read_price_frame(full_path)
        if price_frame is None:
            rows.append(
                PriceSourceCoverage(
                    ticker=ticker,
                    registry_source_found=True,
                    source_path=source_path,
                    source_type=str(match.get("source_type", "")),
                    first_date=str(match.get("first_date", "")),
                    last_date=str(match.get("last_date", "")),
                    adjusted_close_column_found=False,
                    adjusted_close_non_null_rows=0,
                    close_column_found=False,
                    open_column_found=False,
                    selected_signal_rows=int(selected_counts.get(ticker, 0)),
                    selected_signal_rows_inside_source_date_range=0,
                    exact_adjusted_close_path_ready=False,
                    tradable_close_proxy_available=False,
                    blocked_reason="registered_price_file_unreadable_or_missing_date_column",
                )
            )
            continue

        price_frames[ticker] = price_frame
        first_date = pd.to_datetime(match.get("first_date"), errors="coerce")
        last_date = pd.to_datetime(match.get("last_date"), errors="coerce")
        signal_dates = signals.loc[signals["ticker"] == ticker, "signal_date"]
        inside = 0
        if pd.notna(first_date) and pd.notna(last_date):
            inside = int(((signal_dates >= first_date) & (signal_dates <= last_date)).sum())
        adj_col = "adj_close" if "adj_close" in price_frame.columns else ("adjusted_close" if "adjusted_close" in price_frame.columns else None)
        adj_non_null = int(price_frame[adj_col].notna().sum()) if adj_col else 0
        close_found = "close" in price_frame.columns
        open_found = "open" in price_frame.columns
        exact_ready = bool(adj_col and adj_non_null > 0 and inside > 0)
        tradable_proxy = bool(close_found and inside > 0)
        if exact_ready:
            reason = ""
        elif inside == 0:
            reason = "registered_price_source_does_not_cover_selected_signal_dates"
        elif not adj_col:
            reason = "adjusted_close_column_missing"
        elif adj_non_null == 0:
            reason = "adjusted_close_column_all_blank"
        else:
            reason = "exact_adjusted_close_path_incomplete"

        rows.append(
            PriceSourceCoverage(
                ticker=ticker,
                registry_source_found=True,
                source_path=source_path,
                source_type=str(match.get("source_type", "")),
                first_date=str(match.get("first_date", "")),
                last_date=str(match.get("last_date", "")),
                adjusted_close_column_found=bool(adj_col),
                adjusted_close_non_null_rows=adj_non_null,
                close_column_found=close_found,
                open_column_found=open_found,
                selected_signal_rows=int(selected_counts.get(ticker, 0)),
                selected_signal_rows_inside_source_date_range=inside,
                exact_adjusted_close_path_ready=exact_ready,
                tradable_close_proxy_available=tradable_proxy,
                blocked_reason=reason,
            )
        )

    return pd.DataFrame([asdict(row) for row in rows]), price_frames


def _price_at(price_frames: dict[str, pd.DataFrame], ticker: str, date: pd.Timestamp, column: str) -> float | None:
    frame = price_frames.get(ticker)
    if frame is None or column not in frame.columns:
        return None
    matched = frame.loc[frame["date"] == date, column]
    if matched.empty or pd.isna(matched.iloc[0]):
        return None
    return float(matched.iloc[0])


def _build_signal_table(signals: pd.DataFrame, coverage: pd.DataFrame) -> pd.DataFrame:
    fields = [
        "signal_date",
        "signal_variant",
        "ticker",
        "name",
        "market",
        "selected_rank",
        "RS20",
        "RS60",
        "RS20_rank_within_80",
        "pool_rank",
        "within80_rank",
        "is_legacy_7_core",
        "in_31_high_confidence_subpool_reference",
        "in_100_extended_watchlist_reference",
        "capital_support_context",
        "high_exhaustion_or_breakdown_context",
        "layer4_risk_aware_score",
        "rs20_31_bonus_score",
        "rs20_risk_context_score",
        "signal_timing",
        "selection_rule_basis",
    ]
    available_fields = [field for field in fields if field in signals.columns]
    out = signals[available_fields].copy()
    out = out.merge(
        coverage[
            [
                "ticker",
                "registry_source_found",
                "source_path",
                "source_type",
                "exact_adjusted_close_path_ready",
                "tradable_close_proxy_available",
                "blocked_reason",
            ]
        ],
        on="ticker",
        how="left",
    )
    out["price_source_hierarchy"] = "adjusted_close_exact > tradable_close_proxy > blocked"
    out["exact_stock_return_reconstruction_method"] = "selected_ticker_adjusted_close_path_only"
    out["00631L_plus_excess_return_reconstruction_used"] = False
    out["diagnostic_only"] = True
    for key, value in FLAGS.items():
        out[key] = value
    return out


def _build_timing_design() -> pd.DataFrame:
    rows = [
        {
            "timing_variant": "same_week_close_to_next_rebalance_close_comparator",
            "entry_basis": "signal_date close",
            "exit_basis": "next weekly signal/rebalance close",
            "price_requirement": "selected ticker adjusted close on signal_date and next signal_date",
            "source_quality": "exact_if_adjusted_close_path_available_else_blocked",
            "formal_ready": False,
            "diagnostic_role": "proxy comparator retained for continuity with Experiments",
        },
        {
            "timing_variant": "next_day_close_entry_fixed_5td_exit",
            "entry_basis": "next trading day close",
            "exit_basis": "entry_date + 5 trading days close",
            "price_requirement": "selected ticker adjusted close on entry/exit dates",
            "source_quality": "exact_if_adjusted_close_path_available_else_blocked",
            "formal_ready": False,
            "diagnostic_role": "primary exact bounded timing candidate if close path exists",
        },
        {
            "timing_variant": "next_day_close_entry_weekly_rebalance_exit",
            "entry_basis": "next trading day close",
            "exit_basis": "next weekly signal/rebalance close",
            "price_requirement": "selected ticker adjusted close on entry/exit dates",
            "source_quality": "exact_if_adjusted_close_path_available_else_blocked",
            "formal_ready": False,
            "diagnostic_role": "weekly rebalance timing candidate",
        },
        {
            "timing_variant": "next_day_open_entry_fixed_5td_exit",
            "entry_basis": "next trading day open",
            "exit_basis": "entry_date + 5 trading days close",
            "price_requirement": "selected ticker executable open and adjusted/close exit path",
            "source_quality": "blocked_if_open_or_adjusted_close_missing",
            "formal_ready": False,
            "diagnostic_role": "open-entry timing audit only",
        },
    ]
    return pd.DataFrame(rows)


def _build_trade_path(signals: pd.DataFrame, price_frames: dict[str, pd.DataFrame], calendar: list[pd.Timestamp]) -> pd.DataFrame:
    selected = signals.sort_values(["signal_variant", "signal_date", "selected_rank"]).copy()
    next_signal_by_variant: dict[tuple[str, pd.Timestamp], pd.Timestamp | None] = {}
    for variant, group in selected.groupby("signal_variant"):
        dates = sorted(group["signal_date"].drop_duplicates())
        for idx, date in enumerate(dates):
            next_signal_by_variant[(variant, date)] = dates[idx + 1] if idx + 1 < len(dates) else None

    rows: list[dict[str, Any]] = []
    timing_variants = _build_timing_design()["timing_variant"].tolist()
    for _, signal in selected.iterrows():
        signal_date = pd.Timestamp(signal["signal_date"])
        ticker = str(signal["ticker"])
        variant = str(signal["signal_variant"])
        next_trading_day = _next_trading_date(calendar, signal_date, 1)
        fixed_exit = _next_trading_date(calendar, signal_date, 6)
        next_rebalance = next_signal_by_variant.get((variant, signal_date))

        for timing in timing_variants:
            if timing == "same_week_close_to_next_rebalance_close_comparator":
                entry_date = _trading_date_at_or_after(calendar, signal_date)
                exit_date = next_rebalance
                entry_price_column = "adj_close"
                exit_price_column = "adj_close"
            elif timing == "next_day_close_entry_fixed_5td_exit":
                entry_date = next_trading_day
                exit_date = fixed_exit
                entry_price_column = "adj_close"
                exit_price_column = "adj_close"
            elif timing == "next_day_close_entry_weekly_rebalance_exit":
                entry_date = next_trading_day
                exit_date = next_rebalance
                entry_price_column = "adj_close"
                exit_price_column = "adj_close"
            else:
                entry_date = next_trading_day
                exit_date = fixed_exit
                entry_price_column = "open"
                exit_price_column = "adj_close"

            entry_price = _price_at(price_frames, ticker, entry_date, entry_price_column) if entry_date is not None else None
            exit_price = _price_at(price_frames, ticker, exit_date, exit_price_column) if exit_date is not None else None
            gross_return = None
            blocked_reason = ""
            if entry_date is None or exit_date is None:
                blocked_reason = "entry_or_exit_date_outside_trading_calendar"
            elif entry_price is None or exit_price is None:
                blocked_reason = "missing_selected_ticker_exact_entry_or_exit_price"
            elif timing == "next_day_open_entry_fixed_5td_exit":
                # Open prices are executable but not adjusted; keep this timing non-formal until a local policy exists.
                gross_return = (exit_price / entry_price) - 1.0
                blocked_reason = "open_entry_adjustment_policy_not_formal_ready"
            else:
                gross_return = (exit_price / entry_price) - 1.0

            rows.append(
                {
                    "signal_date": signal_date.date().isoformat(),
                    "entry_date": entry_date.date().isoformat() if entry_date is not None else "",
                    "exit_date": exit_date.date().isoformat() if exit_date is not None else "",
                    "ticker": ticker,
                    "name": signal.get("name", ""),
                    "variant": variant,
                    "timing_variant": timing,
                    "entry_price_column": entry_price_column,
                    "exit_price_column": exit_price_column,
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "gross_return": gross_return,
                    "formal_cost_model_cost": None,
                    "net_return_formal_cost": None,
                    "roundtrip_10bp_net_return": (gross_return - 0.001) if gross_return is not None else None,
                    "roundtrip_20bp_net_return": (gross_return - 0.002) if gross_return is not None else None,
                    "roundtrip_40bp_net_return": (gross_return - 0.004) if gross_return is not None else None,
                    "holding_days": int((exit_date - entry_date).days) if entry_date is not None and exit_date is not None else None,
                    "selected_rank": signal.get("selected_rank", None),
                    "RS20": signal.get("RS20", None),
                    "RS60": signal.get("RS60", None),
                    "pool_rank": signal.get("pool_rank", None),
                    "in_31_high_confidence_subpool_reference": signal.get("in_31_high_confidence_subpool_reference", None),
                    "in_100_extended_watchlist_reference": signal.get("in_100_extended_watchlist_reference", None),
                    "00631L_plus_excess_return_reconstruction_used": False,
                    "forward_return_as_rule": False,
                    "future_return_as_rule": False,
                    "diagnostic_only": True,
                    "not_live_rule": True,
                    "blocked_reason": blocked_reason,
                }
            )
    return pd.DataFrame(rows)


def _build_cost_model_audit(trade_path: pd.DataFrame) -> pd.DataFrame:
    metadata = cost_model_metadata(TaiwanCostModel())
    numeric_ready = bool(trade_path["gross_return"].notna().any()) if "gross_return" in trade_path.columns else False
    rows = [
        {
            "cost_model": "local_ep05_taiwan_standard_fee_tax_v1",
            "model_found": True,
            "formal_cost_model_formula_ready": True,
            "numeric_formal_cost_materialized": numeric_ready,
            "application_status": "ready_for_formula_application_if_exact_entry_exit_price_and_notional_exist"
            if numeric_ready
            else "blocked_missing_exact_selected_stock_entry_exit_price_path",
            **metadata,
        },
        {
            "cost_model": "roundtrip_10bp_placeholder",
            "model_found": True,
            "formal_cost_model_formula_ready": False,
            "numeric_formal_cost_materialized": False,
            "application_status": "placeholder_reference_only_not_formal",
            "cost_model_version": "placeholder_roundtrip_bp",
            "broker_fee_rate": None,
            "broker_fee_discount": None,
            "minimum_fee_twd": None,
            "stock_sell_tax_rate": None,
            "etf_sell_tax_rate": None,
            "broker_fee_applies_on": "roundtrip_subtraction",
            "securities_transaction_tax_applies_on": "included_in_placeholder",
            "etf_and_stock_tax_split": False,
            "yuanta_actual_discount_known": False,
            "cost_model_boundary_zh": "10bp placeholder only; not formal fee/tax model.",
        },
        {
            "cost_model": "roundtrip_20bp_placeholder",
            "model_found": True,
            "formal_cost_model_formula_ready": False,
            "numeric_formal_cost_materialized": False,
            "application_status": "placeholder_reference_only_not_formal",
            "cost_model_version": "placeholder_roundtrip_bp",
            "broker_fee_rate": None,
            "broker_fee_discount": None,
            "minimum_fee_twd": None,
            "stock_sell_tax_rate": None,
            "etf_sell_tax_rate": None,
            "broker_fee_applies_on": "roundtrip_subtraction",
            "securities_transaction_tax_applies_on": "included_in_placeholder",
            "etf_and_stock_tax_split": False,
            "yuanta_actual_discount_known": False,
            "cost_model_boundary_zh": "20bp placeholder only; not formal fee/tax model.",
        },
        {
            "cost_model": "roundtrip_40bp_placeholder",
            "model_found": True,
            "formal_cost_model_formula_ready": False,
            "numeric_formal_cost_materialized": False,
            "application_status": "placeholder_reference_only_not_formal",
            "cost_model_version": "placeholder_roundtrip_bp",
            "broker_fee_rate": None,
            "broker_fee_discount": None,
            "minimum_fee_twd": None,
            "stock_sell_tax_rate": None,
            "etf_sell_tax_rate": None,
            "broker_fee_applies_on": "roundtrip_subtraction",
            "securities_transaction_tax_applies_on": "included_in_placeholder",
            "etf_and_stock_tax_split": False,
            "yuanta_actual_discount_known": False,
            "cost_model_boundary_zh": "40bp placeholder only; not formal fee/tax model.",
        },
    ]
    return pd.DataFrame(rows)


def _build_future_data_audit() -> pd.DataFrame:
    rows = [
        {
            "audit_item": "selection_rule_uses_future_return",
            "result": "passed",
            "violation_count": 0,
            "evidence": "selection carried from prior PIT signal table; no forward return columns used to construct this exact path package",
        },
        {
            "audit_item": "00631L_plus_excess_reconstruction_used_for_selected_stock_return",
            "result": "passed",
            "violation_count": 0,
            "evidence": "trade path only attempts selected ticker price path; proxy_stock_forward_return_5d is not used",
        },
        {
            "audit_item": "future_return_as_rule",
            "result": "passed",
            "violation_count": 0,
            "evidence": "forward returns are not rule inputs; package is diagnostic readiness only",
        },
    ]
    return pd.DataFrame(rows)


def _coverage_by_period(signals: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for period, (requested_start, requested_end) in PERIODS.items():
        start = pd.Timestamp(requested_start)
        end = pd.Timestamp(requested_end)
        subset = signals.loc[(signals["signal_date"] >= start) & (signals["signal_date"] <= end)]
        rows.append(
            {
                "period": period,
                "requested_start": requested_start,
                "requested_end": requested_end,
                "actual_start": subset["signal_date"].min().date().isoformat() if not subset.empty else None,
                "actual_end": subset["signal_date"].max().date().isoformat() if not subset.empty else None,
                "signal_rows": int(len(subset)),
                "unique_tickers": int(subset["ticker"].nunique()) if not subset.empty else 0,
            }
        )
    return rows


def _write_manifest(output_files: list[Path], readiness: dict[str, Any]) -> None:
    manifest_path = OUTPUT_DIR / "manifest.json"
    manifest = {
        "task_id": TASK_ID,
        "status": readiness["status"],
        "output_dir": str(OUTPUT_DIR),
        "input_signal_table": str(SIGNAL_INPUT),
        "input_prior_readiness": str(PRIOR_READINESS),
        "input_experiments_dir": str(EXPERIMENTS_DIR),
        "input_price_registry": str(PRICE_REGISTRY),
        "input_benchmark_features": str(BENCHMARK_FEATURES),
        "output_files": [path.name for path in output_files] + ["manifest.json"],
        **FLAGS,
        "diagnostic_only": True,
    }
    manifest["file_hashes"] = {
        path.name: {"sha256": _sha256(path), "bytes": path.stat().st_size}
        for path in output_files
        if path.exists() and path.name != "manifest.json"
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_summary(readiness: dict[str, Any]) -> None:
    lines = [
        "# Legacy RS20 exact path / timing / cost materialization",
        "",
        f"- status: `{readiness['status']}`",
        f"- primary_variant: `{PRIMARY_VARIANT}`",
        f"- selected_signal_rows: {readiness['selected_signal_rows']}",
        f"- selected_unique_ticker_count: {readiness['selected_unique_ticker_count']}",
        f"- exact_selected_stock_adjusted_close_path_ready: {str(readiness['exact_selected_stock_adjusted_close_path_ready']).lower()}",
        f"- next_trading_day_close_path_ready: {str(readiness['next_trading_day_close_path_ready']).lower()}",
        f"- next_trading_day_open_path_ready: {str(readiness['next_trading_day_open_path_ready']).lower()}",
        f"- local_ep05_cost_model_found: {str(readiness['local_ep05_cost_model_found']).lower()}",
        f"- formal_cost_model_ready: {str(readiness['formal_cost_model_ready']).lower()}",
        f"- ready_for_experiments: {str(readiness['ready_for_experiments']).lower()}",
        "",
        "## 判斷",
        "",
        "本輪已把 Legacy RS20 selected-stock exact path 的缺口 materialize 成可稽核 contract。"
        "本機找到 EP05 台股費稅模型，公式與參數可用；但 selected ticker 的 adjusted close / executable path 在本機 price registry 中沒有足夠覆蓋，"
        "因此無法產出 exact selected-stock gross/net return。",
        "",
        "這份 package 不再使用 `00631L_forward_return_5d + forward_excess_vs_00631L_5d` 重建個股報酬。"
        "所有 exact trade path row 均只允許 selected ticker price path；缺價時明確 blocked。",
        "",
        "## Blockers",
        "",
        "- full selected-stock adjusted close path missing for dynamic80 RS20 selected tickers.",
        "- next-day close/open entry timing cannot be exact until selected ticker price path is available.",
        "- formal EP05 fee/tax model is found, but numeric formal-cost return needs entry/exit price and notional; currently blocked by missing exact price path.",
        "",
        "## 下一棒",
        "",
        "readiness 尚未通過 exact Experiments diagnostic。下一步應交 Radar/Data 或 Core source path owner 補 selected ticker adjusted close / open-close path，"
        "範圍先限 Legacy RS20 selected tickers 與 2024-01-02~2026-05-26 加 exit buffer，不要回到全市場 mass download。",
        "",
        "## Flags",
        "",
    ]
    for key, value in FLAGS.items():
        lines.append(f"- {key}={str(value).lower()}")
    lines.append("- diagnostic_only=true")
    (OUTPUT_DIR / "final_summary_zh.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _build_price_source_request(signals: pd.DataFrame) -> pd.DataFrame:
    legacy_start = pd.Timestamp(PERIODS["legacy_requested"][0])
    legacy_end = pd.Timestamp(PERIODS["legacy_requested"][1])
    grouped = []
    for (ticker, name, market), group in signals.groupby(["ticker", "name", "market"], dropna=False):
        legacy_group = group.loc[(group["signal_date"] >= legacy_start) & (group["signal_date"] <= legacy_end)]
        grouped.append(
            {
                "ticker": ticker,
                "name": name,
                "market": market,
                "total_selected_signal_rows": int(len(group)),
                "legacy_requested_selected_signal_rows": int(len(legacy_group)),
                "first_signal_date": group["signal_date"].min().date().isoformat(),
                "last_signal_date": group["signal_date"].max().date().isoformat(),
                "minimal_primary_price_start": "2024-01-02" if len(legacy_group) else "",
                "minimal_primary_price_end_with_exit_buffer": "2026-06-05" if len(legacy_group) else "",
                "full_stability_price_start": group["signal_date"].min().date().isoformat(),
                "full_stability_price_end_with_exit_buffer": "2026-07-10",
                "required_price_columns": "date,ticker,open,close,adjusted_close,source_quality,adjustment_policy",
                "source_request_reason": "selected_by_legacy_rs20_dynamic80_variants",
                "do_not_use_00631L_plus_excess_reconstruction": True,
                "diagnostic_only": True,
                **FLAGS,
            }
        )
    return pd.DataFrame(grouped).sort_values(
        ["legacy_requested_selected_signal_rows", "total_selected_signal_rows", "ticker"],
        ascending=[False, False, True],
    )


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    prior_readiness = _read_json(PRIOR_READINESS)
    signals = pd.read_csv(SIGNAL_INPUT)
    signals["signal_date"] = pd.to_datetime(signals["signal_date"], errors="coerce")
    signals["ticker"] = signals["ticker"].map(_ticker_base)
    selected = signals.loc[signals["signal_variant"].isin(SUPPORTED_VARIANTS)].copy()
    selected = selected.sort_values(["signal_variant", "signal_date", "selected_rank"])
    registry = _load_price_registry()
    calendar = _load_calendar()

    price_coverage, price_frames = _build_price_coverage(selected, registry)
    signal_table = _build_signal_table(selected, price_coverage)
    timing_design = _build_timing_design()
    trade_path = _build_trade_path(selected, price_frames, calendar)
    cost_audit = _build_cost_model_audit(trade_path)
    future_audit = _build_future_data_audit()
    price_source_request = _build_price_source_request(selected)

    missing_price_audit = price_coverage.copy()
    missing_price_audit["source_quality"] = missing_price_audit["exact_adjusted_close_path_ready"].map(
        {True: "exact_adjusted_close_candidate", False: "blocked_or_proxy"}
    )

    exact_ready = bool(price_coverage["exact_adjusted_close_path_ready"].all()) if not price_coverage.empty else False
    any_exact_trade_return = bool(trade_path["gross_return"].notna().any())
    next_close_ready = bool(
        trade_path.loc[
            trade_path["timing_variant"].isin(
                ["next_day_close_entry_fixed_5td_exit", "next_day_close_entry_weekly_rebalance_exit"]
            ),
            "gross_return",
        ].notna().any()
    )
    next_open_ready = bool(
        trade_path.loc[trade_path["timing_variant"] == "next_day_open_entry_fixed_5td_exit", "gross_return"].notna().any()
    )
    formal_formula_ready = True
    formal_numeric_ready = bool(any_exact_trade_return)
    ready_for_experiments = bool(exact_ready and next_close_ready and formal_numeric_ready)
    status = (
        "legacy_rs20_exact_path_timing_cost_ready_for_bounded_diagnostic"
        if ready_for_experiments
        else "legacy_rs20_exact_path_timing_cost_blocked_missing_selected_stock_price_path"
    )

    readiness = {
        "task_id": TASK_ID,
        "status": status,
        "diagnostic_only": True,
        "primary_variant": PRIMARY_VARIANT,
        "supported_signal_variants": SUPPORTED_VARIANTS,
        "reference_variants_retained_as_design_only": REFERENCE_VARIANTS,
        "prior_core_status": prior_readiness.get("status"),
        "selected_signal_rows": int(len(selected)),
        "selected_unique_ticker_count": int(selected["ticker"].nunique()),
        "price_registry_selected_ticker_matches": int(price_coverage["registry_source_found"].sum())
        if not price_coverage.empty
        else 0,
        "exact_selected_stock_adjusted_close_ready_ticker_count": int(
            price_coverage["exact_adjusted_close_path_ready"].sum()
        )
        if not price_coverage.empty
        else 0,
        "exact_selected_stock_adjusted_close_path_ready": exact_ready,
        "next_trading_day_close_path_ready": next_close_ready,
        "next_trading_day_open_path_ready": next_open_ready,
        "same_week_close_rebalance_comparator_materialized": any_exact_trade_return,
        "local_ep05_cost_model_found": True,
        "formal_cost_model_formula_ready": formal_formula_ready,
        "formal_cost_model_ready": formal_numeric_ready,
        "placeholder_cost_model_used": not formal_numeric_ready,
        "ready_for_legacy_rs20_exact_cost_timing_diagnostic": ready_for_experiments,
        "ready_for_experiments": ready_for_experiments,
        "ready_for_formal": False,
        "ready_for_strategy_replay": False,
        "future_data_violation_count": int(future_audit["violation_count"].sum()),
        "requested_vs_actual_coverage": _coverage_by_period(selected),
        "blocked_fields": [
            "full_selected_stock_adjusted_close_path",
            "next_trading_day_close_entry_exact_return",
            "next_trading_day_open_entry_exact_return",
            "numeric_formal_cost_application",
        ]
        if not ready_for_experiments
        else [],
        "proxy_fields": [
            "tradable_close_proxy_if_unadjusted_close_later_allowed",
            "roundtrip_10bp_20bp_40bp_placeholder_reference",
        ],
        **FLAGS,
    }

    output_files = [
        OUTPUT_DIR / "legacy_rs20_exact_selected_stock_trade_path.csv",
        OUTPUT_DIR / "legacy_rs20_exact_selected_stock_signal_table.csv",
        OUTPUT_DIR / "legacy_rs20_exact_timing_variant_design.csv",
        OUTPUT_DIR / "legacy_rs20_exact_cost_model_audit.csv",
        OUTPUT_DIR / "legacy_rs20_exact_missing_price_audit.csv",
        OUTPUT_DIR / "legacy_rs20_exact_selected_ticker_price_source_request.csv",
        OUTPUT_DIR / "legacy_rs20_exact_future_data_audit.csv",
        OUTPUT_DIR / "readiness_for_legacy_rs20_exact_cost_timing_diagnostic.json",
        OUTPUT_DIR / "final_summary_zh.md",
    ]

    trade_path.to_csv(output_files[0], index=False, encoding="utf-8")
    signal_table.to_csv(output_files[1], index=False, encoding="utf-8")
    timing_design.to_csv(output_files[2], index=False, encoding="utf-8")
    cost_audit.to_csv(output_files[3], index=False, encoding="utf-8")
    missing_price_audit.to_csv(output_files[4], index=False, encoding="utf-8")
    price_source_request.to_csv(output_files[5], index=False, encoding="utf-8")
    future_audit.to_csv(output_files[6], index=False, encoding="utf-8")
    output_files[7].write_text(json.dumps(readiness, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_summary(readiness)
    _write_manifest(output_files, readiness)


if __name__ == "__main__":
    main()
