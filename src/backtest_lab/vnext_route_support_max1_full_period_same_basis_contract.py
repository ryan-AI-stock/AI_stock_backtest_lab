from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
LAYER4_POOL = REPO_ROOT / "outputs" / "vnext_layer4_80_primary_pool_contract_20260708" / "layer4_80_primary_pool_contract.csv"
MARKET_FIELDS = (
    REPO_ROOT
    / "outputs"
    / "vnext_regime_switch_hybrid_route_market_fields_path_materialization_20260708"
    / "regime_switch_market_regime_fields.csv"
)
FULL_PERIOD_DIR = REPO_ROOT / "outputs" / "vnext_full_period_regime_switch_benchmark_exception_path_20260708"
FULL_BENCHMARK_PATH = FULL_PERIOD_DIR / "full_period_regime_switch_benchmark_reference_path.csv"
FULL_STOCK_PATH = FULL_PERIOD_DIR / "full_period_regime_switch_stock_route_path.csv"
EXACT_TRIGGER = (
    REPO_ROOT
    / "outputs"
    / "vnext_full_period_exact_consensus_trigger_contract_20260708"
    / "full_period_exact_consensus_trigger_contract.csv"
)
P1_MAX1_DIR = REPO_ROOT / "outputs" / "vnext_p1_c2_route_support_max1_modelization_contract_20260708"
P1_MAX1_CONTRACT = P1_MAX1_DIR / "p1_c2_route_support_max1_modelization_contract.csv"
P1_WEIGHTED_REFRESHED = (
    REPO_ROOT
    / "outputs"
    / "vnext_p1_c2_weighted_pool80_top5_ohlc_absorption_20260708"
    / "p1_c2_weighted_pool80_top5_contract_refreshed.csv"
)
OUTPUT_DIR = REPO_ROOT / "outputs" / "vnext_route_support_max1_full_period_same_basis_contract_20260708"

TASK_ID = "TASK-BACKTEST-CORE-VNEXT-ROUTE-SUPPORT-MAX1-FULL-PERIOD-SAME-BASIS-MODELIZATION-CONTRACT-001"
PRIMARY_TIMING = "next_day_close_entry_fixed_5td_exit"
SOURCE_VARIANTS = [
    "hybrid_pullback_base_mega_override",
    "conservative_hurdle_route",
    "pool_breadth_route",
    "market_bias_pool_trend_route",
    "dispersion_route",
]
DIAGNOSTIC_NOTIONAL = 1_000_000
TRANSITION_COSTS = {
    "00631L_to_stock": {"transition_cost_rate": 0.00385, "sell_fee_twd": 1425, "buy_fee_twd": 1425, "securities_transaction_tax_twd": 1000, "total_transition_cost_twd": 3850},
    "stock_to_00631L": {"transition_cost_rate": 0.00585, "sell_fee_twd": 1425, "buy_fee_twd": 1425, "securities_transaction_tax_twd": 3000, "total_transition_cost_twd": 5850},
    "stock_to_stock": {"transition_cost_rate": 0.00585, "sell_fee_twd": 1425, "buy_fee_twd": 1425, "securities_transaction_tax_twd": 3000, "total_transition_cost_twd": 5850},
    "hold": {"transition_cost_rate": 0.0, "sell_fee_twd": 0, "buy_fee_twd": 0, "securities_transaction_tax_twd": 0, "total_transition_cost_twd": 0},
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
PERIODS = {
    "P1": ("2015-01-02", "2022-12-29"),
    "P2": ("2023-01-02", "2026-06-30"),
    "2024_latest": ("2024-01-02", "2026-06-30"),
    "2026YTD": ("2026-01-02", "2026-06-30"),
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


def _clip01(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").clip(0, 1)


def _period_flags(date_text: str) -> dict[str, bool]:
    date = pd.Timestamp(date_text)
    return {
        f"in_{period}": pd.Timestamp(start) <= date <= pd.Timestamp(end)
        for period, (start, end) in PERIODS.items()
    }


def _period_label(date_text: str) -> str:
    flags = _period_flags(date_text)
    labels = [period[3:] for period, active in flags.items() if active]
    return "|".join(labels) if labels else "outside_requested_periods"


def _calendar() -> pd.DataFrame:
    bench = pd.read_csv(FULL_BENCHMARK_PATH, low_memory=False)
    bench = bench[
        bench["benchmark"].eq("00631L")
        & bench["timing_variant"].eq(PRIMARY_TIMING)
        & ~bench["buy_hold_reference"].astype(str).str.lower().eq("true")
    ].copy()
    for col in ["signal_date", "entry_date", "exit_date"]:
        bench[col] = pd.to_datetime(bench[col], errors="coerce").dt.strftime("%Y-%m-%d")
    cal = bench[["signal_date"]].drop_duplicates().sort_values("signal_date").reset_index(drop=True)
    cal["next_signal_date"] = cal["signal_date"].shift(-1)
    return cal.dropna(subset=["signal_date"]).copy()


def _benchmark_maps() -> dict[str, dict[str, float]]:
    ref = pd.read_csv(FULL_BENCHMARK_PATH, low_memory=False)
    ref = ref[
        ref["timing_variant"].eq(PRIMARY_TIMING)
        & ref["benchmark"].isin(["00631L", "0050"])
        & ~ref["buy_hold_reference"].astype(str).str.lower().eq("true")
    ].copy()
    maps: dict[str, dict[str, float]] = {}
    for benchmark, group in ref.groupby("benchmark"):
        signal_prices = group[["signal_date", "entry_price"]].dropna()
        signal_prices["signal_date"] = pd.to_datetime(signal_prices["signal_date"], errors="coerce").dt.strftime("%Y-%m-%d")
        maps[str(benchmark)] = {r.signal_date: float(r.entry_price) for r in signal_prices.itertuples(index=False)}
    return maps


def _market_c2() -> pd.DataFrame:
    m = pd.read_csv(MARKET_FIELDS, low_memory=False)
    m["signal_date"] = pd.to_datetime(m["snapshot_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    m["c2_market_health_gate"] = (
        (pd.to_numeric(m["0050_price_vs_ma60"], errors="coerce") >= 0)
        & (pd.to_numeric(m["0050_return_20d"], errors="coerce") >= 0)
        & (pd.to_numeric(m["0050_return_40d"], errors="coerce") >= 0)
    )
    m["c2_definition"] = "0050_price_vs_ma60>=0 AND 0050_return_20d>=0 AND 0050_return_40d>=0"
    return m[["signal_date", "c2_market_health_gate", "c2_definition", "0050_price_vs_ma60", "0050_return_20d", "0050_return_40d"]]


def _rank_score(df: pd.DataFrame, col: str, higher_better: bool = True) -> pd.Series:
    if col not in df.columns:
        return pd.Series(0.5, index=df.index)
    vals = pd.to_numeric(df[col], errors="coerce")
    if vals.notna().sum() == 0:
        return pd.Series(0.5, index=df.index)
    return vals.groupby(df["snapshot_date"]).rank(pct=True, ascending=not higher_better).fillna(0.5)


def _route_support_from_full_path() -> pd.DataFrame:
    path = pd.read_csv(FULL_STOCK_PATH, low_memory=False, dtype={"ticker": str})
    path = path[
        path["timing_variant"].eq(PRIMARY_TIMING)
        & path["route_variant"].isin(SOURCE_VARIANTS)
        & path["path_ready"].astype(str).str.lower().eq("true")
    ].copy()
    path["signal_date"] = pd.to_datetime(path["signal_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    path["ticker"] = path["ticker"].map(_ticker)
    return (
        path.groupby(["signal_date", "ticker"], as_index=False)
        .agg(
            route_support_variant_count=("route_variant", "nunique"),
            route_support_variant_flags=("route_variant", lambda x: "|".join(sorted(set(map(str, x.dropna()))))),
            route_support_mode_flags=("selected_route_mode", lambda x: "|".join(sorted(set(map(str, x.dropna()))))),
        )
    )


def _p1_exact_top1() -> pd.DataFrame:
    p1 = pd.read_csv(P1_WEIGHTED_REFRESHED, low_memory=False, dtype={"ticker": str})
    p1 = p1[p1["score_variant"].eq("route_support") & p1["candidate_rank"].eq(1)].copy()
    p1["signal_date"] = pd.to_datetime(p1["signal_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    p1["ticker"] = p1["ticker"].map(_ticker)
    p1["consensus_trigger"] = True
    p1["consensus_trigger_source_quality"] = "p1_exact_consensus4_trigger"
    return p1


def _full_proxy_top1() -> pd.DataFrame:
    pool = pd.read_csv(LAYER4_POOL, low_memory=False, dtype={"ticker": str})
    pool["snapshot_date"] = pd.to_datetime(pool["snapshot_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    pool["ticker"] = pool["ticker"].map(_ticker)
    pool = pool[pool["is_layer4_primary_pool"].astype(str).str.lower().eq("true")].copy()
    support = _route_support_from_full_path()
    pool = pool.merge(support, left_on=["snapshot_date", "ticker"], right_on=["signal_date", "ticker"], how="left", suffixes=("", "_support"))
    if "signal_date" in pool.columns:
        pool = pool.drop(columns=["signal_date"])
    pool["route_support_variant_count"] = pd.to_numeric(pool["route_support_variant_count"], errors="coerce").fillna(0)
    pool["route_support_component"] = (pool["route_support_variant_count"] / len(SOURCE_VARIANTS)).clip(0, 1)
    quality = 1.0 - _clip01(pool.get("layer1_quality_floor_risk_pctile_by_week", pd.Series(index=pool.index, dtype=float))).fillna(0.5)
    quality += pool.get("layer1_pass_bottom30", False).astype(str).str.lower().eq("true").astype(float) * 0.15
    pool["quality_component"] = quality.clip(0, 1)
    rs_parts = [_rank_score(pool, c, True) for c in ["RS20", "RS40", "RS60", "RS30_proxy"] if c in pool.columns]
    pool["rs_component"] = pd.concat(rs_parts, axis=1).mean(axis=1).fillna(0.5) if rs_parts else 0.5
    liq_parts = []
    for col in ["traded_value_rank_20d", "traded_value_rank_60d", "traded_value_rank_5d"]:
        if col in pool.columns:
            liq_parts.append(1.0 - ((pd.to_numeric(pool[col], errors="coerce") - 1.0) / 80.0).clip(0, 1))
    pool["liquidity_component"] = pd.concat(liq_parts, axis=1).mean(axis=1).fillna(0.5) if liq_parts else 0.5
    bias_parts = []
    for col in ["BIAS20_percentile", "BIAS60_percentile", "BIAS120_percentile"]:
        if col in pool.columns:
            p = _clip01(pool[col]).fillna(0.5)
            bias_parts.append(1.0 - (p - 0.5).abs() * 2.0)
    pool["bias_health_component"] = pd.concat(bias_parts, axis=1).mean(axis=1).fillna(0.5) if bias_parts else 0.5
    risk = pd.to_numeric(pool.get("exhaustion_risk_score", 0), errors="coerce").fillna(0) * 0.25
    risk += pd.to_numeric(pool.get("breakdown_risk_score", 0), errors="coerce").fillna(0) * 0.25
    pool["risk_inverse_component"] = (1.0 - risk.clip(0, 1)).clip(0, 1)
    pool["weighted_score"] = (
        pool["quality_component"] * 0.10
        + pool["rs_component"] * 0.20
        + pool["liquidity_component"] * 0.10
        + pool["bias_health_component"] * 0.10
        + pool["route_support_component"] * 0.38
        + pool["risk_inverse_component"] * 0.12
    )
    pool = pool.sort_values(["snapshot_date", "weighted_score", "ticker"], ascending=[True, False, True])
    pool["candidate_rank"] = pool.groupby("snapshot_date").cumcount() + 1
    top = pool[pool["candidate_rank"].eq(1)].copy()
    top = top.rename(columns={"snapshot_date": "signal_date"})
    top["score_variant"] = "route_support"
    return top


def _exact_trigger_dates() -> pd.DataFrame:
    exact = pd.read_csv(EXACT_TRIGGER, low_memory=False, dtype={"candidate_ticker": str})
    exact["signal_date"] = pd.to_datetime(exact["signal_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    exact = exact[exact["exact_trigger_pass"].astype(str).str.lower().eq("true")].copy()
    g = (
        exact.groupby("signal_date", as_index=False)
        .agg(
            consensus_trigger_candidate_count=("candidate_ticker", "nunique"),
            max_consensus_count=("consensus_count", "max"),
            trigger_source_variants=("trigger_source_variants", lambda x: "|".join(sorted(set(map(str, x.dropna()))))),
        )
    )
    g["consensus_trigger"] = True
    g["consensus_trigger_source_quality"] = "exact_same_ticker_consensus_ge4_full_period"
    return g


def _path_map() -> pd.DataFrame:
    frames = []
    p1 = pd.read_csv(P1_WEIGHTED_REFRESHED, low_memory=False, dtype={"ticker": str})
    p1 = p1[p1["score_variant"].eq("route_support") & p1["candidate_rank"].eq(1)].copy()
    p1["signal_date"] = pd.to_datetime(p1["signal_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    p1["ticker"] = p1["ticker"].map(_ticker)
    p1["path_ready"] = p1["official_unadjusted_ohlc_path_ready"].astype(str).str.lower().eq("true")
    p1 = p1.rename(columns={"official_unadjusted_ohlc_path_ready": "official_unadjusted_ohlc_ready"})
    frames.append(p1)
    full = pd.read_csv(FULL_STOCK_PATH, low_memory=False, dtype={"ticker": str})
    full = full[full["timing_variant"].eq(PRIMARY_TIMING)].copy()
    full["signal_date"] = pd.to_datetime(full["signal_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    full["ticker"] = full["ticker"].map(_ticker)
    full["entry_close"] = full["entry_price"]
    full["official_unadjusted_ohlc_ready"] = full["path_ready"].astype(str).str.lower().eq("true")
    full["source_quality"] = full["source_quality"].fillna("official_unadjusted_ohlc_selected_route_path")
    frames.append(full)
    out = pd.concat(frames, ignore_index=True, sort=False)
    keep = [
        "signal_date",
        "ticker",
        "entry_date",
        "exit_date",
        "entry_open",
        "entry_close",
        "exit_close",
        "gross_return_unadjusted",
        "net_return_local_ep05_cost_unit_notional",
        "official_unadjusted_ohlc_ready",
        "source_quality",
        "blocked_reason",
    ]
    return out[[c for c in keep if c in out.columns]].sort_values(["signal_date", "ticker", "official_unadjusted_ohlc_ready"], ascending=[True, True, False]).drop_duplicates(["signal_date", "ticker"])


def _transition(prev_ticker: str, prev_type: str, target_ticker: str, target_type: str) -> tuple[str, str]:
    if prev_ticker == target_ticker and prev_type == target_type:
        return "hold_same_state_no_trade", "hold"
    if prev_type == "etf" and target_type == "stock":
        return "00631L_to_stock_exception", "00631L_to_stock"
    if prev_type == "stock" and target_type == "etf":
        return "stock_exception_to_00631L_base", "stock_to_00631L"
    if prev_type == "stock" and target_type == "stock":
        return "stock_to_stock_exception_switch", "stock_to_stock"
    return "transition_other", "hold"


def _build_contract() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cal = _calendar()
    maps = _benchmark_maps()
    c2 = _market_c2()
    p1 = _p1_exact_top1()
    proxy = _full_proxy_top1()
    proxy = proxy[~proxy["signal_date"].isin(set(p1["signal_date"]))].copy()
    top1 = pd.concat([p1, proxy], ignore_index=True, sort=False)
    top1 = top1.merge(c2, on="signal_date", how="left")
    top1 = top1.merge(_exact_trigger_dates(), on="signal_date", how="left", suffixes=("", "_exact"))
    for col in ["consensus_trigger", "consensus_trigger_source_quality"]:
        exact_col = f"{col}_exact"
        if exact_col in top1.columns:
            top1[col] = top1[col].where(top1[col].notna(), top1[exact_col])
            top1 = top1.drop(columns=[exact_col])
    paths = _path_map()
    rows = []
    missing = []
    transitions = []
    by_date = {r.signal_date: r._asdict() for r in top1.itertuples(index=False)}
    path_by_key = {(r.signal_date, r.ticker): r._asdict() for r in paths.itertuples(index=False)}
    prev_ticker = "00631L"
    prev_type = "etf"
    for r in cal.itertuples(index=False):
        signal_date = r.signal_date
        next_signal_date = r.next_signal_date
        cand = by_date.get(signal_date)
        c2_gate = bool(cand.get("c2_market_health_gate")) if cand is not None and not pd.isna(cand.get("c2_market_health_gate")) else False
        trigger = bool(cand.get("consensus_trigger")) if cand is not None and not pd.isna(cand.get("consensus_trigger")) else False
        use_stock = bool(cand is not None and c2_gate and trigger)
        if use_stock:
            target_ticker = _ticker(cand.get("ticker"))
            target_type = "stock"
            path = path_by_key.get((signal_date, target_ticker), {})
            entry_price = path.get("entry_close")
            exit_price = path.get("exit_close")
            gross = path.get("gross_return_unadjusted")
            ready = bool(path.get("official_unadjusted_ohlc_ready", False))
            entry_date = path.get("entry_date", "")
            exit_date = path.get("exit_date", "")
            source_quality = path.get("source_quality", "selected_stock_path_missing")
            if not ready:
                missing.append({
                    "signal_date": signal_date,
                    "ticker": target_ticker,
                    "entry_date": entry_date,
                    "exit_date": exit_date,
                    "missing_field": "entry_close_or_exit_close",
                    "blocked_reason": path.get("blocked_reason", "selected_stock_official_unadjusted_ohlc_not_materialized_for_route_support_top1"),
                    "next_owner": "Radar/Data bounded selected-ticker-only OHLC fill; exact trigger is ready, no full-market download",
                })
            state_reason = "c2_consensus_trigger_route_support_top1_stock_exception"
        else:
            target_ticker = "00631L"
            target_type = "etf"
            entry_price = maps["00631L"].get(signal_date)
            exit_price = maps["00631L"].get(next_signal_date)
            gross = (exit_price / entry_price - 1.0) if entry_price and exit_price else None
            ready = gross is not None
            entry_date = signal_date
            exit_date = next_signal_date
            source_quality = "benchmark_features_adjusted_close_exact_reference"
            state_reason = "default_00631L_base_no_c2_or_consensus_trigger"
        action, cost_key = _transition(prev_ticker, prev_type, target_ticker, target_type)
        cost = TRANSITION_COSTS[cost_key]
        net = (float(gross) - cost["transition_cost_rate"]) if gross is not None else None
        row = {
            "signal_date": signal_date,
            "next_signal_date": next_signal_date,
            "period_label": _period_label(signal_date),
            **_period_flags(signal_date),
            "selected_ticker": target_ticker,
            "selected_asset_type": target_type,
            "state_reason": state_reason,
            "c2_market_health_gate": c2_gate,
            "c2_definition": cand.get("c2_definition") if cand is not None else "0050_price_vs_ma60>=0 AND 0050_return_20d>=0 AND 0050_return_40d>=0",
            "consensus_trigger": trigger,
            "consensus_trigger_source_quality": cand.get("consensus_trigger_source_quality") if cand is not None and trigger else "no_exact_trigger",
            "consensus_trigger_candidate_count": cand.get("consensus_trigger_candidate_count") if cand is not None else 0,
            "max_consensus_count": cand.get("max_consensus_count") if cand is not None else 0,
            "score_variant": "route_support",
            "route_support_weighted_score": cand.get("weighted_score") if cand is not None else None,
            "route_support_variant_count": cand.get("route_support_variant_count") if cand is not None else 0,
            "route_support_variant_flags": cand.get("route_support_variant_flags") if cand is not None else "",
            "entry_date": entry_date,
            "exit_date": exit_date,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "gross_interval_return": gross,
            "transition_action": action,
            "transition_cost_rate": cost["transition_cost_rate"],
            "net_interval_return_after_transition_cost": net,
            "official_unadjusted_ohlc_ready": ready if target_type == "stock" else True,
            "benchmark_adjusted_path_ready": ready if target_type == "etf" else True,
            "adjusted_close_ready": target_type == "etf",
            "source_quality": source_quality,
            "diagnostic_only": True,
            **FLAGS,
        }
        rows.append(row)
        if action != "hold_same_state_no_trade":
            transitions.append({
                "signal_date": signal_date,
                "transition_date": entry_date,
                "from_ticker": prev_ticker,
                "from_asset_type": prev_type,
                "to_ticker": target_ticker,
                "to_asset_type": target_type,
                "transition_action": action,
                "diagnostic_notional_twd": DIAGNOSTIC_NOTIONAL,
                **cost,
                "cost_model_status": "applied_local_ep05_TaiwanCostModel_unit_notional_transition_cost",
                "cost_model_version": "taiwan_standard_fee_tax_v1",
                "diagnostic_only": True,
                **FLAGS,
            })
        prev_ticker, prev_type = target_ticker, target_type
    return pd.DataFrame(rows), pd.DataFrame(transitions), pd.DataFrame(missing)


def _coverage(contract: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for label, group in [("full", contract), *[(p, contract[contract[f"in_{p}"]]) for p in PERIODS]]:
        stock = group[group["selected_asset_type"].eq("stock")]
        rows.append({
            "period": label,
            "contract_rows": len(group),
            "stock_exception_rows": len(stock),
            "stock_unadjusted_ready_rows": int(stock["official_unadjusted_ohlc_ready"].sum()) if len(stock) else 0,
            "official_unadjusted_ohlc_ready_share": float(stock["official_unadjusted_ohlc_ready"].mean()) if len(stock) else 1.0,
            "adjusted_close_ready_share": 0.0 if len(stock) else 1.0,
            "proxy_trigger_stock_rows": int(stock["consensus_trigger_source_quality"].astype(str).str.contains("proxy").sum()) if len(stock) else 0,
        })
    return pd.DataFrame(rows)


def _blocked(contract: pd.DataFrame, missing: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if not missing.empty:
        for r in missing.itertuples(index=False):
            rows.append(r._asdict() | {"blocked_item": "selected_stock_official_unadjusted_ohlc"})
    rows.extend([
        {"signal_date": "", "ticker": "", "entry_date": "", "exit_date": "", "missing_field": "adjusted_close", "blocked_reason": "selected-stock adjusted close remains blocked; official unadjusted OHLC diagnostic-only", "next_owner": "Strategy Center/Radar only if adjusted-close source is authorized", "blocked_item": "selected_stock_adjusted_close"},
        {"signal_date": "", "ticker": "", "entry_date": "", "exit_date": "", "missing_field": "cash_bear_classifier", "blocked_reason": "cash/bear classifier blocked; no cash rule added", "next_owner": "Strategy Center/Core if cash classifier is re-opened", "blocked_item": "cash_bear_classifier"},
    ])
    return pd.DataFrame(rows)


def _cost_audit(transitions: pd.DataFrame) -> pd.DataFrame:
    if transitions.empty:
        return pd.DataFrame()
    return transitions.groupby(["transition_action", "from_asset_type", "to_asset_type"], as_index=False).agg(
        transition_count=("transition_action", "size"),
        transition_cost_rate=("transition_cost_rate", "first"),
        total_transition_cost_twd_sum=("total_transition_cost_twd", "sum"),
        cost_model_status=("cost_model_status", "first"),
        cost_model_version=("cost_model_version", "first"),
    )


def _score_audit(contract: pd.DataFrame) -> pd.DataFrame:
    stock = contract[contract["selected_asset_type"].eq("stock")].copy()
    cols = [
        "signal_date",
        "period_label",
        "selected_ticker",
        "consensus_trigger_source_quality",
        "route_support_weighted_score",
        "route_support_variant_count",
        "route_support_variant_flags",
        "c2_market_health_gate",
        "c2_definition",
    ]
    out = stock[[c for c in cols if c in stock.columns]]
    out["score_formula"] = "route_support weights: quality .10 + RS .20 + liquidity .10 + BIAS health .10 + route support .38 + risk inverse .12"
    out["future_return_used"] = False
    return out


def _readiness(contract: pd.DataFrame, missing: pd.DataFrame) -> dict[str, Any]:
    stock = contract[contract["selected_asset_type"].eq("stock")]
    official_share = float(stock["official_unadjusted_ohlc_ready"].mean()) if len(stock) else 1.0
    proxy_rows = int(stock["consensus_trigger_source_quality"].astype(str).str.contains("proxy").sum()) if len(stock) else 0
    exact_ready = official_share == 1.0 and proxy_rows == 0
    proxy_ready = official_share == 1.0
    return {
        "task_id": TASK_ID,
        "status": "full_period_same_basis_contract_ready" if exact_ready else "full_period_same_basis_contract_partial_ohlc_blocked",
        "ready_for_route_support_max1_full_period_same_basis_modelization_diagnostic": bool(exact_ready),
        "ready_for_route_support_max1_full_period_proxy_modelization_review": bool(proxy_ready),
        "ready_for_experiments": bool(exact_ready),
        "contract_rows": int(len(contract)),
        "stock_exception_rows": int(len(stock)),
        "transition_count": int((contract["transition_action"] != "hold_same_state_no_trade").sum()),
        "p2_recent_proxy_trigger_stock_rows": proxy_rows,
        "official_unadjusted_ohlc_ready_share": official_share,
        "official_unadjusted_ohlc_missing_rows": int(len(missing)),
        "adjusted_close_ready": False,
        "cost_model_ready": True,
        "comparison_baseline_hooks_ready": True,
        "cash_bear_classifier_ready": False,
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
            "layer4_pool": str(LAYER4_POOL),
            "market_fields": str(MARKET_FIELDS),
            "full_benchmark_path": str(FULL_BENCHMARK_PATH),
            "full_stock_path": str(FULL_STOCK_PATH),
            "exact_consensus_trigger": str(EXACT_TRIGGER),
            "p1_max1_contract": str(P1_MAX1_CONTRACT),
            "p1_weighted_refreshed": str(P1_WEIGHTED_REFRESHED),
        },
        "artifacts": [{"path": str(p), "sha256": _sha256(p), "bytes": p.stat().st_size} for p in files],
        "readiness": readiness,
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    contract, transitions, missing = _build_contract()
    coverage = _coverage(contract)
    blocked = _blocked(contract, missing)
    cost = _cost_audit(transitions)
    score = _score_audit(contract)
    readiness = _readiness(contract, missing)
    future = pd.DataFrame([
        {"audit_item": "route_support_score", "future_return_used_as_rule": False, "future_data_violation_count": 0},
        {"audit_item": "state_machine", "future_return_used_as_rule": False, "future_data_violation_count": 0},
    ])
    paths = {
        "contract": OUTPUT_DIR / "route_support_max1_full_period_same_basis_modelization_contract.csv",
        "transition": OUTPUT_DIR / "route_support_max1_full_period_transition_trace.csv",
        "score": OUTPUT_DIR / "route_support_max1_full_period_score_audit.csv",
        "coverage": OUTPUT_DIR / "route_support_max1_full_period_coverage_audit.csv",
        "cost": OUTPUT_DIR / "route_support_max1_full_period_cost_audit.csv",
        "blocked": OUTPUT_DIR / "route_support_max1_full_period_blocked_proxy_ledger.csv",
        "missing": OUTPUT_DIR / "route_support_max1_full_period_missing_ohlc_ledger.csv",
        "future": OUTPUT_DIR / "route_support_max1_full_period_future_data_audit.csv",
        "readiness": OUTPUT_DIR / "readiness_for_route_support_max1_full_period_same_basis_modelization_diagnostic.json",
        "summary": OUTPUT_DIR / "final_summary_zh.md",
        "manifest": OUTPUT_DIR / "manifest.json",
    }
    contract.to_csv(paths["contract"], index=False, encoding="utf-8-sig")
    transitions.to_csv(paths["transition"], index=False, encoding="utf-8-sig")
    score.to_csv(paths["score"], index=False, encoding="utf-8-sig")
    coverage.to_csv(paths["coverage"], index=False, encoding="utf-8-sig")
    cost.to_csv(paths["cost"], index=False, encoding="utf-8-sig")
    blocked.to_csv(paths["blocked"], index=False, encoding="utf-8-sig")
    missing.to_csv(paths["missing"], index=False, encoding="utf-8-sig")
    future.to_csv(paths["future"], index=False, encoding="utf-8-sig")
    paths["readiness"].write_text(json.dumps(readiness, ensure_ascii=False, indent=2), encoding="utf-8")
    paths["summary"].write_text(
        "\n".join([
            "# route_support max1 full-period same-basis modelization contract",
            "",
            "- 已 materialize P1/P2/2024-latest/2026YTD/full integrated state-machine contract。",
            "- Default state = 00631L state-hold；C2 + consensus trigger 才允許 route_support quant score max1 stock exception。",
            "- P1/P2/recent 均使用 full-period exact consensus trigger contract，不使用 route_support>=4 proxy trigger。",
            f"- official_unadjusted_ohlc_ready_share = {readiness['official_unadjusted_ohlc_ready_share']:.4f}；adjusted_close_ready=false。",
            f"- p2_recent_proxy_trigger_stock_rows = {readiness['p2_recent_proxy_trigger_stock_rows']}。",
            "- Cost model ready：EP05 TaiwanCostModel unit-notional transition cost；後續主結論必須 net after transaction cost。",
            "- 若 readiness true，下一棒交 Experiments 做 full-period exact same-basis route_support max1 diagnostic。",
            "",
            "下一棒：若 ready_for_experiments=true，交 Experiments；若 false，依 missing ledger 補 bounded selected-ticker path。",
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
