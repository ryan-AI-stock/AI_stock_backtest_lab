from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from backtest_lab.costs import TaiwanCostModel, cost_model_metadata


REPO_ROOT = Path(__file__).resolve().parents[2]
LAYER4_80_POOL = REPO_ROOT / "outputs" / "vnext_layer4_80_primary_pool_contract_20260708" / "layer4_80_primary_pool_contract.csv"
C2_STATE_MACHINE = (
    REPO_ROOT
    / "outputs"
    / "vnext_p1_c2_market_health_consensus4_adjusted_state_machine_contract_20260708"
    / "p1_c2_market_health_consensus4_state_machine_contract.csv"
)
P1_UNADJUSTED_PATH = (
    REPO_ROOT
    / "outputs"
    / "vnext_p1_legacy_regime_unadjusted_path_refresh_20260708"
    / "p1_legacy_regime_unadjusted_trade_path_refreshed.csv"
)
OUTPUT_DIR = REPO_ROOT / "outputs" / "vnext_p1_c2_top5_multi_stock_exception_candidate_contract_20260708"

TASK_ID = "TASK-BACKTEST-CORE-VNEXT-P1-C2-TOP5-MULTI-STOCK-EXCEPTION-CANDIDATE-CONTRACT-001"
DIAGNOSTIC_NOTIONAL_TWD = 1_000_000
BASE_ASSET = "00631L"
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


def _ticker_str(value: Any) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    return text.zfill(4) if text.isdigit() and len(text) < 4 else text


def _asset_type(ticker: str) -> str:
    return "etf" if ticker == BASE_ASSET else "stock"


def _c2_signal_frame() -> pd.DataFrame:
    df = pd.read_csv(C2_STATE_MACHINE, low_memory=False)
    df = df.loc[df["signal_date"].notna()].copy()
    df["signal_date"] = pd.to_datetime(df["signal_date"], errors="coerce")
    df["signal_date_key"] = df["signal_date"].dt.date.astype(str)
    bool_cols = ["c2_market_health_gate", "raw_consensus4_exception_active", "exception_allowed_by_c2"]
    for col in bool_cols:
        df[col] = df[col].astype(str).str.lower().eq("true")
    return df[
        [
            "signal_date",
            "signal_date_key",
            "state_start_date",
            "state_end_date",
            "c2_market_health_gate",
            "raw_consensus4_exception_active",
            "exception_allowed_by_c2",
            "holding_ticker",
            "0050_above_ma60_flag",
            "0050_return_20d",
            "0050_return_40d",
        ]
    ].copy()


def _layer4_top5_candidates(signals: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "snapshot_date",
        "ticker",
        "name",
        "market",
        "pool_rank",
        "pool_selection_score",
        "pool_selection_score_col",
        "layer4_pool_role",
        "is_layer4_primary_pool",
        "reference_only",
        "layer4_primary_pool_size",
        "momentum_continuation_high_confidence",
        "pullback_repair_high_confidence",
        "overlap_reacceleration_high_confidence",
        "neutral_quality_liquidity_high_confidence",
        "two_plus_opportunity_labels",
        "opportunity_label_count",
        "layer4_risk_aware_score",
        "layer4_broad_opportunity_net_score",
        "layer4_retention_constrained_score",
        "layer4_c_quota_base_score",
        "RS20",
        "RS40",
        "RS60",
        "traded_value_rank_20d",
        "traded_value_rank_60d",
    ]
    available = pd.read_csv(LAYER4_80_POOL, nrows=0).columns.tolist()
    usecols = [c for c in cols if c in available]
    pool = pd.read_csv(LAYER4_80_POOL, usecols=usecols, dtype={"ticker": str}, low_memory=False)
    pool["snapshot_date"] = pd.to_datetime(pool["snapshot_date"], errors="coerce")
    pool["signal_date_key"] = pool["snapshot_date"].dt.date.astype(str)
    signal_keys = set(signals["signal_date_key"])
    pool = pool.loc[pool["signal_date_key"].isin(signal_keys)].copy()
    pool["candidate_rank"] = pd.to_numeric(pool["pool_rank"], errors="coerce")
    pool = pool.loc[pool["candidate_rank"].between(1, 5, inclusive="both")].copy()
    pool["ticker"] = pool["ticker"].map(_ticker_str)
    pool = pool.sort_values(["signal_date_key", "candidate_rank"])
    candidates = signals.merge(pool, on="signal_date_key", how="left")
    candidates["candidate_rank"] = pd.to_numeric(candidates["candidate_rank"], errors="coerce").astype("Int64")
    candidates["candidate_score"] = pd.to_numeric(candidates["pool_selection_score"], errors="coerce")
    candidates["rank_source"] = "layer4_80_primary_pool.pool_rank"
    candidates["route_source"] = str(LAYER4_80_POOL)
    candidates["candidate_source_policy"] = "PIT Layer4 primary 80 top5 context; consensus4 exact top2-top5 is not available"
    high_cols = [
        "momentum_continuation_high_confidence",
        "pullback_repair_high_confidence",
        "overlap_reacceleration_high_confidence",
        "neutral_quality_liquidity_high_confidence",
    ]
    for col in high_cols:
        if col not in candidates.columns:
            candidates[col] = False
        candidates[col] = candidates[col].fillna(False).astype(bool)
    candidates["any_high_confidence_flag"] = candidates[high_cols].any(axis=1)
    candidates["consensus_high_confidence_flag"] = candidates["two_plus_opportunity_labels"].fillna(False).astype(bool) | candidates["any_high_confidence_flag"]
    candidates["existing_single_exception_ticker"] = candidates["holding_ticker"].map(_ticker_str)
    candidates["matches_existing_single_exception"] = candidates["ticker"].eq(candidates["existing_single_exception_ticker"])
    candidates["c2_exception_candidate_allowed"] = candidates["c2_market_health_gate"].fillna(False).astype(bool)
    candidates["future_return_used_for_rank"] = False
    candidates["diagnostic_only"] = True
    for key, value in FLAGS.items():
        candidates[key] = value
    return candidates[
        [
            "signal_date",
            "state_start_date",
            "state_end_date",
            "candidate_rank",
            "ticker",
            "name",
            "market",
            "candidate_score",
            "rank_source",
            "route_source",
            "candidate_source_policy",
            "c2_market_health_gate",
            "c2_exception_candidate_allowed",
            "raw_consensus4_exception_active",
            "exception_allowed_by_c2",
            "existing_single_exception_ticker",
            "matches_existing_single_exception",
            "consensus_high_confidence_flag",
            "any_high_confidence_flag",
            "momentum_continuation_high_confidence",
            "pullback_repair_high_confidence",
            "overlap_reacceleration_high_confidence",
            "neutral_quality_liquidity_high_confidence",
            "two_plus_opportunity_labels",
            "opportunity_label_count",
            "pool_selection_score_col",
            "layer4_pool_role",
            "is_layer4_primary_pool",
            "reference_only",
            "layer4_primary_pool_size",
            "layer4_risk_aware_score",
            "layer4_broad_opportunity_net_score",
            "layer4_retention_constrained_score",
            "layer4_c_quota_base_score",
            "RS20",
            "RS40",
            "RS60",
            "traded_value_rank_20d",
            "traded_value_rank_60d",
            "0050_above_ma60_flag",
            "0050_return_20d",
            "0050_return_40d",
            "future_return_used_for_rank",
            "diagnostic_only",
            *FLAGS.keys(),
        ]
    ]


def _unadjusted_path_map() -> dict[tuple[str, str], dict[str, Any]]:
    path = pd.read_csv(P1_UNADJUSTED_PATH, low_memory=False, dtype={"ticker": str})
    path = path.loc[
        (path["timing_variant"] == "next_day_close_entry_fixed_5td_exit")
        & (path["path_bucket"] == "ordinary_stock")
    ].copy()
    path["signal_date_key"] = pd.to_datetime(path["signal_date"], errors="coerce").dt.date.astype(str)
    path["ticker_norm"] = path["ticker"].map(_ticker_str)
    path["ready_sort"] = (~path["price_path_ready"].fillna(False).astype(bool)).astype(int)
    path = path.sort_values(["signal_date_key", "ticker_norm", "ready_sort"])
    path = path.drop_duplicates(["signal_date_key", "ticker_norm"], keep="first")
    return {
        (row.signal_date_key, row.ticker_norm): row._asdict()
        for row in path.itertuples(index=False)
    }


def _path_contract(candidates: pd.DataFrame) -> pd.DataFrame:
    path_map = _unadjusted_path_map()
    rows: list[dict[str, Any]] = []
    for row in candidates.itertuples(index=False):
        signal_key = pd.Timestamp(row.signal_date).date().isoformat()
        ticker = _ticker_str(row.ticker)
        required = bool(row.c2_exception_candidate_allowed)
        path = path_map.get((signal_key, ticker)) if required else None
        ready = bool(path and path.get("price_path_ready"))
        rows.append(
            {
                "signal_date": signal_key,
                "entry_date": row.state_start_date,
                "exit_date": row.state_end_date,
                "candidate_rank": row.candidate_rank,
                "ticker": ticker,
                "path_required_for_c2_top5_test": required,
                "timing_variant": "next_day_close_entry_fixed_5td_exit",
                "entry_price_kind": "official_unadjusted_close" if ready else "",
                "exit_price_kind": "official_unadjusted_close" if ready else "",
                "entry_close": path.get("entry_close") if path else None,
                "exit_close": path.get("exit_close") if path else None,
                "entry_open": path.get("entry_open") if path else None,
                "gross_return_unadjusted": path.get("gross_return_unadjusted") if path else None,
                "official_ohlc_path_ready": ready,
                "source_quality": path.get("source_quality") if path else "blocked_missing_selected_candidate_official_ohlc_path" if required else "not_required_when_c2_gate_false",
                "blocked_reason": "" if ready or not required else "missing_official_ohlc_path_for_layer4_top5_candidate",
                "adjusted_close_status": "blocked_not_required_for_this_unadjusted_top5_contract",
                "diagnostic_only": True,
                **FLAGS,
            }
        )
    return pd.DataFrame(rows)


def _transition_cost_design() -> pd.DataFrame:
    model = TaiwanCostModel()
    rows: list[dict[str, Any]] = []
    for max_count in [1, 2, 3, 4, 5]:
        sleeve_weight = 1.0 / max_count
        weighted_notional = DIAGNOSTIC_NOTIONAL_TWD * sleeve_weight
        for transition_action, from_asset, to_asset in [
            ("00631L_to_stock_exception_sleeve", "etf", "stock"),
            ("stock_exception_sleeve_to_00631L", "stock", "etf"),
            ("stock_to_stock_rebalance_within_exception_sleeve", "stock", "stock"),
        ]:
            sell_cost = model.sell_cost(weighted_notional, from_asset)
            buy_cost = model.buy_cost(weighted_notional)
            rows.append(
                {
                    "max_exception_stock_count": max_count,
                    "equal_weight_sleeve_weight": sleeve_weight,
                    "weighted_notional_twd": weighted_notional,
                    "transition_action": transition_action,
                    "from_asset_type": from_asset,
                    "to_asset_type": to_asset,
                    "buy_fee_twd": model.buy_cost_breakdown(weighted_notional)["buy_fee"],
                    "sell_fee_twd": model.sell_cost_breakdown(weighted_notional, from_asset)["sell_fee"],
                    "securities_transaction_tax_twd": model.sell_cost_breakdown(weighted_notional, from_asset)["securities_transaction_tax"],
                    "total_transition_cost_twd": sell_cost + buy_cost,
                    "transition_cost_rate_on_weighted_notional": (sell_cost + buy_cost) / weighted_notional,
                    "transition_cost_rate_on_total_unit_notional": (sell_cost + buy_cost) / DIAGNOSTIC_NOTIONAL_TWD,
                    "cost_model_version": cost_model_metadata()["cost_model_version"],
                    "cost_model_policy": "EP05 TaiwanCostModel; ETF/stock sell tax split; Experiments must apply realized transitions",
                    "diagnostic_only": True,
                    **FLAGS,
                }
            )
    return pd.DataFrame(rows)


def _blocked_proxy_audit(candidates: pd.DataFrame, paths: pd.DataFrame) -> pd.DataFrame:
    required = paths["path_required_for_c2_top5_test"].fillna(False).astype(bool)
    ready = paths["official_ohlc_path_ready"].fillna(False).astype(bool)
    return pd.DataFrame(
        [
            {
                "field_or_component": "top1_to_top5_candidate_rows",
                "status": "ready",
                "ready_rows": len(candidates),
                "requested_rows": len(candidates),
                "reason": "Layer4 80 primary pool covers P1 signal dates and provides PIT pool_rank/score.",
            },
            {
                "field_or_component": "consensus4_exact_top2_to_top5",
                "status": "blocked_proxy",
                "ready_rows": 0,
                "requested_rows": len(candidates),
                "reason": "Existing consensus4 trace has only one exception ticker; top5 candidates use Layer4 high-confidence/risk-aware PIT rank source instead.",
            },
            {
                "field_or_component": "official_ohlc_path_for_c2_true_top5",
                "status": "partial",
                "ready_rows": int((required & ready).sum()),
                "requested_rows": int(required.sum()),
                "reason": "Local selected-stock path source covers only prior selected rows; Layer4 top5 candidate path requires Radar/Data selected-ticker-only OHLC source fill.",
            },
            {
                "field_or_component": "adjusted_close",
                "status": "blocked",
                "ready_rows": 0,
                "requested_rows": int(required.sum()),
                "reason": "Adjusted close remains blocked; this contract is official unadjusted OHLC diagnostic-only where path exists.",
            },
            {
                "field_or_component": "cash_bear_classifier",
                "status": "blocked",
                "ready_rows": 0,
                "requested_rows": len(candidates),
                "reason": "No accepted cash/bear classifier; no cash rule fabricated.",
            },
        ]
    )


def _manifest(files: list[Path], readiness: dict[str, Any]) -> dict[str, Any]:
    return {
        "task_id": TASK_ID,
        "status": readiness["status"],
        "created_at": pd.Timestamp.now(tz="Asia/Taipei").isoformat(),
        "output_dir": str(OUTPUT_DIR),
        "inputs": {
            "layer4_80_pool": str(LAYER4_80_POOL),
            "c2_state_machine": str(C2_STATE_MACHINE),
            "p1_unadjusted_path": str(P1_UNADJUSTED_PATH),
        },
        "artifacts": [
            {
                "name": path.name,
                "path": str(path),
                "sha256": _sha256(path),
                "rows": int(pd.read_csv(path, low_memory=False).shape[0]) if path.suffix == ".csv" else None,
            }
            for path in files
        ],
        "readiness": readiness,
        "flags": FLAGS,
    }


def build() -> dict[str, Any]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    signals = _c2_signal_frame()
    candidates = _layer4_top5_candidates(signals)
    paths = _path_contract(candidates)
    costs = _transition_cost_design()
    missing = paths.loc[
        paths["path_required_for_c2_top5_test"].fillna(False).astype(bool)
        & ~paths["official_ohlc_path_ready"].fillna(False).astype(bool)
    ].copy()
    blocked = _blocked_proxy_audit(candidates, paths)
    future = pd.DataFrame(
        [
            {
                "audit_item": "candidate_rank_source",
                "violation_count": 0,
                "status": "pass",
                "notes": "Ranks use Layer4 PIT pool_rank/pool_selection_score; no future return or winner label is used.",
            },
            {
                "audit_item": "price_path",
                "violation_count": 0,
                "status": "partial_blocked",
                "notes": "Missing path rows are explicit; no silent fill and no 00631L+excess reconstruction.",
            },
        ]
    )
    required = paths["path_required_for_c2_top5_test"].fillna(False).astype(bool)
    ready = paths["official_ohlc_path_ready"].fillna(False).astype(bool)
    readiness = {
        "task_id": TASK_ID,
        "status": "p1_c2_top5_exception_candidate_contract_ready_path_partial_blocked",
        "ready_for_p1_c2_top5_multi_stock_exception_count_diagnostic": False,
        "candidate_contract_ready": True,
        "official_ohlc_path_ready": bool((required & ~ready).sum() == 0),
        "ready_for_radar_top5_selected_candidate_ohlc_source_fill": bool((required & ~ready).sum() > 0),
        "ready_for_experiments": False,
        "ready_for_formal": False,
        "ready_for_strategy_replay": False,
        "signal_rows": int(signals.shape[0]),
        "candidate_rows": int(candidates.shape[0]),
        "c2_true_candidate_path_required_rows": int(required.sum()),
        "official_ohlc_ready_rows": int((required & ready).sum()),
        "official_ohlc_blocked_rows": int((required & ~ready).sum()),
        "unique_blocked_tickers": int(missing["ticker"].nunique()),
        "transition_cost_design_ready": True,
        "adjusted_close_ready": False,
        "cash_bear_classifier_ready": False,
        "future_data_violation_count": 0,
        "not_live_rule": True,
        "forward_returns_live_rule_usage": False,
        **FLAGS,
    }

    artifacts = {
        "p1_c2_top5_exception_candidate_contract.csv": candidates,
        "p1_c2_top5_exception_candidate_path_contract.csv": paths,
        "p1_c2_top5_exception_candidate_missing_path_ledger.csv": missing,
        "p1_c2_top5_exception_candidate_transition_cost_design.csv": costs,
        "p1_c2_top5_exception_candidate_blocked_proxy_audit.csv": blocked,
        "p1_c2_top5_exception_candidate_future_data_audit.csv": future,
    }
    files: list[Path] = []
    for name, df in artifacts.items():
        path = OUTPUT_DIR / name
        df.to_csv(path, index=False, encoding="utf-8-sig")
        files.append(path)

    readiness_path = OUTPUT_DIR / "readiness_for_p1_c2_top5_multi_stock_exception_candidate_contract.json"
    readiness_path.write_text(json.dumps(readiness, ensure_ascii=False, indent=2), encoding="utf-8")
    files.append(readiness_path)

    summary = "\n".join(
        [
            "# P1 C2 top5 multi-stock exception candidate contract",
            "",
            f"- task_id: `{TASK_ID}`",
            "- status: `p1_c2_top5_exception_candidate_contract_ready_path_partial_blocked`",
            f"- candidate_rows: {readiness['candidate_rows']}",
            f"- c2_true_candidate_path_required_rows: {readiness['c2_true_candidate_path_required_rows']}",
            f"- official_ohlc_ready_rows: {readiness['official_ohlc_ready_rows']}",
            f"- official_ohlc_blocked_rows: {readiness['official_ohlc_blocked_rows']}",
            "- candidate ranks use Layer4 80 primary pool PIT pool_rank / pool_selection_score. Existing consensus4 exact top2~top5 is blocked, so this is Layer4 high-confidence/risk-aware top5 exception candidate contract.",
            "- transition cost design uses EP05 TaiwanCostModel with ETF/stock tax split and equal-weight sleeve cost basis for max1~max5.",
            "- adjusted close remains blocked; official unadjusted OHLC is diagnostic-only where available.",
            "- Not ready for Experiments rerun until Radar/Data fills missing official OHLC path rows for C2-true top5 candidates.",
            "",
            "下一棒：交 Radar/Data 做 selected-ticker-only top5 candidate official OHLC source fill，不做 full-market mass download。",
            "",
            "Flags: formal_model_changed=false; trade_decision_changed=false; active_in_trade_decision=false; report_changed=false; portfolio_replay_executed=false; ready_for_strategy_replay=false; ready_for_formal=false; not_live_rule=true; forward_returns_live_rule_usage=false.",
            "",
            "完成後如果下一棒明確，請直接指派下一個 thread；如果下一棒不明確，請回報 Strategy Center 判斷。不要完成後停住不回報。",
        ]
    )
    summary_path = OUTPUT_DIR / "final_summary_zh.md"
    summary_path.write_text(summary, encoding="utf-8")
    files.append(summary_path)

    manifest = _manifest(files, readiness)
    manifest_path = OUTPUT_DIR / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return readiness


if __name__ == "__main__":
    print(json.dumps(build(), ensure_ascii=False, indent=2))
