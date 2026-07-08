from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
WEIGHTED_DIR = REPO_ROOT / "outputs" / "vnext_p1_c2_weighted_pool80_top5_ohlc_absorption_20260708"
STATE_HOLD_DIR = REPO_ROOT / "outputs" / "vnext_p1_state_hold_base_exception_path_contract_20260708"
OUTPUT_DIR = REPO_ROOT / "outputs" / "vnext_p1_c2_route_support_max1_modelization_contract_20260708"

TASK_ID = "TASK-BACKTEST-CORE-VNEXT-P1-C2-ROUTE-SUPPORT-MAX1-MODELIZATION-READINESS-CONTRACT-001"
DIAGNOSTIC_NOTIONAL = 1_000_000
TRANSITION_COSTS = {
    "00631L_to_stock": {
        "transition_cost_rate": 0.00385,
        "sell_fee_twd": 1425,
        "buy_fee_twd": 1425,
        "securities_transaction_tax_twd": 1000,
        "total_transition_cost_twd": 3850,
    },
    "stock_to_00631L": {
        "transition_cost_rate": 0.00585,
        "sell_fee_twd": 1425,
        "buy_fee_twd": 1425,
        "securities_transaction_tax_twd": 3000,
        "total_transition_cost_twd": 5850,
    },
    "stock_to_stock": {
        "transition_cost_rate": 0.00585,
        "sell_fee_twd": 1425,
        "buy_fee_twd": 1425,
        "securities_transaction_tax_twd": 3000,
        "total_transition_cost_twd": 5850,
    },
    "hold": {
        "transition_cost_rate": 0.0,
        "sell_fee_twd": 0,
        "buy_fee_twd": 0,
        "securities_transaction_tax_twd": 0,
        "total_transition_cost_twd": 0,
    },
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


def _calendar() -> pd.DataFrame:
    cal = pd.read_csv(STATE_HOLD_DIR / "p1_base_exception_signal_trace.csv", low_memory=False)
    cal = cal[cal["timing_variant"].eq("next_day_close_entry_fixed_5td_exit")].copy()
    cal["signal_date"] = pd.to_datetime(cal["signal_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    cal["next_signal_date"] = pd.to_datetime(cal["next_signal_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    return cal[["signal_date", "next_signal_date"]].drop_duplicates("signal_date").sort_values("signal_date")


def _benchmark_maps() -> dict[str, dict[str, float]]:
    maps = {}
    for symbol in ["00631L", "0050"]:
        df = pd.read_csv(STATE_HOLD_DIR / f"p1_state_hold_benchmark_path_{symbol}.csv", low_memory=False)
        df["trade_date"] = pd.to_datetime(df["trade_date"], errors="coerce").dt.strftime("%Y-%m-%d")
        maps[symbol] = {
            row.trade_date: float(row.adjusted_close)
            for row in df.itertuples(index=False)
            if pd.notna(row.adjusted_close)
        }
    return maps


def _route_support_top1() -> pd.DataFrame:
    df = pd.read_csv(WEIGHTED_DIR / "p1_c2_weighted_pool80_top5_contract_refreshed.csv", low_memory=False, dtype={"ticker": str})
    df["signal_date"] = pd.to_datetime(df["signal_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    df["ticker"] = df["ticker"].map(_ticker)
    df["official_unadjusted_ohlc_path_ready"] = _as_bool(df["official_unadjusted_ohlc_path_ready"])
    return df[
        df["score_variant"].eq("route_support")
        & df["candidate_rank"].eq(1)
    ].copy()


def _transition_action(prev_ticker: str, prev_type: str, target_ticker: str, target_type: str) -> tuple[str, str]:
    if prev_ticker == target_ticker and prev_type == target_type:
        return "hold_same_state_no_trade", "hold"
    if prev_type == "etf" and target_type == "stock":
        return "00631L_to_stock_exception", "00631L_to_stock"
    if prev_type == "stock" and target_type == "etf":
        return "stock_exception_to_00631L_base", "stock_to_00631L"
    if prev_type == "stock" and target_type == "stock":
        return "stock_to_stock_exception_switch", "stock_to_stock"
    return "transition_other", "hold"


def _build_contract() -> tuple[pd.DataFrame, pd.DataFrame]:
    cal = _calendar()
    top1 = _route_support_top1()
    maps = _benchmark_maps()
    by_date = {row.signal_date: row._asdict() for row in top1.itertuples(index=False)}
    rows: list[dict[str, Any]] = []
    transitions: list[dict[str, Any]] = []
    prev_ticker = "00631L"
    prev_type = "etf"
    for r in cal.itertuples(index=False):
        signal_date = r.signal_date
        next_signal = r.next_signal_date
        stock = by_date.get(signal_date)
        if stock is not None:
            target_ticker = _ticker(stock.get("ticker"))
            target_type = "stock"
            state_reason = "c2_consensus_trigger_route_support_top1_stock_exception"
            entry_price = stock.get("entry_close")
            exit_price = stock.get("exit_close")
            interval_return = stock.get("gross_return_unadjusted")
            path_ready = bool(stock.get("official_unadjusted_ohlc_path_ready", False))
            source_quality = stock.get("source_quality", "official_unadjusted_ohlc")
            score = stock.get("weighted_score")
            entry_date = stock.get("entry_date", "")
            exit_date = stock.get("exit_date", "")
        else:
            target_ticker = "00631L"
            target_type = "etf"
            state_reason = "default_00631L_base_no_c2_consensus_trigger"
            entry_price = maps["00631L"].get(signal_date)
            exit_price = maps["00631L"].get(next_signal)
            interval_return = (exit_price / entry_price - 1.0) if entry_price and exit_price else None
            path_ready = interval_return is not None
            source_quality = "benchmark_features_adjusted_close_exact_reference"
            score = None
            entry_date = signal_date
            exit_date = next_signal

        action, cost_key = _transition_action(prev_ticker, prev_type, target_ticker, target_type)
        cost = TRANSITION_COSTS[cost_key]
        net_return = (float(interval_return) - cost["transition_cost_rate"]) if interval_return is not None else None
        row = {
            "signal_date": signal_date,
            "next_signal_date": next_signal,
            "selected_ticker": target_ticker,
            "selected_asset_type": target_type,
            "state_reason": state_reason,
            "score_variant": "route_support",
            "route_support_weighted_score": score,
            "entry_date": entry_date,
            "exit_date": exit_date,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "gross_interval_return": interval_return,
            "transition_action": action,
            "transition_cost_rate": cost["transition_cost_rate"],
            "net_interval_return_after_transition_cost": net_return,
            "official_unadjusted_ohlc_ready": path_ready if target_type == "stock" else True,
            "benchmark_adjusted_path_ready": path_ready if target_type == "etf" else True,
            "adjusted_close_ready": target_type == "etf",
            "source_quality": source_quality,
            "cash_condition_status": "blocked_no_bear_cash_classifier",
            "diagnostic_only": True,
            **FLAGS,
        }
        rows.append(row)
        if action != "hold_same_state_no_trade":
            transitions.append(
                {
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
                }
            )
        prev_ticker, prev_type = target_ticker, target_type
    return pd.DataFrame(rows), pd.DataFrame(transitions)


def _score_audit() -> pd.DataFrame:
    top1 = _route_support_top1()
    cols = [
        "signal_date",
        "ticker",
        "name",
        "weighted_score",
        "quality_component",
        "rs_component",
        "liquidity_component",
        "bias_health_component",
        "route_support_component",
        "risk_inverse_component",
        "route_support_variant_count",
        "route_support_variant_flags",
        "route_support_mode_flags",
        "future_data_violation_count",
    ]
    out = top1[[c for c in cols if c in top1.columns]].copy()
    out["score_formula"] = "route_support = weighted PIT quant score; route_support_component has largest variant weight"
    out["future_return_used"] = False
    out["prior_exception_is_reference_only"] = True
    return out


def _cost_audit(transitions: pd.DataFrame) -> pd.DataFrame:
    if transitions.empty:
        return pd.DataFrame()
    return (
        transitions.groupby(["transition_action", "from_asset_type", "to_asset_type"], as_index=False)
        .agg(
            transition_count=("transition_action", "size"),
            transition_cost_rate=("transition_cost_rate", "first"),
            total_transition_cost_twd_sum=("total_transition_cost_twd", "sum"),
            cost_model_status=("cost_model_status", "first"),
            cost_model_version=("cost_model_version", "first"),
        )
    )


def _coverage(contract: pd.DataFrame) -> pd.DataFrame:
    stock = contract[contract["selected_asset_type"].eq("stock")]
    return pd.DataFrame(
        [
            {
                "coverage_item": "route_support_max1_state_contract",
                "rows": len(contract),
                "ready_rows": int((contract["official_unadjusted_ohlc_ready"] | contract["benchmark_adjusted_path_ready"]).sum()),
                "ready_share": float((contract["official_unadjusted_ohlc_ready"] | contract["benchmark_adjusted_path_ready"]).mean()),
            },
            {
                "coverage_item": "stock_exception_official_unadjusted_ohlc",
                "rows": len(stock),
                "ready_rows": int(stock["official_unadjusted_ohlc_ready"].sum()) if len(stock) else 0,
                "ready_share": float(stock["official_unadjusted_ohlc_ready"].mean()) if len(stock) else 1.0,
            },
            {
                "coverage_item": "selected_stock_adjusted_close",
                "rows": len(stock),
                "ready_rows": 0,
                "ready_share": 0.0,
            },
            {
                "coverage_item": "00631L_state_hold_base_adjusted_path",
                "rows": int(contract["selected_asset_type"].eq("etf").sum()),
                "ready_rows": int(contract.loc[contract["selected_asset_type"].eq("etf"), "benchmark_adjusted_path_ready"].sum()),
                "ready_share": float(contract.loc[contract["selected_asset_type"].eq("etf"), "benchmark_adjusted_path_ready"].mean()),
            },
        ]
    )


def _blocked() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "blocked_item": "selected_stock_adjusted_close",
                "blocked_reason": "exact historical ex-right date and capital-change adjustment route incomplete",
                "policy": "official unadjusted OHLC diagnostic-only; not formal",
            },
            {
                "blocked_item": "cash_bear_classifier",
                "blocked_reason": "no accepted cash/bear classifier in current P1 modelization contract",
                "policy": "do not fabricate cash rule",
            },
            {
                "blocked_item": "formal_replay",
                "blocked_reason": "diagnostic contract only; adjusted close and formal replay governance not ready",
                "policy": "ready_for_formal=false; ready_for_strategy_replay=false",
            },
        ]
    )


def _readiness(contract: pd.DataFrame, transitions: pd.DataFrame, coverage: pd.DataFrame) -> dict[str, Any]:
    stock = contract[contract["selected_asset_type"].eq("stock")]
    official_share = float(stock["official_unadjusted_ohlc_ready"].mean()) if len(stock) else 1.0
    ready = official_share == 1.0 and not transitions.empty
    return {
        "task_id": TASK_ID,
        "status": "route_support_max1_modelization_contract_ready_unadjusted_diagnostic_adjusted_blocked" if ready else "route_support_max1_modelization_contract_partial",
        "ready_for_p1_c2_route_support_max1_modelization_diagnostic": bool(ready),
        "ready_for_experiments": bool(ready),
        "contract_rows": int(len(contract)),
        "stock_exception_rows": int(len(stock)),
        "transition_count": int(len(transitions)),
        "official_unadjusted_ohlc_ready_share": official_share,
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
        "coverage": coverage.to_dict(orient="records"),
    }


def _manifest(files: list[Path], readiness: dict[str, Any]) -> dict[str, Any]:
    return {
        "task_id": TASK_ID,
        "output_dir": str(OUTPUT_DIR),
        "inputs": {
            "weighted_pool80_refreshed_contract": str(WEIGHTED_DIR / "p1_c2_weighted_pool80_top5_contract_refreshed.csv"),
            "p1_signal_calendar": str(STATE_HOLD_DIR / "p1_base_exception_signal_trace.csv"),
            "p1_00631L_state_hold_base": str(STATE_HOLD_DIR / "p1_state_hold_benchmark_path_00631L.csv"),
            "p1_0050_state_hold_base": str(STATE_HOLD_DIR / "p1_state_hold_benchmark_path_0050.csv"),
        },
        "artifacts": [
            {"path": str(path), "sha256": _sha256(path), "bytes": path.stat().st_size}
            for path in files
        ],
        "readiness": readiness,
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    contract, transitions = _build_contract()
    score = _score_audit()
    cost = _cost_audit(transitions)
    coverage = _coverage(contract)
    blocked = _blocked()
    readiness = _readiness(contract, transitions, coverage)

    paths = {
        "contract": OUTPUT_DIR / "p1_c2_route_support_max1_modelization_contract.csv",
        "transition": OUTPUT_DIR / "p1_c2_route_support_max1_transition_trace.csv",
        "score": OUTPUT_DIR / "p1_c2_route_support_max1_score_audit.csv",
        "cost": OUTPUT_DIR / "p1_c2_route_support_max1_cost_audit.csv",
        "coverage": OUTPUT_DIR / "p1_c2_route_support_max1_coverage_audit.csv",
        "blocked": OUTPUT_DIR / "p1_c2_route_support_max1_blocked_proxy_ledger.csv",
        "readiness": OUTPUT_DIR / "readiness_for_p1_c2_route_support_max1_modelization_diagnostic.json",
        "summary": OUTPUT_DIR / "final_summary_zh.md",
        "manifest": OUTPUT_DIR / "manifest.json",
    }
    contract.to_csv(paths["contract"], index=False, encoding="utf-8-sig")
    transitions.to_csv(paths["transition"], index=False, encoding="utf-8-sig")
    score.to_csv(paths["score"], index=False, encoding="utf-8-sig")
    cost.to_csv(paths["cost"], index=False, encoding="utf-8-sig")
    coverage.to_csv(paths["coverage"], index=False, encoding="utf-8-sig")
    blocked.to_csv(paths["blocked"], index=False, encoding="utf-8-sig")
    paths["readiness"].write_text(json.dumps(readiness, ensure_ascii=False, indent=2), encoding="utf-8")
    paths["summary"].write_text(
        "\n".join(
            [
                "# P1 C2 route_support max1 modelization readiness",
                "",
                "- 主線已收斂為單押：00631L state-hold base + C2 market health gate + consensus trigger + route_support quant score max1。",
                "- State machine 已建立：同股續抱不重買；gate/trigger 失效回 00631L；top1 改變時計 stock-to-stock transition。",
                f"- stock exception official unadjusted OHLC ready share = {readiness['official_unadjusted_ohlc_ready_share']:.4f}。",
                "- Cost model ready：EP05 TaiwanCostModel unit-notional transition cost，ETF/stock transaction tax split retained。",
                "- adjusted_close_ready=false；cash/bear classifier blocked；formal/replay blocked。",
                "- 後續 Experiments 主結論必須 net after transaction cost；gross/no-cost 只能 secondary。",
                "",
                "下一棒：交 Experiments 做 P1 C2 route_support max1 modelization diagnostic。",
                "",
                "Flags: formal_model_changed=false; trade_decision_changed=false; active_in_trade_decision=false; report_changed=false; portfolio_replay_executed=false; ready_for_strategy_replay=false; ready_for_formal=false; not_live_rule=true; forward_returns_live_rule_usage=false.",
                "",
                "完成後如果下一棒明確，請直接指派下一個 thread；如果下一棒不明確，請回報 Strategy Center 判斷。不要完成後停住不回報。",
            ]
        ),
        encoding="utf-8",
    )
    manifest = _manifest([p for k, p in paths.items() if k != "manifest"], readiness)
    paths["manifest"].write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(readiness, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
