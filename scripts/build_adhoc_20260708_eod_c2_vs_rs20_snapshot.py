from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "vnext_adhoc_20260708_eod_c2_vs_rs20_top1_signal_snapshot_20260708"
REQUESTED_DATE = "2026-07-08"


def read_csv(path: Path, **kwargs) -> pd.DataFrame:
    return pd.read_csv(path, **kwargs)


def max_date(path: Path, date_col: str) -> str | None:
    if not path.exists():
        return None
    s = read_csv(path, usecols=[date_col])[date_col].astype(str)
    return str(s.max()) if len(s) else None


def has_date(path: Path, date_col: str, date_value: str) -> bool:
    if not path.exists():
        return False
    s = read_csv(path, usecols=[date_col])[date_col].astype(str)
    return bool((s == date_value).any())


def latest_row(path: Path, date_col: str, columns: list[str] | None = None) -> pd.Series | None:
    if not path.exists():
        return None
    df = read_csv(path, usecols=columns) if columns else read_csv(path)
    if df.empty:
        return None
    latest = df[date_col].astype(str).max()
    rows = df[df[date_col].astype(str) == latest]
    return rows.iloc[0] if len(rows) else None


def as_bool(value) -> bool | None:
    if pd.isna(value):
        return None
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes"}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    layer4_path = ROOT / "outputs/vnext_layer4_80_primary_pool_contract_20260708/layer4_80_primary_pool_contract.csv"
    market_path = ROOT / "outputs/vnext_regime_switch_hybrid_route_market_fields_path_materialization_20260708/regime_switch_market_regime_fields.csv"
    pool_path = ROOT / "outputs/vnext_regime_switch_hybrid_route_market_fields_path_materialization_20260708/regime_switch_pool_regime_fields.csv"
    trigger_path = ROOT / "outputs/vnext_full_period_exact_consensus_trigger_contract_20260708/full_period_exact_consensus_trigger_contract.csv"
    max1_path = ROOT / "outputs/vnext_route_support_max1_full_period_same_basis_contract_20260708/route_support_max1_full_period_same_basis_modelization_contract.csv"
    radar_full_manifest = Path("C:/Users/zergv/Documents/Codex/2026-05-23/ai-stock-rotation-radar-https-docs/outputs/radar_dynamic_pool1_all_listed_liquid_universe_full_sweep_20260703/accepted_liquidity_shard_manifest.csv")
    radar_pit_daily = Path("C:/Users/zergv/Documents/Codex/2026-05-23/ai-stock-rotation-radar-https-docs/outputs/radar_dynamic_pool1_all_listed_liquid_universe_pit_daily_20260703/accepted_liquidity_rows.csv")

    field_dates = {
        "formal_daily_20260708_report_anchor_local_file": None,
        "radar_full_sweep_official_ohlcv": max_date(radar_full_manifest, "last_date"),
        "radar_pit_daily_sample_official_ohlcv": max_date(radar_pit_daily, "date"),
        "layer4_primary80_snapshot": max_date(layer4_path, "snapshot_date"),
        "0050_market_regime_fields": max_date(market_path, "snapshot_date"),
        "dynamic80_pool_regime_fields": max_date(pool_path, "snapshot_date"),
        "exact_consensus_trigger": max_date(trigger_path, "signal_date"),
        "route_support_max1_state_machine": max_date(max1_path, "signal_date"),
    }

    requested_presence = {
        "layer4_primary80_snapshot": has_date(layer4_path, "snapshot_date", REQUESTED_DATE),
        "0050_market_regime_fields": has_date(market_path, "snapshot_date", REQUESTED_DATE),
        "dynamic80_pool_regime_fields": has_date(pool_path, "snapshot_date", REQUESTED_DATE),
        "exact_consensus_trigger": has_date(trigger_path, "signal_date", REQUESTED_DATE),
        "route_support_max1_state_machine": has_date(max1_path, "signal_date", REQUESTED_DATE),
        "radar_pit_daily_sample_official_ohlcv": has_date(radar_pit_daily, "date", REQUESTED_DATE),
    }

    common_reference_date = min(
        d for k, d in field_dates.items()
        if k in {
            "layer4_primary80_snapshot",
            "0050_market_regime_fields",
            "dynamic80_pool_regime_fields",
            "exact_consensus_trigger",
            "route_support_max1_state_machine",
        } and d is not None
    )

    max1_latest = latest_row(max1_path, "signal_date")
    market_latest = latest_row(
        market_path,
        "snapshot_date",
        [
            "snapshot_date",
            "0050_adjusted_close",
            "0050_return_20d",
            "0050_return_40d",
            "0050_return_60d",
            "0050_ma60",
            "0050_price_vs_ma60",
            "0050_bias20",
            "0050_bias60",
        ],
    )

    layer_cols = [
        "snapshot_date",
        "ticker",
        "name",
        "RS20",
        "RS40",
        "RS60",
        "traded_value_rank_20d",
        "traded_value_rank_60d",
        "layer1_pass_bottom30",
        "risk_overheat_penalty_context",
        "rs60_high_short_rs_weakening_exhaustion_context",
        "volatility_pctile_by_week",
    ]
    layer4 = read_csv(layer4_path, usecols=layer_cols)
    layer4_ref = layer4[layer4["snapshot_date"].astype(str) == field_dates["layer4_primary80_snapshot"]].copy()
    rs20_ref = layer4_ref.sort_values(["RS20", "traded_value_rank_20d"], ascending=[False, True]).head(3)

    rs20_top3 = [
        {
            "ticker": str(r["ticker"]),
            "name": r["name"],
            "RS20": None if pd.isna(r["RS20"]) else float(r["RS20"]),
            "RS40": None if pd.isna(r["RS40"]) else float(r["RS40"]),
            "RS60": None if pd.isna(r["RS60"]) else float(r["RS60"]),
            "traded_value_rank_20d": None if pd.isna(r["traded_value_rank_20d"]) else float(r["traded_value_rank_20d"]),
            "risk_overheat_penalty_context": as_bool(r["risk_overheat_penalty_context"]),
            "source_quality": "reference_only_latest_layer4_rs20_sort_not_20260708_not_full_risk_tiebreak",
        }
        for _, r in rs20_ref.iterrows()
    ]

    c2_reference_stock_top1_ticker = None
    c2_reference_stock_top1_name = None
    route_support_score = None
    if max1_latest is not None:
        if str(max1_latest.get("selected_asset_type")) == "stock":
            c2_reference_stock_top1_ticker = str(max1_latest.get("selected_ticker"))
            c2_reference_stock_top1_name = ""
        else:
            # Latest state-machine selected 00631L. Expose the stored route-support
            # score only as context; it is not a selected stock.
            route_support_score = None if pd.isna(max1_latest.get("route_support_weighted_score")) else float(max1_latest.get("route_support_weighted_score"))

    c2_gate_pass_latest = as_bool(max1_latest.get("c2_market_health_gate")) if max1_latest is not None else None
    consensus_trigger_latest = as_bool(max1_latest.get("consensus_trigger")) if max1_latest is not None else None
    rs20_top1 = rs20_top3[0] if rs20_top3 else {}

    market_data_ready = all(requested_presence.values())

    snapshot = {
        "task": "TASK-BACKTEST-CORE-VNEXT-ADHOC-20260708-EOD-C2-VS-RS20-TOP1-SIGNAL-SNAPSHOT-001",
        "status": "blocked_for_vnext_20260708_materialization_missing_but_latest_reference_exported",
        "as_of_requested_date": REQUESTED_DATE,
        "as_of_data_date": REQUESTED_DATE if market_data_ready else common_reference_date,
        "field_as_of_dates": field_dates,
        "requested_date_field_presence": requested_presence,
        "market_data_ready": market_data_ready,
        "formal_daily_report_anchor_status": "user_reports_formal_20260708_report_exists_but_no_local_Core_or_DAILY_STOCK_artifact_found_in_checked_worktrees",
        "c2_gate_pass": False if not market_data_ready else c2_gate_pass_latest,
        "consensus_trigger_pass": False if not market_data_ready else consensus_trigger_latest,
        "c2_selected_asset_type": "blocked" if not market_data_ready else ("stock" if c2_gate_pass_latest and consensus_trigger_latest else "00631L_fallback"),
        "c2_selected_ticker": None if not market_data_ready else str(max1_latest.get("selected_ticker")),
        "c2_selected_name": None if not market_data_ready else ("00631L" if str(max1_latest.get("selected_ticker")) == "00631L" else ""),
        "c2_reference_stock_top1_ticker": c2_reference_stock_top1_ticker,
        "c2_reference_stock_top1_name": c2_reference_stock_top1_name,
        "c2_latest_reference_state_reason": None if max1_latest is None else str(max1_latest.get("state_reason")),
        "c2_latest_reference_route_support_score": route_support_score,
        "c2_definition": "0050_price_vs_ma60>=0 AND 0050_return_20d>=0 AND 0050_return_40d>=0",
        "latest_0050_market_reference": {} if market_latest is None else {
            "snapshot_date": str(market_latest["snapshot_date"]),
            "0050_adjusted_close": float(market_latest["0050_adjusted_close"]),
            "0050_return_20d": float(market_latest["0050_return_20d"]),
            "0050_return_40d": float(market_latest["0050_return_40d"]),
            "0050_return_60d": float(market_latest["0050_return_60d"]),
            "0050_price_vs_ma60": float(market_latest["0050_price_vs_ma60"]),
            "0050_bias20": float(market_latest["0050_bias20"]),
            "0050_bias60": float(market_latest["0050_bias60"]),
        },
        "rs20_top1_ticker": None if not market_data_ready else rs20_top1.get("ticker"),
        "rs20_top1_name": None if not market_data_ready else rs20_top1.get("name"),
        "rs20_reference_top1_ticker": rs20_top1.get("ticker"),
        "rs20_reference_top1_name": rs20_top1.get("name"),
        "rs20_top3_tickers": None if not market_data_ready else [x["ticker"] for x in rs20_top3],
        "rs20_reference_top3": rs20_top3,
        "rs20_route_status": "blocked_for_20260708_vnext_layer4_and_risk_tiebreak_fields_missing; latest_reference_uses_layer4_rs20_sort_only",
        "same_top1_flag": None,
        "diagnostic_only": True,
        "not_live_trade_decision": True,
        "formal_model_changed": False,
        "trade_decision_changed": False,
        "active_in_trade_decision": False,
        "report_changed": False,
        "portfolio_replay_executed": False,
        "ready_for_strategy_replay": False,
        "ready_for_formal": False,
        "not_live_rule": True,
        "forward_returns_live_rule_usage": False,
        "future_data_violation_count": 0,
        "next_owner": "Radar/Data",
        "next_task": "TASK-RADAR-DATA-VNEXT-ADHOC-20260708-EOD-VNEXT-SIGNAL-SNAPSHOT-SOURCE-FILL-001",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    if market_data_ready:
        snapshot["same_top1_flag"] = snapshot["c2_selected_ticker"] == snapshot["rs20_top1_ticker"]

    with (OUT / "adhoc_20260708_eod_c2_vs_rs20_top1_signal_snapshot.json").open("w", encoding="utf-8") as f:
        json.dump(snapshot, f, ensure_ascii=False, indent=2)

    row = {
        "as_of_requested_date": snapshot["as_of_requested_date"],
        "as_of_data_date": snapshot["as_of_data_date"],
        "market_data_ready": snapshot["market_data_ready"],
        "c2_gate_pass": snapshot["c2_gate_pass"],
        "consensus_trigger_pass": snapshot["consensus_trigger_pass"],
        "c2_selected_asset_type": snapshot["c2_selected_asset_type"],
        "c2_selected_ticker": snapshot["c2_selected_ticker"],
        "c2_selected_name": snapshot["c2_selected_name"],
        "c2_reference_stock_top1_ticker": snapshot["c2_reference_stock_top1_ticker"],
        "c2_reference_stock_top1_name": snapshot["c2_reference_stock_top1_name"],
        "rs20_top1_ticker": snapshot["rs20_top1_ticker"],
        "rs20_top1_name": snapshot["rs20_top1_name"],
        "rs20_top3_tickers": "" if snapshot["rs20_top3_tickers"] is None else "|".join(snapshot["rs20_top3_tickers"]),
        "rs20_reference_top1_ticker": snapshot["rs20_reference_top1_ticker"],
        "rs20_reference_top1_name": snapshot["rs20_reference_top1_name"],
        "rs20_reference_top3_tickers": "|".join([x["ticker"] for x in rs20_top3]),
        "same_top1_flag": snapshot["same_top1_flag"],
        "diagnostic_only": True,
        "not_live_trade_decision": True,
        "status_zh": "今日 vNext 欄位尚未 materialize，不能產出 2026-07-08 正式診斷；最新 reference 為 2026-06-29。",
    }
    pd.DataFrame([row]).to_csv(OUT / "adhoc_20260708_eod_c2_vs_rs20_top1_signal_snapshot_zh.csv", index=False, encoding="utf-8-sig")

    blocked_rows = []
    for field, present in requested_presence.items():
        blocked_rows.append({
            "field": field,
            "requested_date": REQUESTED_DATE,
            "field_as_of_date": field_dates.get(field),
            "ready_for_20260708": present,
            "blocked_reason": "missing_20260708_vnext_materialized_field_or_source" if not present else "",
            "next_owner": "Radar/Data" if field in {"radar_pit_daily_sample_official_ohlcv"} else "Core/Data_after_source_anchor",
        })
    blocked_rows.append({
        "field": "formal_daily_20260708_report_anchor_local_file",
        "requested_date": REQUESTED_DATE,
        "field_as_of_date": "",
        "ready_for_20260708": False,
        "blocked_reason": "formal report reportedly exists but no local Core/DAILY_STOCK artifact or cache found in checked worktrees; need report manifest/cache path or Radar source fill",
        "next_owner": "Radar/Data_or_Strategy_Center_provide_formal_report_artifact_path",
    })
    pd.DataFrame(blocked_rows).to_csv(OUT / "adhoc_20260708_eod_c2_vs_rs20_top1_blocked_proxy_audit.csv", index=False, encoding="utf-8-sig")

    summary = f"""# Ad-hoc 2026-07-08 EOD C2 vs RS20 Top1 Signal Snapshot

## 結論

- `as_of_requested_date=2026-07-08`。
- Core 本機沒有可驗證的 vNext 2026-07-08 Layer0-Layer4 / C2 / consensus / route_support / RS20 top3 materialized snapshot。
- 正式版今日報告雖由 Strategy Center 指出已產出，但本 Core 工作樹與 checked DAILY_STOCK local clone 沒有對應 manifest/cache artifact 可作可追溯 anchor。
- 因此本包不輸出今日個股 top1，不把 2026-06-29 reference 冒充 2026-07-08。

## 最新可用 reference

- common vNext reference date: `{common_reference_date}`。
- latest route_support max1 state-machine date: `{field_dates['route_support_max1_state_machine']}`。
- latest state reason: `{snapshot['c2_latest_reference_state_reason']}`。
- latest C2 gate: `{c2_gate_pass_latest}`；latest consensus trigger: `{consensus_trigger_latest}`。
- latest Layer4 RS20 reference top3: {', '.join([x['ticker'] + ' ' + str(x['name']) for x in rs20_top3])}。
- 這些只可作 `reference_only`，不是今日診斷。

## 下一棒

請 Radar/Data 補 bounded source/materialization：
`TASK-RADAR-DATA-VNEXT-ADHOC-20260708-EOD-VNEXT-SIGNAL-SNAPSHOT-SOURCE-FILL-001`

需要補齊：
1. 2026-07-08 official EOD OHLC/成交金額 source for TWSE/TPEx common stocks、0050、00631L。
2. 2026-07-08 Layer0 compact active universe / Layer4 primary80 snapshot inputs。
3. 0050 MA60、20D/40D return、BIAS fields。
4. exact consensus trigger source variants for 2026-07-08。
5. route_support quant score components for primary80。
6. RS20 top3 risk-tiebreak required fields。

## Flags

- formal_model_changed=false
- trade_decision_changed=false
- active_in_trade_decision=false
- report_changed=false
- portfolio_replay_executed=false
- ready_for_strategy_replay=false
- ready_for_formal=false
- not_live_rule=true
- forward_returns_live_rule_usage=false
"""
    (OUT / "final_summary_zh.md").write_text(summary, encoding="utf-8")

    manifest = {
        "output_dir": str(OUT),
        "artifacts": [
            "adhoc_20260708_eod_c2_vs_rs20_top1_signal_snapshot.json",
            "adhoc_20260708_eod_c2_vs_rs20_top1_signal_snapshot_zh.csv",
            "adhoc_20260708_eod_c2_vs_rs20_top1_blocked_proxy_audit.csv",
            "final_summary_zh.md",
        ],
        "diagnostic_only": True,
        "not_live_trade_decision": True,
        "created_at": snapshot["created_at"],
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
