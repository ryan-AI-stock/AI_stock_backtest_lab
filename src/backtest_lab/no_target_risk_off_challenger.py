from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from backtest_lab.execution_layer_next_day_ab_pool1_pool2_formal import (
    INITIAL_CASH,
    VariantSpec as ExecutionVariantSpec,
    _simulate_variant as simulate_next_day_variant,
)
from backtest_lab.formal_model_contract import FORMAL_MODEL_ROUTE, FORMAL_MODEL_TARGET
from backtest_lab.formal_vs_pool1_only_validation import (
    _attach_trade_costs,
    _monthly_performance,
    _normalize_next_day_daily,
    _normalize_next_day_trades,
    _period_performance,
    _target_panel_to_execution_frame,
    _trade_cost_summary,
    _worst_month,
)
from backtest_lab.pool1_pool2_veto_cap_downweight import (
    VariantSpec as DecisionVariantSpec,
    _build_target_weights,
    _load_prices,
    _needed_tickers,
)


DEFAULT_FORMAL_REPLAY_DIR = "outputs/stock_pool_formal_daily_replay_pit_pool2_daily_final_combined_20260624"
DEFAULT_PRICE_CACHE_DIR = "backtest_cache/stock_pool_triad_v1_corrected"
DEFAULT_OUTPUT_DIR = "outputs/no_target_risk_off_challenger_contract_20260702"


@dataclass(frozen=True)
class NoTargetRiskOffVariant:
    variant_id: str
    no_formal_target_policy: str
    description_zh: str
    max_cash_rows: int = 0
    reduce_ratio: float = 0.5
    is_formal_baseline: bool = False


VARIANTS = (
    NoTargetRiskOffVariant(
        "baseline_hold_through",
        "hold_previous",
        "目前正式修正版：沒有新正式目標時沿用上一個已接受正式目標。",
        is_formal_baseline=True,
    ),
    NoTargetRiskOffVariant(
        "no_target_cash_all",
        "exit_to_cash",
        "顯式 challenger：沒有正式目標時空手，直到下一個正式目標出現。",
    ),
    NoTargetRiskOffVariant(
        "no_target_cash_max_3",
        "cash_max_3",
        "顯式 challenger：沒有正式目標時最多空手 3 個交易日，之後恢復前一正式目標。",
        max_cash_rows=3,
    ),
    NoTargetRiskOffVariant(
        "no_target_reduce_exposure_50",
        "reduce_exposure_50",
        "顯式 challenger：沒有正式目標時把前一正式目標曝險降到 50%，不等於正式交易規則。",
        reduce_ratio=0.5,
    ),
)


def run_no_target_risk_off_challenger(
    *,
    formal_replay_dir: str | Path = DEFAULT_FORMAL_REPLAY_DIR,
    price_cache_dir: str | Path = DEFAULT_PRICE_CACHE_DIR,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    initial_cash: float = INITIAL_CASH,
) -> Path:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    run_log: list[dict[str, str]] = []

    def log(step: str, status: str, detail: str = "") -> None:
        run_log.append(
            {
                "timestamp": pd.Timestamp.now(tz="Asia/Taipei").strftime("%Y-%m-%d %H:%M:%S%z"),
                "step": step,
                "status": status,
                "detail": detail,
            }
        )
        pd.DataFrame(run_log).to_csv(output / "run_log.csv", index=False, encoding="utf-8-sig")
        (output / "current_step.txt").write_text(step, encoding="utf-8")

    try:
        replay = Path(formal_replay_dir)
        log("load_inputs", "started", str(replay))
        decision = pd.read_csv(replay / "formal_three_pool_decision_panel.csv").fillna("")
        _validate_decision(decision)
        prices = _load_prices(_needed_tickers(decision), Path(price_cache_dir))
        if not prices:
            raise ValueError("no prices loaded for no-target risk-off challenger")

        daily_frames: list[pd.DataFrame] = []
        trade_frames: list[pd.DataFrame] = []
        blocked_frames: list[pd.DataFrame] = []
        no_target_frames: list[pd.DataFrame] = []

        log("simulate_variants", "started", f"variants={len(VARIANTS)}")
        for variant in VARIANTS:
            decision_spec = DecisionVariantSpec(
                variant.variant_id,
                "confirmation",
                confirmation_days=1,
                no_formal_target_policy="hold_previous" if variant.is_formal_baseline else "exit_to_cash",
            )
            target_panel = _build_target_weights(decision, decision_spec)
            no_target_frames.append(_no_target_event_rows(target_panel, variant))
            frame = _target_panel_to_execution_frame(target_panel)
            daily, trades, _events, blocked = simulate_next_day_variant(
                frame,
                prices,
                ExecutionVariantSpec(
                    variant.variant_id,
                    1,
                    no_formal_target_policy=variant.no_formal_target_policy,
                    no_formal_target_max_cash_rows=variant.max_cash_rows,
                    no_formal_target_reduce_ratio=variant.reduce_ratio,
                    description=variant.description_zh,
                ),
                initial_cash,
            )
            daily_frames.append(_normalize_next_day_daily(daily, variant.variant_id))
            trade_frames.append(_normalize_next_day_trades(trades, variant.variant_id))
            if not blocked.empty:
                blocked_frames.append(blocked.assign(source_variant=variant.variant_id))

        daily_ledger = pd.concat(daily_frames, ignore_index=True)
        trade_ledger = pd.concat(trade_frames, ignore_index=True) if trade_frames else pd.DataFrame()
        blocked = pd.concat(blocked_frames, ignore_index=True) if blocked_frames else pd.DataFrame()
        no_target_events = pd.concat(no_target_frames, ignore_index=True) if no_target_frames else pd.DataFrame()

        log("build_reports", "started", "")
        monthly = _monthly_performance(daily_ledger)
        worst = _worst_month(monthly)
        costs = _trade_cost_summary(trade_ledger)
        performance = _attach_trade_costs(_period_performance(daily_ledger), costs)
        variant_contract = _variant_contract()
        no_target_summary = _no_target_summary(no_target_events, daily_ledger)

        log("write_outputs", "started", "")
        variant_contract.to_csv(output / "variant_contract.csv", index=False, encoding="utf-8-sig")
        daily_ledger.to_csv(output / "daily_equity_by_variant.csv", index=False, encoding="utf-8-sig")
        trade_ledger.to_csv(output / "trade_ledger_by_variant.csv", index=False, encoding="utf-8-sig")
        blocked.to_csv(output / "blocked_execution_events.csv", index=False, encoding="utf-8-sig")
        performance.to_csv(output / "performance_by_variant.csv", index=False, encoding="utf-8-sig")
        monthly.to_csv(output / "monthly_performance.csv", index=False, encoding="utf-8-sig")
        worst.to_csv(output / "worst_month.csv", index=False, encoding="utf-8-sig")
        costs.to_csv(output / "trade_cost_summary.csv", index=False, encoding="utf-8-sig")
        no_target_events.to_csv(output / "no_target_event_panel.csv", index=False, encoding="utf-8-sig")
        no_target_summary.to_csv(output / "no_target_event_summary.csv", index=False, encoding="utf-8-sig")
        (output / "no_target_risk_off_challenger_contract_zh.md").write_text(
            _contract_markdown(variant_contract, performance, no_target_summary),
            encoding="utf-8",
        )
        manifest = {
            "schema_version": 1,
            "task_id": "TASK-BACKTEST-CORE-NO-TARGET-RISK-OFF-CHALLENGER-CONTRACT-20260702",
            "status": "completed_challenger_contract",
            "formal_model_target": FORMAL_MODEL_TARGET,
            "formal_model_route": FORMAL_MODEL_ROUTE,
            "formal_default_no_formal_target_policy": "hold_previous",
            "bug_cash_mapping_used_as_baseline": False,
            "explicit_no_target_risk_off_challenger": True,
            "formal_model_changed": False,
            "trade_decision_changed": False,
            "formal_execution_layer_activated": False,
            "active_in_trade_decision": False,
            "uses_forward_return_as_rule": False,
            "formal_replay_dir": str(replay),
            "price_cache_dir": str(price_cache_dir),
            "start_date": str(decision["date"].iloc[0]) if not decision.empty else "",
            "latest_complete_common_date": str(decision["date"].iloc[-1]) if not decision.empty else "",
            "outputs": {
                "variant_contract": "variant_contract.csv",
                "daily_ledger": "daily_equity_by_variant.csv",
                "trade_ledger": "trade_ledger_by_variant.csv",
                "performance": "performance_by_variant.csv",
                "no_target_events": "no_target_event_panel.csv",
                "summary": "no_target_risk_off_challenger_contract_zh.md",
            },
        }
        (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        pd.DataFrame([{"status": "completed", "output_dir": str(output.resolve())}]).to_csv(output / "completed.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame(columns=["step", "error"]).to_csv(output / "failed.csv", index=False, encoding="utf-8-sig")
        log("completed", "completed", str(output.resolve()))
        (output / "current_step.txt").write_text("completed\n", encoding="utf-8")
        return output
    except Exception as exc:
        pd.DataFrame([{"step": "run_no_target_risk_off_challenger", "error": str(exc)}]).to_csv(output / "failed.csv", index=False, encoding="utf-8-sig")
        log("failed", "failed", str(exc))
        raise


def _validate_decision(decision: pd.DataFrame) -> None:
    required = {"period", "date", "pool1_vote", "pool2_vote"}
    missing = sorted(required - set(decision.columns))
    if missing:
        raise ValueError(f"decision panel missing columns: {missing}")


def _variant_contract() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "variant_id": variant.variant_id,
                "execution_basis": "next_day",
                "no_formal_target_policy": variant.no_formal_target_policy,
                "max_cash_rows": variant.max_cash_rows,
                "reduce_ratio": variant.reduce_ratio,
                "is_formal_baseline": variant.is_formal_baseline,
                "active_in_trade_decision": False,
                "formal_model_changed": False,
                "trade_decision_changed": False,
                "description_zh": variant.description_zh,
            }
            for variant in VARIANTS
        ]
    )


def _no_target_event_rows(target_panel: pd.DataFrame, variant: NoTargetRiskOffVariant) -> pd.DataFrame:
    rows = []
    for item in target_panel.to_dict(orient="records"):
        if str(item.get("target_weights") or "{}").strip() == "{}":
            rows.append(
                {
                    "variant_id": variant.variant_id,
                    "date": item.get("date", ""),
                    "period": item.get("period", ""),
                    "event_reason": item.get("event_reason", ""),
                    "no_formal_target_policy": variant.no_formal_target_policy,
                    "active_in_trade_decision": False,
                }
            )
    return pd.DataFrame(rows)


def _no_target_summary(events: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for variant, group in daily.groupby("variant_id", dropna=False):
        event_count = 0 if events.empty else int(events[events["variant_id"].eq(variant)]["date"].nunique())
        cash_days = int(group["top_holding"].astype(str).eq("cash").sum())
        rows.append(
            {
                "variant_id": variant,
                "no_target_event_days": event_count,
                "cash_top_holding_days": cash_days,
                "active_in_trade_decision": False,
            }
        )
    return pd.DataFrame(rows)


def _contract_markdown(contract: pd.DataFrame, performance: pd.DataFrame, no_target: pd.DataFrame) -> str:
    lines = [
        "# No-target risk-off challenger contract",
        "",
        "本輸出只建立 challenger contract，不啟用正式模型、不改每日正式交易結論。",
        "",
        "## Contract",
        "",
        "- 正式預設：no formal target 時沿用上一個已接受正式目標，不自動清倉。",
        "- `bug_cash_mapping`：舊隱含行為，不可作正式 baseline。",
        "- `explicit_no_target_risk_off_challenger`：只有本 runner 指定 variant 時才可進入 cash / max-3 cash / reduce exposure。",
        "",
        "## Variants",
    ]
    for row in contract.to_dict(orient="records"):
        lines.append(f"- `{row['variant_id']}`：{row['description_zh']}")
    lines.extend(["", "## Main performance rows", ""])
    main = performance[performance["period_label"].eq("2024_now_main")] if "period_label" in performance.columns else pd.DataFrame()
    for row in main.to_dict(orient="records"):
        lines.append(
            f"- `{row['variant_id']}`：return {row.get('return_pct')}%，MDD {row.get('max_drawdown_pct')}%，trade rows {row.get('trade_rows', 0)}。"
        )
    lines.extend(["", "## No-target exposure summary", ""])
    for row in no_target.to_dict(orient="records"):
        lines.append(
            f"- `{row['variant_id']}`：no-target days {row['no_target_event_days']}，cash top holding days {row['cash_top_holding_days']}。"
        )
    lines.extend(["", "## Boundary", "", "- formal_model_changed=false", "- trade_decision_changed=false", "- active_in_trade_decision=false"])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build no-target risk-off challenger contract.")
    parser.add_argument("--formal-replay-dir", default=DEFAULT_FORMAL_REPLAY_DIR)
    parser.add_argument("--price-cache-dir", default=DEFAULT_PRICE_CACHE_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    run_no_target_risk_off_challenger(
        formal_replay_dir=args.formal_replay_dir,
        price_cache_dir=args.price_cache_dir,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
