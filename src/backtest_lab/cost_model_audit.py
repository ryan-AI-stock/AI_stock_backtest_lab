from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from backtest_lab.costs import COST_MODEL_VERSION, TaiwanCostModel, cost_model_metadata


DEFAULT_OUTPUT_DIR = "outputs/cost_model_audit_20260701"
TASK_ID = "TASK-BACKTEST-CORE-COST-MODEL-AUDIT-20260701"


def run_cost_model_audit(*, output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> Path:
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

    log("build_cost_coverage", "started")
    coverage = _current_cost_coverage()
    missing = _missing_cost_fields(coverage)
    manifest = _manifest(coverage, missing)

    coverage.to_csv(output / "current_cost_coverage.csv", index=False, encoding="utf-8-sig")
    missing.to_csv(output / "missing_cost_fields.csv", index=False, encoding="utf-8-sig")
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "final_summary_zh.md").write_text(_summary_zh(manifest, coverage, missing), encoding="utf-8")
    pd.DataFrame([{"status": "completed", "output_dir": str(output.resolve())}]).to_csv(
        output / "completed.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(columns=["step", "error"]).to_csv(output / "failed.csv", index=False, encoding="utf-8-sig")
    log("completed", "completed", str(output.resolve()))
    (output / "current_step.txt").write_text("completed\n", encoding="utf-8")
    return output


def _current_cost_coverage() -> pd.DataFrame:
    rows: list[dict[str, Any]] = [
        {
            "scope": "taiwan_cost_model",
            "source_path": "src/backtest_lab/costs.py",
            "cost_model_version": COST_MODEL_VERSION,
            "buy_fee_included": True,
            "sell_fee_included": True,
            "securities_transaction_tax_included": True,
            "etf_stock_tax_split": True,
            "cost_deducted_from_equity_or_cash": True,
            "cost_fields": "buy_fee;sell_fee;securities_transaction_tax;total_transaction_cost",
            "coverage_status": "complete",
            "notes_zh": "標準台股成本模型：買賣手續費 0.1425%，最低 20 元；賣出證交稅 ETF 0.1%、個股 0.3%。",
        },
        {
            "scope": "current_formal_next_day_replay",
            "source_path": "src/backtest_lab/execution_layer_next_day_ab_pool1_pool2_formal.py",
            "cost_model_version": COST_MODEL_VERSION,
            "buy_fee_included": True,
            "sell_fee_included": True,
            "securities_transaction_tax_included": True,
            "etf_stock_tax_split": True,
            "cost_deducted_from_equity_or_cash": True,
            "cost_fields": "transaction_cost;buy_fee;sell_fee;securities_transaction_tax;total_transaction_cost;cost_model_version",
            "coverage_status": "complete_after_current_fix",
            "notes_zh": "正式 next-day execution ledger 的 trade rows 會拆出手續費與證交稅；交易成本已扣入 cash/equity。",
        },
        {
            "scope": "remove_cap_validation_same_day_and_next_day",
            "source_path": "src/backtest_lab/remove_cap_next_day_validation.py;src/backtest_lab/pool1_pool2_veto_cap_downweight.py",
            "cost_model_version": COST_MODEL_VERSION,
            "buy_fee_included": True,
            "sell_fee_included": True,
            "securities_transaction_tax_included": True,
            "etf_stock_tax_split": True,
            "cost_deducted_from_equity_or_cash": True,
            "cost_fields": "transaction_cost;buy_fee;sell_fee;securities_transaction_tax;total_transaction_cost;cost_model_version",
            "coverage_status": "complete_after_current_fix",
            "notes_zh": "remove-cap apples-to-apples 使用的 same-day / next-day trade ledger 未來輸出皆有拆分成本欄位。",
        },
        {
            "scope": "stock_pool_observation_daily_report",
            "source_path": "src/backtest_lab/stock_pool_observation.py",
            "cost_model_version": COST_MODEL_VERSION,
            "buy_fee_included": False,
            "sell_fee_included": False,
            "securities_transaction_tax_included": False,
            "etf_stock_tax_split": True,
            "cost_deducted_from_equity_or_cash": "not_applicable",
            "cost_fields": "cost_model_boundary only",
            "coverage_status": "not_applicable_no_profit_or_pnl_fields",
            "notes_zh": "每日股票池觀察報告第一層是隔天觀察標的，不是損益對帳單；已補成本口徑說明，避免把未顯示的收益誤解成已扣成本損益。",
        },
        {
            "scope": "legacy_three_pool_formal_daily_replay",
            "source_path": "src/backtest_lab/stock_pool_formal_daily_replay.py",
            "cost_model_version": COST_MODEL_VERSION,
            "buy_fee_included": True,
            "sell_fee_included": True,
            "securities_transaction_tax_included": True,
            "etf_stock_tax_split": True,
            "cost_deducted_from_equity_or_cash": True,
            "cost_fields": "daily transaction_cost total only; no trade ledger split",
            "coverage_status": "partial_legacy_total_cost_only",
            "notes_zh": "舊三池 legacy daily replay 有扣總交易成本，但不輸出 buy_fee/sell_fee/tax 拆分；不屬目前正式日報主線。",
        },
        {
            "scope": "portfolio_dashboard_app",
            "source_path": "src/backtest_lab/portfolio_dashboard.py;src/backtest_lab/portfolio_store.py",
            "cost_model_version": COST_MODEL_VERSION,
            "buy_fee_included": True,
            "sell_fee_included": True,
            "securities_transaction_tax_included": True,
            "etf_stock_tax_split": True,
            "cost_deducted_from_equity_or_cash": True,
            "cost_fields": "estimated_exit_costs_twd;record_trade costs",
            "coverage_status": "covered_for_manual_portfolio_tools",
            "notes_zh": "Portfolio app 估算出場成本與記錄交易時使用同一 TaiwanCostModel。",
        },
    ]
    return pd.DataFrame(rows)


def _missing_cost_fields(coverage: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for item in coverage.to_dict(orient="records"):
        if str(item["coverage_status"]).startswith("partial"):
            rows.append(
                {
                    "scope": item["scope"],
                    "missing_field_or_boundary": "buy_fee/sell_fee/securities_transaction_tax split in legacy output",
                    "blocks_current_formal_report": False,
                    "minimum_fix": "Rerun or modernize only if legacy three-pool output is reused as a formal performance report.",
                }
            )
        if item["coverage_status"] == "not_applicable_no_profit_or_pnl_fields":
            rows.append(
                {
                    "scope": item["scope"],
                    "missing_field_or_boundary": "profit/pnl fields are not present in daily observation report",
                    "blocks_current_formal_report": False,
                    "minimum_fix": "Keep wording clear: daily report is an observation target report, not a YuanTa APP PnL reconciliation.",
                }
            )
    rows.append(
        {
            "scope": "historical_outputs_before_current_fix",
            "missing_field_or_boundary": "previously generated CSV/manifest artifacts may not include split cost fields",
            "blocks_current_formal_report": False,
            "minimum_fix": "Rerun the relevant runner if those historical artifacts are used in a video/report as formal performance evidence.",
        }
    )
    return pd.DataFrame(rows)


def _manifest(coverage: pd.DataFrame, missing: pd.DataFrame) -> dict[str, Any]:
    model = TaiwanCostModel()
    return {
        "task_id": TASK_ID,
        "status": "completed_cost_model_audit",
        "formal_model_changed": False,
        "trade_decision_changed": False,
        "selector_changed": False,
        "cost_model_version": COST_MODEL_VERSION,
        "cost_model": cost_model_metadata(model),
        "buy_fee_rate_included": model.broker_fee_rate,
        "sell_fee_rate_included": model.broker_fee_rate,
        "stock_sell_tax_rate_included": model.stock_sell_tax_rate,
        "etf_sell_tax_rate_included": model.etf_sell_tax_rate,
        "minimum_fee_twd": model.minimum_fee_twd,
        "coverage_rows": int(len(coverage)),
        "missing_or_boundary_rows": int(len(missing)),
        "current_formal_next_day_cost_complete": True,
        "daily_report_has_profit_or_pnl_fields": False,
        "daily_report_cost_boundary_added": True,
        "yuanta_actual_discount_known": False,
        "yuanta_actual_discount_note": "元大實際折扣未知；目前用標準未折扣手續費 0.1425%。",
        "outputs": {
            "current_cost_coverage": "current_cost_coverage.csv",
            "missing_cost_fields": "missing_cost_fields.csv",
            "summary": "final_summary_zh.md",
        },
    }


def _summary_zh(manifest: dict[str, Any], coverage: pd.DataFrame, missing: pd.DataFrame) -> str:
    coverage_table = _markdown_table(coverage[["scope", "coverage_status", "notes_zh"]])
    missing_table = _markdown_table(missing)
    return "\n".join(
        [
            "# 交易成本模型稽核",
            "",
            "## 結論",
            "",
            "- Repo 已有台股成本模型，手續費與證交稅都有納入：買賣手續費 0.1425%，最低 20 元；賣出證交稅 ETF 0.1%、個股 0.3%。",
            "- current formal next-day replay / remove-cap validation 未來輸出會含 buy_fee、sell_fee、securities_transaction_tax、total_transaction_cost 與 cost_model_version。",
            "- 每日 Stock Pool Observation 是隔天觀察標的報告，沒有收益/損益欄位；本次補上成本口徑文字，避免誤讀成元大 APP 未實現損益對帳。",
            "- 舊三池 legacy replay 有扣總交易成本，但只保留 transaction_cost 總額，不拆欄位；若未來重新拿舊輸出作正式績效證據，需重跑或補欄位。",
            "",
            "## 目前成本模型",
            "",
            f"- cost_model_version: `{manifest['cost_model_version']}`",
            f"- broker_fee_rate: `{manifest['buy_fee_rate_included']}`",
            f"- stock_sell_tax_rate: `{manifest['stock_sell_tax_rate_included']}`",
            f"- etf_sell_tax_rate: `{manifest['etf_sell_tax_rate_included']}`",
            "- 元大實際折扣未知；目前採標準未折扣手續費。",
            "",
            "## Coverage",
            "",
            coverage_table,
            "",
            "## Remaining Boundaries",
            "",
            missing_table,
        ]
    )


def _markdown_table(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)
    lines = [
        "| " + " | ".join(str(column) for column in columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in frame.to_dict(orient="records"):
        values = []
        for column in columns:
            value = str(row.get(column, ""))
            values.append(value.replace("|", "/").replace("\n", " "))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit trading-cost coverage for BACKTEST_LAB formal report/replay outputs.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    output = run_cost_model_audit(output_dir=args.output_dir)
    print(f"OUTPUT_DIR={output.resolve()}")


if __name__ == "__main__":
    main()
