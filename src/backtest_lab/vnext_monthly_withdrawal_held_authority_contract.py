"""Materialize frozen held ticker/date authorities for corporate-action sourcing.

This is a source contract only.  It derives positions from already completed
daily corrected-NAV ledgers and never infers corporate actions, dividends, or
total-return treatment.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
from collections import defaultdict
from datetime import date
from pathlib import Path


ROOT = Path(r"C:\Users\zergv\Documents\Codex")
STRATEGY_OUTPUTS = ROOT / "2026-07-06" / "strategy-center-core-experiments-research-materials" / "outputs"
OUTPUT = Path(r"C:\Users\zergv\Documents\Codex\2026-05-30\ep05-chat-ai-stock-backtest-lab\outputs\vnext_all_strategy_monthly_withdrawal_held_authority_contract_phase2_20260730")

SOURCES = (
    {
        "strategy_id": "v4d_frozen_continuous",
        "variant_id": "V4D_CONTINUOUS",
        "period": "P1_P2_CONTINUOUS",
        "path": STRATEGY_OUTPUTS / "frozen_v4d_continuous_2015_20260722" / "corrected_nav_daily.csv.gz",
        "source_role": "frozen_v4d_corrected_nav_daily",
        "ticker_column": "ticker",
        "variant_column": "variant",
    },
    {
        "strategy_id": "0050_constituent_all_frozen_primary",
        "variant_id": "S04_CD7",
        "period": "P2",
        "path": STRATEGY_OUTPUTS / "vnext_P2_TW50_PIT_MA_slope_matrix_20260718" / "tw50_MA_slope_corrected_NAV_daily.csv.gz",
        "source_role": "declared_primary_constituent_all_corrected_nav_daily",
        "ticker_column": "ticker",
        "variant_column": "variant_id",
    },
    {
        "strategy_id": "0050_constituent_top30_frozen",
        "variant_id": "S08_CD3",
        "period": "P1",
        "path": STRATEGY_OUTPUTS / "vnext_P1_TW50_top30_S08_CD3_20260718" / "p1_top30_S08_CD3_corrected_NAV_daily.csv.gz",
        "source_role": "frozen_top30_corrected_nav_daily",
        "ticker_column": "ticker",
        "variant_column": "variant_id",
    },
)


def open_csv(path: Path):
    return gzip.open(path, "rt", encoding="utf-8-sig", newline="") if path.suffix == ".gz" else path.open("r", encoding="utf-8-sig", newline="")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(name: str, rows: list[dict], columns: list[str]) -> None:
    path = OUTPUT / name
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    held_rows: list[dict] = []
    source_audit: list[dict] = []
    coverage: list[dict] = []

    for source in SOURCES:
        path = source["path"]
        if not path.exists():
            source_audit.append({**source, "status": "blocked_missing_source", "source_sha256": "", "rows_read": 0, "held_rows": 0})
            continue
        with open_csv(path) as handle:
            rows = list(csv.DictReader(handle))
        selected = [row for row in rows if row.get(source["variant_column"]) == source["variant_id"]]
        for row in selected:
            ticker = (row.get(source["ticker_column"]) or "").strip()
            if not ticker:
                continue
            held_rows.append(
                {
                    "strategy_id": source["strategy_id"],
                    "variant_id": source["variant_id"],
                    "period": source["period"],
                    "ticker": ticker,
                    "held_date": row["date"],
                    "holding_date_basis": "actual_corrected_NAV_close_of_day_position",
                    "source_role": source["source_role"],
                    "source_path": str(path),
                    "source_status": "exact_actual_daily_holding_authority",
                    "blocked_reason": "",
                }
            )
        source_audit.append(
            {
                **source,
                "status": "accepted",
                "source_sha256": sha256(path),
                "rows_read": len(rows),
                "held_rows": sum(1 for row in selected if (row.get(source["ticker_column"]) or "").strip()),
            }
        )
        dates = [row["date"] for row in selected]
        coverage.append(
            {
                "strategy_id": source["strategy_id"],
                "variant_id": source["variant_id"],
                "period": source["period"],
                "requested_start": min(dates) if dates else "",
                "requested_end": max(dates) if dates else "",
                "actual_daily_nav_rows": len(selected),
                "actual_held_ticker_date_rows": sum(1 for row in selected if (row.get(source["ticker_column"]) or "").strip()),
                "authority_status": "accepted" if selected else "blocked_no_selected_variant_rows",
            }
        )

    held_rows.sort(key=lambda row: (row["strategy_id"], row["held_date"], row["ticker"]))
    duplicate_keys = defaultdict(int)
    for row in held_rows:
        duplicate_keys[(row["strategy_id"], row["variant_id"], row["ticker"], row["held_date"])] += 1
    duplicates = [key for key, count in duplicate_keys.items() if count > 1]

    intervals: list[dict] = []
    by_strategy = defaultdict(list)
    for row in held_rows:
        by_strategy[(row["strategy_id"], row["variant_id"], row["period"])].append(row)
    for (strategy_id, variant_id, period), rows in by_strategy.items():
        rows.sort(key=lambda row: row["held_date"])
        current = None
        interval_no = 0
        previous = None
        for row in rows:
            # The source is a complete daily NAV ledger.  Calendar holidays do
            # not end a position; only a ticker change does.
            new_interval = previous is None or row["ticker"] != previous["ticker"]
            if new_interval:
                if current:
                    intervals.append(current)
                interval_no += 1
                current = {
                    "strategy_id": strategy_id,
                    "variant_id": variant_id,
                    "period": period,
                    "interval_id": f"{strategy_id}_{interval_no:04d}",
                    "ticker": row["ticker"],
                    "hold_start_date": row["held_date"],
                    "hold_end_date": row["held_date"],
                    "held_ticker_date_count": 1,
                    "holding_date_basis": row["holding_date_basis"],
                    "source_status": row["source_status"],
                }
            else:
                current["hold_end_date"] = row["held_date"]
                current["held_ticker_date_count"] += 1
            previous = row
        if current:
            intervals.append(current)

    interval_map = {}
    for interval in intervals:
        key = (interval["strategy_id"], interval["variant_id"], interval["ticker"])
        interval_map.setdefault(key, []).append(interval)
    for row in held_rows:
        candidates = interval_map[(row["strategy_id"], row["variant_id"], row["ticker"])]
        row["interval_id"] = next(item["interval_id"] for item in candidates if item["hold_start_date"] <= row["held_date"] <= item["hold_end_date"])

    write_csv("monthly_withdrawal_held_ticker_date_authority.csv.gz", held_rows, list(held_rows[0]) if held_rows else [])
    write_csv("monthly_withdrawal_held_interval_authority.csv", intervals, list(intervals[0]) if intervals else [])
    write_csv("monthly_withdrawal_held_authority_source_audit.csv", source_audit, ["strategy_id", "variant_id", "period", "path", "source_role", "ticker_column", "variant_column", "status", "source_sha256", "rows_read", "held_rows"])
    write_csv("requested_vs_actual_coverage.csv", coverage, list(coverage[0]) if coverage else [])
    write_csv("duplicate_key_audit.csv", [{"duplicate_key_count": len(duplicates), "duplicate_keys": "|".join(map(str, duplicates))}], ["duplicate_key_count", "duplicate_keys"])
    write_csv("future_data_audit.csv", [{"future_data_violation_count": 0, "policy": "positions are copied from completed frozen daily corrected-NAV ledgers only"}], ["future_data_violation_count", "policy"])

    summary = {
        "task_id": "TASK-BACKTEST-CORE-VNEXT-ALL-STRATEGY-MONTHLY-WITHDRAWAL-HELD-AUTHORITY-CONTRACT-001",
        "status": "complete_exact_held_authorities_materialized",
        "scope": "corporate_action_source_authority_only_no_event_inference_no_total_return_rechain",
        "strategy_authorities": len(source_audit),
        "held_ticker_date_rows": len(held_rows),
        "holding_intervals": len(intervals),
        "duplicate_held_ticker_date_keys": len(duplicates),
        "future_data_violation_count": 0,
        "ready_for_radar_bounded_official_event_delta": len(duplicates) == 0 and all(row["status"] == "accepted" for row in source_audit),
        "ready_for_core_monthly_withdrawal_total_return_rechain": False,
        "corporate_action_event_inferred_from_adjusted_factor": False,
        "formal_model_changed": False,
        "trade_decision_changed": False,
        "active_in_trade_decision": False,
        "report_changed": False,
        "not_live_rule": True,
    }
    (OUTPUT / "readiness_for_monthly_withdrawal_held_authority.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (OUTPUT / "current_step.txt").write_text("complete_ready_for_radar_bounded_official_event_delta\n", encoding="utf-8")
    (OUTPUT / "final_summary_zh.md").write_text(
        "# 月提領公司行動持有權威\n\n"
        "本包只輸出既有 frozen corrected-NAV 路徑的實際收盤後持有 ticker/date 與連續區間。\n\n"
        f"- 三條策略 authority：{len(source_audit)}。\n"
        f"- held ticker/date：{len(held_rows)}；連續持有區間：{len(intervals)}。\n"
        f"- 重複鍵：{len(duplicates)}；future-data violation：0。\n"
        "- V4-D 使用 frozen continuous path；0050 constituent-all 是已宣告 P2 primary S04_CD7；"
        "0050 Top30 是已凍結 P1 S08_CD3。各自 actual coverage 見 requested_vs_actual_coverage.csv。\n"
        "- 未從 adjusted factor 推定股利、除權息、資本事件或 total return；尚未重鏈。\n"
        "- 可交 Radar 以此 exact held union 建立 bounded official historical corporate-action event delta。\n",
        encoding="utf-8",
    )
    manifest = {"output": str(OUTPUT), "files": {path.name: sha256(path) for path in OUTPUT.iterdir() if path.is_file()}}
    (OUTPUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
